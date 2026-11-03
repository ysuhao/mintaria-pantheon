"""Atlas 核心行为验证。用生成的测试信号经 mint_probe 渲染，再测量输出。

    python plugins/comp/tests/verify_atlas.py [--build D:/mint-build/comp] [--only static,attack]

任何一项不达标时退出码非 0。所有数字打印到标准输出，NOTES.md 里的结果就抄自这里。
"""
import argparse
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

RATE = 48000
KNEE = 12.0
CEILING = -0.3
OPTO, ELECTRO = 0.0, 1.0
WARM, SMOOTH = 0.0, 1.0
MANUAL, AUTO = 0.0, 1.0


def skew_norm(v, low, high, centre):
    skew = math.log(0.5) / math.log((centre - low) / (high - low))
    return ((v - low) / (high - low)) ** skew


def settings(threshold=-18.0, ratio=3.0, attack=8.0, release=120.0, makeup=0.0, auto=MANUAL, behavior=ELECTRO,
             character=SMOOTH, lookahead=0.0):
    return {
        "Threshold": (threshold + 60.0) / 60.0,
        "Ratio": skew_norm(ratio, 1.0, 20.0, 4.0),
        "Attack": skew_norm(attack, 0.5, 200.0, 15.0),
        "Release": skew_norm(release, 5.0, 3000.0, 150.0),
        "Makeup": (makeup + 12.0) / 36.0,
        "Auto Release": auto,
        "Behavior": behavior,
        "Character": character,
        "Look-ahead": lookahead / 10.0,
    }


class Bench:
    def __init__(self, build, work):
        self.probe = Path(build) / "tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
        self.plugin = Path(build) / "plugins/comp/mint_comp_artefacts/Release/VST3/Mintaria Atlas.vst3"
        self.work = Path(work)
        self.counter = 0
        self.failures = []

    def render(self, x, params, rate=RATE, block=256):
        """渲染单声道信号（两个声道相同），返回第一声道、第二声道和上报的延迟。"""
        self.counter += 1
        src = self.work / f"in{self.counter}.wav"
        dst = self.work / f"out{self.counter}.wav"
        wavfile.write(src, rate, np.stack([x, x], axis=1).astype(np.float32))
        cmd = [str(self.probe), "render", str(self.plugin), str(src), str(dst), "--block", str(block)]
        for name, value in params.items():
            cmd += ["--set", f"{name}={value:.6f}"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if done.returncode != 0:
            raise RuntimeError(done.stdout + done.stderr)
        latency = int(done.stdout.strip().split()[-1])
        _, y = wavfile.read(dst)
        src.unlink()
        dst.unlink()
        return y[:, 0].astype(np.float64) / 2147483648.0, y[:, 1].astype(np.float64) / 2147483648.0, latency

    def check(self, condition, label):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            self.failures.append(label)


def db(v):
    return 20.0 * np.log10(np.maximum(np.abs(v), 1e-12))


def amp(level_db):
    return 10.0 ** (level_db / 20.0)


def sine(freq, seconds, level_db, rate=RATE):
    n = int(seconds * rate)
    return amp(level_db) * np.sin(2.0 * np.pi * freq * np.arange(n) / rate)


def square(seconds, level_db, rate=RATE, period=48):
    n = int(seconds * rate)
    return amp(level_db) * np.where((np.arange(n) % period) < period // 2, 1.0, -1.0)


def tone_level(y, freq, rate=RATE):
    """加 Hann 窗后该频率的峰值幅度。窗内周期数不用是整数，不会把基频漏进谐波读数。"""
    window = np.hanning(len(y))
    t = np.arange(len(y)) / rate
    return 2.0 * abs(np.sum(y * window * np.exp(-2j * np.pi * freq * t))) / np.sum(window)


def ideal_reduction(level, threshold, ratio):
    over = level - threshold
    slope = 1.0 - 1.0 / ratio
    if over <= -KNEE / 2:
        return 0.0
    if over >= KNEE / 2:
        return slope * over
    return slope * (over + KNEE / 2) ** 2 / (2 * KNEE)


def reduction_trace(x, y):
    """输出和输入逐点相除得到衰减量（dB）。只用于 Smooth 档，它是纯增益。"""
    return db(x) - db(y)


def first_below(trace, start, value):
    hits = np.nonzero(trace[start:] <= value)[0]
    return None if len(hits) == 0 else int(hits[0])


def first_above(trace, start, value):
    hits = np.nonzero(trace[start:] >= value)[0]
    return None if len(hits) == 0 else int(hits[0])


def test_static(bench):
    print("静态曲线：1 kHz 正弦阶梯，手动释放，Smooth")
    levels = np.arange(-60.0, 0.1, 3.0)
    step = 0.4

    def run(case):
        threshold, ratio = case
        x = np.concatenate([sine(1000.0, step, float(l)) for l in levels])
        y, _, _ = bench.render(x, settings(threshold, ratio, attack=1.0, release=40.0))
        worst, table = 0.0, []
        for i, level in enumerate(levels):
            segment = y[int((i + 0.5) * step * RATE):int((i + 1) * step * RATE)]
            measured = 20.0 * math.log10(tone_level(segment, 1000.0))
            expected = level - ideal_reduction(level, threshold, ratio)
            table.append((level, measured, expected))
            worst = max(worst, abs(measured - expected))
        return case, worst, table

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, [(-18.0, 3.0), (-30.0, 2.0), (-30.0, 8.0), (-12.0, 4.0), (-40.0, 20.0), (0.0, 4.0)]))

    for (threshold, ratio), worst, _ in results:
        bench.check(worst < 0.1, f"阈值 {threshold:.0f} dB 比率 {ratio:.0f}:1 最大偏差 {worst:.3f} dB (< 0.1)")


def test_attack(bench):
    print("启动时间：方波从 -40 跳到 -6 dBFS，阈值 -30 比率 4，衰减达到 63% 的时间")
    x = np.concatenate([square(0.2, -40.0), square(0.8, -6.0)])
    start = int(0.2 * RATE)

    def run(attack):
        y, _, _ = bench.render(x, settings(-30.0, 4.0, attack=attack, release=200.0))
        trace = reduction_trace(x, y)
        final = float(np.mean(trace[-2000:]))
        t = first_above(trace, start, 0.632 * final)
        return attack, final, t / RATE * 1000.0

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, [0.5, 1.0, 5.0, 20.0, 100.0]))

    for attack, final, measured in results:
        bench.check(abs(final - 18.0) < 0.1 and abs(measured - attack) < 0.1 * attack + 0.1,
                    f"设定 {attack:g} ms 实测 {measured:.2f} ms（最终衰减 {final:.2f} dB）")


def release_case(bench, release, behavior, auto=MANUAL, level_after=-55.0, ratio=2.0):
    """阈值 -30 比率 2，-6 dBFS 下衰减 12 dB，然后电平落到阈值以下。"""
    x = np.concatenate([square(0.5, -6.0), square(max(3.0, release * 0.012), level_after)])
    y, _, _ = bench.render(x, settings(-30.0, ratio, attack=1.0, release=release, behavior=behavior, auto=auto))
    return x, reduction_trace(x, y), int(0.5 * RATE)


def test_release(bench):
    print("释放时间：衰减 12 dB 回落 63%（到 4.41 dB）的时间，手动释放")

    def run(case):
        release, behavior = case
        x, trace, start = release_case(bench, release, behavior)
        t = first_below(trace, start, 12.0 * (1.0 - 0.632))
        return release, behavior, t / RATE * 1000.0

    cases = [(r, b) for r in (20.0, 50.0, 120.0, 500.0, 1500.0) for b in (OPTO, ELECTRO)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, cases))

    for release, behavior, measured in results:
        name = "Opto" if behavior == OPTO else "Electro"
        bench.check(abs(measured - release) < 0.12 * release,
                    f"{name} 设定 {release:g} ms 实测 {measured:.1f} ms")


def test_shape(bench):
    print("释放曲线：同样 120 ms 设定下，Opto 和 Electro 走完各段的时间（ms）")
    marks = [12.0, 9.0, 6.0, 4.41, 3.0, 1.0, 0.3, 0.1]
    curves = {}
    for name, behavior in (("Opto", OPTO), ("Electro", ELECTRO)):
        _, trace, start = release_case(bench, 120.0, behavior)
        times = [first_below(trace, start, m) / RATE * 1000.0 for m in marks[1:]]
        curves[name] = times
        print("    " + name.ljust(8) + "  ".join(f"到{m:g}dB {t:7.1f}" for m, t in zip(marks[1:], times)))

    opto, electro = curves["Opto"], curves["Electro"]
    # marks[1:] 下标：0=9 dB，1=6 dB，2=4.41 dB，3=3 dB，4=1 dB，5=0.3 dB，6=0.1 dB
    bench.check(opto[0] < electro[0], f"深衰减段 Opto 更快：到 9 dB {opto[0]:.1f} ms 对 {electro[0]:.1f} ms")
    tail_opto, tail_electro = opto[6] - opto[2], electro[6] - electro[2]
    bench.check(tail_opto > 2.0 * tail_electro,
                f"尾段 Opto 更慢：4.41 dB 到 0.1 dB 用 {tail_opto:.0f} ms 对 {tail_electro:.0f} ms（> 2 倍）")


def carrier_release(bench, burst_seconds, auto, behavior=ELECTRO):
    """2 kHz 正弦突发后落到 -60 dBFS 载波上，测衰减回落 63% 的时间。"""
    n_burst = int(burst_seconds * RATE)
    total = int((0.5 + burst_seconds + 2.0) * RATE)
    t = np.arange(total) / RATE
    gain = np.full(total, amp(-60.0))
    start = int(0.5 * RATE)
    gain[start:start + n_burst] = amp(-6.0)
    x = gain * np.sin(2.0 * np.pi * 2000.0 * t)
    y, _, _ = bench.render(x, settings(-30.0, 2.0, attack=1.0, release=120.0, auto=auto, behavior=behavior))
    valid = np.abs(x) > 0.3 * gain
    trace = np.where(valid, reduction_trace(x, y), np.nan)
    end = start + n_burst
    begin = float(np.nanmax(trace[end - 96:end]))
    hits = np.nonzero(np.nan_to_num(trace[end:], nan=99.0) <= begin * (1.0 - 0.632))[0]
    return begin, hits[0] / RATE * 1000.0


def test_auto(bench):
    print("自动释放：同样 120 ms 设定，短突发和长持续音之后的回落时间")
    a_short = carrier_release(bench, 0.03, AUTO)
    a_long = carrier_release(bench, 1.5, AUTO)
    m_short = carrier_release(bench, 0.03, MANUAL)
    m_long = carrier_release(bench, 1.5, MANUAL)
    print(f"    自动 30 ms 突发：起点 {a_short[0]:.1f} dB，回落 {a_short[1]:.1f} ms")
    print(f"    自动 1.5 s 持续：起点 {a_long[0]:.1f} dB，回落 {a_long[1]:.1f} ms")
    print(f"    手动 30 ms 突发：起点 {m_short[0]:.1f} dB，回落 {m_short[1]:.1f} ms")
    print(f"    手动 1.5 s 持续：起点 {m_long[0]:.1f} dB，回落 {m_long[1]:.1f} ms")
    bench.check(a_long[1] > 1.5 * a_short[1], f"自动档随素材变化：持续音 / 突发 = {a_long[1] / a_short[1]:.2f}（> 1.5）")
    bench.check(abs(m_long[1] - m_short[1]) < 0.1 * m_long[1], f"手动档不随素材变化：差 {abs(m_long[1] - m_short[1]):.1f} ms")

    print("自动档不受释放旋钮影响：旋钮 20 ms 和 1500 ms 的输出做差")
    x = np.concatenate([square(0.5, -6.0), square(3.0, -55.0)])
    def knob_difference(auto):
        low, _, _ = bench.render(x, settings(-30.0, 2.0, attack=1.0, release=20.0, auto=auto))
        high, _, _ = bench.render(x, settings(-30.0, 2.0, attack=1.0, release=1500.0, auto=auto))
        return float(np.max(np.abs(low - high)))
    auto_difference = knob_difference(AUTO)
    manual_difference = knob_difference(MANUAL)
    bench.check(auto_difference == 0.0, f"自动档逐点相同：最大差 {auto_difference:.3e}")
    bench.check(manual_difference > 0.001, f"手动档有差别（对照）：最大差 {manual_difference:.3e}")


def harmonics(y, freq, count=5):
    fundamental = tone_level(y, freq)
    return [20.0 * math.log10(max(tone_level(y, freq * k), 1e-9) / fundamental) for k in range(2, count + 1)]


def test_warm(bench):
    print("Warm 谐波：正弦稳态的各次谐波（dBc），阈值 -30，比率 4")

    def run(case):
        freq, level, character = case
        y, _, _ = bench.render(sine(freq, 1.2, level), settings(-30.0, 4.0, attack=2.0, release=120.0, character=character))
        return case, harmonics(y[int(0.6 * RATE):], freq)

    cases = [(f, l, c) for f in (110.0, 220.0, 1000.0, 3000.0) for l in (-40.0, -22.0, -14.0, -6.0) for c in (WARM, SMOOTH)]
    with ThreadPoolExecutor(4) as pool:
        results = dict(pool.map(run, cases))

    for freq in (110.0, 220.0, 1000.0, 3000.0):
        for level in (-40.0, -22.0, -14.0, -6.0):
            gr = ideal_reduction(level, -30.0, 4.0)
            warm, smooth = results[(freq, level, WARM)], results[(freq, level, SMOOTH)]
            print(f"    {freq:6.0f} Hz 输入 {level:5.0f} dB 衰减 {gr:5.1f} dB | Warm H2 {warm[0]:7.1f} H3 {warm[1]:7.1f} "
                  f"| Smooth H2 {smooth[0]:7.1f} H3 {smooth[1]:7.1f}")

    w = results
    bench.check(w[(220.0, -40.0, WARM)][0] < -90.0, f"不压缩时 Warm 无谐波：H2 {w[(220.0, -40.0, WARM)][0]:.1f} dBc")
    bench.check(w[(220.0, -6.0, WARM)][0] > w[(220.0, -22.0, WARM)][0] + 3.0, "谐波随衰减加深而增加")
    bench.check(-60.0 < w[(220.0, -6.0, WARM)][0] < -30.0, f"220 Hz 衰减 18 dB 时 H2 = {w[(220.0, -6.0, WARM)][0]:.1f} dBc（-60 到 -30）")
    bench.check(w[(220.0, -6.0, WARM)][0] > w[(220.0, -6.0, SMOOTH)][0] + 20.0, "Warm 比 Smooth 的谐波高 20 dB 以上")
    bench.check(w[(3000.0, -6.0, WARM)][0] < w[(220.0, -6.0, WARM)][0] - 6.0, "高频产生的谐波明显少于低频")


def test_burst(bench):
    print("突发：-50 dBFS 正弦上叠 0 dBFS 突发（阈值 -30，比率 4，补偿 +12）")
    bed = sine(440.0, 3.0, -50.0)
    start, length = RATE, int(0.2 * RATE)
    loud = sine(440.0, 3.0, 0.0)
    x = bed.copy()
    x[start:start + length] = loud[start:start + length]

    for attack in (0.5, 8.0):
        for character, cname in ((SMOOTH, "Smooth"), (WARM, "Warm")):
            for behavior, bname in ((OPTO, "Opto"), (ELECTRO, "Electro")):
                y, _, _ = bench.render(x, settings(-30.0, 4.0, attack=attack, release=120.0, makeup=12.0,
                                                   character=character, behavior=behavior))
                peak = float(np.max(np.abs(y)))
                bench.check(peak <= amp(CEILING) * 1.0001, f"启动 {attack:g} ms {cname} {bname} 峰值 {20 * math.log10(peak):.2f} dBFS（上限 {CEILING}）")

                # 爆音的判据：波形没有台阶。逐点变化不能超过输入斜率乘补偿增益，增益变化不该额外造出跳变。
                allowed = float(np.max(np.abs(np.diff(x)))) * amp(12.0)
                jump = float(np.max(np.abs(np.diff(y))))
                bench.check(jump < 1.05 * allowed, f"    逐点最大跳变 {jump:.3f}（输入斜率乘补偿 {allowed:.3f}）")

                # 突发结束后的低电平段不能有比稳态更大的跳变，也不能高过稳态。
                after = y[start + length + 16:]
                settled = float(np.max(np.abs(y[-RATE // 4:])))
                after_jump = float(np.max(np.abs(np.diff(after))))
                settled_jump = float(np.max(np.abs(np.diff(y[-RATE // 4:]))))
                bench.check(after_jump <= settled_jump * 1.05,
                            f"    突发结束后没有台阶：逐点 {after_jump:.2e} 对稳态 {settled_jump:.2e}")
                bench.check(float(np.max(np.abs(after))) <= settled * 1.02,
                            f"    突发后单调回升，无过冲：{20 * math.log10(np.max(np.abs(after))):.1f} 对稳态 {20 * math.log10(settled):.1f} dBFS")

    # 增益轨迹本身不能有台阶：方波载波上逐点算衰减，大于一阶启动曲线首点的跳变都算异常。
    # 用 -3 dBFS，输出保护不会动作，量到的只是压缩增益。
    x = np.concatenate([square(0.5, -60.0), square(0.2, -3.0), square(1.0, -60.0)])
    for attack in (0.5, 8.0):
        y, _, _ = bench.render(x, settings(-30.0, 4.0, attack=attack, release=120.0))
        trace = reduction_trace(x, y)
        up = float(np.max(np.diff(trace)))
        down = float(-np.min(np.diff(trace)))
        allowed = 20.25 * (1.0 - math.exp(-1000.0 / (attack * RATE)))
        bench.check(up <= allowed, f"启动 {attack:g} ms 增益逐点最大下压 {up:.3f} dB（一阶上限 {allowed:.3f}）")
        bench.check(down < 0.1, f"    增益逐点最大回升 {down:.4f} dB（< 0.1）")

    # 极端情况：起点是整段静音，单点脉冲，增益不应出现 NaN 或卡死。
    x = np.zeros(RATE)
    x[RATE // 2] = 1.0
    y, _, _ = bench.render(x, settings(-40.0, 20.0, attack=0.5, release=5.0, makeup=24.0, character=WARM))
    bench.check(np.all(np.isfinite(y)) and float(np.max(np.abs(y))) <= amp(CEILING) * 1.0001, "单点脉冲后输出有界")
    bench.check(float(np.max(np.abs(y[-1000:]))) == 0.0, "单点脉冲之后回到全零")


def test_lookahead(bench):
    print("Look-ahead：上报延迟与对齐")
    x = sine(440.0, 1.0, -40.0)

    for ms, expected in ((0.0, 0), (1.0, 48), (5.0, 240), (10.0, 480)):
        y, _, latency = bench.render(x, settings(0.0, 4.0, lookahead=ms))
        residual = float(np.max(np.abs(y[2000:] - x[2000:])))
        bench.check(latency == expected, f"{ms:g} ms 上报 {latency} 采样（应为 {expected}）")
        bench.check(residual < 1e-5, f"    低于阈值时与输入做差 {20 * math.log10(max(residual, 1e-9)):.1f} dBFS（对齐）")

    # 开 Look-ahead 后，突发第一个峰之前增益就该压下去。
    burst = np.concatenate([sine(1000.0, 0.3, -60.0), sine(1000.0, 0.3, -6.0)])
    onset = int(0.3 * RATE)
    for ms in (0.0, 5.0):
        y, _, _ = bench.render(burst, settings(-30.0, 4.0, attack=1.0, release=120.0, lookahead=ms))
        first = float(np.max(np.abs(y[onset:onset + 48])))
        print(f"    预读 {ms:g} ms：突发头 1 ms 峰值 {20 * math.log10(first):.1f} dBFS")
        if ms == 0.0:
            without = first
        else:
            bench.check(first < without * amp(-6.0), f"预读 5 ms 把突发头压低 {20 * math.log10(without / first):.1f} dB（> 6）")

    y, _, _ = bench.render(burst, settings(-30.0, 4.0, attack=1.0, release=120.0, lookahead=5.0))
    level = 20.0 * math.log10(tone_level(y[-RATE // 8:], 1000.0))
    bench.check(abs(level - (-6.0 - ideal_reduction(-6.0, -30.0, 4.0))) < 0.1, f"预读不改变静态曲线：{level:.2f} dBFS")


def test_robust(bench):
    print("健壮性：采样率、缓冲区大小、静音")
    noise = np.random.default_rng(1).standard_normal(int(0.6 * RATE)) * amp(-12.0)
    x = np.concatenate([np.clip(noise, -1.0, 1.0), np.zeros(int(4.0 * RATE))])
    base = settings(-30.0, 4.0, attack=1.0, release=100.0, auto=AUTO, behavior=OPTO, character=WARM)

    # 峰值窗口的相位一直在走，mint_probe 的预热采样数又跟缓冲区有关，
    # 所以只拿预热同为 9600 采样的几种缓冲区比较逐点结果。其余只查有限和归零。
    reference = None
    for block, comparable in ((128, True), (1, True), (32, True), (960, True), (37, False), (1024, False)):
        y, right, _ = bench.render(x, base, block=block)
        finite = np.all(np.isfinite(y))
        tail = float(np.max(np.abs(y[-RATE // 2:])))
        bench.check(finite, f"缓冲区 {block}：输出有限")
        bench.check(tail == 0.0, f"    静音尾部全零（{tail:.2e}）")
        bench.check(np.array_equal(y, right), "    两声道一致")
        if not comparable:
            continue
        if reference is None:
            reference = y
        else:
            diff = float(np.max(np.abs(y - reference)))
            bench.check(diff < 2e-6, f"    与 128 缓冲区做差 {20 * math.log10(max(diff, 1e-9)):.1f} dBFS")

    for rate in (44100, 48000, 96000, 192000):
        tone = np.concatenate([sine(1000.0, 0.4, -40.0, rate), sine(1000.0, 0.6, -6.0, rate)])
        y, _, _ = bench.render(tone, settings(-30.0, 4.0, attack=5.0, release=100.0), rate=rate)
        level = 20.0 * math.log10(tone_level(y[-rate // 4:], 1000.0, rate))
        square_x = np.concatenate([square(0.2, -40.0, rate, rate // 1000), square(0.8, -6.0, rate, rate // 1000)])
        s, _, _ = bench.render(square_x, settings(-30.0, 4.0, attack=20.0, release=100.0), rate=rate)
        trace = reduction_trace(square_x, s)
        start = int(0.2 * rate)
        t = first_above(trace, start, 0.632 * float(np.mean(trace[-rate // 20:]))) / rate * 1000.0
        bench.check(abs(level + 24.0) < 0.1, f"{rate} Hz 静态点 {level:.2f} dBFS（应为 -24）")
        bench.check(abs(t - 20.0) < 2.0, f"    启动 20 ms 实测 {t:.2f} ms")

    # 正弦低频压缩时的调制失真，Smooth 档不该因为检测器纹波明显变脏。
    for freq in (75.0, 82.0, 110.0, 220.0):
        y, _, _ = bench.render(sine(freq, 1.5, -6.0), settings(-30.0, 4.0, attack=2.0, release=120.0))
        h = harmonics(y[int(0.7 * RATE):], freq)
        bench.check(max(h) < -55.0, f"{freq:g} Hz Smooth 衰减 18 dB 时最大谐波 {max(h):.1f} dBc（< -55）")


def test_mono(bench):
    print("单声道：经小工具按单声道布局渲染，与立体声第一声道比较")
    tool = Path(bench.probe).parents[4] / "plugins/comp/tests/shell_probe/atlas_shell_probe_artefacts/Release/atlas_shell_probe.exe"
    x = np.concatenate([sine(440.0, 0.5, -40.0), sine(440.0, 0.5, -6.0), sine(440.0, 0.5, -40.0)])
    params = settings(-30.0, 4.0, attack=1.0, release=100.0, character=WARM)
    src = bench.work / "mono_in.wav"
    dst = bench.work / "mono_out.wav"
    wavfile.write(src, RATE, np.stack([x, x], axis=1).astype(np.float32))
    cmd = [str(tool), "render", str(bench.plugin), "Mintaria Atlas", str(src), str(dst), "--mono", "--block", "128"]
    for name, value in params.items():
        cmd += ["--set", f"{name}={value:.6f}"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        bench.check(False, "单声道渲染失败：" + done.stdout)
        return
    _, mono = wavfile.read(dst)
    stereo, _, _ = bench.render(x, params, block=128)
    mono = mono.astype(np.float64)
    bench.check(mono.ndim == 1, f"输出是单声道（形状 {mono.shape}）")
    level = 20.0 * math.log10(tone_level(mono[int(0.8 * RATE):int(1.0 * RATE)], 440.0))
    bench.check(abs(level + 24.0) < 0.2, f"静态点 {level:.2f} dBFS（应为 -24）")
    bench.check(np.all(np.isfinite(mono)), "输出有限")
    print(f"    单声道与立体声第一声道最大差 {20 * math.log10(max(float(np.max(np.abs(mono - stereo))), 1e-9)):.1f} dBFS（预热时长不同，只作参考）")


def test_automation(bench):
    print("参数自动化：稳态正弦中途改参数，波形不能有台阶")
    tool = Path(bench.probe).parents[4] / "plugins/comp/tests/shell_probe/atlas_shell_probe_artefacts/Release/atlas_shell_probe.exe"
    x = sine(220.0, 2.0, -12.0)
    slope = float(np.max(np.abs(np.diff(x))))
    base = settings(-30.0, 4.0, attack=2.0, release=80.0)
    warm_on = settings(-30.0, 4.0, attack=2.0, release=80.0, character=WARM)
    deep = settings(-50.0, 4.0, attack=2.0, release=80.0)
    loud = settings(-30.0, 4.0, attack=2.0, release=80.0, makeup=12.0)
    ahead = settings(-30.0, 4.0, attack=2.0, release=80.0, lookahead=5.0)

    def run(name, changes, limit):
        src = bench.work / "auto_in.wav"
        dst = bench.work / "auto_out.wav"
        wavfile.write(src, RATE, np.stack([x, x], axis=1).astype(np.float32))
        cmd = [str(tool), "render", str(bench.plugin), "Mintaria Atlas", str(src), str(dst), "--block", "128"]
        for key, value in base.items():
            cmd += ["--set", f"{key}={value:.6f}"]
        for at, key, value in changes:
            cmd += ["--at", f"{at}:{key}={value:.6f}"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if done.returncode != 0:
            bench.check(False, name + " 渲染失败：" + done.stdout)
            return
        _, y = wavfile.read(dst)
        y = y[:, 0].astype(np.float64)
        jump = float(np.max(np.abs(np.diff(y[RATE // 2:]))))
        bench.check(jump < limit, f"{name}：逐点最大跳变 {jump:.4f}（上限 {limit:.4f}）")

    # 稳态输出约为 -30 + 18/4 = -25.5 dBFS，斜率按此算。
    steady = slope * amp(-25.5 + 12.0)
    run("Smooth 切 Warm 再切回", [(1.0, "Character", warm_on["Character"]), (1.5, "Character", base["Character"])], steady * 1.5)
    run("Opto 和 Electro 来回切", [(1.0, "Behavior", OPTO), (1.5, "Behavior", ELECTRO)], steady * 1.5)
    run("阈值跳 20 dB", [(1.0, "Threshold", deep["Threshold"]), (1.5, "Threshold", base["Threshold"])], steady * 1.5)
    run("补偿跳 12 dB", [(1.0, "Makeup", loud["Makeup"]), (1.5, "Makeup", base["Makeup"])], steady * amp(12.0) * 1.05)
    run("Look-ahead 从 0 跳到 5 ms 再回 0", [(1.0, "Look-ahead", ahead["Look-ahead"]), (1.5, "Look-ahead", base["Look-ahead"])], steady * 1.5)
    # 旁通切换时增益要在 20 ms 内走 13.5 dB，上限放宽到干信号斜率。
    run("旁通开关", [(1.0, "Bypass", 1.0), (1.5, "Bypass", 0.0)], slope * 1.05)

    # 旁通稳定后输出等于输入，Look-ahead 开着时也要对齐。
    for ms, ahead_params in ((0.0, base), (5.0, ahead)):
        src = bench.work / "bypass_in.wav"
        dst = bench.work / "bypass_out.wav"
        wavfile.write(src, RATE, np.stack([x, x], axis=1).astype(np.float32))
        cmd = [str(tool), "render", str(bench.plugin), "Mintaria Atlas", str(src), str(dst), "--block", "128", "--at", "0.5:Bypass=1.0"]
        for key, value in ahead_params.items():
            cmd += ["--set", f"{key}={value:.6f}"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        _, y = wavfile.read(dst)
        y = y[:, 0].astype(np.float64)
        residual = float(np.max(np.abs(y[RATE:] - x[RATE:])))
        bench.check(residual < 1e-6, f"旁通（预读 {ms:g} ms）与输入做差 {20 * math.log10(max(residual, 1e-9)):.1f} dBFS")


def test_bypass_like(bench):
    print("直通：比率 1:1，Smooth，补偿 0")
    x = np.clip(np.random.default_rng(2).standard_normal(RATE) * amp(-18.0), -0.9, 0.9)
    y, _, _ = bench.render(x, settings(-60.0, 1.0))
    diff = float(np.max(np.abs(y - x)))
    bench.check(diff < 1e-5, f"与输入做差 {20 * math.log10(max(diff, 1e-9)):.1f} dBFS")


TESTS = {
    "static": test_static,
    "attack": test_attack,
    "release": test_release,
    "shape": test_shape,
    "auto": test_auto,
    "warm": test_warm,
    "burst": test_burst,
    "lookahead": test_lookahead,
    "robust": test_robust,
    "mono": test_mono,
    "automation": test_automation,
    "unity": test_bypass_like,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="D:/mint-build/comp")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    names = args.only.split(",") if args.only else list(TESTS)
    with tempfile.TemporaryDirectory() as work:
        bench = Bench(args.build, work)
        for name in names:
            TESTS[name](bench)

    print(f"\n{'全部通过' if not bench.failures else '未通过 ' + str(len(bench.failures)) + ' 项'}")
    for label in bench.failures:
        print("  - " + label.strip())
    return 1 if bench.failures else 0


if __name__ == "__main__":
    sys.exit(main())
