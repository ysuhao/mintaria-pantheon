#!/usr/bin/env python3
"""Janus 的可重复验证脚本：用生成的信号渲染，再按波形判断核心行为。

    python plugins/gate/tests/run_tests.py [--build D:/mint-build/gate] [--keep DIR]

依赖 numpy 和 gate_host（mint_probe 不喂侧链，也不能中途改参数）。
gate_host 的目标在 plugins/gate/tests/host 下，构建时要单独点名。
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RATE = 48000
THRESHOLD_MIN, THRESHOLD_MAX = -80.0, 0.0
RANGE_MAX = 60.0


def thr(db):
    return (db - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)


def rng(db):
    return db / RANGE_MAX


def release_norm(ms):
    # 与插件参数相同的斜率：20-2000 ms，中点 200 ms
    lo, hi, centre = 20.0, 2000.0, 200.0
    skew = np.log(0.5) / np.log((centre - lo) / (hi - lo))
    return float(((ms - lo) / (hi - lo)) ** skew)


def la(ms):
    return ms / 10.0


MODE = {"vocal": 0.0, "speech": 1.0 / 3.0, "instrument": 2.0 / 3.0, "duck": 1.0}


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def write_wav(path, data, rate=RATE):
    """写 32 位浮点 WAV。data 为一维或 (n, 声道)。"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    payload = data.tobytes()
    with open(path, "wb") as f:
        fmt = (3).to_bytes(2, "little") + channels.to_bytes(2, "little") + rate.to_bytes(4, "little")
        fmt += (rate * channels * 4).to_bytes(4, "little") + (channels * 4).to_bytes(2, "little") + (32).to_bytes(2, "little")
        f.write(b"RIFF" + (36 + len(payload)).to_bytes(4, "little") + b"WAVE")
        f.write(b"fmt " + (16).to_bytes(4, "little") + fmt)
        f.write(b"data" + len(payload).to_bytes(4, "little") + payload)


def read_wav(path):
    raw = Path(path).read_bytes()
    pos = 12
    fmt_tag = channels = bits = None
    while pos < len(raw):
        tag, size = raw[pos:pos + 4], int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            fmt_tag = int.from_bytes(body[:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            bits = int.from_bytes(body[14:16], "little")
            if fmt_tag == 0xFFFE:
                fmt_tag = int.from_bytes(body[24:26], "little")
        elif tag == b"data":
            if fmt_tag == 3 and bits == 32:
                arr = np.frombuffer(body, dtype=np.float32)
            elif bits == 32:
                arr = np.frombuffer(body, dtype=np.int32) / 2147483648.0
            elif bits == 24:
                b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
                v = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
                v = np.where(v >= 1 << 23, v - (1 << 24), v)
                arr = v / 8388608.0
            else:
                raise ValueError(f"unsupported wav: tag {fmt_tag}, {bits} bits")
            return arr.reshape(-1, channels).astype(np.float64)
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def sine(freq, seconds, level_db, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return 10 ** (level_db / 20) * np.sin(2 * np.pi * freq * t)


class Rig:
    def __init__(self, build, workdir):
        self.build = Path(build)
        self.host = Path(build) / "plugins/gate/tests/host/gate_host_artefacts/Release/gate_host.exe"
        self.plugin = Path(build) / "plugins/gate/mint_gate_artefacts/Release/VST3/Mintaria Janus.vst3"
        self.dir = Path(workdir)
        self.count = 0
        self.results = []

    def render(self, signal, params=None, sc=None, block=256, rate=RATE, extra=()):
        self.count += 1
        i, o = self.dir / f"in{self.count}.wav", self.dir / f"out{self.count}.wav"
        write_wav(i, signal, rate)
        cmd = [str(self.host), "render", str(self.plugin), str(i), str(o), "--block", str(block)]
        for name, value in (params or {}).items():
            cmd += ["--set", f"{name}={value}"]
        if sc is not None:
            s = self.dir / f"sc{self.count}.wav"
            write_wav(s, sc, rate)
            cmd += ["--sc", str(s)]
        cmd += list(extra)
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if run.returncode != 0 or "rendered" not in run.stdout or "WARN" in run.stdout:
            raise RuntimeError(f"gate_host failed: {run.stdout} {run.stderr}")
        latency = int(run.stdout.split("reported latency")[1].split()[0])
        return read_wav(o), latency

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def gain_db(out, inp):
    """逐段估计增益（dB）：每 5 ms 的 RMS 比。"""
    win = int(0.005 * RATE)
    n = min(len(out), len(inp)) // win
    a = np.sqrt(np.mean(out[:n * win].reshape(n, win) ** 2, axis=1))
    b = np.sqrt(np.mean(inp[:n * win].reshape(n, win) ** 2, axis=1))
    return db(a) - db(b)


def test_attenuation(r):
    """阈值以下的衰减量：扫电平和衰减量，读稳态增益。"""
    for mode in ["vocal", "speech", "instrument"]:
        for range_db in [6, 18, 40, 60]:
            sig = np.concatenate([sine(1000, 0.3, -20), sine(1000, 2.0, -70)])
            out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(range_db), "Mode": MODE[mode], "Release": release_norm(100)})
            tail = out[-int(0.3 * RATE):, 0]
            ref = sig[-int(0.3 * RATE):]
            got = float(db(np.sqrt(np.mean(tail ** 2))) - db(np.sqrt(np.mean(ref ** 2))))
            r.check(f"attenuation {mode} range {range_db} dB", abs(got + range_db) < 0.1, f"measured {got:.2f} dB")

    # 阈值以上逐采样透明
    sig = sine(1000, 1.0, -20)
    out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(40)})
    settled = int(0.05 * RATE)  # 宿主预跑的静音让门先关上了，开头是一次正常的打开
    diff = np.max(np.abs(out[settled:, 0] - sig[settled:]))
    r.check("transparent above threshold", diff < 1e-7, f"max diff {diff:.2e}")

    # 衰减量 0：什么都不变
    sig = np.concatenate([sine(1000, 0.3, -20), sine(1000, 1.0, -70)])
    out, _ = r.render(sig, {"Threshold": thr(-40), "Range": 0.0})
    r.check("range 0 is transparent", np.max(np.abs(out[:, 0] - sig)) < 1e-6, f"max diff {np.max(np.abs(out[:, 0] - sig)):.2e}")

    # 阈值校准：门关着时，1 kHz 高于阈值 1.5 dB 能打开，低于 1.5 dB 打不开
    for mode in ["vocal", "speech", "instrument"]:
        gains = {}
        for offset in (1.5, -1.5):
            sig = np.concatenate([np.zeros(RATE), sine(1000, 1.5, -40 + offset)])
            out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(30), "Mode": MODE[mode], "Release": release_norm(100)})
            gains[offset] = float(db(np.sqrt(np.mean(out[-int(0.3 * RATE):, 0] ** 2))) - (-40 + offset - 3.0103))
        r.check(f"threshold calibration {mode}", abs(gains[1.5]) < 0.01 and gains[-1.5] < -0.5, f"T+1.5: {gains[1.5]:.2f} dB, T-1.5: {gains[-1.5]:.2f} dB")

    # 静态曲线：从打开状态走到稳态，记录每个电平的增益
    curve = []
    for level in [-30, -38, -40, -42, -44, -46, -50, -55, -60]:
        sig = np.concatenate([sine(1000, 0.3, -20), sine(1000, 1.5, level)])
        out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(30), "Release": release_norm(100)})
        tail = out[-int(0.2 * RATE):, 0]
        curve.append((level, float(db(np.sqrt(np.mean(tail ** 2))) - (level - 3.0103))))
    print("      static curve (input dB: gain dB) " + ", ".join(f"{a}: {b:.1f}" for a, b in curve))
    r.check("static curve is monotonic", all(curve[i][1] >= curve[i + 1][1] - 0.2 for i in range(len(curve) - 1)))
    r.check("full range well below threshold", abs(curve[-1][1] + 30) < 0.2, f"{curve[-1][1]:.2f} dB at -60")
    return curve


def keyed_click_signal(tone_hz=110.0, seconds=4.0):
    """低电平恒定低频音（永远低于阈值）+ 高频按键音触发门。按键边沿 30 ms 升余弦，输入自身没有溅射。"""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    tone = 10 ** (-46 / 20) * np.sin(2 * np.pi * tone_hz * t)
    key = np.zeros(n)
    edge = int(0.03 * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    for start in np.arange(0.5, seconds - 0.5, 1.0):
        i, j = int(start * RATE), int((start + 0.4) * RATE)
        key[i:j] = 1.0
        key[i:i + edge] = ramp
        key[j - edge:j] = ramp[::-1]
    return tone + 10 ** (-20 / 20) * key * np.sin(2 * np.pi * 9000.0 * t), tone, key


def band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return float(np.sum(spec[(f >= lo) & (f <= hi)]))


def test_clicks(r):
    """开合无咔嗒：门的增益运动产生的边带，要明显低于同样开合的硬切换。"""
    for range_db in (18, 40, 60):
        sig, tone, key = keyed_click_signal()
        params = {"Threshold": thr(-30), "Range": rng(range_db), "Release": release_norm(50), "Mode": MODE["instrument"]}
        out, _ = r.render(sig, params)
        y = out[:, 0]
        hard = tone * np.where(key > 0.32, 1.0, 10 ** (-range_db / 20))
        # 110 Hz 和 9 kHz 之外的频段里只剩增益运动的边带
        lo, hi = 800.0, 5000.0
        e_in, e_out, e_hard = band_energy(sig, lo, hi), band_energy(y, lo, hi), band_energy(hard, lo, hi)
        below_hard = 10 * np.log10(e_out / e_hard)
        r.check(f"open/close sidebands far below hard switch, range {range_db}", below_hard < -20.0,
                f"{below_hard:.1f} dB vs hard switch; input floor {10 * np.log10(e_in / e_hard):.1f} dB")

    for mode in ("vocal", "speech"):
        sig, tone, key = keyed_click_signal()
        out, _ = r.render(sig, {"Threshold": thr(-30), "Range": rng(40), "Release": release_norm(100), "Mode": MODE[mode]})
        hard = tone * np.where(key > 0.32, 1.0, 10 ** (-40 / 20))
        lo, hi = 800.0, 5000.0
        below_hard = 10 * np.log10(band_energy(out[:, 0], lo, hi) / band_energy(hard, lo, hi))
        r.check(f"open/close sidebands far below hard switch, {mode}", below_hard < -20.0, f"{below_hard:.1f} dB vs hard switch")

    # 闪避模式的增益完全由侧链决定，载波恒定，能直接读增益轨迹：不能超过设定的下压速度，也不能有台阶
    carrier = sine(1000, 3.0, -12)
    sc = np.zeros_like(carrier)
    sc[int(0.5 * RATE):int(1.5 * RATE)] = sine(1000, 1.0, -20)
    out, _ = r.render(carrier, {"Threshold": thr(-40), "Range": rng(30), "Mode": MODE["duck"], "Release": release_norm(100)}, sc=sc)
    y = out[:, 0]
    period = RATE // 1000
    env = np.array([np.max(np.abs(y[i:i + period])) for i in range(0, len(y) - period, period)])
    step = np.abs(np.diff(db(env)))
    r.check("duck gain trajectory stays on its slew limit", np.max(step) <= 3.2, f"max {np.max(step):.2f} dB per ms")
    curvature = np.abs(np.diff(step))
    r.check("duck gain trajectory has no corner", np.max(curvature) <= 3.2, f"max change {np.max(curvature):.2f} dB per ms per ms")

    # 信号结束后：静音接入点无尖峰
    sig = np.concatenate([sine(1000, 0.5, -20), np.zeros(RATE)])
    out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(60), "Release": release_norm(50)})
    r.check("silence after signal stays finite and quiet", np.max(np.abs(out[-RATE // 2:, 0])) < 1e-9, f"peak {np.max(np.abs(out[-RATE // 2:, 0])):.1e}")


def count_crossings(g, level):
    above = g > level
    return int(np.sum(above[1:] != above[:-1]))


def test_chatter(r):
    """不抖动：电平在阈值附近徘徊时增益不能来回翻。"""
    params = {"Threshold": thr(-40), "Range": rng(30), "Release": release_norm(100)}

    def tremolo(centre, depth, rate_hz, carrier=220.0, seconds=4.0):
        t = np.arange(int(seconds * RATE)) / RATE
        level = 10 ** ((centre + depth * np.sin(2 * np.pi * rate_hz * t)) / 20)
        sig = level * np.sin(2 * np.pi * carrier * t)
        lead = int(0.5 * RATE)
        sig[:lead] = 0.1 * np.sin(2 * np.pi * carrier * t[:lead])
        return sig

    # 打开状态下，中心落在阈值上：颤音不能把门带关
    sig = tremolo(-40, 2, 8)
    out, _ = r.render(sig, params)
    g = gain_db(out[RATE:, 0], sig[RATE:])
    r.check("tremolo on the threshold stays open", np.ptp(g) < 1.0, f"swing {np.ptp(g):.2f} dB")

    # 中心落在关闭点上下（阈值 - 迟滞），8 Hz 颤音跨过关闭点：保持时间要把它撑住
    sig = tremolo(-44, 2, 8)
    out, _ = r.render(sig, params)
    g = gain_db(out[RATE:, 0], sig[RATE:])
    r.check("tremolo across the close point does not flap", np.ptp(g) < 4.0, f"swing {np.ptp(g):.2f} dB, -3 dB crossings {count_crossings(g, -3)}")

    # 已经关闭、落在膝内：颤音只允许平缓地带动增益，不能跟着翻满衰减量
    sig = np.concatenate([np.zeros(RATE), tremolo(-46, 1.5, 6)[RATE // 2:]])
    out, _ = r.render(sig, params)
    g = gain_db(out[int(2.0 * RATE):, 0], sig[int(2.0 * RATE):])
    r.check("tremolo inside the knee moves gain gently", np.ptp(g) < 6.0, f"swing {np.ptp(g):.2f} dB of 30 dB range")

    # 低频正弦的波形纹波不能让增益抖
    # 电平选在检测滤波之后正好落进膝内的位置，增益处于中段时最容易把纹波放大出来
    for freq, mode, level in [(80, "vocal", -42), (50, "instrument", -44)]:
        sig = np.concatenate([sine(freq, 0.3, -20), np.zeros(RATE // 2), sine(freq, 3.0, level)])
        out, _ = r.render(sig, {**params, "Mode": MODE[mode]})
        g = gain_db(out[int(2.0 * RATE):, 0], sig[int(2.0 * RATE):])
        r.check(f"{freq} Hz inside the knee has no ripple ({mode})", np.ptp(g) < 1.0 and -28.0 < np.mean(g) < -2.0,
                f"swing {np.ptp(g):.2f} dB, mean {np.mean(g):.1f} dB")

    # 噪声贴着关闭点：这是门最容易抖的情况。扫几个电平，逐个看翻转次数
    rs = np.random.RandomState(7)
    base = rs.standard_normal(int(5 * RATE))
    base /= np.max(np.abs(base))
    worst_swing = 0.0
    report = []
    for peak in (-36, -38, -40, -42, -44, -46, -48, -50):
        sig = base * 10 ** (peak / 20)
        sig[:int(0.3 * RATE)] *= 30
        out, _ = r.render(sig, params)
        g = gain_db(out[RATE:, 0], sig[RATE:])
        worst_swing = max(worst_swing, float(np.ptp(g)))
        report.append(f"{peak}:{np.ptp(g):.1f}")
    r.check("noise around the threshold does not flutter", worst_swing < 10.0, "peak dB:gain swing dB of 30 dB range  " + " ".join(report))

    # 底噪的读数要按 RMS 算：峰值碰到阈值、RMS 低 13 dB 的噪声应当被压下去
    sig = base * 10 ** (-40 / 20)
    sig[:int(0.3 * RATE)] *= 30
    out, _ = r.render(sig, params)
    g = gain_db(out[int(1.5 * RATE):, 0], sig[int(1.5 * RATE):])
    r.check("noise floor with peaks at threshold gets attenuated", np.mean(g) < -15.0, f"mean {np.mean(g):.1f} dB of 30 dB range")

    # 单个尖峰不该把门打开
    tone = sine(1000, 2.0, -60)
    spiky = tone.copy()
    spiky[RATE] += 10 ** (-30 / 20)
    out, _ = r.render(spiky, params)
    g = gain_db(out[int(1.1 * RATE):int(1.8 * RATE), 0], tone[int(1.1 * RATE):int(1.8 * RATE)])
    r.check("single spike does not open the gate", np.max(g) < -29.0, f"gain after spike max {np.max(g):.2f} dB")

    # 词间短停顿不应掉下去：语音模式 80 ms 停顿
    parts = []
    for _ in range(8):
        parts += [sine(300, 0.2, -20), np.zeros(int(0.08 * RATE))]
    sig = np.concatenate(parts + [np.zeros(int(0.5 * RATE))])
    out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(30), "Mode": MODE["speech"], "Release": release_norm(200)})
    start = int(0.1 * RATE)
    stop = int(8 * 0.28 * RATE) - int(0.1 * RATE)
    y = out[start:stop, 0]
    voiced = np.abs(sig[start:stop]) > 0
    r.check("speech mode holds through 80 ms pauses", np.max(np.abs(y[voiced] - sig[start:stop][voiced])) < 1e-5,
            f"max error {np.max(np.abs(y[voiced] - sig[start:stop][voiced])):.1e}")


def onset_loss(out, sig, onset):
    """起音后 N ms 内相对输入的能量损失（dB）。"""
    res = {}
    for ms in (1, 2, 5, 10):
        n = int(ms * 0.001 * RATE)
        a, b = out[onset:onset + n], sig[onset:onset + n]
        res[ms] = float(10 * np.log10(np.sum(a ** 2) / np.sum(b ** 2)))
    return res


def test_lookahead(r):
    """Look-ahead：上报准确、对齐、起音不吃字头。"""
    # 0 延迟默认
    click = np.zeros(RATE)
    click[10000] = 0.5
    out, latency = r.render(click, {"Threshold": 0.0, "Range": 0.0})
    r.check("default latency is 0", latency == 0, f"latency {latency}")
    r.check("default output aligned", int(np.argmax(np.abs(out[:, 0]))) == 10000, f"peak at {int(np.argmax(np.abs(out[:, 0])))}")

    for ms, rate in [(1.0, 48000), (5.0, 48000), (10.0, 48000), (5.0, 44100), (7.3, 96000), (10.0, 192000)]:
        click = np.zeros(rate)
        click[rate // 4] = 0.5
        out, latency = r.render(click, {"Threshold": 0.0, "Range": 0.0, "Look-ahead": la(ms)}, rate=rate)
        want = int(np.floor(ms * 0.001 * rate + 0.5))  # 与 std::round 相同，.5 向上
        peak = int(np.argmax(np.abs(out[:, 0])))
        r.check(f"look-ahead {ms} ms at {rate}: latency and alignment", latency == want and peak == rate // 4,
                f"latency {latency} (want {want}), aligned peak at {peak} (want {rate // 4})")

    # 字头：静音后突然起音（低频最考验），高于阈值
    onset = int(0.5 * RATE)
    for freq in (150, 1000, 5000):
        sig = np.zeros(RATE)
        sig[onset:] = sine(freq, (RATE - onset) / RATE, -20)
        common = {"Threshold": thr(-40), "Range": rng(40), "Release": release_norm(100)}
        out0, _ = r.render(sig, common)
        loss0 = onset_loss(out0[:, 0], sig, onset)
        out5, _ = r.render(sig, {**common, "Look-ahead": la(5)})
        loss5 = onset_loss(out5[:, 0], sig, onset)
        print(f"      onset {freq} Hz: look-ahead off {loss0}, 5 ms {loss5}")
        if freq >= 1000:
            r.check(f"look-ahead 5 ms keeps onset {freq} Hz", abs(loss5[1]) < 0.5, f"first 1 ms {loss5[1]:.2f} dB")
        else:
            r.check(f"look-ahead 5 ms keeps onset {freq} Hz", abs(loss5[5]) < 0.5, f"first 5 ms {loss5[5]:.2f} dB")
        r.check(f"look-ahead improves onset {freq} Hz", loss5[2] > loss0[2] or abs(loss0[2]) < 0.1, f"first 2 ms {loss0[2]:.2f} -> {loss5[2]:.2f} dB")

    sig = np.zeros(RATE)
    sig[onset:] = sine(1000, (RATE - onset) / RATE, -20)
    out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(18)})
    print(f"      onset 1000 Hz, default range 18 dB, look-ahead off: {onset_loss(out[:, 0], sig, onset)}")

    # 中途开关 Look-ahead 不应出现旧音频或爆音
    sig = sine(440, 3.0, -20)
    out, _ = r.render(sig, {"Threshold": thr(-60), "Range": rng(20)},
                      extra=["--at", f"1.0:Look-ahead={la(5)}", "--at", f"2.0:Look-ahead=0"])
    y = out[:, 0]
    r.check("look-ahead switch never exceeds input peak", np.max(np.abs(y)) <= np.max(np.abs(sig)) * 1.01, f"out peak {np.max(np.abs(y)):.4f} / in {np.max(np.abs(sig)):.4f}")
    period = RATE // 440
    jump = np.max(np.abs(np.diff(y)))
    natural = np.max(np.abs(np.diff(sig)))
    r.check("look-ahead switch makes no discontinuity", jump <= natural * 1.2, f"max step {jump:.4f}, natural {natural:.4f}")


def test_release(r):
    """释放旋钮：下落时间跟设定成比例。"""
    measured = {}
    for ms in (50, 200, 800):
        sig = np.concatenate([sine(1000, 0.5, -20), sine(1000, 3.0, -70)])
        out, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(30), "Release": release_norm(ms)})
        g = gain_db(out[:, 0], sig)
        start = int(0.5 * RATE) // int(0.005 * RATE)
        t10 = np.argmax(g[start:] < -3) * 5
        t90 = np.argmax(g[start:] < -27) * 5
        measured[ms] = t90 - t10
    print(f"      release dial -> -3..-27 dB fall time: {measured}")
    for ms, got in measured.items():
        want = ms * 0.8
        r.check(f"release {ms} ms fall time", abs(got - want) <= max(15, want * 0.25), f"{got} ms for 24 of 30 dB (want about {want:.0f})")


def test_duck(r):
    """闪避：外部侧链压低，未接时直通。"""
    carrier = sine(1000, 3.0, -20)
    sc = np.zeros_like(carrier)
    sc[RATE:int(2 * RATE)] = sine(500, 1.0, -20)
    params = {"Threshold": thr(-40), "Range": rng(18), "Mode": MODE["duck"], "Release": release_norm(100)}

    out, _ = r.render(carrier, params, sc=sc)
    g = gain_db(out[:, 0], carrier)
    idx = lambda s: int(s * RATE) // int(0.005 * RATE)
    during = np.mean(g[idx(1.3):idx(1.9)])
    before = np.mean(g[idx(0.3):idx(0.9)])
    after = np.mean(g[idx(2.6):idx(2.95)])
    r.check("duck reduces by range while sidechain plays", abs(during + 18) < 0.1, f"{during:.2f} dB")
    r.check("duck passes before sidechain", abs(before) < 0.01, f"{before:.2f} dB")
    r.check("duck recovers after sidechain", abs(after) < 0.01, f"{after:.2f} dB")

    # 侧链低于阈值：不动作
    out, _ = r.render(carrier, params, sc=sc * 10 ** (-40 / 20))
    r.check("duck ignores sidechain below threshold", np.max(np.abs(out[:, 0] - carrier)) < 1e-6)

    # 未接：总线关闭 / 总线开但静音，都要直通
    out, _ = r.render(carrier, params, extra=["--no-sc-bus"])
    r.check("duck with sidechain bus disabled passes through", np.max(np.abs(out[:, 0] - carrier)) < 1e-6)
    out, _ = r.render(carrier, params)
    r.check("duck with silent sidechain passes through", np.max(np.abs(out[:, 0] - carrier)) < 1e-6)

    # 运行中切进闪避，侧链没接：不能因为切换压一下
    for extra in (["--no-sc-bus"], []):
        steady = sine(1000, 3.0, -20)
        out, _ = r.render(steady, {"Threshold": thr(-40), "Range": rng(18), "Release": release_norm(100)},
                          extra=extra + ["--at", f"1.0:Mode={MODE['duck']}", "--at", f"2.0:Mode={MODE['vocal']}"])
        diff = np.max(np.abs(out[RATE // 2:, 0] - steady[RATE // 2:]))
        r.check(f"switching modes live does not dip {extra}", diff < 1e-6, f"max diff {diff:.1e}")

    # 内部模式不受侧链影响
    sig = np.concatenate([sine(1000, 0.3, -20), sine(1000, 1.5, -70)])
    out_a, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(18)}, sc=sine(1000, 1.8, -10))
    out_b, _ = r.render(sig, {"Threshold": thr(-40), "Range": rng(18)}, extra=["--no-sc-bus"])
    r.check("internal modes ignore the sidechain", np.max(np.abs(out_a - out_b)) < 1e-6, f"max diff {np.max(np.abs(out_a - out_b)):.1e}")


def test_formats(r):
    """声道、缓冲区、采样率、静音和异常输入。"""
    sig = np.concatenate([sine(1000, 0.3, -20), sine(1000, 1.0, -70), sine(1000, 0.3, -20)])
    params = {"Threshold": thr(-40), "Range": rng(30), "Release": release_norm(100)}
    reference, _ = r.render(sig, params, block=64)
    for block in (1, 7, 32, 480, 4096):
        out, _ = r.render(sig, params, block=block)
        diff = np.max(np.abs(out[:, 0] - reference[:, 0]))
        r.check(f"block {block} matches block 64", diff < 1e-3, f"max diff {diff:.1e}")

    # Look-ahead 下也不受缓冲区影响，包括比延迟短的 1 采样缓冲
    params_la = {**params, "Look-ahead": la(5)}
    reference, _ = r.render(sig, params_la, block=64)
    out, _ = r.render(sig, params_la, block=1)
    r.check("look-ahead block 1 matches block 64", np.max(np.abs(out[:, 0] - reference[:, 0])) < 1e-3)

    out, _ = r.render(sig, params, extra=["--mono"])
    r.check("mono layout works", out.shape[1] == 1 and abs(float(db(np.sqrt(np.mean(out[int(0.9 * RATE):int(1.2 * RATE), 0] ** 2)))) - (-70 - 3.01 - 30)) < 0.2)

    # 立体声联动：一路有声，两路同时打开
    stereo = np.zeros((int(1.3 * RATE), 2))
    stereo[:, 1] = np.concatenate([sine(1000, 0.3, -20), sine(1000, 1.0, -70)])
    stereo[:, 0] = sine(1000, 1.3, -70)
    out, _ = r.render(stereo, params)
    g_left_early = float(db(np.sqrt(np.mean(out[int(0.1 * RATE):int(0.28 * RATE), 0] ** 2))) - (-70 - 3.01))
    g_left_late = float(db(np.sqrt(np.mean(out[-int(0.2 * RATE):, 0] ** 2))) - (-70 - 3.01))
    r.check("stereo detection is linked", abs(g_left_early) < 0.1 and abs(g_left_late + 30) < 0.2, f"quiet channel early {g_left_early:.2f}, late {g_left_late:.2f} dB")

    for rate in (44100, 96000, 192000):
        s = np.concatenate([sine(1000, 0.3, -20, rate), sine(1000, 1.0, -70, rate)])
        out, _ = r.render(s, params, rate=rate)
        tail = out[-int(0.2 * rate):, 0]
        got = float(db(np.sqrt(np.mean(tail ** 2))) + 70 + 3.0103)
        r.check(f"rate {rate} attenuation", abs(got + 30) < 0.2, f"{got:.2f} dB")

    # 全静音：没有 denormal、NaN、Inf
    out, _ = r.render(np.zeros(RATE * 2), params)
    r.check("silence stays exactly zero", np.max(np.abs(out)) == 0.0, f"peak {np.max(np.abs(out)):.1e}")
    tiny = np.full(RATE, 1e-30)
    out, _ = r.render(tiny, params)
    r.check("denormal-size input stays finite", np.all(np.isfinite(out)))

    # 所有模式快速切换：不能有 NaN，也不能超过输入峰值
    sig = sine(300, 2.0, -20)
    extra = []
    for i, m in enumerate([0.0, 1 / 3, 2 / 3, 1.0, 0.0, 2 / 3, 1 / 3, 1.0]):
        extra += ["--at", f"{0.2 * (i + 1):.1f}:Mode={m}"]
    out, _ = r.render(sig, params, extra=extra, sc=sine(500, 2.0, -20))
    r.check("mode switching stays finite and bounded", np.all(np.isfinite(out)) and np.max(np.abs(out)) <= np.max(np.abs(sig)) * 1.001)
    jump = np.max(np.abs(np.diff(out[:, 0])))
    natural = np.max(np.abs(np.diff(sig)))
    r.check("mode switching makes no discontinuity", jump <= natural * 1.2, f"max step {jump:.5f}, natural {natural:.5f}")

    # 宿主旁路（含 Look-ahead）对齐且无跳变
    sig = sine(440, 3.0, -20)
    extra = ["--bypass-at", "1.0:1", "--bypass-at", "2.0:0"]
    out, _ = r.render(sig, {"Threshold": thr(-10), "Range": rng(30), "Release": release_norm(100), "Look-ahead": la(5)}, extra=extra)
    tail = out[int(1.5 * RATE):int(1.9 * RATE), 0]
    r.check("bypass passes signal aligned under look-ahead", np.max(np.abs(tail - sig[int(1.5 * RATE):int(1.9 * RATE)])) < 1e-4,
            f"max diff {np.max(np.abs(tail - sig[int(1.5 * RATE):int(1.9 * RATE)])):.1e}")
    jump = np.max(np.abs(np.diff(out[:, 0])))
    r.check("bypass switch makes no discontinuity", jump <= np.max(np.abs(np.diff(sig))) * 1.2, f"max step {jump:.5f}")


def test_probe(r):
    """mint_probe：项目约定点名的检查、参数和 Look-ahead 渲染对齐。"""
    probe = str(r.build / "tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe")
    plugin = str(r.plugin)

    run = subprocess.run([probe, "check", plugin, "--expect-latency", "0"], capture_output=True, text=True, timeout=300)
    r.check("mint_probe check --expect-latency 0", run.returncode == 0 and "PASS" in run.stdout, run.stdout.strip().replace("\n", " | "))

    click = np.zeros(RATE)
    click[RATE // 4] = 0.5
    i, o = r.dir / "probe_in.wav", r.dir / "probe_out.wav"
    write_wav(i, click)
    for ms, want in [(0.0, 0), (5.0, 240), (10.0, 480)]:
        run = subprocess.run([probe, "render", plugin, str(i), str(o), "--set", f"Threshold=0", "--set", "Range=0", "--set", f"Look-ahead={la(ms)}"],
                             capture_output=True, text=True, timeout=120)
        y = read_wav(o)
        peak = int(np.argmax(np.abs(y[:, 0])))
        r.check(f"mint_probe render look-ahead {ms} ms", f"reported latency {want}" in run.stdout and peak == RATE // 4, f"{run.stdout.strip()}; peak at {peak}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="D:/mint-build/gate")
    parser.add_argument("--keep", help="保留渲染文件的目录")
    parser.add_argument("--only", help="只跑名字里包含这个词的测试")
    args = parser.parse_args()

    workdir = args.keep or tempfile.mkdtemp(prefix="gate_tests_")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    rig = Rig(args.build, workdir)

    tests = [test_probe, test_attenuation, test_clicks, test_chatter, test_lookahead, test_release, test_duck, test_formats]
    for test in tests:
        if args.only and args.only not in test.__name__:
            continue
        print(f"== {test.__name__}: {test.__doc__.strip()}")
        test(rig)

    failed = [name for name, ok, _ in rig.results if not ok]
    print(f"\n{len(rig.results) - len(failed)}/{len(rig.results)} passed")
    for name in failed:
        print(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
