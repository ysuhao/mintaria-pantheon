#!/usr/bin/env python3
"""Helios 核心行为验证。先构建 mint_probe、mint_rider_VST3、mint_rider_bench，再运行：

    python plugins/rider/tests/run_tests.py [--real path/to/vocal.wav]

每项给出实测数字和 PASS/FAIL，任一失败时退出码为 1。
"""
import argparse
import subprocess
import sys
import numpy as np
from scipy import signal as sps

import riderlib as rl
from riderlib import SR

failures = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)

def section(title):
    print(f"\n== {title}")

def test_probe():
    section("mint_probe check")
    r = subprocess.run([rl.PROBE, "check", rl.PLUGIN, "--expect-latency", "0"], capture_output=True, text=True, timeout=600)
    print("  " + r.stdout.strip().replace("\n", "\n  "))
    check("check 默认延迟 0", r.returncode == 0 and "PASS" in r.stdout)

def test_accuracy():
    section("电平精度：目标 -18 dB，范围 ±24（末尾 1 秒电平减目标）")
    for speed in [50, 100]:
        worst = 0.0
        for db in [-42, -36, -30, -24, -18, -12, -6]:
            x = rl.sine(1000, 5, db)
            y = rl.run_signal("acc", x, rl.helios_sets(up=24, down=24, speed=speed))[:, 0]
            err = rl.level_db(y[-SR:]) + 18
            worst = max(worst, abs(err))
            print(f"  速度 {speed} 正弦 1 kHz {db:>4} dB → {rl.level_db(y[-SR:]):7.2f} dB，误差 {err:+.3f}")
        check(f"速度 {speed} 正弦误差 ≤ 0.1 dB", worst <= 0.1, f"最大 {worst:.3f} dB")

    for freq in [100, 220, 4000]:
        x = rl.sine(freq, 5, -30)
        y = rl.run_signal("acc", x, rl.helios_sets(up=24, down=24, speed=100, gate=0))[:, 0]
        err = rl.level_db(y[-SR:]) + 18
        check(f"速度 100 正弦 {freq} Hz 误差 ≤ 0.5 dB", abs(err) <= 0.5, f"{err:+.3f} dB")

    worst = 0.0
    for db in [-36, -30, -24, -18, -12, -6]:
        x = rl.phrase(6, db)
        y = rl.run_signal("acc", x, rl.helios_sets(up=24, down=24, speed=75))[:, 0]
        err = rl.level_db(y[3 * SR:-SR // 2]) + 18
        worst = max(worst, abs(err))
        print(f"  合成人声（音节起伏 ±1.5 dB）{db:>4} dB → 误差 {err:+.3f}")
    check("合成人声误差 ≤ 0.5 dB", worst <= 0.5, f"最大 {worst:.3f} dB")

    for target in [-30, -24, -12, -6]:
        x = rl.sine(1000, 4, -18)
        y = rl.run_signal("acc", x, rl.helios_sets(target=target, up=24, down=24))[:, 0]
        err = rl.level_db(y[-SR:]) - target
        check(f"目标 {target} dB", abs(err) <= 0.1, f"误差 {err:+.3f}")

def test_ranges():
    section("范围：逐采样增益的最大最小值")
    seq = [-48, -36, -6, -3, -30, -30]
    x = rl.steps(seq, 3)
    x = x + 0.0
    for up, down in [(3, 9), (12, 2), (6, 6), (24, 24), (0, 6), (6, 0)]:
        for speed in [50, 100]:
            y = rl.run_signal("rng", x, rl.helios_sets(up=up, down=down, speed=speed))[:, 0]
            g = rl.sample_gain_db(x, y)
            ok = g.max() <= up + 0.01 and g.min() >= -down - 0.01
            check(f"上 {up} 下 {down} 速度 {speed}", ok, f"实测 {g.max():+.3f} / {g.min():+.3f} dB")

    y = rl.run_signal("rng", x, rl.helios_sets(up=0, down=0))[:, 0]
    check("上下都为 0 时完全直通", np.max(np.abs(y - x.astype(np.float32))) == 0.0, f"最大差 {np.max(np.abs(y - x.astype(np.float32))):.2e}")

def hold_signal(voice_db=-30.0):
    """人声 → 静音 → 呼吸 → 静音 → 齿音 → 静音 → 人声。全程叠 -90 dB 导频。"""
    parts = []
    marks = {}
    def add(name, data):
        start = sum(len(p) for p in parts)
        parts.append(data)
        marks[name] = (start, start + len(data))
    add("v1", rl.phrase(4, voice_db, syllables=False))
    add("s1", np.zeros(2 * SR))
    add("br1", rl.breath(0.5, voice_db - 18))
    add("s2", np.zeros(SR))
    add("br2", rl.breath(0.5, voice_db - 10))
    add("s3", np.zeros(SR))
    add("sib", rl.sibilant(0.2, voice_db - 8))
    add("s4", np.zeros(2 * SR))
    add("v2", rl.phrase(3, voice_db, syllables=False))
    x = np.concatenate(parts)
    return x + rl.pilot(len(x)), marks

def test_hold():
    section("静音和呼吸：人声 -30 dB，目标 -18，范围 ±24（用 -90 dB 导频读静音段增益）")
    for speed in [0, 25, 50, 75, 100]:
        x, m = hold_signal()
        y = rl.run_signal("hold", x, rl.helios_sets(speed=speed, up=24, down=24))[:, 0]
        g = rl.sample_gain_db(x, y)
        end_v1 = m["v1"][1]
        ref = np.median(g[end_v1 - int(0.25 * SR):end_v1 - int(0.1 * SR)])
        region = g[end_v1:m["v2"][0]]
        rise = region.max() - ref
        fall = region.min() - ref
        silence = g[m["s1"][0] + SR // 2:m["s1"][1]]
        drift = silence.max() - silence.min()
        print(f"  速度 {speed:>3}：句尾增益 {ref:+6.2f}，静音/呼吸段相对句尾 {rise:+.2f} / {fall:+.2f} dB，静音内漂移 {drift:.3f} dB")
        # 句尾淡出（60 ms）期间门还没关，快速档会跟上一点，这部分单独给上限。门关之后必须完全不动。
        limit = 0.5 if speed <= 50 else 3.0
        check(f"速度 {speed} 句尾上冲 ≤ {limit} dB", rise <= limit, f"{rise:+.2f} dB")
        after = g[m["s1"][0] + SR // 10:m["v2"][0]]
        check(f"速度 {speed} 静音、呼吸、齿音全段不动（≤ 0.05 dB）", np.ptp(after) <= 0.05, f"{np.ptp(after):.3f} dB")

    x, m = hold_signal()
    y = rl.run_signal("hold", x, rl.helios_sets(speed=75, gate=0, up=24, down=24))[:, 0]
    g = rl.sample_gain_db(x, y)
    end_v1 = m["v1"][1]
    ref = np.median(g[end_v1 - int(0.25 * SR):end_v1 - int(0.1 * SR)])
    print(f"  对照（人声门 0%，速度 75）：呼吸段相对句尾上升 {g[m['br2'][0]:m['s3'][0]].max() - ref:+.2f} dB")

def test_false_open():
    section("门的误开：60 段呼吸和 60 段齿音，每段随机电平，速度 100（最容易暴露）")
    parts = [rl.phrase(3, -30, syllables=False), np.zeros(SR // 2)]
    for i in range(60):
        parts.append(rl.breath(0.4, -30 - float(rl.RNG.uniform(6, 25))))
        parts.append(np.zeros(SR // 5))
        parts.append(rl.sibilant(0.15, -30 - float(rl.RNG.uniform(4, 20))))
        parts.append(np.zeros(SR // 5))
    start = len(parts[0]) + len(parts[1])
    x = np.concatenate(parts)
    x = x + rl.pilot(len(x))
    for gate in [50, 100]:
        y = rl.run_signal("false", x, rl.helios_sets(speed=100, up=24, down=24, gate=gate))[:, 0]
        g = rl.sample_gain_db(x, y)[start:]
        check(f"人声门 {gate}% 全程增益不动（≤ 0.05 dB）", np.ptp(g) <= 0.05, f"摆动 {np.ptp(g):.3f} dB")

def test_loud_noise():
    section("响的非人声：粉噪，目标 -18，范围 ±24")
    for db in [-3, -6, -12, -30]:
        x = rl.pink(6, db)
        y = rl.run_signal("noise", x, rl.helios_sets(up=24, down=24))[:, 0]
        g = np.median(rl.sample_gain_db(x, y)[-SR:])
        # 检测带 60 Hz 高通，粉噪的读数比整段电平低，理论值按同样的高通计算。
        heard = rl.level_db(sps.sosfilt(sps.butter(2, 60, "highpass", fs=SR, output="sos"), x)[SR:])
        expect = min(0.0, -18.0 + 6.0 - heard)
        check(f"粉噪 {db} dB：只压到目标 +6 dB，不往上推", abs(g - expect) <= 1.0, f"增益 {g:+.2f} dB，理论 {expect:+.2f}")

def test_speed():
    section("响应时间：1 kHz 正弦 -24 ↔ -12 dB，目标 -18，范围 ±12")
    x = rl.steps([-24, -12, -24], 10)
    rows = {}
    for speed in [0, 25, 50, 75, 100]:
        y = rl.run_signal("spd", x, rl.helios_sets(speed=speed))[:, 0]
        g = rl.sample_gain_db(x, y)
        down = rl.response_ms(g, 10 * SR, 20 * SR)
        up = rl.response_ms(g, 20 * SR, 30 * SR)
        rows[speed] = (down, up)
        print(f"  速度 {speed:>3}：下压 10–90% {down['rise']:8.1f} ms，过冲 {down['overshoot']:+.2f} dB；"
              f"上推 10–90% {up['rise']:8.1f} ms，过冲 {up['overshoot']:+.2f} dB；稳态 {down['settled']:+.2f} / {up['settled']:+.2f}")
    speeds = sorted(rows)
    mono_down = all(rows[a][0]["rise"] > rows[b][0]["rise"] for a, b in zip(speeds, speeds[1:]))
    mono_up = all(rows[a][1]["rise"] > rows[b][1]["rise"] for a, b in zip(speeds, speeds[1:]))
    check("响应时间随速度单调变短", mono_down and mono_up)

def test_zipper():
    section("拉链噪声")
    x = rl.phrase(4, -30, syllables=True)
    x = np.concatenate([x, rl.phrase(4, -12, syllables=True)])
    ref = rl.run_signal("zip", x, rl.helios_sets(speed=100), block=256)[:, 0]
    for block in [1, 7, 64, 1024]:
        y = rl.run_signal("zip", x, rl.helios_sets(speed=100), block=block)[:, 0]
        diff = np.max(np.abs(y - ref))
        check(f"缓冲区 {block} 与 256 的输出逐采样一致", diff == 0.0, f"最大差 {diff:.2e}")
    y = rl.run_signal("zip", x, rl.helios_sets(speed=100), block=512, extra=["--random-blocks"])[:, 0]
    diff = np.max(np.abs(y - ref))
    check("随机缓冲区长度与 256 一致", diff == 0.0, f"最大差 {diff:.2e}")

    # 正弦幅度跟 1 Hz 慢变，增益一直在动。杂散只算离载波 200 Hz 以外的能量。
    t = np.arange(8 * SR) / SR
    env = rl.db_to_amp(-24 + 8 * np.sin(2 * np.pi * 1.0 * t))
    x = env * np.sin(2 * np.pi * 1000 * t)
    for speed in [50, 100]:
        y = rl.run_signal("zip", x, rl.helios_sets(speed=speed), block=256)[:, 0]
        seg = y[2 * SR:]
        win = np.blackman(len(seg))
        spec = np.abs(np.fft.rfft(seg * win)) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        carrier = spec[np.abs(f - 1000) <= 200].sum()
        rest = spec[np.abs(f - 1000) > 200].sum()
        ratio = 10 * np.log10(max(rest, 1e-30) / carrier)
        g = rl.sample_gain_db(x, y, floor=1e-3)[2 * SR:]
        dg = np.diff(g)
        check(f"速度 {speed} 增益持续移动时的杂散 ≤ -80 dBc", ratio <= -80, f"{ratio:.1f} dBc，最大每采样 {np.max(np.abs(dg)):.4f} dB")

    # 阶跃时最容易听出拉链。用原始输入的同一包络比较，去掉阶跃本身的频谱。
    x = rl.steps([-30, -12], 2)
    y = rl.run_signal("zip", x, rl.helios_sets(speed=100), block=256)[:, 0]
    g = rl.sample_gain_db(x, y, floor=1e-4)
    # 从阶跃开始算，块边界（第 255、256 个差分）上的二阶差分不该比旁边的采样大。
    start = 2 * SR
    d2 = np.abs(np.diff(g[start:start + SR // 2], 2))
    boundary = np.arange(256 - start % 256 - 1, len(d2), 256)
    neighbours = np.concatenate([boundary - 8, boundary + 8])
    neighbours = neighbours[(neighbours >= 0) & (neighbours < len(d2))]
    edge, near = np.mean(d2[boundary[boundary < len(d2)]]), np.mean(d2[neighbours])
    print(f"  阶跃后增益二阶差分：块边界平均 {edge:.6f} dB，相隔 8 采样的位置平均 {near:.6f} dB")
    check("块边界没有台阶（边界不大于旁边的 1.5 倍）", edge <= near * 1.5 + 1e-6)

    section("稳态正弦的失真（增益跟着波形动的程度）：载波 ±20 Hz 之外的能量")
    for freq in [100, 220, 1000]:
        for speed in [50, 100]:
            x = rl.sine(freq, 4, -24)
            y = rl.run_signal("thd", x, rl.helios_sets(speed=speed))[:, 0]
            seg = y[2 * SR:]
            spec = np.abs(np.fft.rfft(seg * np.blackman(len(seg)))) ** 2
            f = np.fft.rfftfreq(len(seg), 1 / SR)
            near = np.abs(f - freq) <= 20
            ratio = 10 * np.log10(max(spec[~near].sum(), 1e-30) / spec[near].sum())
            check(f"{freq} Hz 速度 {speed} 失真 ≤ -50 dBc", ratio <= -50, f"{ratio:.1f} dBc")

def test_sidechain():
    section("音乐感知：主输入 -18 dB 正弦，目标 -18，范围 ±24")
    levels = [-30, -18, -12, -6, 0, -18]
    x = rl.sine(1000, 6 * len(levels), -18)
    sc = np.concatenate([rl.music_bed(6, l) for l in levels])
    sc_path = rl.write_wav("sc.wav", sc)
    for follow in [100, 50, 0]:
        y = rl.run_signal("sc", x, rl.helios_sets(up=24, down=24, speed=75, follow=follow), sc=sc_path)[:, 0]
        g = rl.sample_gain_db(x, y)
        worst = 0.0
        out = []
        for i, l in enumerate(levels):
            measured = np.median(g[6 * SR * i + 4 * SR:6 * SR * (i + 1)])
            expect = follow / 100 * np.clip(l + 18, -12, 12)
            worst = max(worst, abs(measured - expect))
            out.append(f"{l}:{measured:+.2f}/{expect:+.1f}")
        check(f"跟随 {follow}% 目标位移（伴奏电平:实测/理论）", worst <= 0.5, f"{' '.join(out)}，最大误差 {worst:.2f} dB")

    # 伴奏响应时间。
    y = rl.run_signal("sc", x, rl.helios_sets(up=24, down=24, speed=75), sc=sc_path)[:, 0]
    g = rl.sample_gain_db(x, y)
    res = rl.response_ms(g, 12 * SR, 18 * SR)
    print(f"  伴奏 -18 → -12 dB 时目标的 10–90%：{res['rise']:.0f} ms（含速度 75 的推子响应）")

    # 真实伴奏的起伏用粉噪近似，看增益的抖动大小。
    sc_noise = rl.write_wav("sc_pink.wav", rl.pink(12, -12))
    x_short = rl.sine(1000, 12, -18)
    y = rl.run_signal("sc", x_short, rl.helios_sets(up=24, down=24, speed=75), sc=sc_noise)[:, 0]
    g = rl.sample_gain_db(x_short, y)[4 * SR:]
    print(f"  粉噪伴奏 -12 dB：位移中位 {np.median(g):+.2f} dB，5–95% 摆动 {np.percentile(g, 95) - np.percentile(g, 5):.2f} dB")

    x2, _ = hold_signal()
    no_bus = rl.run_signal("sc2", x2, rl.helios_sets(), extra=["--no-sc-bus"])[:, 0]
    silent_path = rl.write_wav("sc_silent.wav", np.zeros(len(x2)))
    silent = rl.run_signal("sc2", x2, rl.helios_sets(), sc=silent_path)[:, 0]
    no_data = rl.run_signal("sc2", x2, rl.helios_sets())[:, 0]
    check("侧链总线未启用 = 接了但静音", np.max(np.abs(no_bus - silent)) == 0.0, f"最大差 {np.max(np.abs(no_bus - silent)):.2e}")
    check("侧链启用未送信号 = 接了但静音", np.max(np.abs(no_data - silent)) == 0.0)

    # 伴奏消失后目标回到旋钮值。
    sc = np.concatenate([rl.music_bed(6, -6), np.zeros(6 * SR)])
    x = rl.sine(1000, 12, -18)
    sc_path = rl.write_wav("sc3.wav", sc)
    y = rl.run_signal("sc3", x, rl.helios_sets(up=24, down=24, speed=75), sc=sc_path)[:, 0]
    g = rl.sample_gain_db(x, y)
    check("伴奏停止后位移回零", abs(np.median(g[-SR:])) <= 0.1, f"有伴奏 {np.median(g[5 * SR:6 * SR]):+.2f}，停止后 {np.median(g[-SR:]):+.2f}")
    undershoot = g[6 * SR:].min()
    check("伴奏停止时目标不被拖到负值（≥ -1 dB）", undershoot >= -1.0, f"最低 {undershoot:+.2f} dB")

    # 单声道主输入 + 立体声侧链。
    y = rl.run_signal("sc4", x, rl.helios_sets(up=24, down=24, speed=75), sc=sc_path, extra=["--mono"])[:, 0]
    g = rl.sample_gain_db(x, y)
    check("单声道主输入带立体声侧链", abs(np.median(g[5 * SR:6 * SR]) - 12) <= 0.5, f"{np.median(g[5 * SR:6 * SR]):+.2f} dB")

def test_lookahead():
    section("Look-ahead")
    for rate in [44100, 48000, 96000]:
        for ms, norm in [(5.0, 0.5), (10.0, 1.0)]:
            x = rl.phrase(2, -18, sr=rate) if rate == 48000 else rl.sine(700, 2, -18, sr=rate)
            path = rl.write_wav("la_in.wav", x, rate)
            out = f"{rl.TMP}/la_out.wav"
            sets = [f"Look-ahead={norm}", "Range Up=0", "Range Down=0"]
            cmd = [rl.PROBE, "render", rl.PLUGIN, path, out]
            for s in sets:
                cmd += ["--set", s]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            reported = int(r.stdout.strip().split()[-1])
            expect = int(ms * 0.001 * rate + 0.5)
            y = rl.read_wav(out)[0][:, 0]
            # probe 去掉了上报的延迟，所以输出应当和输入逐采样对齐。24 位写出，误差限定在 1 LSB 左右。
            diff = np.max(np.abs(y[:len(x) - expect] - x[:len(x) - expect]))
            check(f"{rate} Hz {ms} ms 上报 {reported}（应 {expect}）并对齐", reported == expect and diff < 2e-6, f"对齐后最大差 {diff:.2e}")

    # 增益提前量：-30 → -12 的阶跃，开了 Look-ahead 后增益应当提前开始下压。
    x = rl.steps([-30, -12], 3)
    y0 = rl.run_signal("la", x, rl.helios_sets(speed=100, look=0))[:, 0]
    y10 = rl.run_signal("la", np.concatenate([x, np.zeros(480)]), rl.helios_sets(speed=100, look=10))[:, 0]
    y10 = y10[480:]
    g0 = rl.sample_gain_db(x, y0, floor=1e-4)
    g10 = rl.sample_gain_db(x, y10[:len(x)], floor=1e-4)
    start = 3 * SR
    settled = np.median(g0[start - 4800:start - 480])
    t0 = int(np.argmax(g0[start - 960:] < settled - 3)) - 960
    t10 = int(np.argmax(g10[start - 960:] < settled - 3)) - 960
    print(f"  阶跃到来时增益下落 3 dB 的时刻（相对阶跃）：不开 {t0 / 48:.2f} ms，10 ms 时 {t10 / 48:.2f} ms")
    check("Look-ahead 让增益提前约 10 ms", abs((t0 - t10) / 48 - 10.0) <= 1.0, f"提前 {(t0 - t10) / 48:.2f} ms")

def test_live_changes():
    section("播放中改参数：逐采样增益的连续性")
    x = rl.sine(1000, 8, -30) + rl.pilot(8 * SR)
    jumps = {
        "目标 -18 → -30": [f"3:Target={rl.n_target(-30)}"],
        "上推范围 24 → 3（增益正在 +12）": [f"3:Range Up={rl.n_range(3)}"],
        "速度 50 → 100": [f"3:Speed={rl.n_pct(100)}", f"3:Target={rl.n_target(-24)}"],
    }
    for name, events in jumps.items():
        extra = []
        for e in events:
            extra += ["--at", e]
        y = rl.run_signal("live", x, rl.helios_sets(up=24, down=24), extra=extra)[:, 0]
        g = rl.sample_gain_db(x, y, floor=1e-4)[3 * SR - 480:]
        dg = np.max(np.abs(np.diff(g[::32]))) / 32
        check(f"{name}：每采样变化 ≤ 0.02 dB", dg <= 0.02, f"最大 {dg:.4f} dB，终值 {np.median(g[-SR:]):+.2f}")

    # 静音时缩小范围，保持中的增益也要回到新范围内。
    x = np.concatenate([rl.phrase(3, -30, syllables=False), np.zeros(5 * SR)])
    x = x + rl.pilot(len(x))
    y = rl.run_signal("live", x, rl.helios_sets(up=24, down=24), extra=["--at", f"4:Range Up={rl.n_range(3)}"])[:, 0]
    g = rl.sample_gain_db(x, y)
    check("静音中缩小上推范围后增益回到范围内", g[-SR:].max() <= 3.01, f"之前 {np.median(g[int(3.5 * SR):4 * SR]):+.2f}，之后 {g[-SR:].max():+.2f}")

def test_robust():
    section("健壮性")
    x = rl.phrase(20, -24)
    _, text = rl.render(rl.write_wav("cpu.wav", x), "cpu_out.wav", rl.helios_sets(), block=64)
    print(f"  CPU：{text}（单线程，1 个实例，缓冲区 64）")
    for rate in [44100, 48000, 88200, 96000, 192000]:
        x = rl.steps([-24, -12, -24], 6, sr=rate)
        y = rl.run_signal("rob", x, rl.helios_sets(speed=50), sr=rate)[:, 0]
        g = rl.sample_gain_db(x, y)
        res = rl.response_ms(g, 6 * rate, 12 * rate, sr=rate)
        print(f"  {rate} Hz：下压 10–90% {res['rise']:.1f} ms，稳态 {res['settled']:+.2f} dB")
        check(f"{rate} Hz 稳态正确", abs(res["settled"] + 6) <= 0.1)

    for rate in [44100, 192000]:
        x = rl.steps([-24, -12], 1.5, sr=rate)
        whole = rl.run_signal("rob1", x, rl.helios_sets(speed=75), sr=rate, block=256)[:, 0]
        single = rl.run_signal("rob1", x, rl.helios_sets(speed=75), sr=rate, block=1)[:, 0]
        check(f"{rate} Hz 下 1 采样缓冲区与 256 一致", np.max(np.abs(whole - single)) == 0.0, f"最大差 {np.max(np.abs(whole - single)):.2e}")

    x = np.zeros(SR * 2)
    y = rl.run_signal("rob", x, rl.helios_sets(), block=1)
    check("纯静音输出全零（1 采样缓冲区）", np.max(np.abs(y)) == 0.0)

    x = rl.phrase(1, -18)
    x = np.concatenate([x, np.zeros(SR * 30)])
    y = rl.run_signal("rob", x, rl.helios_sets(speed=100))
    tail = y[-SR:]
    denormal = np.sum((np.abs(tail) > 0) & (np.abs(tail) < 1.2e-38))
    check("长静音后无 denormal/NaN", denormal == 0 and np.all(np.isfinite(y)))

    x = rl.sine(1000, 3, -18)
    x[SR] = np.nan
    x[SR + 10] = np.inf
    y = rl.run_signal("rob", x, rl.helios_sets())[:, 0]
    check("输入坏值不污染后续处理", np.all(np.isfinite(y[SR + 100:])), f"末段电平 {rl.level_db(y[-SR:]):.2f} dB")

    x = rl.phrase(3, -24)
    y = rl.run_signal("rob", x, rl.helios_sets(), extra=["--mono"])[:, 0]
    check("单声道工作", rl.level_db(y[SR:-SR // 2]) > -20)

    # 左右反相的人声。
    y = rl.run_signal("rob", np.stack([x, -x], axis=1), rl.helios_sets())
    check("左右反相的人声不被判成静音", rl.level_db(y[SR:-SR // 2, 0]) > -20, f"{rl.level_db(y[SR:-SR // 2, 0]):.2f} dB")

def test_automation():
    section("写自动化（独立音频线程，消息线程正常运转）")
    # 幅度一直在变，增益就一直在动，能看出写入的密度。
    t = np.arange(int(7.5 * SR)) / SR
    x = rl.db_to_amp(-24 + 9 * np.sin(2 * np.pi * 0.7 * t)) * np.sin(2 * np.pi * 1000 * t)
    path = rl.write_wav("auto_in.wav", x)
    out = f"{rl.TMP}/auto_out.wav"
    log = f"{rl.TMP}/auto_log.csv"
    cmd = [rl.BENCH, "write", rl.PLUGIN, path, out, log, "--block", "256"]
    for s in rl.helios_sets(speed=75):
        cmd += ["--set", s]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print("  " + r.stdout.strip())
    if r.returncode != 0:
        check("write 跑通", False, r.stdout)
        return
    rows = np.loadtxt(log, delimiter=",", skiprows=1, ndmin=2)
    values = rows[rows[:, 0] == 0]
    check("音频线程上没有参数通知", rows[:, 3].sum() == 0)
    check("全部通知都在消息线程", rows[:, 4].min() == 1)
    check("手势成对且只有一次", (rows[:, 0] == 1).sum() == 1 and (rows[:, 0] == 2).sum() == 1)
    written_seconds = (values[-1, 1] - values[0, 1]) / SR
    check("写入密度 ≥ 15 点/秒", len(values) / max(written_seconds, 1e-3) >= 15, f"{len(values)} 点 / {written_seconds:.2f} s")

    y = rl.read_wav(out)[0][:, 0]
    g = rl.sample_gain_db(x, y)
    written = values[:, 2] * 48 - 24
    pos = values[:, 1].astype(int) - 1
    errors = written - g[np.clip(pos, 0, len(g) - 1)]
    check("写入值和实际增益吻合（中位绝对误差 ≤ 0.2 dB）", np.median(np.abs(errors)) <= 0.2,
          f"中位 {np.median(np.abs(errors)):.3f}，最大 {np.max(np.abs(errors)):.3f} dB，写入范围 {written.min():+.2f}…{written.max():+.2f}")
    off_at = int(len(x) * 0.9)
    check("开关关闭后不再写入", values[-1, 1] <= off_at + SR // 10, f"最后写入 {values[-1, 1] / SR:.2f} s，关闭 {off_at / SR:.2f} s")

    y2 = rl.run_signal("auto", x, rl.helios_sets(speed=75))[:, 0]
    check("写自动化不改变音频", np.max(np.abs(y - y2)) == 0.0, f"最大差 {np.max(np.abs(y - y2)):.2e}")

    cmd = [rl.BENCH, "write", rl.PLUGIN, path, out, log, "--block", "256", "--offline"]
    for s in rl.helios_sets(speed=75):
        cmd += ["--set", s]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print("  离线渲染：" + r.stdout.strip())
    check("离线渲染时不写自动化", "values 0" in r.stdout)

def test_real(path):
    section(f"真实人声：{path}")
    x, sr = rl.read_wav(path)
    x = x[:min(len(x), 60 * sr)]
    p = rl.write_wav("real_in.wav", x, sr)
    vpos, vmask = rl.voiced_frames(x[:, 0], sr)
    for speed in [25, 50, 75, 100]:
        y, _ = rl.render(p, "real_out.wav", rl.helios_sets(speed=speed))
        g = rl.sample_gain_db(x[:, 0], y[:, 0], floor=1e-4)
        pos, lin = rl.short_levels(x[:, 0], sr)
        _, lout = rl.short_levels(y[:, 0], sr)
        voiced = lin > np.percentile(lin, 95) - 20
        print(f"  速度 {speed}：浊音帧 50 ms 电平标准差 {np.std(rl.frame_levels(x[:, 0], vpos, sr)[vmask]):.2f} → "
              f"{np.std(rl.frame_levels(y[:, 0], vpos, sr)[vmask]):.2f} dB，中位 {np.median(rl.frame_levels(y[:, 0], vpos, sr)[vmask]):.1f} dB，"
              f"增益范围 {g.min():+.1f}…{g.max():+.1f}")
        # 静音段（400 ms 电平低于人声 30 dB）的增益在相邻窗之间的移动。
        quiet = lin < np.percentile(lin, 95) - 30
        if quiet.sum() > 3:
            print(f"    低电平窗 {quiet.sum()} 个，这些窗内增益的最大摆动中位数 "
                  f"{np.median([np.ptp(g[a:a + int(0.4 * sr)]) for a in pos[quiet]]):.3f} dB")
    print("  这些数字只说明行为，听感由用户判断。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="真实人声 WAV，可选，不进仓库")
    ap.add_argument("--only", help="只跑某项，如 hold")
    args = ap.parse_args()

    tests = dict(probe=test_probe, accuracy=test_accuracy, ranges=test_ranges, hold=test_hold, false_open=test_false_open, loud_noise=test_loud_noise, speed=test_speed,
                 zipper=test_zipper, sidechain=test_sidechain, lookahead=test_lookahead, live=test_live_changes, robust=test_robust,
                 automation=test_automation)
    for name, fn in tests.items():
        if args.only is None or args.only == name:
            fn()
    if args.real and args.only in (None, "real"):
        test_real(args.real)

    print()
    if failures:
        print(f"FAILED {len(failures)}: " + "; ".join(failures))
        return 1
    print("ALL PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
