"""行为验证：直通、缓冲区无关、静音、动态频段、Look-ahead、立体声模式、扫频和快速拖动。

每一项打印 PASS/FAIL 和数字。退出码为失败项数。
"""
import os
import re
import sys

import numpy as np
from scipy import signal

import iris_common as ic

RATE = 48000
failures = 0


def check(name, ok, detail):
    global failures
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        failures += 1


def noise(seconds, level=0.1, seed=1, stereo=True):
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    x = rng.standard_normal((n, 2 if stereo else 1)) * level
    return x if stereo else x[:, 0]


def tone(freq, seconds, level_db, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return 10 ** (level_db / 20) * np.sin(2 * np.pi * freq * t)


def rms_db(x):
    return 20 * np.log10(max(np.sqrt(np.mean(np.square(x))), 1e-12))


def render(name, x, sets, sweeps=(), block=256, mono=False):
    in_path = ic.write_wav(f"{name}_in.wav", x, RATE)
    out_path = os.path.join(ic.WORK, f"{name}_out.wav")
    log = ic.render_bench(in_path, out_path, sets, sweeps, block=block, mono=mono)
    _, y = ic.read_wav(out_path)
    return y, log


def test_null():
    x = noise(1.0)
    y, _ = render("null", x, [])
    check("无频段时完全直通", np.max(np.abs(y - x.astype(np.float32))) == 0.0, f"最大差 {np.max(np.abs(y - x.astype(np.float32))):.3e}")


def rich_sets():
    sets = []
    sets += ic.band_sets(1, kind="lowCut", freq=60.0, slope=48, q=1.0)
    sets += ic.band_sets(2, kind="bell", freq=300.0, gain=-5.0, q=2.0, dynamic=True, thresh=-30.0, rng=-6.0)
    sets += ic.band_sets(3, kind="bell", freq=3000.0, gain=4.0, q=1.0, stereo="mid")
    sets += ic.band_sets(4, kind="highShelf", freq=9000.0, gain=3.0, q=1.0, stereo="side", dynamic=True, thresh=-40.0, rng=-4.0)
    sets += ic.band_sets(5, kind="notch", freq=7000.0, q=6.0, stereo="left")
    sets += ic.band_sets(6, kind="tilt", freq=1000.0, gain=2.0, q=1.0, stereo="right")
    sets += ic.band_sets(7, kind="highCut", freq=18000.0, slope=18, q=1.0)
    return sets


def test_block_independence():
    x = noise(1.0)
    sets = rich_sets()
    base, _ = render("block1024", x, sets, block=1024)
    for block in (1, 32, 113):
        y, _ = render(f"block{block}", x, sets, block=block)
        diff = np.max(np.abs(y - base))
        check(f"缓冲区 {block} 与 1024 输出一致", diff < 1e-5, f"最大差 {diff:.2e}")


def test_silence():
    x = np.zeros((RATE * 2, 2))
    x[0, :] = 0.5
    y, _ = render("silence", x, rich_sets())
    tail = y[-RATE // 2:]
    finite = np.isfinite(y).all()
    check("静音后没有 NaN/Inf，尾部归零", finite and np.max(np.abs(tail)) < 1e-12, f"尾部峰值 {np.max(np.abs(tail)):.2e}，全部有限 {finite}")
    tiny = np.abs(tail[tail != 0.0])
    denormal = tiny[tiny < 1.17549e-38]
    check("尾部没有 denormal", denormal.size == 0, f"{denormal.size} 个")


def test_dynamic():
    # 钟形静态 0 dB，阈值 -24，范围 -6。过渡宽度 max(6, 12) = 12 dB。
    freq = 1000.0
    sets = ic.band_sets(1, kind="bell", freq=freq, gain=0.0, q=2.0, dynamic=True, thresh=-24.0, rng=-6.0)
    print("动态频段：1 kHz 钟形，阈值 -24 dB，范围 -6 dB，过渡 12 dB")
    measured = []
    for level in (-48, -36, -24, -18, -12, -6, 0):
        y, _ = render(f"dyn_{-level}", tone(freq, 1.0, level), sets)
        gain = rms_db(y[RATE // 2:, 0]) - rms_db(tone(freq, 1.0, level)[RATE // 2:])
        measured.append((level, gain))
        print(f"  输入 {level:+4d} dB -> 增益 {gain:+6.2f} dB")
    below = dict(measured)[-36]
    full = dict(measured)[-6]
    check("低于阈值不动作", abs(below) < 0.05, f"{below:+.2f} dB")
    # 检测器看的是整流后的包络，正弦平均值低于峰值，所以用区间判断。
    check("高于阈值达到范围", -6.3 < full < -5.0, f"{full:+.2f} dB（目标 -6）")
    gains = [g for _, g in measured]
    check("增益随电平单调变化", all(b <= a + 0.02 for a, b in zip(gains, gains[1:])), " ".join(f"{g:+.2f}" for g in gains))

    # 带外信号不触发。
    y, _ = render("dyn_outband", tone(100.0, 1.0, -6), sets)
    gain = rms_db(y[RATE // 2:, 0]) - rms_db(tone(100.0, 1.0, -6)[RATE // 2:])
    check("带外信号不触发", abs(gain) < 0.1, f"100 Hz -6 dB 增益 {gain:+.2f} dB")

    # 增益方向可以反过来：高电平时提升。
    up = ic.band_sets(1, kind="bell", freq=freq, gain=0.0, q=2.0, dynamic=True, thresh=-24.0, rng=6.0)
    y, _ = render("dyn_up", tone(freq, 1.0, -6), up)
    gain = rms_db(y[RATE // 2:, 0]) - rms_db(tone(freq, 1.0, -6)[RATE // 2:])
    check("范围为正时高电平提升", 5.0 < gain < 6.3, f"{gain:+.2f} dB")

    # 突发音：响度从 -40 跳到 -6，看增益下降的时间过程，并确认没有开关噪声。
    x = np.concatenate([tone(freq, 0.5, -40), tone(freq, 0.5, -6)])
    y, _ = render("dyn_burst", x, sets)
    envelope = np.abs(signal.hilbert(y[:, 0]))
    input_envelope = 10 ** (-6 / 20)
    after = envelope[RATE // 2 + int(0.1 * RATE):]
    gain_after = 20 * np.log10(np.mean(after) / input_envelope)
    check("突发音稳定后增益到位", -6.5 < gain_after < -5.0, f"{gain_after:+.2f} dB")


def test_lookahead():
    x = noise(0.5)
    sets = ["Look-ahead=0.5"]  # 5 ms
    y, log = render("look", x, sets)
    match = re.search(r"reported latency (\d+)", log)
    reported = int(match.group(1)) if match else -1
    expected = round(5.0 * 0.001 * RATE)
    check("Look-ahead 5 ms 上报延迟", reported == expected, f"上报 {reported}，期望 {expected}")
    correlate = np.max(np.abs(y[:, 0] - x[:, 0]))
    check("补偿延迟后与输入对齐", correlate < 1e-5, f"最大差 {correlate:.2e}")

    # 打开后增益应该提前到达：比较同一突发音在开/关 Look-ahead 下的包络。
    x = np.concatenate([tone(1000.0, 0.5, -40), tone(1000.0, 0.5, -6)])
    base = ic.band_sets(1, kind="bell", freq=1000.0, gain=0.0, q=2.0, dynamic=True, thresh=-24.0, rng=-12.0)
    off, _ = render("look_off", x, base)
    on, _ = render("look_on", x, base + ["Look-ahead=0.5"])
    onset = RATE // 2
    peak_off = np.max(np.abs(off[onset:onset + int(0.002 * RATE), 0]))
    peak_on = np.max(np.abs(on[onset:onset + int(0.002 * RATE), 0]))
    check("开启后起音更早被压低", peak_on < peak_off, f"起音 2 ms 内峰值：关 {peak_off:.3f}，开 {peak_on:.3f}")


def test_stereo_modes():
    x = noise(1.0, seed=3)
    x[:, 1] = noise(1.0, seed=4)[:, 1]
    common = dict(kind="bell", freq=1000.0, gain=12.0, q=1.0)

    y, _ = render("st_left", x, ic.band_sets(1, stereo="left", **common))
    check("左：右声道不变", np.max(np.abs(y[:, 1] - x[:, 1])) < 1e-6, f"{np.max(np.abs(y[:, 1] - x[:, 1])):.2e}")
    check("左：左声道改变", np.max(np.abs(y[:, 0] - x[:, 0])) > 0.01, f"{np.max(np.abs(y[:, 0] - x[:, 0])):.2e}")

    y, _ = render("st_right", x, ic.band_sets(1, stereo="right", **common))
    check("右：左声道不变", np.max(np.abs(y[:, 0] - x[:, 0])) < 1e-6, f"{np.max(np.abs(y[:, 0] - x[:, 0])):.2e}")
    check("右：右声道改变", np.max(np.abs(y[:, 1] - x[:, 1])) > 0.01, f"{np.max(np.abs(y[:, 1] - x[:, 1])):.2e}")

    y, _ = render("st_mid", x, ic.band_sets(1, stereo="mid", **common))
    side_diff = np.max(np.abs((y[:, 0] - y[:, 1]) - (x[:, 0] - x[:, 1])))
    mid_diff = np.max(np.abs((y[:, 0] + y[:, 1]) - (x[:, 0] + x[:, 1])))
    check("中：侧信号不变，中信号改变", side_diff < 1e-5 and mid_diff > 0.01, f"侧差 {side_diff:.2e}，中差 {mid_diff:.2e}")

    y, _ = render("st_side", x, ic.band_sets(1, stereo="side", **common))
    side_diff = np.max(np.abs((y[:, 0] - y[:, 1]) - (x[:, 0] - x[:, 1])))
    mid_diff = np.max(np.abs((y[:, 0] + y[:, 1]) - (x[:, 0] + x[:, 1])))
    check("侧：中信号不变，侧信号改变", mid_diff < 1e-5 and side_diff > 0.01, f"中差 {mid_diff:.2e}，侧差 {side_diff:.2e}")


def test_output_gain():
    x = noise(0.5)
    y, _ = render("output", x, ["Output=" + str(ic.norm_linear(6.0, -30.0, 30.0))])
    gain = rms_db(y[RATE // 4:, 0]) - rms_db(x[RATE // 4:, 0])
    check("输出增益 +6 dB", abs(gain - 6.0) < 0.01, f"{gain:+.3f} dB")


def wideband_peak_db(y, cutoff):
    sos = signal.butter(8, cutoff, btype="high", fs=RATE, output="sos")
    filtered = signal.sosfilt(sos, y)[RATE // 10:]
    return 20 * np.log10(max(np.max(np.abs(filtered)), 1e-12))


def naive_sweep(x, sweep_period, low, high, gain, q, block):
    """对照：每个缓冲区直接换直接型 biquad 系数的普通做法。"""
    out = np.zeros_like(x)
    z = np.zeros(2)
    n = 0
    while n < len(x):
        phase = (n / RATE / sweep_period) % 1.0
        tri = 2 * phase if phase < 0.5 else 2 - 2 * phase
        f = low * (high / low) ** tri
        a = 10 ** (gain / 40)
        w = 2 * np.pi * f / RATE
        alpha = np.sin(w) / (2 * q)
        b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
        aa = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
        out[n:n + block], z = signal.lfilter(b / aa[0], aa / aa[0], x[n:n + block], zi=z)
        n += block
    return out


def test_sweeps():
    # 1 kHz 正弦过一个来回扫动的窄钟形。纯净的话 8 kHz 以上应该几乎没有能量。
    x = tone(1000.0, 4.0, -12)
    low, high = 100.0, 8000.0
    sets = ic.band_sets(1, kind="bell", freq=1000.0, gain=12.0, q=4.0)
    sweep_period = 0.2  # 每秒 5 个来回，约 6 oct，远快于手拖
    sweeps = [f"B01 Freq={ic.norm_log(low, 10.0, 30000.0)}:{ic.norm_log(high, 10.0, 30000.0)}:{sweep_period}"]
    y, _ = render("sweep_fast", x, sets, sweeps)
    ref = wideband_peak_db(x, 8000.0)
    peak = wideband_peak_db(y[:, 0], 8000.0)
    naive = naive_sweep(x, sweep_period, low, high, 12.0, 4.0, 256)
    naive_peak = wideband_peak_db(naive, 8000.0)
    print(f"快速扫频（1 kHz 正弦，钟形 Q4 +12 dB，100 Hz 到 8 kHz，每 0.2 s 一个来回）8 kHz 以上峰值：Iris {peak:.1f} dBFS，换系数对照 {naive_peak:.1f} dBFS，输入 {ref:.1f} dBFS")
    check("快速扫频无宽带咔嗒（8 kHz 以上峰值 < -90 dBFS）", peak < -90.0, f"{peak:.1f} dBFS")

    # 增益每 50 ms 一个来回（±12 dB），远快于手拖。
    sweeps = ["B01 Gain=0.3:0.7:0.05"]
    y, _ = render("sweep_gain", x, ic.band_sets(1, kind="bell", freq=1000.0, gain=0.0, q=2.0), sweeps)
    peak = wideband_peak_db(y[:, 0], 8000.0)
    check("快速扫增益无宽带咔嗒（8 kHz 以上 < -80 dBFS）", peak < -80.0, f"{peak:.1f} dBFS")

    # 低频高 Q：SVF 结构的优势场景。
    x_low = tone(120.0, 4.0, -12)
    sweeps = [f"B01 Freq={ic.norm_log(40.0, 10.0, 30000.0)}:{ic.norm_log(400.0, 10.0, 30000.0)}:0.2"]
    y, _ = render("sweep_lowq", x_low, ic.band_sets(1, kind="bell", freq=100.0, gain=18.0, q=20.0), sweeps)
    peak = wideband_peak_db(y[:, 0], 4000.0)
    check("低频高 Q 快速扫频稳定且无宽带咔嗒（4 kHz 以上 < -80 dBFS）", np.isfinite(y).all() and peak < -80.0, f"{peak:.1f} dBFS，输出峰值 {np.max(np.abs(y)):.2f}")

    # 高陡切除扫频。
    sweeps = [f"B01 Freq={ic.norm_log(200.0, 10.0, 30000.0)}:{ic.norm_log(2000.0, 10.0, 30000.0)}:0.2"]
    y, _ = render("sweep_cut", tone(5000.0, 4.0, -12), ic.band_sets(1, kind="lowCut", freq=400.0, slope=96, q=1.0), sweeps)
    peak = wideband_peak_db(y[:, 0], 12000.0)
    check("96 dB/oct 切除快速扫频（12 kHz 以上 < -80 dBFS）", np.isfinite(y).all() and peak < -80.0, f"{peak:.1f} dBFS")

    # 开关频段、改类型：应该是淡入淡出，没有阶跃。
    sweeps = ["B01 On=0:1:0.1"]
    y, _ = render("toggle", x, ic.band_sets(1, kind="bell", freq=1000.0, gain=12.0, q=1.0), sweeps)
    peak = wideband_peak_db(y[:, 0], 8000.0)
    check("快速开关频段（8 kHz 以上 < -80 dBFS）", peak < -80.0, f"{peak:.1f} dBFS")

    sweeps = ["B01 Type=0:1:0.2"]
    y, _ = render("type_toggle", x, ic.band_sets(1, kind="bell", freq=1000.0, gain=12.0, q=1.0), sweeps)
    peak = wideband_peak_db(y[:, 0], 8000.0)
    # 每 200 ms 把七种类型过一遍，1 kHz 的电平本身就在大幅跳变，这里只要求不出现硬切。
    check("连续改类型（8 kHz 以上 < -60 dBFS）", peak < -60.0, f"{peak:.1f} dBFS")

    # 动态频段的阈值被自动化时，增益量也要平滑。
    sets = ic.band_sets(1, kind="bell", freq=1000.0, gain=0.0, q=2.0, dynamic=True, thresh=-24.0, rng=-12.0)
    sweeps = [f"B01 Threshold={ic.norm_linear(-30.0, -80.0, 0.0)}:{ic.norm_linear(0.0, -80.0, 0.0)}:0.1"]
    y, _ = render("sweep_thresh", x, sets, sweeps)
    peak = wideband_peak_db(y[:, 0], 8000.0)
    check("动态频段快速扫阈值（8 kHz 以上 < -80 dBFS）", peak < -80.0, f"{peak:.1f} dBFS")


def test_dynamic_timing():
    # 默认起音 5 ms、释放 120 ms。量增益变化到 90% 所用的时间。
    freq = 1000.0
    sets = ic.band_sets(1, kind="bell", freq=freq, gain=0.0, q=2.0, dynamic=True, thresh=-24.0, rng=-6.0)
    x = np.concatenate([tone(freq, 0.5, -40), tone(freq, 0.5, -6), tone(freq, 1.0, -40)])
    y, _ = render("dyn_time", x, sets)
    window = int(0.0005 * RATE)
    env = np.sqrt(np.convolve(y[:, 0] ** 2, np.ones(window * 2) / (window * 2), mode="same")) * np.sqrt(2)
    ref = np.sqrt(np.convolve(x ** 2, np.ones(window * 2) / (window * 2), mode="same")) * np.sqrt(2)
    gain = 20 * np.log10(np.maximum(env, 1e-9) / np.maximum(ref, 1e-9))
    onset, fall = RATE // 2, RATE
    attack = np.argmax(gain[onset:fall] < -5.4) / RATE * 1000
    skip = int(0.003 * RATE)  # 测量窗口跨过电平跳变时比值不可信
    release = (np.argmax(gain[fall + skip:] > -0.6) + skip) / RATE * 1000
    print(f"动态时间（默认起音 5 ms，释放 120 ms）：增益到 90% 的实测起音 {attack:.1f} ms，释放 {release:.1f} ms")
    check("动态起音和释放有响应", 0.0 < attack < 60.0 and 20.0 < release < 900.0, f"{attack:.1f} ms / {release:.1f} ms")


def test_mono():
    x = noise(0.5)[:, 0]
    sets = ic.band_sets(1, kind="bell", freq=1000.0, gain=6.0, q=1.0)
    stereo, _ = render("mono_ref", x, sets)
    y, _ = render("mono", x, sets, mono=True)
    diff = np.max(np.abs(y[:, 0] - stereo[:, 0]))
    check("单声道总线与立体声左声道一致", diff < 1e-6, f"最大差 {diff:.2e}")
    y, _ = render("mono_side", x, ic.band_sets(1, kind="bell", freq=1000.0, gain=6.0, q=1.0, stereo="side"), mono=True)
    diff = np.max(np.abs(y[:, 0] - x))
    check("单声道总线上侧频段不起作用", diff < 1e-6, f"最大差 {diff:.2e}")
    y, _ = render("mono_block1", x, sets, block=1, mono=True)
    check("单声道缓冲区 1 采样", np.isfinite(y).all(), "全部有限")


def main():
    for test in (test_null, test_output_gain, test_block_independence, test_silence, test_dynamic, test_lookahead, test_stereo_modes, test_sweeps, test_dynamic_timing, test_mono):
        test()
    print(f"\n{failures} 项失败")
    return failures


if __name__ == "__main__":
    sys.exit(main())
