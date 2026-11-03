#!/usr/bin/env python
"""Aegis 的可重复验证脚本。需要 numpy 和 scipy。

    cmake --build D:/mint-build/limiter --config Release --target mint_probe mint_limiter_VST3 aegis_bench --parallel 4
    python plugins/limiter/tests/run_tests.py [--build D:/mint-build/limiter] [--only peaks,align,...] [--reference ... --reference-name ...]

--reference <容器路径> --reference-name <插件名> 同时跑对标插件的黑盒对比（只调用插件处理音频，不读它的任何文件）。
具体路径和插件名写在 NOTES.md。
硬指标不过时退出码非 0。结果写到 <build>/aegis_tests/results.txt。
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

RATE = 48000
LOOKAHEADS = {"0": 0.0, "1.5": 1 / 3, "5": 2 / 3, "10": 1.0}
LOOKAHEAD_MS = {"0": 0.0, "1.5": 1.5, "5": 5.0, "10": 10.0}
DEFAULT_LOOKAHEAD_MS = 1.5
DEFAULT_LATENCY_48K = 72
DEFAULT_CEILING_DB = -1.0
FULL_SCALE_24 = 2.0 ** 23
LSB_DB = 20 * np.log10(1 + 1 / (0.5 * FULL_SCALE_24))


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def norm_db(value):
    return (value + 30.0) / 30.0


def norm_release(ms):
    return ((ms - 1.0) / 999.0) ** 0.3


class Runner:
    def __init__(self, build):
        self.work = build / "aegis_tests"
        self.work.mkdir(parents=True, exist_ok=True)
        self.probe = build / "tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
        self.bench = build / "plugins/limiter/tests/aegis_bench_artefacts/Release/aegis_bench.exe"
        self.plugin = build / "plugins/limiter/mint_limiter_artefacts/Release/VST3/Mintaria Aegis.vst3"
        self.lines = []
        self.failures = 0

    def say(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def check(self, ok, text):
        self.say(("PASS " if ok else "FAIL ") + text)
        self.failures += 0 if ok else 1

    def write(self, name, x, rate=RATE):
        path = self.work / name
        wavfile.write(path, rate, np.asarray(x, dtype=np.float32))
        return path

    def render(self, source, lookahead="0", threshold=0.0, ceiling=-1.0, release=60.0, automatic=True,
               block=256, tool="bench", extra=()):
        out = self.work / "out.wav"
        settings = ["--set", f"Threshold={norm_db(threshold)}", "--set", f"Ceiling={norm_db(ceiling)}",
                    "--set", f"Release={norm_release(release)}", "--set", f"Auto Release={1 if automatic else 0}",
                    "--set", f"Look-ahead={LOOKAHEADS[lookahead]}", "--block", str(block)] + list(extra)
        exe = self.bench if tool == "bench" else self.probe
        result = subprocess.run([str(exe), "render", str(self.plugin), str(source), str(out)] + settings,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        latency = int(result.stdout.strip().split()[-1])
        return read(out), latency


def read(path):
    rate, x = wavfile.read(path)
    if x.dtype == np.int32:
        x = x / 2147483648.0
    return x.astype(np.float64)


def true_peak(x, factor=8):
    data = x if x.ndim > 1 else x[:, None]
    peak = 0.0
    for ch in range(data.shape[1]):
        padded = np.concatenate([np.zeros(256), data[:, ch], np.zeros(256)])
        peak = max(peak, np.max(np.abs(resample_poly(padded, factor, 1, window=("kaiser", 12.0)))))
    return peak


def stereo(x, y=None):
    return np.stack([x, x if y is None else y], axis=1)


def sine(freq, seconds, amp=1.0, phase=0.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t + phase)


def hostile_signals():
    rng = np.random.default_rng(10)
    n = RATE
    burst = np.zeros(n)
    for start in range(4800, n - 480, 7200):
        burst[start:start + 192] = rng.choice([-1.0, 1.0], 192)
    t = np.arange(n) / RATE
    square = np.sign(np.sin(2 * np.pi * 100.0 * t) + 1e-12)
    square_hi = np.sign(np.sin(2 * np.pi * 3001.0 * t) + 1e-12)
    clipped = np.clip(8.0 * np.sin(2 * np.pi * 220.0 * t), -1, 1)
    # fs/4 且相位 45 度：采样值只有 0.707，真实峰值 1.0，是 True Peak 的经典最坏情况。
    tp_worst = np.sin(2 * np.pi * (RATE / 4) * t + np.pi / 4)
    # 交替满幅，再周期性翻转，样本之间的连续波形能冲得比采样值高得多。
    isp = np.tile([1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0], n // 10)
    return {
        "满幅突发": stereo(burst, -burst),
        "方波 100 Hz": stereo(square),
        "方波 3 kHz": stereo(square_hi),
        "削波正弦": stereo(clipped),
        "满幅噪声": stereo(rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)),
        "fs/4 45°": stereo(tp_worst),
        "交替样本串": stereo(isp),
        "元音代理": stereo(vowel_signal()),
    }


def test_host(r):
    r.say("\n## 0. mint_probe check、截图和浮点压力测试")
    result = subprocess.run([str(r.probe), "check", str(r.plugin), "--expect-latency", str(DEFAULT_LATENCY_48K)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    r.check(result.returncode == 0, f"mint_probe check --expect-latency {DEFAULT_LATENCY_48K}：" + " ".join(result.stdout.split()))

    shot = r.work / "aegis_shot.png"
    result = subprocess.run([str(r.probe), "shot", str(r.plugin), str(shot)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    r.check(result.returncode == 0, f"mint_probe shot：{result.stdout.strip()}，{shot}")

    result = subprocess.run([str(r.bench), "stress", str(r.plugin), "--cases", "3000"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in result.stdout.strip().splitlines():
        r.say(line)
    r.check(result.returncode == 0, "aegis_bench stress")


def test_peaks(r, reference=None):
    r.say("\n## 1. 输出峰值（mint_probe render，24 位 WAV）")
    r.say("上限 -1.0 dB 和 -0.3 dB，阈值 0 / -12 dB，手动 1 ms 和自动释放。sample 峰值超过上限 1 个 24 位 LSB 判失败。")
    r.say("| 信号 | Look-ahead | 最大 sample 峰值相对上限 (dB) | 最大 True Peak 相对上限 (dB) |")
    r.say("|---|---|---|---|")
    worst_sample = {k: -999.0 for k in LOOKAHEADS}
    worst_tp = {k: -999.0 for k in LOOKAHEADS}
    for name, x in hostile_signals().items():
        source = r.write("in.wav", x)
        for look in LOOKAHEADS:
            sample_max = tp_max = -999.0
            for threshold in (0.0, -12.0):
                for ceiling in (-1.0, -0.3):
                    for release, automatic in ((1.0, False), (60.0, True)):
                        y, _ = r.render(source, look, threshold, ceiling, release, automatic, tool="probe")
                        sample_max = max(sample_max, db(np.max(np.abs(y))) - ceiling)
                        tp_max = max(tp_max, db(true_peak(y)) - ceiling)
            worst_sample[look] = max(worst_sample[look], sample_max)
            worst_tp[look] = max(worst_tp[look], tp_max)
            r.say(f"| {name} | {look} ms | {sample_max:+.5f} | {tp_max:+.2f} |")
    for look in LOOKAHEADS:
        r.check(worst_sample[look] <= LSB_DB,
                f"Look-ahead {look} ms：sample 峰值最大 {worst_sample[look]:+.6f} dB 相对上限，True Peak 最大 {worst_tp[look]:+.2f} dB")


def test_alignment(r):
    r.say("\n## 2. 延迟上报与对齐")
    # 与插件的 std::round 一致：.5 向远离 0 取整（Python 的 round 是银行家取整）。
    expected = lambda rate, ms: int(np.floor(ms * 0.001 * rate + 0.5))
    for rate in (44100, 48000, 96000, 192000):
        x = stereo(0.1 * np.random.default_rng(3).normal(size=rate // 4).clip(-3, 3))
        source = r.write("in.wav", x, rate)
        for look in LOOKAHEADS:
            y, latency = r.render(source, look, 0.0, 0.0, tool="probe")
            ok = latency == expected(rate, LOOKAHEAD_MS[look])
            residual = np.max(np.abs(y - x))
            r.check(ok and residual <= 2 / FULL_SCALE_24,
                    f"{rate} Hz，{look} ms：上报 {latency} 采样，低电平残差峰值 {db(residual):.1f} dBFS")

        # 默认状态，不设任何参数：前瞻 1.5 ms、上限 -1 dB。
        out = r.work / "out.wav"
        result = subprocess.run([str(r.probe), "render", str(r.plugin), str(source), str(out)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        latency = int(result.stdout.strip().split()[-1])
        residual = np.max(np.abs(read(out) - x * 10 ** (DEFAULT_CEILING_DB / 20)))
        r.check(latency == expected(rate, DEFAULT_LOOKAHEAD_MS) and residual <= 3 / FULL_SCALE_24,
                f"{rate} Hz，默认状态：上报 {latency} 采样，对齐后残差峰值 {db(residual):.1f} dBFS")

    # 正在限制时也要对齐：单个超阈值脉冲的位置不能动。
    x = np.zeros(RATE // 2)
    x[6000] = 1.0
    x[12000] = -1.0
    source = r.write("in.wav", stereo(x))
    for look in LOOKAHEADS:
        y, _ = r.render(source, look, -12.0, -1.0, tool="probe")
        positions = sorted(int(i) for i in np.argsort(np.abs(y[:, 0]))[-2:])
        r.check(positions == [6000, 12000], f"{look} ms：超阈值脉冲输出位置 {positions}")

    # 拐点以下完全透明：0 延迟低于 -1 dB、前瞻低于 0 dB 时输出就是输入。
    for look, amp in (("0", 0.89), ("1.5", 0.999), ("5", 0.999), ("10", 0.999)):
        x = stereo(sine(997, 0.5, amp))
        y, _ = r.render(r.write("in.wav", x), look, 0.0, 0.0)
        residual = np.max(np.abs(y - x))
        r.check(residual < 1e-6, f"{look} ms：{db(amp):.2f} dBFS 正弦的残差峰值 {db(residual):.1f} dBFS")

    # 旁通：不限制，但延迟仍要与上报一致。
    x = stereo(0.9 * np.random.default_rng(4).uniform(-1, 1, RATE // 4))
    source = r.write("in.wav", x)
    for look in LOOKAHEADS:
        y, latency = r.render(source, look, -12.0, -1.0, extra=("--set", "Bypass=1"))
        residual = np.max(np.abs(y - x))
        r.check(residual < 1e-6, f"旁通 {look} ms：上报 {latency} 采样，与输入的残差峰值 {db(residual):.1f} dBFS")

    # 任意缓冲区大小：结果与块大小无关。
    x = stereo(np.concatenate([sine(300, 0.1, 0.2), sine(300, 0.2, 1.0), sine(300, 0.2, 0.2)]))
    source = r.write("in.wav", x)
    for look in LOOKAHEADS:
        base, _ = r.render(source, look, -12.0, -1.0, block=256)
        worst = 0.0
        for block in (1, 7, 1024):
            y, _ = r.render(source, look, -12.0, -1.0, block=block)
            worst = max(worst, np.max(np.abs(y - base)))
        r.check(worst < 1e-4, f"{look} ms：块大小 1/7/1024 与 256 的最大差 {db(worst):.1f} dBFS")


def harmonic_distortion(y, fundamental):
    """THD 和 THD+N。y 是单声道、整数周期的稳态片段。"""
    window = np.hanning(len(y))
    spectrum = np.abs(np.fft.rfft(y * window)) ** 2
    bin_hz = RATE / len(y)
    k = int(round(fundamental / bin_hz))
    fund = np.sum(spectrum[k - 2:k + 3])
    harm = 0.0
    h = 2
    while h * fundamental < RATE / 2 - 3 * bin_hz:
        i = int(round(h * fundamental / bin_hz))
        harm += np.sum(spectrum[i - 2:i + 3])
        h += 1
    rest = np.sum(spectrum) - fund
    return 10 * np.log10(max(harm, 1e-30) / fund), 10 * np.log10(max(rest, 1e-30) / fund)


def distortion_row(y, freq):
    seg = y[RATE:RATE + RATE // 2, 0]
    return harmonic_distortion(seg, freq)


def high_band_residual(y, start, length, cutoff):
    """以起音为中心的窗口内，cutoff 以上的能量相对总能量（dB）。"""
    seg = y[start - length // 2:start + length // 2, 0]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / RATE)
    return 10 * np.log10(max(np.sum(spectrum[f >= cutoff]), 1e-30) / np.sum(spectrum))


def null_residual(y, x, start, seconds):
    """零差残差：用 20 ms 滑窗估计慢变增益 g，y - g*x 的能量相对 y（dB）。
    慢变的增益骑行不算失真，波形被改形和快速调制才算。"""
    a, b = start, start + int(seconds * RATE)
    window = int(0.02 * RATE)
    gain = uniform_filter1d(y * x, window) / np.maximum(uniform_filter1d(x * x, window), 1e-12)
    residual = y[a:b] - gain[a:b] * x[a:b]
    return 10 * np.log10(max(np.sum(residual ** 2), 1e-30) / np.sum(y[a:b] ** 2))


def imd(y, f1, f2):
    seg = y[RATE:RATE + RATE // 2 + RATE // 2, 0]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    bin_hz = RATE / len(seg)
    idx = lambda f: int(round(f / bin_hz))
    tones = sum(np.sum(spectrum[idx(f) - 2:idx(f) + 3]) for f in (f1, f2))
    rest = np.sum(spectrum) - tones
    return 10 * np.log10(max(rest, 1e-30) / tones)


ONSET_AT = int(0.2 * RATE)
VOWEL_F0 = 220.0


def onset_signal():
    # 1 kHz 从 -20 dB 用 2 ms 升余弦抬到 0 dB。输入本身在 3 kHz 以上几乎没有能量，残差基本都来自限制器。
    n = int(0.5 * RATE)
    env = np.full(n, 0.1)
    ramp = int(0.002 * RATE)
    env[ONSET_AT:ONSET_AT + ramp] = 0.1 + 0.9 * 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    env[ONSET_AT + ramp:ONSET_AT + int(0.1 * RATE)] = 1.0
    return env * sine(1000, 0.5)


def vowel_signal():
    # 元音代理：220 Hz 的 12 次谐波，每秒 4 个音节的平滑起伏。起伏的边带很窄，
    # 所以离开谐波的能量基本都是限制器造成的。没有真实人声素材时用它比较。
    n = 3 * RATE
    t = np.arange(n) / RATE
    x = sum(np.sin(2 * np.pi * VOWEL_F0 * h * t + h * 1.3) / h for h in range(1, 13))
    x = x * (0.3 + 0.7 * np.sin(np.pi * 4.0 * t) ** 2)
    return x / np.max(np.abs(x))


def distortion_signals():
    return {
        "1k": stereo(sine(1000, 2.0)),
        "100": stereo(sine(100, 2.0)),
        "onset": stereo(onset_signal()),
        "imd": stereo(sine(1000, 2.0, 0.5) + sine(1300, 2.0, 0.5)),
        "vowel": stereo(vowel_signal()),
    }


def distortion_table(r, render, label_rows):
    """render(source, label) -> y，label_rows 是 (标签, 模式关键字) 列表。"""
    signals = {name: r.write(f"dist_{name}.wav", x) for name, x in distortion_signals().items()}
    for label, key in label_rows:
        y = render(signals["1k"], key)
        thd1k, _ = distortion_row(y, 1000)
        level = db(np.max(np.abs(y[RATE:RATE + RATE // 2, 0])))
        y = render(signals["100"], key)
        thd100, _ = distortion_row(y, 100)
        y = render(signals["onset"], key)
        onset = high_band_residual(y, ONSET_AT, int(0.02 * RATE), 3000)
        y = render(signals["imd"], key)
        two = imd(y, 1000, 1300)
        y = render(signals["vowel"], key)
        vowel = null_residual(y[:, 0], vowel_signal(), RATE // 2, 2.0)
        r.say(f"| {label} | {thd1k:.1f} | {thd100:.1f} | {onset:.1f} | {two:.1f} | {vowel:.1f} | {level:.2f} |")


DIST_HEADER = ("| 模式 | 1 kHz THD | 100 Hz THD | 起音高频残差 | 双音 IMD+N | 元音零差残差 | 1 kHz 稳态峰值 (dBFS) |\n"
               "|---|---|---|---|---|---|---|")


def test_distortion(r):
    r.say("\n## 3. 失真（dB 相对基频或总能量，越低越好）")
    r.say("满幅信号、阈值 -12 dB（12 dB 增益衰减）、上限 -1 dB。稳态取第 1.0–1.5 秒。起音高频残差：1 kHz 2 ms 起音，以起音为中心 20 ms 窗内 3 kHz 以上能量。")
    r.say(f"输入自身的起音高频残差 {high_band_residual(stereo(onset_signal()), ONSET_AT, int(0.02 * RATE), 3000):.1f} dB。元音零差残差：220 Hz 12 次谐波、每秒 4 次起伏，20 ms 滑窗去掉慢变增益后的残差。")
    for automatic, release, title in ((True, 60.0, "自动释放"), (False, 60.0, "手动 60 ms"), (False, 1.0, "手动 1 ms")):
        r.say(f"\n### {title}\n{DIST_HEADER}")
        rows = [(f"{look} ms", look) for look in LOOKAHEADS]
        distortion_table(r, lambda s, look: r.render(s, look, -12.0, -1.0, release, automatic)[0], rows)


def recovery_ms(y, tone_amp_out, start):
    """探针音回到稳态 1 dB 以内所需的时间。"""
    freq = 8000
    env = np.abs(y[:, 0])
    win = int(RATE / freq) * 8
    level = np.array([np.max(env[i:i + win]) for i in range(start, len(env) - win, win)])
    below = np.where(level < tone_amp_out * 10 ** (-1 / 20))[0]
    if len(below) == 0:
        return 0.0
    return (below[-1] + 1) * win / RATE * 1000


def release_signals():
    probe_amp = 0.05
    def make(over_seconds):
        n = int((0.2 + over_seconds + 3.5) * RATE)
        t = np.arange(n) / RATE
        probe = probe_amp * np.sin(2 * np.pi * 8000 * t)
        loud = np.zeros(n)
        a, b = int(0.2 * RATE), int((0.2 + over_seconds) * RATE)
        loud[a:b] = np.sin(2 * np.pi * 200 * t[a:b]) * 0.95
        return stereo(probe + loud), b
    return {"5 ms 峰值": make(0.005), "100 ms": make(0.1), "2 s 持续": make(2.0)}, probe_amp


def release_table(r, render, modes):
    signals, probe_amp = release_signals()
    r.say("| 模式 | " + " | ".join(signals) + " |")
    r.say("|---|" + "---|" * len(signals))
    for label, key in modes:
        cells = []
        for index, (name, (x, end)) in enumerate(signals.items()):
            y = render(r.write(f"rel_{index}.wav", x), key)
            steady = np.max(np.abs(y[int(0.05 * RATE):int(0.15 * RATE), 0]))
            cells.append(f"{recovery_ms(y, steady, end):.0f} ms")
        r.say(f"| {label} | " + " | ".join(cells) + " |")


def test_release(r):
    r.say("\n## 4. 释放恢复时间（回到稳态 1 dB 内）")
    r.say("-26 dB 的 8 kHz 探针音叠加不同长度的 12 dB 过载。自动释放应当对短峰值恢复快、对持续过载恢复慢。")
    modes = [("自动 / 0 ms", ("0", True, 60.0)), ("自动，旋钮 1000 ms / 0 ms", ("0", True, 1000.0)), ("自动 / 5 ms", ("5", True, 60.0)),
             ("手动 1 / 0 ms", ("0", False, 1.0)), ("手动 60 / 0 ms", ("0", False, 60.0)), ("手动 300 / 0 ms", ("0", False, 300.0)),
             ("手动 1000 / 0 ms", ("0", False, 1000.0)), ("手动 60 / 5 ms", ("5", False, 60.0))]
    release_table(r, lambda s, m: r.render(s, m[0], -12.0, -1.0, m[2], m[1])[0], modes)


def reference_settings(threshold, ceiling, release, automatic):
    # 量化留在 24 位，抖动和噪声整形都关掉，只比限制本身。
    return [f"Thresh Slider={norm_db(threshold)}", f"Ceiling Slider={norm_db(ceiling)}",
            f"Release Slider={0.4 + np.log10(release) / 5}", f"ARC={1 if automatic else 0}",
            "Quantize=0", "IDR Type=0", "Noise Shaping=0"]


def test_reference(r, shell, name):
    """黑盒对比：同一信号、同样设定，只喂音频、取音频。结果不代表听感。"""
    r.say("\n## 5. 对标黑盒对比")
    modes = {"auto": (60.0, True, "自动释放"), "m60": (60.0, False, "手动 60 ms"), "m1": (1.0, False, "手动 1 ms")}

    dist = {name: r.write(f"dist_{name}.wav", x) for name, x in distortion_signals().items()}
    release_inputs, _ = release_signals()
    rel = [r.write(f"rel_{i}.wav", x) for i, (name, (x, end)) in enumerate(release_inputs.items())]
    hostile = {name: x for name, x in hostile_signals().items()}
    host = {name: r.write(f"host_{i}.wav", x) for i, (name, x) in enumerate(hostile.items())}
    low = stereo(sine(1000, 1.0, 0.01))
    low_path = r.write("low.wav", low)
    align_input = 0.05 * np.random.default_rng(5).normal(size=RATE // 2).clip(-3, 3)
    align_path = r.write("align.wav", stereo(align_input))

    def out(tag, source):
        return r.work / f"ref_{tag}_{Path(source).stem}.wav"

    # 这个托管方式下，对标插件的释放值只在实例第一次设置时生效（改值后显示变了，声音没变），
    # 所以每个释放值单独加载一次实例，一次约 30 秒。
    batches = {60.0: [], 1.0: []}
    def add(tag, source, release, settings):
        batches[release].append("\t".join([str(source), str(out(tag, source))] + settings))

    for tag, (release, automatic, _) in modes.items():
        for source in list(dist.values()) + rel:
            add(tag, source, release, reference_settings(-12.0, -1.0, release, automatic))
    for source in host.values():
        add("p0", source, 60.0, reference_settings(0.0, -1.0, 60.0, True))
        add("p12", source, 60.0, reference_settings(-12.0, -1.0, 60.0, True))
    add("gain", low_path, 60.0, reference_settings(-12.0, -1.0, 60.0, True))
    add("align", align_path, 60.0, reference_settings(0.0, 0.0, 60.0, True))

    latency = "?"
    for release, jobs in batches.items():
        jobs_file = r.work / "ref_jobs.txt"
        jobs_file.write_text("\n".join(jobs), encoding="utf-8")
        result = subprocess.run([str(r.bench), "batch", shell, str(jobs_file), "--name", name],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)
        if result.returncode != 0 or "FAIL" in result.stdout:
            r.say(f"对标插件没有跑通：{result.stdout[-400:]}{result.stderr[-200:]}")
            return
        latency = result.stdout.strip().split()[-1]
    r.say(f"对标：{name}（{Path(shell).name}），上报延迟 {latency} 采样，量化 24 位，抖动和噪声整形关闭。")

    r.say("\n### 增益律（-40 dBFS 1 kHz，阈值 -12、上限 -1，期望 +11 dB）")
    y, _ = r.render(low_path, "0", -12.0, -1.0)
    ours = db(np.max(np.abs(y))) - db(0.01)
    theirs = read(out("gain", low_path))
    r.say(f"Aegis {ours:+.3f} dB，对标 {db(np.max(np.abs(theirs))) - db(0.01):+.3f} dB")

    # 上报延迟是否真的等于音频延迟：非周期信号的互相关。
    y = read(out("align", align_path))[:, 0]
    corr = np.correlate(y[4096:12288], align_input[4096 - 512:12288 + 512], mode="valid")
    r.say(f"对标按上报延迟对齐后的残余偏移：{512 - int(np.argmax(corr))} 采样")

    r.say("\n### 峰值（自动释放，上限 -1 dB；相对上限 dB）")
    r.say("| 信号 | Aegis 0 ms sample / TP | Aegis 1.5 ms sample / TP | 对标 sample / TP |")
    r.say("|---|---|---|---|")
    for name, source in host.items():
        row = []
        for look in ("0", "1.5"):
            s_max = t_max = -999.0
            for threshold in (0.0, -12.0):
                y, _ = r.render(source, look, threshold, -1.0)
                s_max = max(s_max, db(np.max(np.abs(y))) + 1.0)
                t_max = max(t_max, db(true_peak(y)) + 1.0)
            row.append(f"{s_max:+.3f} / {t_max:+.2f}")
        s_max = t_max = -999.0
        for tag in ("p0", "p12"):
            y = read(out(tag, source))
            s_max = max(s_max, db(np.max(np.abs(y))) + 1.0)
            t_max = max(t_max, db(true_peak(y)) + 1.0)
        row.append(f"{s_max:+.3f} / {t_max:+.2f}")
        r.say(f"| {name} | " + " | ".join(row) + " |")

    def side_by_side(source, key):
        tag, look, release, automatic = key
        if look is None:
            return read(out(tag, source))
        return r.render(source, look, -12.0, -1.0, release, automatic)[0]

    for tag, (release, automatic, title) in modes.items():
        r.say(f"\n### 失真：{title}\n{DIST_HEADER}")
        distortion_table(r, side_by_side, [("Aegis 0 ms", (tag, "0", release, automatic)),
                                           ("Aegis 1.5 ms", (tag, "1.5", release, automatic)),
                                           ("对标", (tag, None, release, automatic))])

    r.say("\n### 释放恢复时间（回到稳态 1 dB 内）")
    rows = []
    for tag, (release, automatic, title) in modes.items():
        rows += [(f"Aegis 0 ms {title}", (tag, "0", release, automatic)), (f"对标 {title}", (tag, None, release, automatic))]
    release_table(r, side_by_side, rows)

    r.say("\n### 与对标的零差（元音代理，阈值 -12、上限 -1，第 0.5–2.5 秒）")
    theirs = read(out("auto", dist["vowel"]))[:, 0]
    for look in ("0", "1.5"):
        y = r.render(dist["vowel"], look, -12.0, -1.0)[0][:, 0]
        a, b = RATE // 2, RATE // 2 + 2 * RATE
        difference = y[a:b] - theirs[a:b]
        rms = lambda v: np.sqrt(np.mean(v ** 2))
        r.say(f"Aegis {look} ms 自动释放：响度差 {db(rms(y[a:b])) - db(rms(theirs[a:b])):+.2f} dB RMS，差信号相对对标 {db(rms(difference)) - db(rms(theirs[a:b])):.1f} dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="D:/mint-build/limiter")
    parser.add_argument("--only", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--reference-name", default="")
    args = parser.parse_args()

    r = Runner(Path(args.build))
    tests = {"host": test_host, "peaks": test_peaks, "align": test_alignment, "dist": test_distortion, "release": test_release}
    for name, test in tests.items():
        if not args.only or name in args.only.split(","):
            test(r)
    if args.reference:
        test_reference(r, args.reference, args.reference_name)

    r.say(f"\n失败项：{r.failures}")
    (r.work / "results.txt").write_text("\n".join(r.lines), encoding="utf-8")
    return 0 if r.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
