#!/usr/bin/env python3
"""Lethe 核心行为的回归测试，全部经过 mint_probe 走真实的 VST3。

用法：
    python -I run_tests.py --probe <mint_probe.exe> --plugin <Mintaria Lethe.vst3> --work <临时目录>
                           [--mode 0|1|2] [--speech clean.wav] [--noise noise.wav]
                           [--deep-filter deep-filter.exe --model DeepFilterNet3_onnx.tar.gz]

依赖 numpy、scipy、soundfile。没给 --speech 时用合成的类人声信号，结果只能当回归，不代表降噪质量。
真实语音可以用 DeepFilterNet 仓库 assets/ 下的样本，33711 是 CC BY-NC，只用于测试，不要提交进来。
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODE_NAMES = {0: "标准", 1: "低延迟", 2: "轻量"}
# 三档的纯噪声降噪量差别很大，这是模型特性（见 NOTES.md）。下限只用来防回归。
MIN_NOISE_REDUCTION = {0: 12, 1: 8, 2: 20}
# 全新加载（不设 Mode）时的上报延迟。默认档是低延迟，44.1 kHz / 128 是用户宿主的实际设置。
DEFAULT_LATENCY = {(48000, 256): 1520, (44100, 128): 1387}


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


class Probe:
    def __init__(self, exe, plugin, work, mode):
        self.exe, self.plugin, self.work, self.mode = exe, plugin, Path(work), mode
        self.work.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def render(self, audio, rate, voice=1.0, ambience=0.0, block=256, default_mode=False):
        """audio: [samples, channels]。返回与输入对齐（mint_probe 已扣除上报延迟）的输出和上报延迟。"""
        self.counter += 1
        src = self.work / f"in_{self.counter}.wav"
        dst = self.work / f"out_{self.counter}.wav"
        sf.write(src, audio, rate, subtype="FLOAT")
        mode_norm = self.mode / 2
        cmd = [self.exe, "render", self.plugin, str(src), str(dst), "--block", str(block),
               "--set", f"Voice={voice}", "--set", f"Ambience={ambience}"]
        if not default_mode:
            cmd += ["--set", f"Mode={mode_norm}"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"mint_probe failed: {result.stdout}\n{result.stderr}")
        latency = int(result.stdout.strip().split()[-1])
        out, _ = sf.read(dst, dtype="float64", always_2d=True)
        return out, latency


def lag_of(reference, signal, search):
    """signal 相对 reference 的延迟（采样）。"""
    n = min(len(reference), len(signal))
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(signal[:n], size) * np.conj(np.fft.rfft(reference[:n], size)))
    corr = np.concatenate([corr[-search:], corr[:search + 1]])
    return int(np.argmax(np.abs(corr))) - search


def si_sdr(reference, estimate):
    scale = np.dot(estimate, reference) / np.dot(reference, reference)
    return 10 * np.log10(np.sum((scale * reference) ** 2) / np.sum((estimate - scale * reference) ** 2)), scale


def synthetic_speech(rate, seconds):
    """有基频和共振峰、带停顿的合成信号。"""
    t = np.arange(int(rate * seconds)) / rate
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voiced = sum(np.sin(k * phase) * np.exp(-k / 8.0) for k in range(1, 25))
    envelope = (np.sin(2 * np.pi * 1.3 * t) > -0.2).astype(float)
    kernel = np.hanning(int(rate * 0.02))
    envelope = np.convolve(envelope, kernel / kernel.sum(), mode="same")
    return 0.2 * voiced * envelope / np.abs(voiced).max()


def load_mono(path, rate):
    audio, source_rate = sf.read(path, dtype="float64", always_2d=True)
    audio = audio[:, 0]
    if source_rate != rate:
        audio = resample_poly(audio, rate, source_rate)
    return audio


class Results:
    def __init__(self):
        self.failures = 0

    def check(self, name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        self.failures += 0 if ok else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--speech")
    parser.add_argument("--noise")
    parser.add_argument("--deep-filter", help="官方 deep-filter 可执行文件，用于逐样点对比")
    parser.add_argument("--model", help="官方模型 tar.gz")
    args = parser.parse_args()

    probe = Probe(args.probe, args.plugin, args.work, args.mode)
    res = Results()
    print(f"mode {args.mode} ({MODE_NAMES[args.mode]})")

    rate = 48000
    rng = np.random.default_rng(7)
    speech = load_mono(args.speech, rate)[: rate * 10] if args.speech else synthetic_speech(rate, 10)
    speech = speech / np.abs(speech).max() * 0.5
    noise = load_mono(args.noise, rate) if args.noise else rng.normal(size=len(speech)) * 0.05
    noise = np.resize(noise, len(speech))
    noise *= np.sqrt(np.mean(speech**2) / np.mean(noise**2)) * 10 ** (-5 / 20)  # 5 dB SNR
    noisy = speech + noise

    # 0. 全新加载时的默认档。
    for (test_rate, block), expected in DEFAULT_LATENCY.items():
        signal = resample_poly(noisy, test_rate, 48000) if test_rate != 48000 else noisy
        _, latency = probe.render(np.stack([signal, signal], 1), test_rate, 1.0, 0.0, block, default_mode=True)
        res.check(f"默认档的上报延迟 {test_rate}/{block}", latency == expected, f"{latency} 采样，期望 {expected}")

    # 1. 上报延迟和样点对齐：两旋钮拉满时必须和原声一致。
    for test_rate, block in ((48000, 256), (48000, 64), (48000, 1024), (44100, 256), (96000, 512)):
        signal = resample_poly(noisy, test_rate, 48000) if test_rate != 48000 else noisy
        out, latency = probe.render(np.stack([signal, signal], 1), test_rate, 1.0, 1.0, block)
        residual = out[:, 0] - signal[: len(out)]
        worst = float(np.max(np.abs(residual)))
        res.check(f"两旋钮拉满与原声的残差 {test_rate}/{block}", worst < 1e-5,
                  f"峰值 {db(worst):.1f} dBFS，上报延迟 {latency} 采样（{latency * 1000 / test_rate:.2f} ms）")

    # 2. 人声对齐：只开人声时输出和输入的互相关峰必须在 0 延迟。
    for test_rate, block in ((48000, 256), (44100, 256), (96000, 256)):
        signal = resample_poly(noisy, test_rate, 48000) if test_rate != 48000 else noisy
        clean = resample_poly(speech, test_rate, 48000) if test_rate != 48000 else speech
        out, _ = probe.render(np.stack([signal, signal], 1), test_rate, 1.0, 0.0, block)
        lag = lag_of(clean, out[:, 0], 2000)
        res.check(f"人声相对干声的延迟 {test_rate}", lag == 0, f"{lag} 采样")

    # 3. 降噪量。
    out, _ = probe.render(np.stack([noise, noise], 1), rate, 1.0, 0.0)
    skip = rate  # 前 1 秒是模型收敛
    reduction = db(rms(noise[skip:])) - db(rms(out[skip:, 0]))
    res.check("纯噪声的降噪量", reduction > MIN_NOISE_REDUCTION[args.mode], f"{reduction:.1f} dB")

    out, _ = probe.render(np.stack([noisy, noisy], 1), rate, 1.0, 0.0)
    voice = out[:, 0]
    sdr_in, _ = si_sdr(speech[skip:], noisy[skip:])
    sdr_out, scale = si_sdr(speech[skip:], voice[skip:])
    res.check("带噪语音 SI-SDR", sdr_out > sdr_in + 3, f"{sdr_in:.2f} -> {sdr_out:.2f} dB，人声增益 {scale:.3f}")

    # 4. 环境音 = 干声 - 人声：只开环境音时噪声保留，人声大部分被减掉。
    out, _ = probe.render(np.stack([noisy, noisy], 1), rate, 0.0, 1.0)
    ambience = out[:, 0]
    res.check("人声 + 环境音 = 原声（分别渲染再相加）", np.max(np.abs(voice + ambience - noisy)) < 1e-5,
              f"峰值 {db(float(np.max(np.abs(voice + ambience - noisy)))):.1f} dBFS")
    speech_gain = np.dot(ambience[skip:], speech[skip:]) / np.dot(speech[skip:], speech[skip:])
    noise_gain = np.dot(ambience[skip:], noise[skip:]) / np.dot(noise[skip:], noise[skip:])
    res.check("环境音里的人声泄漏", abs(speech_gain) < 0.25, f"人声系数 {speech_gain:.3f}，噪声系数 {noise_gain:.3f}")

    # 5. 两个旋钮都关：严格静音。
    out, _ = probe.render(np.stack([noisy, noisy], 1), rate, 0.0, 0.0)
    res.check("两旋钮都关", float(np.max(np.abs(out[rate // 2:]))) == 0.0, f"峰值 {db(float(np.max(np.abs(out[rate // 2:])))):.1f} dBFS")

    # 6. 静音输入：不产生 NaN，也不产生噪声。
    silence = np.zeros((rate * 3, 2))
    out, _ = probe.render(silence, rate, 1.0, 1.0)
    res.check("静音输入", np.isfinite(out).all() and float(np.max(np.abs(out))) < 1e-6, f"峰值 {db(float(np.max(np.abs(out)))):.1f} dBFS")

    # 7. 立体声声像：左右增益不同的同一信号，人声输出的声像不能变。
    panned = np.stack([noisy, noisy * 0.25], 1)
    out, _ = probe.render(panned, rate, 1.0, 0.0)
    balance = rms(out[skip:, 1]) / rms(out[skip:, 0])
    res.check("立体声声像保持", abs(balance - 0.25) < 0.02, f"右/左 = {balance:.3f}，输入 0.25")

    # 8. 和官方实现逐样点比较。
    if args.deep_filter and args.mode != 2:
        official_dir = Path(args.work) / "official"
        official_dir.mkdir(exist_ok=True)
        official_in = Path(args.work) / "noisy.wav"
        sf.write(official_in, noisy, rate, subtype="FLOAT")
        subprocess.run([args.deep_filter, "-m", args.model, "-o", str(official_dir), str(official_in)], check=True, capture_output=True)
        reference = load_mono(official_dir / "noisy.wav", rate)
        delay = {0: 1440, 1: 480}[args.mode]
        reference = reference[delay:]
        out, _ = probe.render(noisy[:, None].repeat(2, 1), rate, 1.0, 0.0)
        n = min(len(reference), len(out))
        # mint_probe 先跑一段静音，帧网格和官方不同，所以这里不能要求逐样点相同，只看质量相当。
        # 逐样点比较见 lethe_rt_test --engine-only。
        sdr_official, _ = si_sdr(speech[skip:n], reference[skip:n])
        sdr_ours, _ = si_sdr(speech[skip:n], out[skip:n, 0])
        res.check("与官方 deep-filter 的 SI-SDR 差", abs(sdr_official - sdr_ours) < 1.0,
                  f"官方 {sdr_official:.2f} dB，本插件 {sdr_ours:.2f} dB，波形差 {db(rms(out[skip:n, 0] - reference[skip:n])) - db(rms(reference[skip:n])):.1f} dB")

    print(f"\n{res.failures} failure(s)")
    return 1 if res.failures else 0


if __name__ == "__main__":
    sys.exit(main())
