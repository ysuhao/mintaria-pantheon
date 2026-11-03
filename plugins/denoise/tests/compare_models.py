#!/usr/bin/env python3
"""对比候选模型的降噪效果和算法延迟，结果写进 NOTES.md。

用法：python -I compare_models.py --speech clean.wav --noise a.wav --noise b.wav --work <目录> \
          --candidate "DF3=<cmd>" --candidate "RNNoise=<cmd>" ...

<cmd> 是一条命令模板，{in} 和 {out} 换成 48 kHz 单声道 32 位浮点 wav 的路径，{raw_in}、{raw_out} 是裸浮点文件。
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load(path):
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    audio = audio[:, 0]
    return resample_poly(audio, 48000, rate) if rate != 48000 else audio


def si_sdr(reference, estimate):
    scale = np.dot(estimate, reference) / np.dot(reference, reference)
    return 10 * np.log10(np.sum((scale * reference) ** 2) / np.sum((estimate - scale * reference) ** 2)), scale


def lag(reference, signal):
    n = min(len(reference), len(signal))
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(signal[:n], size) * np.conj(np.fft.rfft(reference[:n], size)))
    return int(np.argmax(corr[:6000]))


def run(template, work, audio):
    source = work / "cand_in.wav"
    target = work / "cand_out.wav"
    raw_in, raw_out = work / "cand_in.raw", work / "cand_out.raw"
    sf.write(source, audio, 48000, subtype="FLOAT")
    audio.astype(np.float32).tofile(raw_in)
    target.unlink(missing_ok=True)
    raw_out.unlink(missing_ok=True)
    command = template.format(**{"in": source, "out": target, "raw_in": raw_in, "raw_out": raw_out, "work": work})
    subprocess.run(shlex.split(command, posix=False), check=True, capture_output=True)
    if target.exists():
        out, _ = sf.read(target, dtype="float64", always_2d=True)
        return out[:, 0]
    return np.fromfile(raw_out, np.float32).astype(np.float64)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech", required=True)
    parser.add_argument("--noise", action="append", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    speech = load(args.speech)
    speech = speech / np.abs(speech).max() * 0.5
    skip = 48000

    print("| 模型 | 噪声 | 输入 SNR | 算法延迟（采样） | SI-SDR 输入 → 输出 | 人声增益 | 纯噪声降噪量 |")
    print("|---|---|---|---|---|---|---|")
    for spec in args.candidate:
        name, template = spec.split("=", 1)
        for noise_path in args.noise:
            raw_noise = np.resize(load(noise_path), len(speech))
            for snr in (5, 0):
                noise = raw_noise * np.sqrt(np.mean(speech**2) / np.mean(raw_noise**2)) * 10 ** (-snr / 20)
                noisy = speech + noise

                out = run(template, work, noisy)
                delay = lag(speech, out)
                aligned = out[delay:]
                n = min(len(aligned), len(speech))
                sdr_in, _ = si_sdr(speech[skip:n], noisy[skip:n])
                sdr_out, gain = si_sdr(speech[skip:n], aligned[skip:n])

                noise_out = run(template, work, noise)
                m = min(len(noise_out), len(noise))
                reduction = 10 * np.log10(np.mean(noise[skip:m] ** 2) / np.maximum(np.mean(noise_out[skip:m] ** 2), 1e-20))

                print(f"| {name} | {Path(noise_path).stem} | {snr} dB | {delay} | {sdr_in:.2f} → {sdr_out:.2f} dB | {gain:.3f} | {reduction:.1f} dB |", flush=True)


if __name__ == "__main__":
    sys.exit(main())
