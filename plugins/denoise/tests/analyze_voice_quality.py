#!/usr/bin/env python3
"""测量三档引擎的“发闷”和“水声”线索。结果只是线索，听感由人判断。

用法：python -I analyze_voice_quality.py --test <lethe_rt_test.exe> --speech clean.wav --noise a.wav --noise b.wav --work <目录>

- 频段增益：干净语音单独进模型，输出和输入的长期谱差（只算有语音的帧）。负值大说明这段频率被削了。
- 影子滤波：带噪输入时，把模型的时频增益 G = |Y| / |X| 分别乘到语音和噪声上，得到语音损失和残留噪声。
- 音乐噪声：稳态噪声单独进模型，比较输出和输入功率谱的峰度。log 峰度比越大，残留噪声越像孤立闪烁的亮点。
  这是该现象的常用客观指标，和“水声”不是一回事，只能当线索。
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

RATE = 48000
ENGINES = [("标准", "standard", 1440), ("低延迟", "low", 480), ("轻量", "light", 960)]
BANDS = [(0, 250), (250, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 20000)]
NPERSEG = 1024
HOP = 256


def load(path):
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    audio = audio[:, 0]
    return resample_poly(audio, RATE, rate) if rate != RATE else audio


def spectrum(x):
    freqs, _, z = stft(x, RATE, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    return freqs, z


class Runner:
    def __init__(self, exe, work):
        self.exe, self.work = exe, Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    def run(self, kind, delay, audio):
        """返回和输入对齐的输出，尾部不足一帧的部分补零。"""
        source, target = self.work / "q_in.wav", self.work / "q_out.wav"
        sf.write(source, audio, RATE, subtype="FLOAT")
        subprocess.run([self.exe, str(source), "--kind", kind, "--engine-only", "1", "--out", str(target)],
                       check=True, capture_output=True)
        out, _ = sf.read(target, dtype="float64", always_2d=True)
        out = out[delay:, 0]
        return np.pad(out, (0, len(audio) - len(out)))


def active_frames(z, floor_db):
    """语音帧：帧能量距离峰值帧不超过 floor_db。"""
    energy = np.sum(np.abs(z) ** 2, axis=0)
    return 10 * np.log10(energy / energy.max() + 1e-20) > -floor_db


def band_db(power, freqs, low, high):
    return 10 * np.log10(np.sum(power[(freqs >= low) & (freqs < high)]) + 1e-20)


def report_clean_speech(runner, speech):
    print("\n### 干净语音进模型：各频段的长期谱变化（dB）\n")
    print("| 档位 | " + " | ".join(f"{a / 1000:g}-{b / 1000:g}k" for a, b in BANDS) + " | 总电平 | 频谱重心 Hz（输入 → 输出） |")
    print("|---|" + "---|" * (len(BANDS) + 2))
    freqs, clean = spectrum(speech)
    mask = active_frames(clean, 40)
    power_in = np.mean(np.abs(clean[:, mask]) ** 2, axis=1)
    centroid_in = np.sum(freqs * power_in) / np.sum(power_in)
    for name, kind, delay in ENGINES:
        _, out = spectrum(runner.run(kind, delay, speech))
        power_out = np.mean(np.abs(out[:, mask]) ** 2, axis=1)
        cells = [band_db(power_out, freqs, a, b) - band_db(power_in, freqs, a, b) for a, b in BANDS]
        total = 10 * np.log10(np.sum(power_out) / np.sum(power_in))
        centroid = np.sum(freqs * power_out) / np.sum(power_out)
        print(f"| {name} | " + " | ".join(f"{c:+.1f}" for c in cells) + f" | {total:+.1f} | {centroid_in:.0f} → {centroid:.0f} |")


def report_shadow(runner, speech, noise, label, snr):
    noise = np.resize(noise, len(speech))
    noise = noise * np.sqrt(np.mean(speech**2) / np.mean(noise**2)) * 10 ** (-snr / 20)
    noisy = speech + noise
    freqs, s = spectrum(speech)
    _, n = spectrum(noise)
    _, x = spectrum(noisy)
    voiced = active_frames(s, 30)
    pause = ~active_frames(s, 45)

    print(f"\n### {label}，输入 SNR {snr} dB（语音帧 {voiced.sum()}，停顿帧 {pause.sum()}）\n")
    print("| 档位 | 语音损失 0-4k | 语音损失 4-8k | 语音损失 8-16k | 残留噪声（语音段）| 残留噪声 4-16k（语音段）| 残留噪声（停顿段）| 语音段 SNR 提升 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, kind, delay in ENGINES:
        _, y = spectrum(runner.run(kind, delay, noisy))
        gain = np.minimum(np.abs(y) / (np.abs(x) + 1e-9), 4.0)
        speech_power = np.abs(s) ** 2
        noise_power = np.abs(n) ** 2

        def ratio(power, frames, low=0, high=24000):
            bins = (freqs >= low) & (freqs < high)
            return 10 * np.log10(np.sum((gain[bins][:, frames] ** 2) * power[bins][:, frames]) / (np.sum(power[bins][:, frames]) + 1e-20) + 1e-20)

        speech_gain = ratio(speech_power, voiced)
        noise_gain = ratio(noise_power, voiced)
        print(f"| {name} | {ratio(speech_power, voiced, 0, 4000):+.1f} | {ratio(speech_power, voiced, 4000, 8000):+.1f} "
              f"| {ratio(speech_power, voiced, 8000, 16000):+.1f} | {noise_gain:+.1f} | {ratio(noise_power, voiced, 4000, 16000):+.1f} "
              f"| {ratio(noise_power, pause):+.1f} | {speech_gain - noise_gain:+.1f} |")


def kurtosis(z):
    power = np.abs(z.ravel()) ** 2
    power = power[power > 0]
    return np.mean(power**2) / np.mean(power) ** 2


def report_musical_noise(runner):
    rng = np.random.default_rng(11)
    white = rng.normal(size=RATE * 8)
    spectrum_shape = np.fft.rfft(rng.normal(size=RATE * 8))
    spectrum_shape /= np.sqrt(np.maximum(np.fft.rfftfreq(RATE * 8, 1 / RATE), 20.0))
    pink = np.fft.irfft(spectrum_shape)

    print("\n### 稳态噪声单独进模型：残留电平和 log 峰度比（前 2 秒不算）\n")
    print("| 档位 | 噪声 | 输入电平 | 残留电平 | 降噪量 | log 峰度比 0-4k | log 峰度比 4-12k | 帧间增益抖动（dB/帧，按残留能量加权）|")
    print("|---|---|---|---|---|---|---|---|")
    for noise_name, base in (("白噪声", white), ("粉噪声", pink)):
        for level_db in (-40, -60):
            noise = base / np.sqrt(np.mean(base**2)) * 10 ** (level_db / 20)
            freqs, x = spectrum(noise)
            start = 2 * RATE // HOP
            for name, kind, delay in ENGINES:
                y_time = runner.run(kind, delay, noise)
                _, y = spectrum(y_time)
                y, ref = y[:, start:], x[:, start:]
                low, high = freqs < 4000, (freqs >= 4000) & (freqs < 12000)
                lkr_low = np.log(max(kurtosis(y[low]), 1e-9) / kurtosis(ref[low]))
                lkr_high = np.log(max(kurtosis(y[high]), 1e-9) / kurtosis(ref[high]))

                gain_db = 20 * np.log10(np.abs(y) / (np.abs(ref) + 1e-12) + 1e-6)
                weight = np.abs(y[:, 1:]) ** 2
                flicker = np.sum(np.abs(np.diff(gain_db, axis=1)) * weight) / (np.sum(weight) + 1e-30)

                residual = 20 * np.log10(np.sqrt(np.mean(y_time[2 * RATE:] ** 2)) + 1e-12)
                print(f"| {name} | {noise_name} | {level_db} dBFS | {residual:.1f} dBFS | {level_db - residual:.1f} dB "
                      f"| {lkr_low:+.2f} | {lkr_high:+.2f} | {flicker:.1f} |")


def pink_noise(length, seed):
    rng = np.random.default_rng(seed)
    shape = np.fft.rfft(rng.normal(size=length))
    shape /= np.sqrt(np.maximum(np.fft.rfftfreq(length, 1 / RATE), 20.0))
    return np.fft.irfft(shape, length)


def report_speech_detail(runner, speech):
    """说话时的细节：辅音帧的损失（长期平均会被元音盖掉），以及语音段里残留噪声的尖峰度和抖动。"""
    for snr in (15, 5):
        noise = pink_noise(len(speech), 21)
        noise = noise * np.sqrt(np.mean(speech**2) / np.mean(noise**2)) * 10 ** (-snr / 20)
        noisy = speech + noise
        freqs, s = spectrum(speech)
        _, n = spectrum(noise)
        _, x = spectrum(noisy)

        power = np.abs(s) ** 2
        low = freqs < 4000
        band = (freqs >= 4000) & (freqs < 12000)
        energy = np.sum(power, axis=0)
        loud = 10 * np.log10(energy / energy.max() + 1e-20) > -40
        fricative = loud & (np.sum(power[band], axis=0) > np.sum(power[low], axis=0))
        voiced = active_frames(s, 30)

        print(f"\n### 粉噪声 + 语音，输入 SNR {snr} dB（辅音帧 {fricative.sum()}，语音帧 {voiced.sum()}）\n")
        print("| 档位 | 辅音帧 4-12k 语音损失 | 损失最重的 10% 辅音帧 | 语音段残留噪声 4-12k | 语音段 log 峰度比 0-4k | 语音段 log 峰度比 4-12k | 噪声主导时频点的增益抖动（dB/帧）|")
        print("|---|---|---|---|---|---|---|")
        for name, kind, delay in ENGINES:
            _, y = spectrum(runner.run(kind, delay, noisy))
            gain = np.minimum(np.abs(y) / (np.abs(x) + 1e-9), 4.0)

            frame_in = np.sum(power[band][:, fricative], axis=0)
            frame_out = np.sum((gain[band][:, fricative] ** 2) * power[band][:, fricative], axis=0)
            frame_loss = 10 * np.log10(frame_out / (frame_in + 1e-20) + 1e-20)
            total_loss = 10 * np.log10(np.sum(frame_out) / np.sum(frame_in))
            worst = np.mean(np.sort(frame_loss)[: max(1, len(frame_loss) // 10)])

            shadow = gain * np.abs(n)
            frames = voiced
            lkr = [np.log(max(kurtosis(shadow[b][:, frames]), 1e-9) / kurtosis(n[b][:, frames])) for b in (low, band)]
            residual = 10 * np.log10(np.sum(shadow[band][:, frames] ** 2) / np.sum(np.abs(n[band][:, frames]) ** 2))

            # 只看噪声比语音强 6 dB 的时频点，这些地方的增益闪烁最容易听成水声。
            gain_db = 20 * np.log10(gain + 1e-6)
            noisy_bins = (np.abs(n) > 2 * np.abs(s))[:, 1:] & (np.abs(n) > 2 * np.abs(s))[:, :-1]
            noisy_bins[:, ~voiced[1:]] = False
            weight = (shadow[:, 1:] ** 2) * noisy_bins
            flicker = np.sum(np.abs(np.diff(gain_db, axis=1)) * weight) / (np.sum(weight) + 1e-30)

            print(f"| {name} | {total_loss:+.1f} dB | {worst:+.1f} dB | {residual:+.1f} dB | {lkr[0]:+.2f} | {lkr[1]:+.2f} | {flicker:.1f} |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--noise", action="append", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()

    runner = Runner(args.test, args.work)
    speech = load(args.speech)
    speech = speech / np.abs(speech).max() * 0.5

    report_clean_speech(runner, speech)
    for path in args.noise:
        raw = load(path)
        for snr in (15, 5, 0):
            report_shadow(runner, speech, raw, Path(path).stem, snr)
    report_musical_noise(runner)
    report_speech_detail(runner, speech)


if __name__ == "__main__":
    sys.exit(main())
