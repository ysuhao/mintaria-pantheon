#!/usr/bin/env python3
"""试验吞音的修复方法，同时记录代价。只是试验，不改产品。

用法：python -I singing_experiments.py --test <lethe_rt_test.exe> --models <plugins/denoise/models> --work <仓库外目录> \
          --engine df3|df3_nodf|df3_tau10|df3_g6|df3_g12|df3_ll|df3_ll_gate|rnnoise \
          --song <a.m4a> ... --speech clean.wav --noise a.wav --noise b.wav

每种引擎配置单独跑一次（可以并行），输出 Markdown 表格行。同一引擎的输出上再叠加后处理：
- ceil<N>：衰减上限 N dB，等同把环境音旋钮放在 -N dB。
- 音高保护：只在输出端工作，见 pitch_guard()。
引擎配置里的 g6、g12 是只在送进模型前加增益、出来再除回去；gate 见 lsnr_gate()。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))  # python -I 不把脚本目录放进搜索路径

from analyze_singing import pick_segments  # noqa: E402
from df_prototype import DfPrototype, Options  # noqa: E402
import singing_common as sc  # noqa: E402

CEILINGS = (24, 18, 12, 6)
HOP = 480


def pitch_guard(voice, dry, trigger_db=-12.0, keep_db=-6.0):
    """音高保护原型：有明显音高、电平高出噪底，而模型输出比干声低 trigger_db 以上时，补回到 keep_db。

    正常语音 V 和 D 电平差不多，不会触发，所以平时不损失降噪。
    实时实现时，对齐后的干声比输入晚一个算法延迟，这里用到的都是已有数据，不增加延迟。
    """
    frames = min(len(voice), len(dry)) // HOP
    low = resample_poly(dry, 1, 6)  # 8 kHz
    level = np.zeros(frames)
    deficit = np.zeros(frames)
    pitched = np.zeros(frames, bool)
    window = 320  # 40 ms
    for f in range(frames):
        d = dry[f * HOP : (f + 1) * HOP]
        v = voice[f * HOP : (f + 1) * HOP]
        energy_d = np.mean(d**2) + 1e-14
        level[f] = 10 * np.log10(energy_d)
        deficit[f] = np.sqrt((np.mean(v**2) + 1e-14) / energy_d)
        end = (f + 1) * 80
        if end < window + 100:
            continue
        segment = low[end - window : end]
        delayed = np.stack([low[end - window - lag : end - lag] for lag in range(8, 101)])
        norm = np.sqrt(np.sum(segment**2) * np.sum(delayed**2, axis=1)) + 1e-12
        pitched[f] = np.max(delayed @ segment / norm) > 0.7

    # 噪底：最近 3 秒电平的最小值。持续的嗡声几秒内就被算进噪底，不再触发。
    floor = np.array([level[max(0, f - 300) : f + 1].min() for f in range(frames)])
    trigger = 10 ** (trigger_db / 20)
    target = 10 ** (keep_db / 20)
    amount = np.zeros(frames)
    hold = 0.0
    for f in range(frames):
        wanted = 0.0
        if pitched[f] and level[f] - floor[f] > 12.0 and deficit[f] < trigger:
            wanted = (target - deficit[f]) / (1.0 - deficit[f] + 1e-9)
        hold = max(wanted, hold * 0.9)  # 释放约 100 ms
        amount[f] = hold

    curve = np.interp(np.arange(frames * HOP), (np.arange(frames) + 0.5) * HOP, amount)
    return voice[: frames * HOP] + curve * (dry[: frames * HOP] - voice[: frames * HOP])


def lsnr_gate(voice, lsnr, low_db=-5.0, high_db=5.0, floor_db=-15.0):
    """lsnr 门：模型自己估计的局部信噪比很低时，在输出上再多压一些。上升 10 ms，下降约 200 ms。"""
    frames = min(len(lsnr), len(voice) // HOP)
    wanted = np.interp(lsnr[:frames], [low_db, high_db], [floor_db, 0.0])
    gains = np.zeros(frames)
    current = 0.0
    for f in range(frames):
        current = wanted[f] if wanted[f] > current else current + 0.05 * (wanted[f] - current)
        gains[f] = current
    curve = 10 ** (np.interp(np.arange(frames * HOP), (np.arange(frames) + 0.5) * HOP, gains) / 20)
    return voice[: frames * HOP] * curve


def rms_db(x):
    return 10 * np.log10(np.mean(x**2) + 1e-20)


def si_sdr(reference, estimate):
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-20)
    return 10 * np.log10(np.sum((scale * reference) ** 2) / (np.sum((estimate - scale * reference) ** 2) + 1e-20))


def fixture_noises(rate=sc.RATE, seconds=8):
    t = np.arange(rate * seconds) / rate
    rng = np.random.default_rng(5)
    white = rng.normal(size=len(t))
    shape = np.fft.rfft(rng.normal(size=len(t)))
    shape /= np.sqrt(np.maximum(np.fft.rfftfreq(len(t), 1 / rate), 20.0))
    pink = np.fft.irfft(shape, len(t))
    hum = sum(np.sin(2 * np.pi * 100 * k * t) / k for k in range(1, 8))
    beep = np.sin(2 * np.pi * 1000 * t) * (np.mod(t, 1.5) < 0.5)
    level = lambda x, db: x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)
    return {"白噪声": level(white, -40), "粉噪声": level(pink, -40), "嗡声": level(hum, -40), "间歇提示音": level(beep, -30)}


def make_engine(args):
    if args.engine == "rnnoise":
        return lambda audio: sc.run_engine(args.test, "light", 960, audio, args.work)

    configs = {
        "df3": ("df3", 2, Options(), 0.0, False),
        "df3_nodf": ("df3", 2, Options(use_df=False), 0.0, False),
        "df3_tau10": ("df3", 2, Options(norm_alpha=0.999), 0.0, False),
        "df3_g6": ("df3", 2, Options(), 6.0, False),
        "df3_g12": ("df3", 2, Options(), 12.0, False),
        "df3_ll": ("df3_ll", 0, Options(), 0.0, False),
        "df3_ll_gate": ("df3_ll", 0, Options(), 0.0, True),
    }
    directory, lookahead, options, gain_db, gate = configs[args.engine]
    delay = 480 + lookahead * 480
    gain = 10 ** (gain_db / 20)

    def run(audio):
        out, trace = DfPrototype(Path(args.models) / directory, lookahead, options).process(audio * gain)
        out = out / gain
        if gate:
            # 跟踪信息比输出早 delay，门控要作用在对齐之前。
            lsnr = np.nan_to_num(np.roll(trace["lsnr"], lookahead + 1), nan=-15.0)
            out = lsnr_gate(out, lsnr)
        out = out[delay:]
        return np.pad(out, (0, max(0, len(audio) - len(out))))[: len(audio)]

    return run


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--song", action="append", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--noise", action="append", required=True)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=16.0)
    args = parser.parse_args()

    args.work = str(Path(args.work) / args.engine)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    engine = make_engine(args)

    post = {"原样": lambda v, d: v}
    for ceiling in CEILINGS:
        gain = 10 ** (-ceiling / 20)
        post[f"上限 {ceiling} dB"] = lambda v, d, g=gain: v + g * (d - v)
    post["音高保护"] = lambda v, d: pitch_guard(v, d)

    songs = [sc.decode(p, s, args.seconds) for p in args.song for s in pick_segments(p, args.segments, args.seconds)]
    speech = sc.decode(args.speech, 0, 10.0)
    speech = speech / np.abs(speech).max() * 0.5
    real_noises = [sc.decode(p, 0, 34.0) for p in args.noise]

    # 干净演唱：吞音。
    swallow = {k: [0, 0, 0.0, []] for k in post}
    outputs = []
    for dry in songs:
        voice = engine(dry)
        outputs.append(voice)
        for key, fn in post.items():
            stats = sc.swallow_stats(dry, fn(voice, dry))
            swallow[key][0] += stats["loud"]
            swallow[key][1] += stats["swallowed"]
            swallow[key][2] = min(swallow[key][2], stats["deepest"])
            swallow[key][3].append(stats["worst10"])

    # 带噪语音和带噪演唱：SI-SDR（5 dB 输入）。跳过前 2 秒。
    def noisy_sdr(clean, noise):
        noise = np.resize(noise, len(clean))
        noise = noise * np.sqrt(np.mean(clean**2) / np.mean(noise**2)) * 10 ** (-5 / 20)
        mix = clean + noise
        voice = engine(mix)
        skip = 2 * sc.RATE
        return {k: si_sdr(clean[skip:], fn(voice, mix)[skip:]) for k, fn in post.items()}

    speech_sdr = [noisy_sdr(speech, n) for n in real_noises]
    song_sdr = [noisy_sdr(songs[i], real_noises[0]) for i in range(0, len(songs), args.segments)]

    # 纯噪声：降噪量。跳过前 4 秒，让噪底跟踪稳定。
    pure = {}
    noises = dict(fixture_noises())
    for index, n in enumerate(real_noises, 1):
        noises[f"录音{index}"] = np.resize(n, 8 * sc.RATE) / np.sqrt(np.mean(n**2)) * 10 ** (-30 / 20)
    for name, noise in noises.items():
        voice = engine(noise)
        skip = 4 * sc.RATE
        pure[name] = {k: rms_db(noise[skip:]) - rms_db(fn(voice, noise)[skip:]) for k, fn in post.items()}

    for key in post:
        loud, swallowed, deepest, worst = swallow[key]
        reduction = " | ".join(f"{pure[n][key]:.1f}" for n in noises)
        print(f"| {args.engine} {key} | {100 * swallowed / max(loud, 1):.1f}% | {deepest:.0f} | {np.mean(worst):.1f} "
              f"| {np.mean([s[key] for s in speech_sdr]):.2f} | {np.mean([s[key] for s in song_sdr]):.2f} | {reduction} |", flush=True)
    print("# noises: " + ", ".join(noises), flush=True)


if __name__ == "__main__":
    sys.exit(main())
