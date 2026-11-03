#!/usr/bin/env python3
"""定位吞音的原因：电平依赖、lsnr、基频和起音后时间、Deep Filtering 和掩码各自的作用。

用法：python -I singing_diagnose.py --test <lethe_rt_test.exe> --models <plugins/denoise/models> --work <仓库外目录> \
          --song <a.m4a> ... [--control <说话素材.m4a>] [--gains 24,12,6,0,-6,-12,-18,-24,-36]

素材只在本机使用，不要提交，也不要用于训练。
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

HOP = 480


def level_sweep(args, songs, talks):
    print("\n## 整体增益扫描（吞音比例 / 最差 10% 帧的平均跌落）\n")
    names = list(sc.ENGINES)
    print("| 增益 dB | " + " | ".join(f"{n} 歌" for n in names) + " | " + " | ".join(f"{n} 说话" for n in names) + " |")
    print("|---|" + "---|" * (2 * len(names)))
    for gain in args.gains:
        cells = []
        for group in (songs, talks):
            for kind, delay in sc.ENGINES.values():
                loud, swallowed, worst = 0, 0, []
                for dry in group:
                    scaled = dry * 10 ** (gain / 20)
                    stats = sc.swallow_stats(scaled, sc.run_engine(args.test, kind, delay, scaled, args.work))
                    loud += stats["loud"]
                    swallowed += stats["swallowed"]
                    worst.append(stats["worst10"])
                cells.append(f"{100 * swallowed / max(loud, 1):.1f}% / {np.mean(worst):.0f}" if group else "-")
        print(f"| {gain:+d} | " + " | ".join(cells) + " |", flush=True)


def run_prototype(args, dry, directory="df3", lookahead=2, options=None):
    out, trace = DfPrototype(Path(args.models) / directory, lookahead, options).process(dry)
    delay = HOP + lookahead * HOP
    return np.pad(out[delay:], (0, delay))[: len(dry)], trace


def lsnr_bins(args, songs):
    print("\n## lsnr 和吞音（100 ms 响亮帧）\n")
    print("| 引擎 | lsnr 范围 dB | 平均跌落 dB | 吞音比例 | 帧数 |")
    print("|---|---|---|---|---|")
    for name, directory, lookahead in (("标准", "df3", 2), ("低延迟", "df3_ll", 0)):
        all_lsnr, all_diff, zero_stage = [], [], []
        for dry in songs:
            wet, trace = run_prototype(args, dry, directory, lookahead)
            stats = sc.swallow_stats(dry, wet)
            frames = len(stats["diff"])
            lsnr = np.array([np.nanmean(trace["lsnr"][i * 10 : (i + 1) * 10]) for i in range(frames)])
            zero_stage.append(np.mean(trace["stage"] == 0))
            all_lsnr.append(lsnr[stats["mask"]])
            all_diff.append(stats["diff"][stats["mask"]])
        lsnr, diff = np.concatenate(all_lsnr), np.concatenate(all_diff)
        for low, high in ((-15.1, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 36)):
            picked = (lsnr >= low) & (lsnr < high)
            if picked.any():
                print(f"| {name} | {low:.0f} 到 {high} | {diff[picked].mean():.1f} | {100 * np.mean(diff[picked] < sc.DROP_DB):.1f}% | {picked.sum()} |")
        print(f"\n{name}：lsnr 和跌落的相关系数 {np.corrcoef(lsnr, diff)[0, 1]:.2f}，lsnr 低于 -15 的零掩码阶段占全部帧 {100 * np.mean(zero_stage):.2f}%\n")


def pitch_track(x):
    low = resample_poly(x, 1, 6)
    frames = len(low) // 80
    f0, strength = np.zeros(frames), np.zeros(frames)
    for f in range(frames):
        end = (f + 1) * 80
        if end < 420:
            continue
        segment = low[end - 320 : end]
        delayed = np.stack([low[end - 320 - lag : end - lag] for lag in range(8, 101)])
        corr = delayed @ segment / (np.sqrt(np.sum(segment**2) * np.sum(delayed**2, axis=1)) + 1e-12)
        best = int(np.argmax(corr))
        strength[f], f0[f] = corr[best], 8000 / (best + 8)
    return f0, strength


def pitch_and_onset(args, songs):
    by_f0, by_onset = {}, {}
    for dry in songs:
        kind, delay = sc.ENGINES["标准"]
        wet = sc.run_engine(args.test, kind, delay, dry, args.work)
        f0, strength = pitch_track(dry)
        frames = min(len(f0), len(wet) // HOP)
        level = 10 * np.log10(np.array([np.mean(dry[i * HOP : (i + 1) * HOP] ** 2) for i in range(frames)]) + 1e-14)
        out = 10 * np.log10(np.array([np.mean(wet[i * HOP : (i + 1) * HOP] ** 2) for i in range(frames)]) + 1e-14)
        loud = level > level.max() - 20
        since = 0
        for i in range(100, frames):
            since = 0 if (level[i] - level[max(0, i - 5)] > 6 or not loud[i]) else since + 1
            if loud[i] and strength[i] >= 0.7:
                by_f0.setdefault(min(int(np.log2(f0[i] / 100) * 2), 6), []).append(out[i] - level[i])
                by_onset.setdefault(min(since // 20, 8), []).append(out[i] - level[i])

    print("\n## 标准档：吞音和基频（有音高的响亮 10 ms 帧）\n")
    print("| 基频 Hz | 平均跌落 dB | 吞音比例 | 帧数 |\n|---|---|---|---|")
    for key in sorted(by_f0):
        v = np.array(by_f0[key])
        print(f"| {100 * 2 ** (key / 2):.0f}-{100 * 2 ** ((key + 1) / 2):.0f} | {v.mean():.1f} | {100 * np.mean(v < sc.DROP_DB):.1f}% | {len(v)} |")
    print("\n## 标准档：吞音和起音后时间\n")
    print("| 起音后 s | 平均跌落 dB | 吞音比例 | 帧数 |\n|---|---|---|---|")
    for key in sorted(by_onset):
        v = np.array(by_onset[key])
        print(f"| {key * 0.2:.1f}-{(key + 1) * 0.2:.1f} | {v.mean():.1f} | {100 * np.mean(v < sc.DROP_DB):.1f}% | {len(v)} |")


def filter_vs_mask(args, songs):
    print("\n## 标准档：吞音帧上 Deep Filtering 和掩码的作用\n")
    print("| 片段 | 吞音帧 | 这些帧原始跌落 dB | 这些帧只用掩码的跌落 dB |\n|---|---|---|---|")
    for index, dry in enumerate(songs, 1):
        wet, _ = run_prototype(args, dry)
        stats = sc.swallow_stats(dry, wet)
        bad = stats["mask"] & (stats["diff"] < sc.DROP_DB)
        if not bad.any():
            continue
        masked, _ = run_prototype(args, dry, options=Options(use_df=False))
        only_mask = sc.swallow_stats(dry, masked)
        print(f"| {index} | {bad.sum()} | {stats['diff'][bad].mean():.1f} | {only_mask['diff'][bad].mean():.1f} |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--song", action="append", required=True)
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--gains", default="24,12,6,0,-6,-12,-18,-24,-36")
    args = parser.parse_args()
    args.gains = [int(g) for g in args.gains.split(",") if g]
    Path(args.work).mkdir(parents=True, exist_ok=True)

    songs = [sc.decode(p, s, 16.0) for p in args.song for s in pick_segments(p, 2, 16.0)]
    talks = [sc.decode(p, s, 16.0) for p in args.control for s in pick_segments(p, 3, 16.0)]
    print("歌片段 rms dBFS：" + " ".join(f"{10 * np.log10(np.mean(x**2)):.1f}" for x in songs))
    print("说话片段 rms dBFS：" + " ".join(f"{10 * np.log10(np.mean(x**2)):.1f}" for x in talks))

    if args.gains:
        level_sweep(args, songs, talks)
    lsnr_bins(args, songs)
    pitch_and_onset(args, songs)
    filter_vs_mask(args, songs)


if __name__ == "__main__":
    sys.exit(main())
