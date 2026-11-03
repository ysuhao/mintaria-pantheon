#!/usr/bin/env python3
"""三档引擎对演唱的吞音统计。

用法：python -I analyze_singing.py --test <lethe_rt_test.exe> --work <仓库外的临时目录> \
          --song <a.m4a> --song <b.m4a> ... [--control <说话素材.m4a>] [--segments 3] [--seconds 16]

每首歌自动挑响亮帧最多的几段。输出里歌名用序号代替，素材只在本机使用，不要提交，也不要用于训练。
吞音帧：比片段峰值帧低 20 dB 以内的 100 ms 帧，输出比输入低 12 dB 以上。前 1 秒不算。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # python -I 不把脚本目录放进搜索路径

import singing_common as sc  # noqa: E402


def pick_segments(path, count, seconds):
    """按响亮帧数挑互不重叠的片段，返回起始秒。"""
    total = sc.duration(path)
    audio = sc.decode(path, 0, total)
    level = sc.frame_db(audio)
    loud = level > level.max() - 25
    window = int(seconds * 10)
    score = np.convolve(loud.astype(float), np.ones(window), mode="valid")
    starts = []
    for _ in range(count):
        best = int(np.argmax(score))
        if score[best] <= 0:
            break
        starts.append(best / 10)
        score[max(0, best - window) : best + window] = 0
    return starts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--song", action="append", required=True)
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=16.0)
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    print("| 素材 | 起始 s | 响亮帧 | " + " | ".join(f"{n} 吞音帧 / 最深 dB" for n in sc.ENGINES) + " |")
    print("|---|---|---|---|---|---|")

    def run_group(paths, prefix):
        totals = {n: [0, 0, 0.0] for n in sc.ENGINES}
        per_song = []
        for index, path in enumerate(paths, 1):
            song = {n: [0, 0] for n in sc.ENGINES}
            for start in pick_segments(path, args.segments, args.seconds):
                dry = sc.decode(path, start, args.seconds)
                cells = []
                loud = 0
                for name, (kind, delay) in sc.ENGINES.items():
                    stats = sc.swallow_stats(dry, sc.run_engine(args.test, kind, delay, dry, work))
                    loud = stats["loud"]
                    cells.append(f"{stats['swallowed']} / {stats['deepest']:.0f}")
                    totals[name][0] += stats["loud"]
                    totals[name][1] += stats["swallowed"]
                    totals[name][2] = min(totals[name][2], stats["deepest"])
                    song[name][0] += stats["loud"]
                    song[name][1] += stats["swallowed"]
                print(f"| {prefix}{index} | {start:.1f} | {loud} | " + " | ".join(cells) + " |", flush=True)
            per_song.append(song)
        return totals, per_song

    totals, per_song = run_group(args.song, "歌")
    print("\n汇总（演唱）：")
    for name, (loud, swallowed, deepest) in totals.items():
        print(f"- {name}：{swallowed}/{loud} = {100 * swallowed / max(loud, 1):.1f}%，最深 {deepest:.0f} dB")
    print("每首歌的吞音比例（%）：")
    for index, song in enumerate(per_song, 1):
        print(f"- 歌{index}：" + "，".join(f"{n} {100 * s / max(l, 1):.1f}" for n, (l, s) in song.items()))

    if args.control:
        print("\n| 对照 | 起始 s | 响亮帧 | " + " | ".join(sc.ENGINES) + " |")
        print("|---|---|---|---|---|---|")
        totals, _ = run_group(args.control, "说话")
        print("\n汇总（说话）：")
        for name, (loud, swallowed, deepest) in totals.items():
            print(f"- {name}：{swallowed}/{loud} = {100 * swallowed / max(loud, 1):.1f}%，最深 {deepest:.0f} dB")


if __name__ == "__main__":
    sys.exit(main())
