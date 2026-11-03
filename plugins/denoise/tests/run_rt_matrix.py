#!/usr/bin/env python3
"""按真实时间跑一组缓冲区、采样率和负载组合，汇总延迟、丢帧、回退和跳变。

用法：python -I run_rt_matrix.py <lethe_rt_test.exe> <noisy.wav> [--seconds 15]

判定：输出不含 NaN，最大样点跳变不超过输入最大跳变的 1.2 倍（标记 sine 的用例放宽到 2 倍，
因为切换本身要在 1 ms 内走完幅度差）。回退次数只记录，高负载下回退是设计行为，要看的是有没有爆音。
语音文件的循环点本身有 0.29 的跳变，会盖住小爆音，所以强制回退和切档的用例用纯正弦。
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

CASES = [
    # 名称, 参数
    ("标准 48k/1", ["--block", "1"]),
    ("标准 48k/32", ["--block", "32"]),
    ("标准 48k/64", ["--block", "64"]),
    ("标准 48k/128", ["--block", "128"]),
    ("标准 48k/256", ["--block", "256"]),
    ("标准 48k/512", ["--block", "512"]),
    ("标准 48k/1024", ["--block", "1024"]),
    ("标准 44.1k/256", ["--rate", "44100", "--block", "256"]),
    ("标准 96k/256", ["--rate", "96000", "--block", "256"]),
    ("标准 192k/512", ["--rate", "192000", "--block", "512"]),
    ("标准 48k/256 单声道", ["--channels", "1"]),
    ("低延迟 48k/64", ["--kind", "low", "--block", "64"]),
    ("低延迟 48k/128", ["--kind", "low", "--block", "128"]),
    ("低延迟 48k/256", ["--kind", "low", "--block", "256"]),
    ("低延迟 48k/1024", ["--kind", "low", "--block", "1024"]),
    ("轻量 48k/64", ["--kind", "light", "--block", "64"]),
    ("轻量 48k/256", ["--kind", "light", "--block", "256"]),
    ("轻量 48k/1024", ["--kind", "light", "--block", "1024"]),
    ("轻量 48k/256 单声道", ["--kind", "light", "--channels", "1"]),
    ("标准 48k/256 全核满载", ["--burn", "16"]),
    ("低延迟 48k/256 全核满载", ["--kind", "low", "--burn", "16"]),
    ("标准 48k/256 工作线程迟到 40 ms", ["--force-late-ms", "40"], "sine"),
    ("低延迟 48k/256 工作线程迟到 40 ms", ["--kind", "low", "--force-late-ms", "40"], "sine"),
    ("轻量 48k/256 工作线程迟到 40 ms", ["--kind", "light", "--force-late-ms", "40"], "sine"),
    ("标准 96k/512 工作线程迟到 40 ms", ["--rate", "96000", "--block", "512", "--force-late-ms", "40"], "sine96"),
    ("标准 48k/256 四实例", ["--instances", "4"]),
    ("48k/256 每 3 秒切档（正弦）", ["--switch-sec", "3"], "sine"),
]


def write_sine(path, rate, seconds=20):
    t = np.arange(int(rate * seconds)) / rate
    sf.write(path, 0.25 * np.sin(2 * np.pi * 440 * t), rate, subtype="FLOAT")


def field(pattern, text, default=""):
    match = re.search(pattern, text)
    return match.group(1) if match else default


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("exe")
    parser.add_argument("wav")
    parser.add_argument("--seconds", default="15")
    args = parser.parse_args()

    print("| 组合 | 上报延迟 ms | 帧 | 丢帧 | 回退 | 推理均值/峰值 ms | 工作线程占单核 | 回调占比 | 最大跳变 输入/输出 | 结果 |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    temp = Path(tempfile.mkdtemp(prefix="lethe_rt_"))
    write_sine(temp / "sine.wav", 48000)
    write_sine(temp / "sine96.wav", 96000)

    failures = 0
    for name, extra, *source in CASES:
        wav = str(temp / f"{source[0]}.wav") if source else args.wav
        limit = 2.0 if source else 1.2
        result = subprocess.run([args.exe, wav, "--seconds", args.seconds, *extra],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = result.stdout
        step_in = float(field(r"largest step input ([\d.]+)", text, "0"))
        step_out = float(field(r"output ([\d.]+), non-finite", text, "99"))
        bad = int(field(r"non-finite (\d+)", text, "1"))
        ok = result.returncode == 0 and bad == 0 and step_out <= step_in * limit
        failures += 0 if ok else 1
        print(f"| {name} | {field(r'samples \(([\d.]+) ms', text)} | {field(r'frames (\d+)', text)} "
              f"| {field(r'dropped (\d+)', text)} | {field(r'underruns (\d+)', text)} "
              f"| {field(r'infer avg ([\d.]+)', text)}/{field(r'max ([\d.]+) ms; worker', text)} "
              f"| {field(r'one core ([\d.]+)%', text)}% | {field(r'share of period ([\d.]+)%', text)}% "
              f"| {step_in:.3f}/{step_out:.3f} | {'PASS' if ok else 'FAIL'} |", flush=True)
        if not ok:
            print(text)

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
