"""演唱吞音测试的公共部分：解码、引擎调用和吞音统计。

真人演唱素材只在本机使用。这里不写死任何路径，解码出的临时文件由调用方放在仓库外的工作目录，
用完删除。不要把音频或歌名提交进仓库，也不要用于训练。
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 48000
FRAME = RATE // 10          # 100 ms
LOUD_DB = -20.0             # 响亮帧：距离片段峰值帧不超过 20 dB
DROP_DB = -12.0             # 吞音帧：比输入低 12 dB 以上
ENGINES = {"标准": ("standard", 1440), "低延迟": ("low", 480), "轻量": ("light", 960)}


def decode(path, start, seconds):
    """解码成 48 kHz 单声道（左右平均，和插件的特征一致）。"""
    ffmpeg = shutil.which("ffmpeg")
    command = [ffmpeg, "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(path),
               "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def duration(path):
    ffprobe = shutil.which("ffprobe")
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def run_engine(exe, kind, delay, audio, work):
    """用真实的 C++ 引擎处理，返回和输入对齐的输出。"""
    work = Path(work)
    source, target = work / "engine_in.wav", work / "engine_out.wav"
    sf.write(source, audio, RATE, subtype="FLOAT")
    subprocess.run([exe, str(source), "--kind", kind, "--engine-only", "1", "--out", str(target)], check=True, capture_output=True)
    out, _ = sf.read(target, dtype="float64", always_2d=True)
    source.unlink()
    target.unlink()
    out = out[delay:, 0]
    return np.pad(out, (0, max(0, len(audio) - len(out))))[: len(audio)]


def frame_db(x):
    n = len(x) // FRAME
    frames = x[: n * FRAME].reshape(n, FRAME)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-14)


def swallow_stats(dry, wet):
    """返回响亮帧数、吞音帧数、最深跌落和最差 10% 帧的平均跌落，以及每帧的差值。"""
    level = frame_db(dry)
    diff = frame_db(wet) - level
    # 前 1 秒是模型收敛，不算。
    loud = (level > level.max() + LOUD_DB) & (np.arange(len(level)) >= 10)
    picked = diff[loud]
    if len(picked) == 0:
        return {"loud": 0, "swallowed": 0, "deepest": 0.0, "worst10": 0.0, "diff": diff, "mask": loud}
    return {
        "loud": int(loud.sum()),
        "swallowed": int(np.sum(picked < DROP_DB)),
        "deepest": float(picked.min()),
        "worst10": float(np.mean(np.sort(picked)[: max(1, len(picked) // 10)])),
        "diff": diff,
        "mask": loud,
    }
