"""Iris 验证脚本共用的工具：路径、参数映射、渲染和模拟目标曲线。"""
import math
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import wavfile

BUILD = os.environ.get("MINT_EQ_BUILD", "D:/mint-build/eq")
PROBE = f"{BUILD}/tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
BENCH = f"{BUILD}/plugins/eq/mint_eq_bench_artefacts/Release/mint_eq_bench.exe"
PLUGIN = f"{BUILD}/plugins/eq/mint_eq_artefacts/Release/VST3/Mintaria Iris.vst3"
WORK = os.environ.get("MINT_EQ_WORK", os.path.join(tempfile.gettempdir(), "mint_eq_tests"))
os.makedirs(WORK, exist_ok=True)

TYPES = ["bell", "lowShelf", "highShelf", "lowCut", "highCut", "notch", "tilt"]
STEREO = ["stereo", "left", "right", "mid", "side"]
SLOPES = [6, 12, 18, 24, 30, 36, 48, 72, 96]
NEUTRAL_Q = 0.70710678118654752


def norm_log(value, low, high):
    return math.log(value / low) / math.log(high / low)


def norm_linear(value, low, high):
    return (value - low) / (high - low)


def band_sets(index, kind="bell", freq=1000.0, gain=0.0, q=1.0, slope=12, stereo="stereo", dynamic=False, thresh=-24.0, rng=-6.0):
    """生成一个频段的 --set 参数列表，index 从 1 开始。"""
    p = f"B{index:02d} "
    return [
        f"{p}Used=1",
        f"{p}On=1",
        f"{p}Type={TYPES.index(kind) / (len(TYPES) - 1)}",
        f"{p}Slope={SLOPES.index(slope) / (len(SLOPES) - 1)}",
        f"{p}Stereo={STEREO.index(stereo) / (len(STEREO) - 1)}",
        f"{p}Freq={norm_log(freq, 10.0, 30000.0)}",
        f"{p}Gain={norm_linear(gain, -30.0, 30.0)}",
        f"{p}Q={norm_log(q, 0.025, 40.0)}",
        f"{p}Dynamic={1 if dynamic else 0}",
        f"{p}Threshold={norm_linear(thresh, -80.0, 0.0)}",
        f"{p}Range={norm_linear(rng, -30.0, 30.0)}",
    ]


def set_args(sets):
    out = []
    for item in sets:
        out += ["--set", item]
    return out


def write_wav(name, data, rate):
    path = os.path.join(WORK, name)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    wavfile.write(path, rate, data)
    return path


def read_wav(path):
    rate, data = wavfile.read(path)
    if data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype == np.int16:
        data = data / 32768.0
    return rate, np.asarray(data, dtype=np.float64)


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(" ".join(cmd) + "\n" + result.stdout + result.stderr)
    return result.stdout


def render_probe(in_path, out_path, sets, plugin=PLUGIN, block=256):
    return run([PROBE, "render", plugin, in_path, out_path, "--block", str(block)] + set_args(sets))


def render_bench(in_path, out_path, sets, sweeps=(), plugin=PLUGIN, block=256, mono=False):
    cmd = [BENCH, "render", plugin, in_path, out_path, "--block", str(block)] + set_args(sets)
    if mono:
        cmd.append("--mono")
    for sweep in sweeps:
        cmd += ["--sweep", sweep]
    return run(cmd)


_impulse_lock = threading.Lock()
_impulses = {}


def impulse_path(rate, length=1 << 17, amplitude=0.1):
    # 并行渲染共用同一个文件，只写一次。
    with _impulse_lock:
        if rate not in _impulses:
            x = np.zeros(length, dtype=np.float32)
            x[0] = amplitude
            _impulses[rate] = write_wav(f"impulse_{rate}.wav", x, rate)
    return _impulses[rate], amplitude


def impulse_response_db(rate, sets, name, plugin=PLUGIN, channel=0, use_probe=False):
    """跑脉冲，返回频率轴和幅频 (dB)。

    mint_probe render 输出 24 位，脉冲幅度 0.1 时噪声底约 -73 dB，量 -60 dB 附近的阻带会有 2 dB 左右的假误差。
    所以默认用 mint_eq_bench render：流程相同，输出 32 位浮点。verify_response.py 里抽查两者一致。
    """
    in_path, amplitude = impulse_path(rate)
    out_path = os.path.join(WORK, f"{name}.wav")
    if use_probe:
        render_probe(in_path, out_path, sets, plugin=plugin)
    else:
        render_bench(in_path, out_path, sets, plugin=plugin)
    _, y = read_wav(out_path)
    if y.ndim > 1:
        y = y[:, channel]
    spectrum = np.fft.rfft(y / amplitude)
    freqs = np.fft.rfftfreq(len(y), 1.0 / rate)
    return freqs, 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))


def _biquad_db(f, f0, n2, n1, n0, d2, d1, d0):
    x = f / f0
    v = x * x
    num = (n0 - n2 * v) ** 2 + (n1 * x) ** 2
    den = (d0 - d2 * v) ** 2 + (d1 * x) ** 2
    return 10.0 * np.log10(np.maximum(num, 1e-30) / np.maximum(den, 1e-30))


def analog_target_db(f, kind, f0, gain=0.0, q=1.0, slope=12):
    """设计目标曲线。q 是面板上的 Q，换算和原型与 IrisDsp.h、IrisFilterDesign.h 一致。"""
    f = np.asarray(f, dtype=np.float64)
    if kind in ("bell", "lowShelf", "highShelf", "tilt"):
        g = abs(gain)
        sign = 1.0 if gain >= 0 else -1.0
        if kind == "bell":
            q = q * NEUTRAL_Q
        else:
            q = min(max(NEUTRAL_Q * q ** 0.47, 0.1), 4.0)
        if kind == "bell":
            a = 10 ** (g / 40)
            out = _biquad_db(f, f0, 1, a / q, 1, 1, 1 / (a * q), 1)
        elif kind == "lowShelf":
            a = 10 ** (g / 40)
            r = math.sqrt(a) / q
            out = _biquad_db(f, f0, a, a * r, a * a, a, r, 1)
        elif kind == "highShelf":
            a = 10 ** (g / 40)
            r = math.sqrt(a) / q
            out = _biquad_db(f, f0, a * a, a * r, a, 1, r, a)
        else:
            a = 10 ** (g / 20)
            r = math.sqrt(a) / q
            out = _biquad_db(f, f0, a, r, 1, 1, r, a)
        return sign * out
    if kind == "notch":
        return _biquad_db(f, f0, 1, 0, 1, 1, 1 / (q * NEUTRAL_Q), 1)

    order = slope // 6
    high = kind == "lowCut"
    out = np.zeros_like(f)
    if order % 2 == 1:
        out += _biquad_db(f, f0, 0, 1, 0, 0, 1, 1) if high else _biquad_db(f, f0, 0, 0, 1, 0, 1, 1)
    pairs = order // 2
    for k in range(1, pairs + 1):
        qs = 0.5 / math.cos(math.pi * (2 * k - 1) / (2 * order))
        if k == pairs:
            qs *= q
        out += _biquad_db(f, f0, 1, 0, 0, 1, 1 / qs, 1) if high else _biquad_db(f, f0, 0, 0, 1, 1, 1 / qs, 1)
    return out


def blt_bell_db(f, f0, gain, q, rate):
    """普通双线性变换钟形（常见的 cookbook 公式），只做对照。q 是面板上的 Q。"""
    a = 10 ** (gain / 40)
    w0 = 2 * math.pi * f0 / rate
    alpha = math.sin(w0) / (2 * q * NEUTRAL_Q)
    b = np.array([1 + alpha * a, -2 * math.cos(w0), 1 - alpha * a])
    aa = np.array([1 + alpha / a, -2 * math.cos(w0), 1 - alpha / a])
    z = np.exp(-2j * np.pi * np.asarray(f) / rate)
    num = b[0] + b[1] * z + b[2] * z * z
    den = aa[0] + aa[1] * z + aa[2] * z * z
    return 20 * np.log10(np.abs(num / den))


def parallel(fn, items, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
