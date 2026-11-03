#!/usr/bin/env python
"""Mintaria Echo 的可重复验证。用生成的测试信号，经 VST3 接口把插件当黑盒跑。

用法：python plugins/delay/tests/test_echo.py [--build D:/mint-build/delay] [--out 目录]

mint_probe render 不提供节拍、不能中途改参数，所以节拍同步和改时间的测试走 mint_delay_host。
需要 numpy 和 scipy。退出码非 0 表示有测试没过。
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="D:/mint-build/delay")
parser.add_argument("--out", default=None)
options = parser.parse_args()

PLUGIN = f"{options.build}/plugins/delay/mint_delay_artefacts/Release/VST3/Mintaria Echo.vst3"
PROBE = f"{options.build}/tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
HOST = f"{options.build}/plugins/delay/mint_delay_host_artefacts/Release/mint_delay_host.exe"
OUT = options.out or tempfile.mkdtemp(prefix="echo_test_")
os.makedirs(OUT, exist_ok=True)

failures = []
counter = 0


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def norm_skewed(value, low, high, centre):
    skew = math.log(0.5) / math.log((centre - low) / (high - low))
    return ((value - low) / (high - low)) ** skew


def time_norm(ms):
    return norm_skewed(ms, 1.0, 4000.0, 300.0)


def hi_norm(hz):
    return norm_skewed(hz, 1000.0, 20000.0, 6000.0)


def lo_norm(hz):
    return norm_skewed(hz, 20.0, 2000.0, 200.0)


DIVISIONS = ["1/32T", "1/32", "1/16T", "1/32D", "1/16", "1/8T", "1/16D", "1/8", "1/4T", "1/8D",
             "1/4", "1/2T", "1/4D", "1/2", "1/1T", "1/2D", "1/1", "1/1D"]
BEATS = [1 / 12, 1 / 8, 1 / 6, 3 / 16, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1, 4 / 3, 3 / 2, 2, 8 / 3, 3, 4, 6]


def division_norm(name):
    return DIVISIONS.index(name) / (len(DIVISIONS) - 1)


MODE = {"stereo": 0.0, "pingpong": 0.5, "mono": 1.0}

# 把音色影响降到最低，专门测时间和增益。
NEUTRAL = {"High Cut": 1.0, "Low Cut": 0.0, "Drive": 0.0, "Mix": 1.0, "LoFi": 0.0, "Mod Depth": 0.0}


def write_wav(signal, rate):
    global counter
    counter += 1
    path = f"{OUT}/in_{counter}.wav"
    data = signal if signal.ndim == 2 else np.stack([signal, signal], axis=1)
    wavfile.write(path, rate, data.astype(np.float32))
    return path


def read_wav(path):
    rate, data = wavfile.read(path)
    if data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    return rate, np.asarray(data, dtype=np.float64)


def render_host(signal, rate, params=None, at=(), bpm=None, bpm_at=(), block=256, mono=False):
    """走测试宿主，32 位浮点输出，能提供节拍信息和中途改参数。"""
    source = write_wav(signal, rate)
    target = source.replace("in_", "out_")
    command = [HOST, "render", PLUGIN, source, target, "--block", str(block)]
    if mono:
        command.append("--mono")
    for name, value in (params or {}).items():
        command += ["--set", f"{name}={value}"]
    for seconds, name, value in at:
        command += ["--at", str(seconds), f"{name}={value}"]
    if bpm is not None:
        command += ["--bpm", str(bpm)]
    for seconds, value in bpm_at:
        command += ["--bpm-at", str(seconds), str(value)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or "FAIL" in result.stdout:
        raise RuntimeError(result.stdout + result.stderr)
    info = dict(line.split("\t") for line in result.stdout.splitlines() if "\t" in line)
    return read_wav(target)[1], info


def render_probe(signal, rate, params=None, block=256):
    """走 mint_probe render，24 位输出。"""
    source = write_wav(signal, rate)
    target = source.replace("in_", "probe_")
    command = [PROBE, "render", PLUGIN, source, target, "--block", str(block)]
    for name, value in (params or {}).items():
        command += ["--set", f"{name}={value}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or "FAIL" in result.stdout:
        raise RuntimeError(result.stdout + result.stderr)
    return read_wav(target)[1]


def host_info(params=None, bpm=None):
    command = [HOST, "params", PLUGIN]
    for name, value in (params or {}).items():
        command += ["--set", f"{name}={value}"]
    if bpm is not None:
        command += ["--bpm", str(bpm)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return dict(line.split("\t")[:2] for line in result.stdout.splitlines() if line.startswith(("tail", "latency")))


def impulse(rate, seconds, at=0.1, level=0.5):
    x = np.zeros(int(rate * seconds))
    x[int(rate * at)] = level
    return x


def peak_position(x):
    """峰值位置，用抛物线插值到亚采样。"""
    i = int(np.argmax(np.abs(x)))
    if 0 < i < len(x) - 1:
        a, b, c = np.abs(x[i - 1]), np.abs(x[i]), np.abs(x[i + 1])
        denominator = a - 2 * b + c
        if abs(denominator) > 1e-15:
            return i + 0.5 * (a - c) / denominator
    return float(i)


def burst(rate, seconds, frequency=500.0, length=0.03, at=0.05, level=0.1):
    """汉宁窗短音，能量集中，用来按窗口量每次回声的电平。"""
    x = np.zeros(int(rate * seconds))
    n = int(rate * length)
    start = int(rate * at)
    t = np.arange(n) / rate
    x[start:start + n] = level * np.hanning(n) * np.sin(2 * np.pi * frequency * t)
    return x


def echo_rms(y, rate, time_s, index, length=0.03, at=0.05):
    start = int(rate * (at + time_s * index))
    return np.sqrt(np.mean(y[start:start + int(rate * length) + 8] ** 2))


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


# ------------------------------------------------------------------ 探针自检
result = subprocess.run([PROBE, "check", PLUGIN, "--expect-latency", "0"], capture_output=True, text=True, timeout=600)
report("mint_probe check", result.returncode == 0 and "latency: 0 samples" in result.stdout,
       " / ".join(result.stdout.split("\n")[:3]))

# ------------------------------------------------------------------ 干声直通
rate = 48000
noise = np.random.default_rng(1).normal(0, 0.1, rate)
y = render_probe(noise, rate, {"Mix": 0.0, "Feedback": 0.5})
error = np.max(np.abs(y[:, 0] - noise))
report("干湿 0% 时干声直通（mint_probe，24 位）", error < 3e-6, f"最大差 {db(error):.1f} dBFS，0 采样对齐")

y, _ = render_host(noise, rate, {"Mix": 0.2})
error = np.max(np.abs(y[:int(0.3 * rate) - 1, 0] - noise[:int(0.3 * rate) - 1].astype(np.float32)))
report("默认设置第一次回声之前等于干声", error == 0.0, f"最大差 {error:.2e}")

# ------------------------------------------------------------------ 毫秒模式时间误差
worst = 0.0
rows = []
for ms in [1.0, 10.0, 37.3, 100.0, 300.0, 1234.5, 4000.0]:
    x = impulse(48000, ms * 0.001 + 0.5)
    y = render_probe(x, 48000, {**NEUTRAL, "Time": time_norm(ms), "Feedback": 0.0})
    # 脉冲本身不在湿声里（Mix=100%），第一个峰就是第一次回声。
    measured = (peak_position(y[:, 0]) - int(0.1 * 48000)) / 48
    rows.append(f"{ms}->{measured:.3f}")
    worst = max(worst, abs(measured - ms))
report("毫秒模式实测时间（48 kHz，mint_probe）", worst < 0.05, f"最大误差 {worst:.4f} ms；" + "，".join(rows))

# ------------------------------------------------------------------ 默认音色下的时间偏移（环内滤波的群延迟）
x = impulse(48000, 1.0)
y, _ = render_host(x, 48000, {"Feedback": 0.0, "Mix": 1.0, "Drive": 0.0})
measured = (peak_position(y[:, 0]) - 4800) / 48
report("默认高切低切下的时间（300 ms）", abs(measured - 300) < 0.1, f"实测 {measured:.3f} ms，含滤波器群延迟")

# ------------------------------------------------------------------ 不同采样率
rows = []
worst = 0.0
for sr in [44100, 48000, 88200, 96000, 176400, 192000]:
    for ms in [250.0, 1234.5]:
        x = impulse(sr, 1.6)
        y = render_probe(x, sr, {**NEUTRAL, "Time": time_norm(ms), "Feedback": 0.0})
        measured = (peak_position(y[:, 0]) - int(0.1 * sr)) * 1000 / sr
        worst = max(worst, abs(measured - ms))
        rows.append(f"{sr}/{ms}:{measured - ms:+.4f}")
report("不同采样率时间一致", worst < 0.02, f"最大误差 {worst:.4f} ms；" + " ".join(rows))

# ------------------------------------------------------------------ 节拍同步：真实经 VST3 传 BPM
rows = []
worst = 0.0
for bpm in [60.0, 97.3, 120.0, 174.0]:
    for name in ["1/4", "1/8", "1/8D", "1/4T", "1/16", "1/2D", "1/32T"]:
        expected = BEATS[DIVISIONS.index(name)] * 60000 / bpm
        x = impulse(48000, expected * 0.001 + 0.3)
        y, _ = render_host(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm(name), "Feedback": 0.0}, bpm=bpm)
        measured = (peak_position(y[:, 0]) - 4800) / 48
        worst = max(worst, abs(measured - expected))
        rows.append(f"{bpm:g}/{name}:{measured - expected:+.4f}")
report("节拍同步实测时间（含附点、三连）", worst < 0.02, f"最大误差 {worst:.4f} ms；" + " ".join(rows[:7]) + " ...")

# 全部 18 个时值在 120 BPM 下
worst = 0.0
for name, beats in zip(DIVISIONS, BEATS):
    expected = beats * 500.0
    x = impulse(48000, expected * 0.001 + 0.3)
    y, _ = render_host(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm(name), "Feedback": 0.0}, bpm=120)
    measured = (peak_position(y[:, 0]) - 4800) / 48
    worst = max(worst, abs(measured - expected))
report("18 个时值（120 BPM）", worst < 0.02, f"最大误差 {worst:.4f} ms")

# 没有宿主速度时的回退
x = impulse(48000, 0.8)
y = render_probe(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm("1/4"), "Feedback": 0.0})
measured = (peak_position(y[:, 0]) - 4800) / 48
report("无宿主速度回退到备用 120 BPM（mint_probe）", abs(measured - 500) < 0.02, f"1/4 实测 {measured:.3f} ms")

fallback_norm = (90.0 - 20.0) / 280.0
y, _ = render_host(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm("1/4"), "Feedback": 0.0, "Fallback BPM": fallback_norm})
measured = (peak_position(y[:, 0]) - 4800) / 48
report("备用速度旋钮生效（90 BPM）", abs(measured - 666.667) < 0.05, f"1/4 实测 {measured:.3f} ms")

# 宿主先给后断：保持最后的速度
x = impulse(48000, 1.5, at=1.0)
y, _ = render_host(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm("1/4"), "Feedback": 0.0},
                   bpm=150, bpm_at=[(0.5, 0)])
measured = (peak_position(y[:, 0]) - 48000) / 48
report("宿主中途停止提供速度时沿用上次（150 BPM）", abs(measured - 400) < 0.05, f"1/4 实测 {measured:.3f} ms")

# 超长时值对折
x = impulse(48000, 9.0, at=0.1)
y, _ = render_host(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm("1/1D"), "Feedback": 0.0}, bpm=30)
measured = (peak_position(y[:, 0]) - 4800) / 48
report("超过 8 秒对折（30 BPM 的 1/1D=12 s -> 6 s）", abs(measured - 6000) < 0.05, f"实测 {measured:.3f} ms")

# ------------------------------------------------------------------ 反馈衰减
rows = []
worst = 0.0
for fb in [0.25, 0.5, 0.75, 0.9]:
    time_s = 0.1
    y, _ = render_host(burst(48000, 1.2), 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": fb})
    levels = [echo_rms(y[:, 0], 48000, time_s, n) for n in range(1, 8)]
    steps = np.diff(db(np.array(levels)))
    # 500 Hz 过环内一阶 6 kHz 带限的损耗按公式算出来，再加低切和高切几乎为 0 的影响
    loss = 20 * np.log10(1 / math.sqrt(1 + (500 / 6000) ** 2))
    expected = 20 * math.log10(fb) + loss
    error = float(np.max(np.abs(steps - expected)))
    worst = max(worst, error)
    rows.append(f"{fb:.0%}: 每次 {np.mean(steps):.3f} dB，预期 {expected:.3f}")
report("反馈衰减", worst < 0.1, f"最大偏差 {worst:.3f} dB；" + "；".join(rows))

y, _ = render_host(burst(48000, 1.0), 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0})
first = echo_rms(y[:, 0], 48000, 0.1, 1)
second = echo_rms(y[:, 0], 48000, 0.1, 2)
report("反馈 0% 只有一次回声", db(second) < -120, f"第 1 次 {db(first):.1f} dB，第 2 次 {db(second):.1f} dB")

# ------------------------------------------------------------------ 音色随反馈变暗
y, _ = render_host(impulse(48000, 1.0, level=0.3), 48000,
                   {"Time": time_norm(50.0), "Feedback": 0.8, "Mix": 1.0, "Drive": 0.0, "High Cut": hi_norm(12000.0), "Low Cut": lo_norm(40.0)})
centroids = []
for n in range(1, 8):
    start = 4800 + int(48000 * 0.05 * n) - 16
    segment = y[start:start + 1024, 0] * np.hanning(1024)
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(1024, 1 / 48000)
    centroids.append(float(np.sum(freqs * spectrum) / np.sum(spectrum)))
report("每次回声比上一次暗（频谱重心递减）", all(a > b for a, b in zip(centroids, centroids[1:])),
       " > ".join(f"{c:.0f}" for c in centroids) + " Hz")

# ------------------------------------------------------------------ 反馈 100% 不发散
rng = np.random.default_rng(2)
for label, params, time_ms in [
    ("中性", {**NEUTRAL}, 50.0),
    ("极短 1 ms", {**NEUTRAL}, 1.0),
    ("驱动最大", {**NEUTRAL, "Drive": 1.0}, 50.0),
    ("全开", {**NEUTRAL, "Drive": 1.0, "LoFi": 1.0, "Mod Depth": 1.0, "Mod Rate": 1.0}, 10.0),
]:
    x = np.zeros(48000 * 30)
    x[4800:4800 + 4800] = rng.uniform(-1.0, 1.0, 4800) * 2.0
    y, _ = render_host(x, 48000, {**params, "Time": time_norm(time_ms), "Feedback": 1.0})
    windows = np.array([np.max(np.abs(y[i:i + 48000])) for i in range(0, len(y) - 48000, 48000)])
    finite = np.all(np.isfinite(y))
    growing = windows[-1] > windows[2:].max() * 0.999 and windows[-1] > windows[2] * 1.05
    report(f"反馈 100% 不发散（{label}，+6 dBFS 噪声）", finite and windows.max() < 4.0 and not growing,
           f"峰值 {windows.max():.3f}，第 2 秒 {windows[1]:.3f}，第 29 秒 {windows[-1]:.4f}")

# 乒乓 100%
x = np.zeros(48000 * 10)
x[4800:4800 + 4800] = rng.uniform(-1.0, 1.0, 4800)
y, _ = render_host(np.stack([x, x], axis=1), 48000, {**NEUTRAL, "Mode": MODE["pingpong"], "Time": time_norm(30.0), "Feedback": 1.0})
report("反馈 100% 不发散（乒乓）", np.all(np.isfinite(y)) and np.max(np.abs(y)) < 4.0, f"峰值 {np.max(np.abs(y)):.3f}")

# ------------------------------------------------------------------ 改变时间无爆音
# 持续 1 kHz 正弦，中途改时间。正弦本身没有高频，爆音只会出现在高频残差里。
# 电平压低到 -34 dBFS，饱和器保持线性，否则反馈叠加后的谐波会盖住要看的东西。
# 对照组用同样的信号做硬切，判定标准是残差比硬切低 40 dB 以上。
HF_SOS = butter(6, 6000, "highpass", fs=48000, output="sos")
LEVEL = 0.02


def hf_peak(signal, skip=4800):
    return float(np.max(np.abs(sosfilt(HF_SOS, signal)[skip:])))


t = np.arange(48000 * 3) / 48000
sine = LEVEL * np.sin(2 * np.pi * 1000 * t)
# 淡入，否则信号起点的折角会随反馈一圈圈回来，被算成爆音。
sine[:2400] *= np.hanning(4800)[:2400]
hard =np.concatenate([sine[:48000], sine[48000 - 777:48000 * 3 - 777]])
baseline = hf_peak(hard)
QUIET = baseline * 0.01


def check_clicks(label, params, at=(), bpm=None, bpm_at=()):
    y, _ = render_host(sine, 48000, params, at=at, bpm=bpm, bpm_at=bpm_at, block=64)
    steady, _ = render_host(sine, 48000, params, bpm=bpm, block=64)
    peak = max(hf_peak(y[:, 0], skip=9600), hf_peak(y[:, 1], skip=9600))
    floor = max(hf_peak(steady[:, 0], skip=9600), hf_peak(steady[:, 1], skip=9600))
    report(f"无爆音（{label}）", peak < QUIET,
           f">6 kHz 残差峰值 {db(peak):.1f} dBFS，稳态基底 {db(floor):.1f}，硬切对照 {db(baseline):.1f}")


drag = [(1.0 + i * 0.03, "Time", time_norm(100.0 + 37.0 * i)) for i in range(20)]
for label, at, feedback in [
    ("100 -> 900 ms 单次", [(1.0, "Time", time_norm(900.0))], 0.0),
    ("100 -> 900 ms 单次，反馈 60%", [(1.0, "Time", time_norm(900.0))], 0.6),
    ("900 -> 100 ms 单次，反馈 60%", [(0.5, "Time", time_norm(900.0)), (1.5, "Time", time_norm(100.0))], 0.6),
    ("连续拖动 20 步", drag, 0.0),
    ("连续拖动 20 步，反馈 90%", drag, 0.9),
]:
    check_clicks(label, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": feedback}, at=at)

check_clicks("切模式", {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.3},
             at=[(1.0, "Mode", MODE["pingpong"]), (1.5, "Mode", MODE["mono"]), (2.0, "Mode", MODE["stereo"])])
check_clicks("宿主变速和换时值", {**NEUTRAL, "Time Base": 1.0, "Division": division_norm("1/8"), "Feedback": 0.3},
             at=[(2.0, "Division", division_norm("1/4T"))], bpm=120, bpm_at=[(1.0, 90), (1.5, 140)])
check_clicks("毫秒和同步互切", {**NEUTRAL, "Time": time_norm(100.0), "Division": division_norm("1/8"), "Feedback": 0.3},
             at=[(1.0, "Time Base", 1.0), (2.0, "Time Base", 0.0)], bpm=120)
check_clicks("反馈、干湿、高切低切跳变", {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.3, "Mix": 0.5},
             at=[(1.0, "Feedback", 0.9), (1.2, "Mix", 1.0), (1.4, "High Cut", 0.2), (1.6, "Low Cut", 0.6)])

# 驱动拉满本身就会产生谐波，这里只看跳变的那一刻比之后的稳态多出多少。
y, _ = render_host(sine, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0}, at=[(1.5, "Drive", 1.0)], block=64)
steady, _ = render_host(sine, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0, "Drive": 1.0}, block=64)
peak, floor = hf_peak(y[:, 0], skip=9600), hf_peak(steady[:, 0], skip=9600)
report("无爆音（驱动跳变）", peak < max(QUIET, floor * 2), f">6 kHz 残差峰值 {db(peak):.1f} dBFS，驱动拉满的稳态 {db(floor):.1f}")

# 改时间之后新时间要生效
x = impulse(48000, 2.0, at=1.0)
y, _ = render_host(x, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0}, at=[(0.5, "Time", time_norm(250.0))])
measured = (peak_position(y[:, 0]) - 48000) / 48
report("改时间后按新时间回声", abs(measured - 250) < 0.02, f"实测 {measured:.3f} ms")

y, _ = render_host(x, 48000, {**NEUTRAL, "Time Base": 1.0, "Division": division_norm("1/4"), "Feedback": 0.0},
                   bpm=120, bpm_at=[(0.5, 90)])
measured = (peak_position(y[:, 0]) - 48000) / 48
report("宿主变速后按新速度回声（120 -> 90 BPM）", abs(measured - 666.667) < 0.02, f"实测 {measured:.3f} ms")

# ------------------------------------------------------------------ 乒乓和立体声路由
for label, source_left, source_right in [("中置信号", 1, 1), ("只有左声道", 1, 0), ("只有右声道", 0, 1)]:
    x = impulse(48000, 1.0, level=0.5)
    stereo_x = np.stack([x * source_left, x * source_right], axis=1)
    y, _ = render_host(stereo_x, 48000, {**NEUTRAL, "Mode": MODE["pingpong"], "Time": time_norm(100.0), "Feedback": 0.7})
    sides = []
    ok = True
    for n in range(1, 7):
        start = 4800 + 4800 * n - 48
        left = np.max(np.abs(y[start:start + 480, 0]))
        right = np.max(np.abs(y[start:start + 480, 1]))
        sides.append(("L" if left > right else "R") + f"{20 * np.log10(max(left, right) / max(min(left, right), 1e-9)):.0f}")
        ok = ok and (left > right * 1000 if n % 2 == 1 else right > left * 1000)
    report(f"乒乓左右交替（{label}）", ok, " ".join(sides) + "（侧别和隔离 dB）")

x = impulse(48000, 0.6, level=0.5)
y, _ = render_host(np.stack([x, 0 * x], axis=1), 48000, {**NEUTRAL, "Mode": MODE["stereo"], "Time": time_norm(100.0), "Feedback": 0.5})
report("立体声左右独立", np.max(np.abs(y[:, 1])) == 0.0 and np.max(np.abs(y[:, 0])) > 0.1,
       f"右声道最大 {np.max(np.abs(y[:, 1])):.2e}")

y, _ = render_host(np.stack([x, 0 * x], axis=1), 48000, {**NEUTRAL, "Mode": MODE["mono"], "Time": time_norm(100.0), "Feedback": 0.5})
report("单声道两边相同", np.max(np.abs(y[:, 0] - y[:, 1])) == 0.0 and np.max(np.abs(y[:, 0])) > 0.1,
       f"左右差 {np.max(np.abs(y[:, 0] - y[:, 1])):.2e}")

# 单声道总线：乒乓和立体声都退化成单线，每次回声都要有
x = impulse(48000, 1.0, level=0.5)
for label, value in [("立体声", MODE["stereo"]), ("乒乓", MODE["pingpong"]), ("单声道", MODE["mono"])]:
    y, _ = render_host(x, 48000, {**NEUTRAL, "Mode": value, "Time": time_norm(100.0), "Feedback": 0.5}, mono=True)
    peaks = [np.max(np.abs(y[4800 + 4800 * n - 48:4800 + 4800 * n + 432, 0])) for n in range(1, 5)]
    report(f"单声道总线（模式={label}）", all(a > b > 0 for a, b in zip(peaks, peaks[1:])), " ".join(f"{db(p):.1f}" for p in peaks) + " dB")

# ------------------------------------------------------------------ 不同采样率音色
spreads = []
rows = []
for frequency in [1000.0, 5000.0]:
    levels = []
    for sr in [44100, 48000, 96000, 192000]:
        y, _ = render_host(burst(sr, 0.4, frequency=frequency), sr, {"Time": time_norm(100.0), "Feedback": 0.5, "Mix": 1.0, "Drive": 0.0})
        levels.append([echo_rms(y[:, 0], sr, 0.1, n) for n in (1, 2)])
    levels = db(np.array(levels))
    spread = levels.max(axis=0) - levels.min(axis=0)
    spreads.append(float(spread.max()))
    rows.append(f"{frequency:g} Hz 第 1/2 次回声极差 {spread[0]:.2f}/{spread[1]:.2f} dB")
report("默认音色跨采样率基本一致", max(spreads) < 1.0, "；".join(rows))

# ------------------------------------------------------------------ 缓冲区大小无关
x = np.zeros(48000)
x[100:2500] = np.random.default_rng(3).normal(0, 0.2, 2400)
params = {"Time": time_norm(57.3), "Feedback": 0.6, "Mix": 0.5, "Mod Depth": 0.3, "LoFi": 0.5, "Mode": MODE["pingpong"]}
reference, _ = render_host(x, 48000, params, block=1024)
worst = 0.0
for block in [1, 7, 64, 256, 8192]:
    y, _ = render_host(x, 48000, params, block=block)
    worst = max(worst, float(np.max(np.abs(y - reference))))
report("缓冲区 1/7/64/256/8192 与 1024 输出相同", worst < 2e-3 or db(worst) < -60, f"最大差 {db(worst):.1f} dBFS")

# ------------------------------------------------------------------ 静音、denormal、NaN
x = np.zeros(48000 * 8)
x[4800] = 0.5
y, _ = render_host(x, 48000, {"Time": time_norm(1.0), "Feedback": 0.999, "Mix": 1.0}, block=64)
ok = np.all(np.isfinite(y))
tiny = np.abs(y[np.nonzero(y)])
subnormal = int(np.sum(tiny < 1.175e-38))
report("静音尾部无 NaN/Inf/denormal", ok and subnormal == 0, f"非零样本 {tiny.size}，denormal {subnormal} 个")

x = np.zeros(4800)
y, _ = render_host(x, 48000, {"Mix": 1.0, "Feedback": 0.9, "Drive": 1.0})
report("纯静音输入输出全零（驱动最大，有偏置）", np.max(np.abs(y)) == 0.0, f"最大 {np.max(np.abs(y)):.2e}")

# ------------------------------------------------------------------ getTailLengthSeconds
rows = []
ok = True
for fb_norm, expected in [(0.0, 0.34), (0.5, 0.3 * (1 + 10) + 0.04), (1.0, math.inf)]:
    info = host_info({"Time": time_norm(300.0), "Feedback": fb_norm})
    tail = float(info.get("tail", "nan"))
    good = math.isinf(tail) if math.isinf(expected) else abs(tail - expected) < 0.01
    ok = ok and good and info.get("latency") == "0"
    rows.append(f"反馈 {fb_norm:.0%}: {tail:g} s")
report("getTailLengthSeconds 经 VST3 读回", ok, "；".join(rows))

# 同步模式跟随速度
info = host_info({"Time Base": 1.0, "Division": division_norm("1/4"), "Feedback": 0.0}, bpm=60)
report("同步模式的尾音跟随宿主速度", abs(float(info.get("tail", "nan")) - 1.04) < 0.01, f"60 BPM 1/4 反馈 0%: {info.get('tail')} s")

# 报的尾音要不短于实际
x = impulse(48000, 6.0)
y, info = render_host(x, 48000, {"Time": time_norm(100.0), "Feedback": 0.6, "Mix": 1.0, "High Cut": 1.0, "Low Cut": 0.0, "Drive": 0.0})
above = np.nonzero(np.abs(y[:, 0]) > 0.5 * 0.001)[0]
actual = (above[-1] - 4800) / 48000 if above.size else 0.0
report("实际 -60 dB 时间不超过上报的尾音", actual <= float(info["tail"]), f"实测 {actual:.3f} s，上报 {info['tail']} s")

# ------------------------------------------------------------------ 其余功能的基本行为
sine = 0.05 * np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
y0, _ = render_host(sine, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0})
y1, _ = render_host(sine, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0, "Mod Depth": 1.0, "Mod Rate": norm_skewed(5.0, 0.1, 10.0, 1.0)})
report("调制在湿声里产生变化", np.max(np.abs(y1[9600:, 0] - y0[9600:, 0])) > 0.01, f"差值峰值 {np.max(np.abs(y1[9600:, 0] - y0[9600:, 0])):.3f}")
report("调制下左右声道不同", np.max(np.abs(y1[9600:, 0] - y1[9600:, 1])) > 0.01, f"差值峰值 {np.max(np.abs(y1[9600:, 0] - y1[9600:, 1])):.3f}")

loud = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
clean, _ = render_host(loud, 48000, {**NEUTRAL, "Time": time_norm(10.0), "Feedback": 0.0})
driven, _ = render_host(loud, 48000, {**NEUTRAL, "Time": time_norm(10.0), "Feedback": 0.0, "Drive": 1.0})


def thd(signal):
    spectrum = np.abs(np.fft.rfft(signal[4800:4800 + 43200] * np.hanning(43200)))
    bin_of = lambda hz: int(hz * 43200 / 48000)
    fundamental = spectrum[bin_of(1000) - 5:bin_of(1000) + 5].max()
    harmonics = [spectrum[bin_of(1000 * k) - 5:bin_of(1000 * k) + 5].max() for k in range(2, 8)]
    return math.sqrt(sum(h * h for h in harmonics)) / fundamental


report("驱动增加谐波", thd(driven[:, 0]) > thd(clean[:, 0]) * 5, f"THD {thd(clean[:, 0]):.4f} -> {thd(driven[:, 0]):.4f}")
crushed, _ = render_host(loud, 48000, {**NEUTRAL, "Time": time_norm(10.0), "Feedback": 0.0, "LoFi": 1.0})
report("低保真改变信号", np.max(np.abs(crushed[9600:, 0] - clean[9600:, 0])) > 0.05, f"差值峰值 {np.max(np.abs(crushed[9600:, 0] - clean[9600:, 0])):.3f}")

# 低保真拉满也不能把安静的人声吃成静音（曾经量化得太粗，-27 dBFS 的 3 kHz 以上被整段门掉）
rows = []
quiet_ok = True
for frequency, floor_db in [(1000.0, -3.0), (3000.0, -30.0)]:
    source = burst(48000, 0.6, frequency=frequency)
    y, _ = render_host(source, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0, "LoFi": 1.0})
    gain = db(echo_rms(y[:, 0], 48000, 0.1, 1) / echo_rms(source, 48000, 0.1, 0))
    half, _ = render_host(source, 48000, {**NEUTRAL, "Time": time_norm(100.0), "Feedback": 0.0, "LoFi": 0.5})
    rows.append(f"{frequency:g} Hz: 50% {db(echo_rms(half[:, 0], 48000, 0.1, 1) / echo_rms(source, 48000, 0.1, 0)):+.1f}，100% {gain:+.1f} dB")
    quiet_ok = quiet_ok and gain > floor_db
report("低保真 100% 保留 -27 dBFS 信号", quiet_ok, "；".join(rows))

# 无过采样，量一下饱和的混叠（驱动最大，5 kHz，高切打开）
tone = 0.5 * np.sin(2 * np.pi * 5000 * np.arange(48000) / 48000)
y, _ = render_host(tone, 48000, {**NEUTRAL, "Time": time_norm(10.0), "Feedback": 0.0, "Drive": 1.0})
spectrum = np.abs(np.fft.rfft(y[4800:4800 + 43200, 0] * np.hanning(43200)))
mask = np.ones_like(spectrum, dtype=bool)
for k in range(1, 5):
    centre = int(k * 5000 * 43200 / 48000)
    mask[max(0, centre - 6):centre + 7] = False
print(f"[INFO] 驱动最大时 5 kHz 正弦的最大非谐波分量（混叠）: {db(spectrum[mask].max() / spectrum.max()):.1f} dB（相对基波）")

# CPU 粗测：默认设置 + 调制和低保真全开
x = np.random.default_rng(4).normal(0, 0.1, 48000 * 5)
_, info = render_host(x, 48000, {"Mod Depth": 0.5, "LoFi": 0.5, "Feedback": 0.6, "Mode": MODE["pingpong"]})
print(f"[INFO] 处理耗时占实时的比例（48 kHz，块 256，经 VST3）: {float(info['realtime_ratio']) * 100:.2f}%")

print()
print(f"{len(failures)} 项未通过" if failures else "全部通过")
print(f"中间文件: {OUT}")
sys.exit(1 if failures else 0)
