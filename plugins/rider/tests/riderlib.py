"""run_tests.py 和 compare_reference.py 共用的信号生成、渲染和测量工具。"""
import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal as sps

BUILD = os.environ.get("MINT_RIDER_BUILD", "D:/mint-build/rider")
PLUGIN = f"{BUILD}/plugins/rider/mint_rider_artefacts/Release/VST3/Mintaria Helios.vst3"
PROBE = f"{BUILD}/tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
BENCH = f"{BUILD}/plugins/rider/mint_rider_bench_artefacts/Release/mint_rider_bench.exe"
TMP = os.environ.get("MINT_RIDER_TMP", f"{BUILD}/test-tmp")
os.makedirs(TMP, exist_ok=True)

SR = 48000
# 电平刻度与插件一致：RMS，正弦读数等于它的峰值。
CAL = 10 * np.log10(2.0)
RNG = np.random.default_rng(20261102)

# 参数名到归一化值的换算，全部是线性映射。
def n_target(db):   return (db + 36.0) / 30.0
def n_range(db):    return db / 24.0
def n_pct(p):       return p / 100.0
def n_look(ms):     return ms / 10.0

def helios_sets(target=-18, up=12, down=12, speed=50, gate=50, follow=100, look=0):
    return [f"Target={n_target(target)}", f"Range Up={n_range(up)}", f"Range Down={n_range(down)}",
            f"Speed={n_pct(speed)}", f"Vocal Gate={n_pct(gate)}", f"Music Follow={n_pct(follow)}",
            f"Look-ahead={n_look(look)}"]

def db_to_amp(db): return 10.0 ** (db / 20.0)

def level_db(x):
    """与插件相同刻度的整段电平。"""
    x = np.asarray(x, dtype=np.float64)
    return 10 * np.log10(max(np.mean(x ** 2), 1e-20)) + CAL

def sine(freq, seconds, db, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return db_to_amp(db) * np.sin(2 * np.pi * freq * t)

def steps(levels, seconds, freq=1000.0, sr=SR):
    """按段改变电平的正弦。相位连续，换挡点没有毛刺以外的成分。"""
    env = np.concatenate([np.full(int(seconds * sr), db_to_amp(l)) for l in levels])
    return env * np.sin(2 * np.pi * freq * np.arange(len(env)) / sr)

def voice(seconds, db, f0=150.0, vibrato=0.01, sr=SR):
    """合成浊音：带抖动的谐波列加三个共振峰。没有真人的呼吸和辅音。"""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    drift = np.cumsum(RNG.normal(0, 1, n))
    drift = drift / max(np.max(np.abs(drift)), 1e-9) * 0.015
    freq = f0 * (1 + vibrato * np.sin(2 * np.pi * 5.2 * t) + drift)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    out = np.zeros(n)
    formants = [(600, 110), (1200, 130), (2600, 200)]
    for h in range(1, int(5000 / f0)):
        hz = h * f0
        gain = 1.0 / h
        for fc, bw in formants:
            gain *= 1 + 3.0 / (1 + ((hz - fc) / bw) ** 2)
        out += gain * np.sin(h * phase)
    return out * db_to_amp(db - CAL) / np.sqrt(np.mean(out ** 2))

def phrase_env(n, attack=0.02, release=0.06, sr=SR):
    a, r = int(attack * sr), int(release * sr)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    return env

def phrase(seconds, db, f0=150.0, syllables=True, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    v = voice(seconds, db, f0, sr=sr) * phrase_env(len(t), sr=sr)
    if syllables:
        v *= 10 ** (1.5 * np.sin(2 * np.pi * 3.7 * t) / 20)
    return v

def breath(seconds, db, sr=SR):
    """呼吸：300 Hz–6 kHz 带噪声，缓起缓落。"""
    n = int(seconds * sr)
    sos = sps.butter(2, [300, 6000], btype="bandpass", fs=sr, output="sos")
    x = sps.sosfilt(sos, RNG.normal(0, 1, n + 4800))[4800:]
    x *= np.hanning(n) ** 0.5
    return x * db_to_amp(db - CAL) / np.sqrt(np.mean(x ** 2))

def sibilant(seconds, db, sr=SR):
    n = int(seconds * sr)
    sos = sps.butter(4, [4500, 11000], btype="bandpass", fs=sr, output="sos")
    x = sps.sosfilt(sos, RNG.normal(0, 1, n + 4800))[4800:] * phrase_env(n, 0.01, 0.02, sr)
    return x * db_to_amp(db - CAL) / np.sqrt(np.mean(x ** 2))

def pink(seconds, db, sr=SR):
    n = int(seconds * sr)
    spec = np.fft.rfft(RNG.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x * db_to_amp(db - CAL) / np.sqrt(np.mean(x ** 2))

def music_bed(seconds, db, sr=SR):
    """五个不成谐波的正弦，电平稳定，用来精确读目标位移。"""
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate([97.0, 233.0, 587.0, 1499.0, 3301.0]))
    return x * db_to_amp(db - CAL) / np.sqrt(np.mean(x ** 2))

def pilot(n, db=-90.0, sr=SR):
    """低于所有门限的 19 kHz 导频，让静音段的增益也能直接读出来。"""
    return db_to_amp(db) * np.sin(2 * np.pi * 19000 * np.arange(n) / sr)

def write_wav(name, x, sr=SR):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    path = f"{TMP}/{name}"
    wavfile.write(path, sr, x)
    return path

def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int32:
        x = x / 2147483648.0
    elif x.dtype == np.int16:
        x = x / 32768.0
    x = x.astype(np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return x, sr

def render(in_path, out_name, sets=(), sc=None, block=256, plugin=PLUGIN, extra=()):
    """用 mint_rider_bench 渲染，32 位浮点输出，不跑预热段。"""
    out = f"{TMP}/{out_name}"
    cmd = [BENCH, "render", plugin, in_path, out, "--block", str(block)]
    for s in sets:
        cmd += ["--set", s]
    if sc:
        cmd += ["--sc", sc]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or "rendered" not in r.stdout:
        raise RuntimeError(f"render failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return read_wav(out)[0], r.stdout.strip()

def run_signal(name, x, sets=(), **kw):
    path = write_wav(f"{name}_in.wav", x, kw.pop("sr", SR))
    out, _ = render(path, f"{name}_out.wav", sets, **kw)
    return out

def sample_gain_db(x_in, x_out, floor=1e-5):
    """逐采样增益（延迟为 0 时）。输入太小的采样无法除，用前一个有效值填上。"""
    x_in = np.asarray(x_in, dtype=np.float64)
    g = np.full(len(x_in), np.nan)
    ok = np.abs(x_in) > floor
    g[ok] = 20 * np.log10(np.maximum(np.abs(x_out[ok] / x_in[ok]), 1e-9))
    idx = np.where(ok, np.arange(len(g)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = g[idx]
    filled[np.isnan(filled)] = 0.0
    return filled

def response_ms(g, start, end, sr=SR):
    """从 start 开始的 10%–90% 和到 90% 的时间。稳态取 [end-0.5s, end)。"""
    settled = np.median(g[end - sr // 2:end])
    origin = np.median(g[start - sr // 10:start])
    span = settled - origin
    if abs(span) < 0.2:
        return None
    y = (g[start:end] - origin) / span
    t10 = int(np.argmax(y >= 0.1))
    t90 = int(np.argmax(y >= 0.9))
    return dict(origin=origin, settled=settled, t10=t10 * 1000 / sr, t90=t90 * 1000 / sr,
                rise=(t90 - t10) * 1000 / sr, overshoot=float((np.max(y) - 1) * abs(span)))

def voiced_frames(x, sr, level_below_p90=20.0):
    """用和插件同样的思路（1 kHz 低通、降到 4 kHz、40 ms 窗自相关）在测试里单独挑出明显的浊音帧。"""
    d_factor = round(sr / 4000)
    low_rate = sr / d_factor
    d = sps.sosfilt(sps.butter(4, 1000, fs=sr, output="sos"), x)[::d_factor]
    win, lo, hi, hop = round(0.04 * low_rate), round(low_rate / 800), round(low_rate / 60), round(0.01 * low_rate)
    pos, value = [], []
    for n in range(win + hi, len(d), hop):
        cur = d[n - win:n]
        energy = (cur * cur).sum()
        best = 0.0
        if energy > 1e-10:
            for lag in range(lo, hi + 1):
                past = d[n - win - lag:n - lag]
                ep, c = (past * past).sum(), (cur * past).sum()
                if c > 0 and ep > 1e-12:
                    best = max(best, c / np.sqrt(energy * ep))
        pos.append(n * d_factor)
        value.append(best)
    pos, value = np.array(pos), np.array(value)
    level = frame_levels(x, pos, sr)
    return pos, (value > 0.8) & (level > np.percentile(level, 90) - level_below_p90)

def frame_levels(x, pos, sr, win=0.05):
    w = int(win * sr)
    cum = np.concatenate([[0.0], np.cumsum(x * x)])
    return 10 * np.log10(np.maximum((cum[pos] - cum[np.maximum(pos - w, 0)]) / w, 1e-14)) + CAL

def short_levels(x, sr, win=0.4, hop=0.1):
    w, h = int(win * sr), int(hop * sr)
    cum = np.concatenate([[0.0], np.cumsum(x * x)])
    pos = np.arange(0, len(x) - w, h)
    p = (cum[pos + w] - cum[pos]) / w
    return pos, 10 * np.log10(np.maximum(p, 1e-14)) + CAL
