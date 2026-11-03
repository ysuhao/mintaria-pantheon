"""脉冲响应的测量函数：RT60、回声密度、频谱起伏、自相关、声道相关。只依赖 numpy 和 scipy。"""
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.special import erfc

OCTAVES = [125, 250, 500, 1000, 2000, 4000, 8000]


def read_wav(path):
    rate, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float64) / (2.0 ** (8 * data.dtype.itemsize - 1))
    elif data.dtype.kind == "u":
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return rate, data


def write_wav(path, rate, data):
    wavfile.write(path, rate, np.asarray(data, dtype=np.float32))


def impulse(rate, seconds, amplitude=0.5):
    x = np.zeros((int(rate * seconds), 2), dtype=np.float32)
    x[0, :] = amplitude
    return x


def sweep(rate, seconds=6.0, low=20.0, high=20000.0, amplitude=0.25):
    """指数扫频（Farina 法）及其逆滤波器。"""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    k = np.log(high / low)
    x = np.sin(2.0 * np.pi * low * seconds / k * (np.exp(t * k / seconds) - 1.0))
    fade = int(0.02 * rate)
    window = np.ones(n)
    window[:fade] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    window[-fade:] = window[:fade][::-1]
    x = x * window
    inverse = x[::-1] * np.exp(-t * k / seconds)
    # 归一化：x 与 inverse 的卷积峰值为 1，即单位冲激响应对应幅度 1。
    norm = np.max(np.abs(signal.fftconvolve(x, inverse)))
    return amplitude * x, inverse / norm / amplitude


def deconvolve(recorded, inverse, sweep_samples, ir_seconds, rate):
    """反卷积后从直达时刻取 ir_seconds 秒。"""
    out = []
    for ch in range(recorded.shape[1]):
        full = signal.fftconvolve(recorded[:, ch], inverse)
        start = sweep_samples - 1
        out.append(full[start:start + int(ir_seconds * rate)])
    return np.stack(out, axis=1)


def octave_band(x, rate, centre):
    """时间反转后滤波，把滤波器振铃对衰减测量的影响压到最小。"""
    low = centre / np.sqrt(2.0)
    high = min(centre * np.sqrt(2.0), 0.45 * rate)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    return signal.sosfilt(sos, x[::-1])[::-1]


def decay_time(x, rate, start_db=-5.0, end_db=-35.0):
    """Schroeder 反向积分。尾部本底噪声用最后 5 % 的能量估计并减掉。"""
    energy = np.asarray(x, dtype=np.float64) ** 2
    tail = energy[int(len(energy) * 0.95):]
    floor = float(np.mean(tail)) if len(tail) else 0.0
    edc = np.cumsum((energy - floor)[::-1])[::-1]
    edc = np.maximum(edc, 1e-30)
    if edc[0] <= 1e-28:
        return float("nan")
    db = 10.0 * np.log10(edc / edc[0])
    below_start = np.argmax(db <= start_db)
    below_end = np.argmax(db <= end_db)
    if db[below_start] > start_db or db[below_end] > end_db or below_end <= below_start + 2:
        return float("nan")
    t = np.arange(below_start, below_end) / rate
    slope, _ = np.polyfit(t, db[below_start:below_end], 1)
    return -60.0 / slope if slope < 0 else float("nan")


def band_rt(ir, rate, centre, start_db=-5.0, end_db=-35.0):
    values = [decay_time(octave_band(ir[:, ch], rate, centre), rate, start_db, end_db) for ch in range(ir.shape[1])]
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def echo_density(x, rate, window_ms=20.0, hop_ms=1.0):
    """Abel & Huang 2006 的归一化回声密度。1 表示局部统计已接近高斯噪声。"""
    n = int(rate * window_ms * 0.001)
    hop = max(1, int(rate * hop_ms * 0.001))
    win = np.hanning(n)
    win /= win.sum()
    out = []
    for pos in range(0, len(x) - n, hop):
        seg = x[pos:pos + n]
        sigma = np.sqrt(np.sum(win * seg * seg))
        out.append(np.sum(win * (np.abs(seg) > sigma)) if sigma > 0 else 0.0)
    return np.arange(len(out)) * hop / rate + 0.5 * n / rate, np.array(out) / erfc(1.0 / np.sqrt(2.0))


def mixing_time(x, rate, threshold=0.9):
    """回声密度首次达到阈值且其后 50 ms 不再回落的时间，返回秒。"""
    t, ned = echo_density(x, rate)
    run = int(0.05 / (t[1] - t[0]))
    for i in range(len(ned) - run):
        if np.all(ned[i:i + run] >= threshold):
            return float(t[i])
    return float("nan")


def onset(x, rate, threshold_db=-40.0):
    peak = np.max(np.abs(x))
    return float(np.argmax(np.abs(x) >= peak * 10 ** (threshold_db / 20.0)) / rate) if peak > 0 else float("nan")


def spectral_roughness(x, rate, start, length=1.0, low=200.0, high=6000.0):
    """局部中值归一化后的窄带谱。纯扩散尾音的 |X|^2 服从指数分布，dB 标准差约 5.6，最大峰约高出中值 11–12 dB。"""
    a = int(start * rate)
    n = int(length * rate)
    seg = x[a:a + n]
    if len(seg) < n // 2:
        return dict(std=float("nan"), p999=float("nan"), peak=float("nan"), peak_hz=float("nan"))
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / rate)
    keep = (freqs >= low) & (freqs <= high)
    db = 10.0 * np.log10(np.maximum(spectrum, 1e-30))
    width = int(np.ceil(100.0 / (freqs[1] - freqs[0]))) | 1
    local = signal.medfilt(db, width)
    rel = (db - local)[keep]
    idx = int(np.argmax(rel))
    return dict(std=float(np.std(rel)), p999=float(np.percentile(rel, 99.9)), peak=float(rel[idx]), peak_hz=float(freqs[keep][idx]))


def persistence(x, rate, t1, t2, length=0.5, low=200.0, high=6000.0):
    """两个时间点窄带谱的相关系数。模式孤立（金属声）时相同频率一直最响，相关高。"""
    def spec(start):
        a = int(start * rate)
        n = int(length * rate)
        seg = x[a:a + n]
        if len(seg) < n:
            return None, None
        s = 10.0 * np.log10(np.maximum(np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2, 1e-30))
        return s, np.fft.rfftfreq(n, 1.0 / rate)

    s1, f = spec(t1)
    s2, _ = spec(t2)
    if s1 is None or s2 is None:
        return float("nan")
    keep = (f >= low) & (f <= high)
    width = int(np.ceil(100.0 / (f[1] - f[0]))) | 1
    r1 = (s1 - signal.medfilt(s1, width))[keep]
    r2 = (s2 - signal.medfilt(s2, width))[keep]
    return float(np.corrcoef(r1, r2)[0, 1])


def flatten_decay(x, rate, window_ms=30.0):
    """去掉衰减包络，留下载波，再看自相关。"""
    n = int(rate * window_ms * 0.001) | 1
    env = np.sqrt(signal.fftconvolve(x * x, np.hanning(n) / np.hanning(n).sum(), mode="same"))
    return x / np.maximum(env, np.max(env) * 1e-4)


def comb_peak(x, rate, start=0.02, length=0.6, min_lag_ms=2.0, max_lag_ms=300.0):
    """去包络后的归一化自相关峰。周期性重复（梳状、颤动回声）会冒出来。"""
    a = int(start * rate)
    n = int(length * rate)
    seg = flatten_decay(x[a:a + n], rate)
    seg = seg - np.mean(seg)
    spec = np.fft.rfft(seg, 2 * len(seg))
    ac = np.fft.irfft(np.abs(spec) ** 2)[:len(seg)]
    ac /= ac[0] if ac[0] > 0 else 1.0
    lo = int(min_lag_ms * 0.001 * rate)
    hi = min(int(max_lag_ms * 0.001 * rate), len(ac) - 1)
    # 自相关样本数随滞后减少，按有效长度校正。
    ac = ac * len(seg) / np.maximum(len(seg) - np.arange(len(ac)), 1)
    idx = lo + int(np.argmax(np.abs(ac[lo:hi])))
    return float(np.abs(ac[idx])), 1000.0 * idx / rate


def stereo_correlation(ir, rate, start=0.05, length=1.0):
    a = int(start * rate)
    n = int(length * rate)
    l = ir[a:a + n, 0]
    r = ir[a:a + n, 1]
    denom = np.sqrt(np.sum(l * l) * np.sum(r * r))
    if denom <= 0:
        return float("nan")
    full = signal.correlate(l, r, mode="full", method="fft") / denom
    mid = len(full) // 2
    span = int(0.001 * rate)
    return float(np.max(np.abs(full[mid - span:mid + span + 1])))


def band_levels(x, rate, bands=OCTAVES):
    """各倍频程的能量，相对 1 kHz，单位 dB。"""
    levels = {}
    for centre in bands:
        band = octave_band(x, rate, centre)
        levels[centre] = 10.0 * np.log10(max(float(np.sum(band * band)), 1e-30))
    ref = levels.get(1000, 0.0)
    return {k: v - ref for k, v in levels.items()}


def energy_db(ir):
    """IR 总能量，每声道平均。0 dB 表示稳态宽带噪声经过后功率不变。"""
    return 10.0 * np.log10(max(float(np.mean(np.sum(ir * ir, axis=0))), 1e-30))
