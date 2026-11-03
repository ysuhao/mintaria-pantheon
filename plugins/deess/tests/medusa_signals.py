"""Medusa 测试共用：合成信号、WAV 读写、调用 mint_probe、测量函数。只依赖 numpy / scipy。"""
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

BUILD = Path('D:/mint-build/deess')
PROBE = BUILD / 'tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe'
PLUGIN = BUILD / 'plugins/deess/mint_deess_artefacts/Release/VST3/Mintaria Medusa.vst3'

VOWELS = {
    # 名称: (基频, [(共振峰频率, 带宽, 增益 dB)...], 谱斜率)
    'a_male': (120.0, [(750, 90, 0), (1200, 110, -4), (2600, 170, -14), (3400, 250, -20)], 1.0),
    'a_female': (220.0, [(900, 100, 0), (1400, 130, -5), (2900, 190, -13), (4000, 260, -20)], 1.0),
    'i_bright': (240.0, [(300, 60, 0), (2500, 120, -4), (3300, 170, -4), (4400, 240, -8)], 0.7),
    'belt': (440.0, [(1000, 120, 0), (1600, 150, -2), (3100, 180, -5), (4200, 260, -9), (6200, 500, -18)], 0.6),
}


def vowel(n, fs, name='a_male', rms_db=-18.0, vibrato=True, seed=1):
    f0, formants, slope = VOWELS[name]
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    pitch = f0 * (1 + (0.012 * np.sin(2 * np.pi * 5.2 * t) if vibrato else 0))
    phase = 2 * np.pi * np.cumsum(pitch) / fs
    x = np.zeros(n)
    for h in range(1, int(0.45 * fs / f0)):
        freq = h * f0
        amp = 1.0 / h ** slope
        gain = sum(10 ** (g / 20) / (1 + ((freq - f) / b) ** 2) for f, b, g in formants)
        x += amp * (gain + 0.002) * np.sin(h * phase + rng.uniform(0, 6.28))
    x *= 10 ** (rms_db / 20) / np.sqrt(np.mean(x ** 2))
    return x


def sibilant_noise(n, fs, lo=4000.0, hi=10000.0, rms_db=-18.0, seed=2):
    rng = np.random.default_rng(seed)
    sos = signal.butter(4, [lo, min(hi, 0.45 * fs)], 'bandpass', fs=fs, output='sos')
    x = signal.sosfilt(sos, rng.standard_normal(n + 2000))[2000:]
    return x * 10 ** (rms_db / 20) / np.sqrt(np.mean(x ** 2))


def fades(n, fs, attack_ms, release_ms):
    env = np.ones(n)
    a = max(1, int(attack_ms * 0.001 * fs))
    r = max(1, int(release_ms * 0.001 * fs))
    env[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    env[-r:] = np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    return env


def speech_like(fs=48000, vowel_name='a_male', sib_rel_db=-3.0, level_db=0.0, cycles=8, seed=3, sib_band=(4000.0, 10000.0), gap_ms=35.0):
    """元音与齿音交替。返回 (信号, 元音区间, 齿音区间)。sib_rel_db 是齿音 RMS 相对元音 RMS。"""
    vowel_len, sib_len = int(0.42 * fs), int(0.15 * fs)
    # gap_ms 为负时元音和齿音交叠，接近连续语流。
    gap = int(gap_ms * 0.001 * fs)
    total = cycles * (vowel_len + gap + sib_len + gap) + int(0.1 * fs)
    v_all = vowel(total, fs, vowel_name, -18.0 + level_db)
    s_all = sibilant_noise(total, fs, *sib_band, rms_db=-18.0 + level_db + sib_rel_db, seed=seed)
    x = np.zeros(total)
    vowels, sibs = [], []
    pos = int(0.05 * fs)
    while pos + vowel_len + gap + sib_len < total:
        x[pos:pos + vowel_len] += v_all[pos:pos + vowel_len] * fades(vowel_len, fs, 15, 25)
        vowels.append((pos, pos + vowel_len))
        pos += vowel_len + gap
        x[pos:pos + sib_len] += s_all[pos:pos + sib_len] * fades(sib_len, fs, 10, 15)
        sibs.append((pos, pos + sib_len))
        pos += sib_len + gap
    return x, vowels, sibs


def write_wav24(path, x, fs):
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    ints = np.clip(np.round(x * 8388607.0), -8388608, 8388607).astype(np.int32)
    raw = ints.astype('<i4').tobytes()
    packed = bytearray(len(raw) // 4 * 3)
    packed[0::3], packed[1::3], packed[2::3] = raw[0::4], raw[1::4], raw[2::4]
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(3)
        w.setframerate(int(fs))
        w.writeframes(bytes(packed))


def read_wav(path):
    fs, data = wavfile.read(str(path))
    if data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    return fs, data.astype(np.float64)


def render(in_path, out_path, plugin=None, params=None, block=256):
    cmd = [str(PROBE), 'render', str(plugin or PLUGIN), str(in_path), str(out_path), '--block', str(block)]
    for name, value in (params or {}).items():
        cmd += ['--set', f'{name}={value}']
    done = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
    if done.returncode != 0:
        raise RuntimeError(done.stdout + done.stderr)
    return done.stdout.strip()


def freq_norm(hz):
    """频率旋钮 3–12 kHz、skew 0.5 的归一化值。"""
    return ((hz - 3000.0) / 9000.0) ** 0.5


def band_rms_db(x, fs, lo, hi):
    sos = signal.butter(6, [lo, min(hi, 0.47 * fs)], 'bandpass', fs=fs, output='sos')
    return 20 * np.log10(np.sqrt(np.mean(signal.sosfilt(sos, x) ** 2)) + 1e-12)


def band_attenuation(x_in, x_out, seg, fs, lo=4000.0, hi=10000.0, skip_ms=25, tail_ms=20):
    """齿音段内频带能量的衰减量（dB，正值表示被削掉）。"""
    a = seg[0] + int(skip_ms * 0.001 * fs)
    b = seg[1] - int(tail_ms * 0.001 * fs)
    pad = int(0.01 * fs)
    sos = signal.butter(6, [lo, min(hi, 0.47 * fs)], 'bandpass', fs=fs, output='sos')
    fi = signal.sosfilt(sos, x_in[max(0, a - pad):b])[pad:]
    fo = signal.sosfilt(sos, x_out[max(0, a - pad):b])[pad:]
    return 10 * np.log10((np.mean(fi ** 2) + 1e-20) / (np.mean(fo ** 2) + 1e-20))


def change_db(x_in, x_out, a, b):
    """差信号相对原信号的电平（dB），-200 表示逐采样相同。"""
    d = x_out[a:b] - x_in[a:b]
    e = np.sqrt(np.mean(d ** 2))
    if e == 0.0:
        return -200.0
    return 20 * np.log10(e / (np.sqrt(np.mean(x_in[a:b] ** 2)) + 1e-20))


def gain_trace_db(x_in, x_out, fs, lo=5000.0, hi=8000.0, win=128, hop=32):
    """用短时频谱比估计某频段内的增益曲线，噪声本身的起伏被约掉。"""
    f, _, si = signal.stft(x_in, fs, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
    _, t, so = signal.stft(x_out, fs, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
    m = (f >= lo) & (f < hi)
    pi = np.sum(np.abs(si[m]) ** 2, axis=0)
    po = np.sum(np.abs(so[m]) ** 2, axis=0)
    return t, 10 * np.log10((po + 1e-20) / (pi + 1e-20))
