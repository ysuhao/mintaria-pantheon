#!/usr/bin/env python3
"""Medusa 结构选型：每种音频通路都调到 4–10 kHz 宽带噪声衰减 9 dB，再比较空闲零点和频带外的代价。

  python structure_compare.py

只依赖 numpy / scipy。这里只比较音频通路，检测器在插件里，所有结构喂同形状的增益曲线。
"""
import numpy as np
from scipy import signal

FS = 48000
rng = np.random.default_rng(7)


def band_db(x, lo, hi):
    spec = np.fft.rfft(x * np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(np.sum(np.abs(spec[m]) ** 2) + 1e-30)


def bp_coefs(f0, q):
    # 峰值增益为 1 的二阶带通（RBJ constant 0 dB peak）。
    w = 2 * np.pi * f0 / FS
    alpha = np.sin(w) / (2 * q)
    b = np.array([alpha, 0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w), 1 - alpha])
    return b / a[0], a / a[0]


def subtractive(x, g, f0, q):
    b, a = bp_coefs(f0, q)
    return x - (1 - g) * signal.lfilter(b, a, x)


def cascade(x, g_db, f0, q, spread):
    # 三段级联，每段同样深（插件里三段各有检测器，这是宽带噪声下的情况）。
    y = x
    g = 10 ** (g_db / 20)
    for f in (f0 / spread, f0, f0 * spread):
        y = subtractive(y, g, f, q)
    return y


def lr4_split(x, g, fc):
    b, a = signal.butter(2, fc, 'low', fs=FS)
    b2, a2 = signal.butter(2, fc, 'high', fs=FS)
    low = signal.lfilter(b, a, signal.lfilter(b, a, x))
    high = signal.lfilter(b2, a2, signal.lfilter(b2, a2, x))
    return low + g * high


def modulated_biquad(x, g_db, f0, q, every=32):
    # 直接调制钟形 biquad 系数，每 32 采样更新一次（常见动态 EQ 写法）。
    y = np.zeros_like(x)
    z = np.zeros(2)
    w = 2 * np.pi * f0 / FS
    for i in range(0, len(x), every):
        A = 10 ** (g_db[i] / 40)
        alpha = np.sin(w) / (2 * q)
        b = [1 + alpha * A, -2 * np.cos(w), 1 - alpha * A]
        a = [1 + alpha / A, -2 * np.cos(w), 1 - alpha / A]
        y[i:i + every], z = signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x[i:i + every], zi=z)
    return y


def gain_curve(n, start, stop, depth_db):
    # 与插件相同的非对称平滑：起 1.5 ms，收 40 ms。
    tgt = np.zeros(n)
    tgt[start:stop] = 1.0
    env = np.zeros(n)
    atk, rel = np.exp(-1 / (0.0015 * FS)), np.exp(-1 / (0.040 * FS))
    s = 0.0
    for i in range(n):
        c = atk if tgt[i] > s else rel
        s = tgt[i] + c * (s - tgt[i])
        env[i] = s
    return 1 - env * (1 - 10 ** (-depth_db / 20))


def main():
    n = FS
    start, stop = FS // 4, FS // 2
    noise = rng.standard_normal(n)
    gi = np.ones(n)

    structures = {
        '单钟形减法 Q=0.7': lambda x, g, gdb: subtractive(x, g, 6500, 0.7),
        '单钟形减法 Q=1.0': lambda x, g, gdb: subtractive(x, g, 6500, 1.0),
        '三段级联减法（采用）': lambda x, g, gdb: cascade(x, gdb, 6500, 1.2, 1.45),
        'LR4 分频高段压缩': lambda x, g, gdb: lr4_split(x, g, 4000),
        '系数调制钟形 Q=0.7': lambda x, g, gdb: modulated_biquad(x, gdb, 6500, 0.7),
    }

    seg = slice(start + 2400, stop - 240)
    print(f'{"结构":<22}{"所需深度":>8}{"空闲残差":>10}{"4-10k":>8}{"10-12k":>8}{"12-18k":>8}{"<3k":>8}{"<1k":>8}{"尾段频外":>10}')
    for name, fn in structures.items():
        # 二分找到让 4–10 kHz 衰减 9 dB 的增益深度。
        lo_depth, hi_depth = 0.5, 40.0
        for _ in range(12):
            depth = 0.5 * (lo_depth + hi_depth)
            g = gain_curve(n, start, stop, depth)
            out = fn(noise, g, 20 * np.log10(g))
            cut = band_db(noise[seg], 4000, 10000) - band_db(out[seg], 4000, 10000)
            lo_depth, hi_depth = (depth, hi_depth) if cut < 9.0 else (lo_depth, depth)
        # 空闲残差：增益恒为 0 dB 时输出与输入之差。
        residual = fn(noise, gi, np.zeros(n)) - noise
        idle_db = 20 * np.log10(np.std(residual[2000:]) / np.std(noise) + 1e-12)
        row = [idle_db]
        for lo, hi in ((4000, 10000), (10000, 12000), (12000, 18000), (100, 3000), (100, 1000)):
            row.append(band_db(out[seg], lo, hi) - band_db(noise[seg], lo, hi))
        # 增益变化段的频外能量：只看增益上下沿内 >18 kHz 与 <500 Hz 的差信号，反映调制产物。
        edge = slice(stop - 100, stop + 4000)
        diff = out[edge] - noise[edge]
        spread = band_db(diff, 16000, 24000) - band_db(noise[edge], 16000, 24000)
        row.append(spread)
        print(f'{name:<22}{depth:>8.1f}' + ''.join(f'{v:>10.1f}' if i in (0, 6) else f'{v:>8.2f}' for i, v in enumerate(row)))


if __name__ == '__main__':
    main()
