#!/usr/bin/env python
"""Mintaria Vesta 的离线验证。全部信号由脚本生成，经 mint_probe 过真实的 VST3。

    python plugins/tube/tests/verify_tube.py [--build D:/mint-build/tube] [--out results.txt]

依赖 numpy、scipy。先构建 mint_probe、mint_tube_VST3、mint_tube_model_bench。
"""
import argparse
import ctypes
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt

RATE = 48000
DRIVES = (0, 6, 12, 18, 24, 30, 36)
failures = []
lines = []


def say(text=''):
    print(text, flush=True)
    lines.append(text)


def check(name, ok, detail=''):
    say(f"{'PASS' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


class Plugin:
    def __init__(self, build):
        self.probe = os.path.join(build, 'tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe')
        self.vst3 = os.path.join(build, 'plugins/tube/mint_tube_artefacts/Release/VST3/Mintaria Vesta.vst3')
        self.temp = tempfile.TemporaryDirectory()
        self.count = 0

    def run(self, *args):
        return subprocess.run([self.probe, *args], capture_output=True, text=True, errors='replace')

    def render(self, x, rate=RATE, drive=12.0, output=0.0, recovery=120.0, hq=False, block=256):
        self.count += 1
        source = os.path.join(self.temp.name, f'in{self.count}.wav')
        target = os.path.join(self.temp.name, f'out{self.count}.wav')
        wavfile.write(source, int(rate), np.asarray(x, dtype=np.float32))
        result = self.run('render', self.vst3, source, target,
                          '--set', f'Drive={drive / 36.0}',
                          '--set', f'Output={(output + 12.0) / 24.0}',
                          '--set', f'Recovery={math.log(recovery / 10.0) / math.log(200.0)}',
                          '--set', f'Quality={1.0 if hq else 0.0}',
                          '--block', str(block))
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        _, data = wavfile.read(target)
        data = data.astype(np.float64) / 2147483648.0
        os.remove(source)
        os.remove(target)
        self.reported = int(result.stdout.split('reported latency')[1].strip())
        self.last_stereo = data
        return data[:, 0] if data.ndim > 1 else data


def tone(hz, level_db, seconds=1.0, rate=RATE):
    t = np.arange(int(rate * seconds)) / rate
    return 10.0 ** (level_db / 20.0) * np.sin(2.0 * np.pi * hz * t)


def voice_like(rms_db, seconds=3.0, rate=RATE, seed=3):
    """基频滑动的带谐波信号，叠加共振峰、音节包络和少量齿音噪声，幅度分布接近人声。"""
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    t = np.arange(n) / rate
    f0 = 165.0 + 25.0 * np.sin(2.0 * np.pi * 0.7 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / rate
    x = np.zeros(n)
    for h in range(1, 41):
        hz = h * f0
        formant = sum(a / (1.0 + ((hz - c) / b) ** 2) for c, b, a in ((500, 120, 1.0), (1500, 200, 0.5), (2800, 300, 0.25)))
        x += np.sin(h * phase + 0.9 * h * h) * (0.25 / h + formant)
    envelope = np.maximum(0.0, np.sin(np.pi * 3.0 * t)) ** 1.5
    hiss = np.diff(rng.standard_normal(n + 1)) * (np.maximum(0.0, np.sin(np.pi * 3.0 * t + 2.2)) ** 8)
    x = x * envelope + 0.25 * hiss
    return x * 10.0 ** (rms_db / 20.0) / rms(x)


def k_weight(x):
    """BS.1770 的 K 计权，48 kHz 系数。"""
    x = lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585], x)
    return lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], x)


def spectrum(y, rate):
    y = y[len(y) // 2:]
    window = np.kaiser(len(y), 24.0)    # 旁瓣低于 -180 dB，主瓣约正负 8 个频点
    return np.fft.rfftfreq(len(y), 1.0 / rate), np.abs(np.fft.rfft(y * window)) / (window.sum() / 2.0)


def band(mags, index):
    return float(np.sqrt(np.sum(mags[max(index - 9, 0):index + 10] ** 2)))


def harmonics(y, f0, rate=RATE):
    freq, mags = spectrum(y, rate)
    df = freq[1]
    return np.array([band(mags, int(round(h * f0 / df))) for h in range(1, 31) if h * f0 < rate / 2 - 100])


def alias_levels(y, f0, rate=RATE):
    """返回低于基频的 dB：8 kHz 以下、20 kHz 以下最强的非谐波分量。"""
    freq, mags = spectrum(y, rate)
    df = freq[1]
    mask = np.ones(len(mags), dtype=bool)
    h = 1
    while h * f0 < rate / 2:
        i = int(round(h * f0 / df))
        mask[max(i - 20, 0):i + 21] = False
        h += 1
    mask[:int(10.0 / df)] = False
    fund = band(mags, int(round(f0 / df)))
    return float(db(mags[mask & (freq <= 8000)].max() / fund)), float(db(mags[mask & (freq <= 20000)].max() / fund))


def section(title):
    say()
    say(f'## {title}')


def test_probe(plugin):
    section('mint_probe check')
    result = plugin.run('check', plugin.vst3, '--expect-latency', '0')
    say(result.stdout.strip())
    check('mint_probe check 默认 0 延迟', result.returncode == 0 and 'PASS' in result.stdout)
    say(plugin.run('params', plugin.vst3).stdout.strip())


def test_curve(bench):
    section('传输曲线')
    if bench is None:
        say('跳过：没有 mint_tube_model_bench.dll')
        return
    P = ctypes.POINTER(ctypes.c_double)
    bench.curve.argtypes = [P, P, P, P, ctypes.c_int, ctypes.c_double]
    bench.even_at.argtypes = [ctypes.c_double]
    bench.even_at.restype = ctypes.c_double
    w = np.linspace(-40.0, 40.0, 800001)
    y, first, second = (np.zeros_like(w) for _ in range(3))
    even = bench.even_at(36.0)
    bench.curve(*(a.ctypes.data_as(P) for a in (w, y, first, second)), len(w), even)
    dw = w[1] - w[0]
    check('最大驱动下曲线单调', np.all(np.diff(y) > 0.0), f'最小斜率 {np.min(np.diff(y) / dw):.4f}，二次项 {even:.3f}')
    numeric = np.diff(first + even * second) / dw
    middle = 0.5 * (y[1:] + y[:-1])
    check('原函数与曲线一致', np.max(np.abs(numeric - middle)) < 1e-5, f'最大误差 {np.max(np.abs(numeric - middle)):.2e}')
    check('曲线过原点，斜率为 1', abs(y[len(y) // 2]) < 1e-12 and abs((y[len(y) // 2 + 1] - y[len(y) // 2 - 1]) / (2 * dw) - 1.0) < 1e-3)
    at = lambda value: float(y[int(np.argmin(np.abs(w - value)))])
    say(f'最大驱动下 w=+3 处 {at(3.0):.3f}，w=-3 处 {at(-3.0):.3f}')


def test_harmonics(plugin):
    section('THD 与谐波分布（1 kHz，实时档）')
    for level in (-30.0, -18.0, -6.0):
        say(f'输入峰值 {level:.0f} dBFS')
        say('驱动 dB   THD %    H2     H3     H4     H5  偶/奇 dB')
        thds = []
        ratios = []
        for drive in DRIVES:
            y = plugin.render(tone(1000.0, level), drive=drive)
            h = harmonics(y, 1000.0)
            even = np.sqrt(np.sum(h[1::2] ** 2))
            odd = np.sqrt(np.sum(h[2::2] ** 2))
            thd = 100.0 * np.sqrt(even ** 2 + odd ** 2) / h[0]
            thds.append(thd)
            ratios.append(float(db(even / max(odd, 1e-12))))
            rel = [max(float(db(h[i] / h[0])), -140.0) for i in (1, 2, 3, 4)]
            say(f'{drive:6d} {thd:7.3f} ' + ' '.join(f'{r:6.1f}' for r in rel) + f' {ratios[-1]:8.1f}')
        check(f'{level:.0f} dBFS：THD 随驱动单调上升', all(b > a for a, b in zip(thds, thds[1:])))
        check(f'{level:.0f} dBFS：偶次多于奇次', min(ratios) > 0.0, f'最小 {min(ratios):.1f} dB')

    say()
    say('低频（变压器）：100 Hz 和 40 Hz，0 dB 驱动，看输入变压器')
    for hz in (100.0, 40.0, 20.0):
        for level in (-18.0, -6.0, 0.0):
            y = plugin.render(tone(hz, level, 2.0), drive=0.0)
            h = harmonics(y, hz)
            say(f'{hz:5.0f} Hz {level:5.0f} dBFS  H2 {float(db(h[1] / h[0])):7.1f}  H3 {float(db(h[2] / h[0])):7.1f}')
    say('低频（变压器）：最大驱动下输出变压器被推满')
    for hz in (1000.0, 100.0, 40.0):
        y = plugin.render(tone(hz, -18.0, 2.0), drive=36.0)
        h = harmonics(y, hz)
        say(f'{hz:5.0f} Hz -18 dBFS 驱动 36  H2 {float(db(h[1] / h[0])):7.1f}  H3 {float(db(h[2] / h[0])):7.1f}')
    say('低频（变压器）：电子管还没削波时，H3 只来自变压器（驱动 18，-24 dBFS）')
    third = {}
    for hz in (1000.0, 100.0, 40.0, 20.0):
        h = harmonics(plugin.render(tone(hz, -24.0, 2.0), drive=18.0), hz)
        third[hz] = float(db(h[2] / h[0]))
        say(f'{hz:5.0f} Hz  H2 {float(db(h[1] / h[0])):7.1f}  H3 {third[hz]:7.1f}')
    check('变压器的 H3 随频率降低而上升', third[20.0] > third[40.0] > third[100.0])


def test_alias(plugin, bench):
    section('混叠（48 kHz，最强非谐波分量，相对基频 dB）')
    cases = ((1009.0, -6.0, 36.0), (3001.0, -12.0, 30.0), (5003.0, -12.0, 24.0), (5003.0, -6.0, 36.0),
             (9973.0, -18.0, 24.0), (9973.0, -6.0, 36.0), (14983.0, -18.0, 24.0))
    say('频率 Hz  输入  驱动 | 未抗混叠 <8k <20k | 实时档 <8k <20k | 高质量 <8k <20k')
    improvement = []
    hq_better = []
    for hz, level, drive in cases:
        x = tone(hz, level, 2.0)
        rt = alias_levels(plugin.render(x, drive=drive), hz)
        hq = alias_levels(plugin.render(x, drive=drive, hq=True), hz)
        if bench is not None:
            P = ctypes.POINTER(ctypes.c_double)
            bench.run_naive.argtypes = [P, P, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
            source = np.asarray(x, dtype=np.float64)
            naive_out = np.zeros_like(source)
            bench.run_naive(source.ctypes.data_as(P), naive_out.ctypes.data_as(P), len(source), float(RATE), drive, 0.12)
            naive = alias_levels(naive_out, hz)
            improvement.append(naive[0] - rt[0])
        else:
            naive = (float('nan'), float('nan'))
        hq_better.append(rt[1] - hq[1])
        say(f'{hz:7.0f} {level:5.0f} {drive:5.0f} | {naive[0]:7.1f} {naive[1]:6.1f} | {rt[0]:7.1f} {rt[1]:6.1f} | {hq[0]:7.1f} {hq[1]:6.1f}')
    if improvement:
        check('实时档 8 kHz 以下的混叠低于未抗混叠的基线', min(improvement) > 6.0, f'改善 {min(improvement):.1f} 到 {max(improvement):.1f} dB')
    check('高质量档的混叠低于实时档', min(hq_better) > 6.0, f'20 kHz 以下改善 {min(hq_better):.1f} 到 {max(hq_better):.1f} dB')


def test_compensation(plugin):
    section('增益补偿误差（输出 - 输入，dB，只开驱动，输出旋钮 0 dB）')
    signals = [('人声式', voice_like(level), level) for level in (-30.0, -24.0, -18.0, -12.0, -6.0)]
    signals += [('正弦 1 kHz', tone(1000.0, level + 3.01, 2.0), level) for level in (-30.0, -18.0, -6.0)]
    header = '信号           输入 RMS | ' + ' '.join(f'{d:6d}' for d in DRIVES)
    say(header)
    at_reference = []
    for name, x, level in signals:
        errors = []
        weighted = []
        for drive in DRIVES:
            y = plugin.render(x, drive=drive)
            errors.append(float(db(rms(y) / rms(x))))
            weighted.append(float(db(rms(k_weight(y)) / rms(k_weight(x)))))
        say(f'{name:10s} {level:6.0f} RMS  | ' + ' '.join(f'{e:6.1f}' for e in errors))
        say(f'{"":10s} {"":6s} K 计权 | ' + ' '.join(f'{e:6.1f}' for e in weighted))
        if name == '人声式' and level == -18.0:
            at_reference = errors, weighted
    check('人声式 -18 dBFS：各档 RMS 误差不超过 1.5 dB', max(abs(e) for e in at_reference[0]) <= 1.5,
          f'最大 {max(abs(e) for e in at_reference[0]):.2f} dB')
    check('人声式 -18 dBFS：各档 K 计权误差不超过 2.5 dB', max(abs(e) for e in at_reference[1]) <= 2.5,
          f'最大 {max(abs(e) for e in at_reference[1]):.2f} dB')


def probe_envelope(y, rate, probe_hz, start):
    """在 start 之后解调探测音的幅度，窗长 1 ms（2 kHz 的两个周期）。"""
    size = int(round(rate * 2.0 / probe_hz))
    # 先减去一个周期的滑动平均：大信号结束后的低频尾巴会漏进解调结果。
    y = y - np.convolve(y, np.ones(size // 2) / (size // 2), mode='same')
    t = np.arange(len(y)) / rate
    carrier = np.exp(-2j * np.pi * probe_hz * t)
    demod = np.convolve(y * carrier, np.ones(size) / size, mode='same')
    # 末尾半个窗口是卷积的边缘效应，不能用。
    return t[start:-size], 2.0 * np.abs(demod[start:-size])


def test_recovery(plugin):
    section('偏置恢复时间常数（大信号后探测音增益的指数拟合）')
    say('设定 ms  实测 ms  误差 %  结束时增益下陷 dB  档位')
    for setting, hq in ((10.0, False), (50.0, False), (120.0, False), (500.0, False), (2000.0, False), (120.0, True)):
        burst_seconds = 0.3
        rest_seconds = max(1.0, 12.0 * setting * 0.001)
        n = int(RATE * (burst_seconds + rest_seconds))
        t = np.arange(n) / RATE
        x = 0.01 * np.sin(2.0 * np.pi * 2000.0 * t)
        burst_end = int(burst_seconds * RATE)
        x[:burst_end] += 0.5 * np.sin(2.0 * np.pi * 5003.0 * t[:burst_end])
        y = plugin.render(x, drive=18.0, recovery=setting, hq=hq)

        times, env = probe_envelope(y, RATE, 2000.0, burst_end + int(0.001 * RATE))
        steady = float(np.median(env[-int(0.1 * RATE):]))
        deviation = np.abs(env - steady) / steady
        # 线性区内增益偏移正比于偏置，取 5% 到 1% 这段，避开拐点和本底。
        first_low = int(np.nonzero(deviation >= 0.05)[0][-1]) + 1
        last = first_low + int(np.nonzero(deviation[first_low:] < 0.01)[0][0])
        mask = np.zeros(len(deviation), dtype=bool)
        mask[first_low:last] = True
        slope, _ = np.polyfit(times[mask], np.log(deviation[mask]), 1)
        measured = -1000.0 / slope
        error = 100.0 * (measured - setting) / setting
        say(f'{setting:7.0f} {measured:8.1f} {error:7.1f} {float(db(np.min(env) / steady)):13.2f}       {"高质量" if hq else "实时"}')
        check(f'恢复 {setting:.0f} ms{"（高质量）" if hq else ""} 误差不超过 15%', abs(error) <= 15.0)


def test_blocking(plugin):
    section('大信号后的堵塞：-40 dBFS 探测音回到稳态 1 dB 内需要多久（默认恢复 120 ms）')
    say('驱动  突发 dBFS  峰值输出 dBFS  回到 1 dB 内 ms')
    for drive, level in ((12.0, -6.0), (24.0, -6.0), (36.0, -18.0), (36.0, -6.0), (36.0, 0.0)):
        n = int(RATE * 1.5)
        t = np.arange(n) / RATE
        x = 0.01 * np.sin(2.0 * np.pi * 2000.0 * t)
        burst_end = int(0.3 * RATE)
        x[:burst_end] += 10.0 ** (level / 20.0) * np.sin(2.0 * np.pi * 5003.0 * t[:burst_end])
        y = plugin.render(x, drive=drive)
        times, env = probe_envelope(y, RATE, 2000.0, burst_end + int(0.001 * RATE))
        steady = float(np.median(env[-int(0.1 * RATE):]))
        off = np.nonzero(np.abs(db(env / steady)) > 1.0)[0]
        recovered = (times[off[-1]] - burst_end / RATE) * 1000.0 if len(off) else 0.0
        say(f'{drive:4.0f} {level:9.0f} {float(db(np.max(np.abs(y)))):13.1f} {recovered:14.1f}')


def test_silence(plugin):
    section('静音、噪声与直流')
    silent = np.zeros(RATE * 3)
    for hq in (False, True):
        label = '高质量' if hq else '实时'
        y = plugin.render(silent, drive=36.0, output=12.0, hq=hq)
        check(f'{label}：静音进，输出全 0', np.max(np.abs(y)) == 0.0, f'峰值 {np.max(np.abs(y)):.2e}')

        x = np.zeros(RATE * 5)
        x[:RATE] = tone(220.0, -3.0, 1.0)
        y = plugin.render(x, drive=36.0, hq=hq)
        tail = y[RATE:]
        check(f'{label}：大信号结束 1 秒后低于 -120 dBFS', rms(tail[RATE:]) < 1e-6, f'RMS {float(db(rms(tail[RATE:]))):.1f} dBFS')
        check(f'{label}：大信号结束 3 秒后无残余直流', abs(np.mean(tail[3 * RATE:])) < 1e-6, f'均值 {abs(np.mean(tail[3 * RATE:])):.2e}')
        say(f'{label}：大信号结束后 50 ms 内的最大残留 {float(db(np.max(np.abs(tail[:int(0.05 * RATE)])))):.1f} dBFS（信号期间峰值 {float(db(np.max(np.abs(y[:RATE])))):.1f} dBFS）')

    say('大信号结束后的低频冲击（220 Hz，-3 dBFS，结束后 50 ms 内的最大残留）')
    for drive in (12.0, 24.0, 36.0):
        x = np.zeros(RATE * 2)
        x[:RATE] = tone(220.0, -3.0, 1.0)
        y = plugin.render(x, drive=drive)
        say(f'驱动 {drive:.0f}：{float(db(np.max(np.abs(y[RATE:RATE + int(0.05 * RATE)])))):.1f} dBFS，信号期间峰值 {float(db(np.max(np.abs(y[:RATE])))):.1f} dBFS')

    say('人声式信号（-18 dBFS RMS）30 Hz 以下的成分，由非对称削波整流产生')
    low = butter(4, 30.0, 'low', fs=RATE, output='sos')
    voice = voice_like(-18.0)
    for drive in (12.0, 24.0, 36.0):
        y = plugin.render(voice, drive=drive)
        say(f'驱动 {drive:.0f}：相对总电平 {float(db(rms(sosfilt(low, y)) / rms(y))):.1f} dB，峰值 {float(db(np.max(np.abs(sosfilt(low, y))))):.1f} dBFS')

    say('稳态直流（1 kHz 正弦，输出均值相对输出峰值）')
    for drive, level in ((12.0, -6.0), (24.0, -6.0), (36.0, -6.0), (36.0, 0.0)):
        y = plugin.render(tone(1000.0, level, 2.0), drive=drive)
        settled = y[RATE:]
        ratio = abs(np.mean(settled)) / np.max(np.abs(settled))
        check(f'驱动 {drive:.0f} 输入 {level:.0f}：稳态直流低于 -80 dB', ratio < 1e-4, f'{float(db(ratio)):.1f} dB')

    x = np.zeros(RATE * 2)
    x[:] = 0.1
    y = plugin.render(x, drive=12.0)
    check('输入 0.1 的直流被隔断', abs(y[-1]) < 1e-3, f'末尾 {abs(y[-1]):.2e}')


def lag_between(a, b):
    corr = np.fft.irfft(np.fft.rfft(a, 2 * len(a)) * np.conj(np.fft.rfft(b, 2 * len(a))))
    lag = int(np.argmax(np.abs(corr)))
    return lag - 2 * len(a) if lag > len(a) else lag


def sweep(rate, seconds=8, f0=1.0, level=0.01):
    """指数扫频。-40 dBFS 在最大驱动下也不到拐点，测到的是线性响应。"""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    k = math.log(0.48 * rate / f0) / seconds
    return level * np.sin(2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k)


def transfer(plugin, rate, drive, hq):
    x = sweep(rate)
    y = plugin.render(x, rate=rate, drive=drive, hq=hq)
    return np.fft.rfftfreq(len(x), 1.0 / rate), np.fft.rfft(y) / np.fft.rfft(x)


def test_latency(plugin):
    section('延迟与对齐')
    for rate in (44100, 48000, 96000, 192000):
        # 300 Hz 到 15 kHz 的噪声。变压器的低频相移和过采样滤波器在 Nyquist 附近的相位都不算延迟，
        # 不能让它们把互相关峰值拖偏；这两部分由下面的群延迟表单独报告。
        spectrum_x = np.fft.rfft(np.random.default_rng(5).standard_normal(rate))
        freq_x = np.fft.rfftfreq(rate, 1.0 / rate)
        spectrum_x[(freq_x < 300.0) | (freq_x > 15000.0)] = 0.0
        x = np.fft.irfft(spectrum_x)
        x *= 0.02 / np.max(np.abs(x))
        realtime = plugin.render(x, rate=rate, drive=0.0)
        check(f'{rate} Hz 实时：上报 {plugin.reported} 采样，输出与输入对齐', plugin.reported == 0 and lag_between(realtime, x) == 0, f'残余偏移 {lag_between(realtime, x)}')
        quality = plugin.render(x, rate=rate, drive=0.0, hq=True)
        check(f'{rate} Hz 高质量：上报 {plugin.reported} 采样，输出与实时档对齐', plugin.reported > 0 and lag_between(quality, realtime) == 0, f'残余偏移 {lag_between(quality, realtime)}')

    say('群延迟（采样）。低频部分来自变压器高通，属于模拟电路本身的相移')
    points = (200.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0)
    say('采样率 档位   | ' + ' '.join(f'{p:>7.0f}' for p in points))
    delays = {}
    for rate, hq in ((44100, False), (44100, True), (48000, False), (48000, True), (96000, False), (96000, True),
                     (192000, False), (192000, True)):
        freq, response = transfer(plugin, rate, 0.0, hq)
        phase = np.unwrap(np.angle(response))
        df = freq[1] - freq[0]
        width = int(100.0 / df)
        row = []
        for hz in points:
            i = int(np.argmin(np.abs(freq - hz)))
            row.append(-(phase[i + width] - phase[i - width]) / (2.0 * np.pi * (freq[i + width] - freq[i - width])) * rate)
        delays[(rate, hq)] = row
        say(f'{rate:6d} {"高质量" if hq else "实时"} | ' + ' '.join(f'{d:7.2f}' for d in row))
    for rate in (44100, 48000, 96000, 192000):
        gap = max(abs(a - b) for a, b in zip(delays[(rate, False)][:4], delays[(rate, True)][:4]))
        check(f'{rate} Hz：扣除上报延迟后，10 kHz 以下两档群延迟差小于 0.5 采样', gap < 0.5, f'最大差 {gap:.2f}')
    check('实时档 1 kHz 到 10 kHz 的群延迟不超过 1 采样（48 kHz）', max(abs(d) for d in delays[(48000, False)][1:4]) <= 1.0)


def test_response(plugin):
    section('小信号频响（相对 1 kHz，dB）')
    points = (5, 7, 10, 15, 20, 40, 100, 5000, 10000, 15000, 20000, 30000, 40000)
    say('采样率  驱动 档位   | ' + ' '.join(f'{p:>7d}' for p in points))
    flat = []
    corners = []
    for rate, drive, hq in ((44100, 0, False), (48000, 0, False), (48000, 12, False), (48000, 36, False), (96000, 0, False),
                            (96000, 36, False), (192000, 0, False), (48000, 0, True), (48000, 36, True)):
        freq, response = transfer(plugin, rate, drive, hq)
        power = np.abs(response) ** 2
        ref = np.mean(power[np.abs(freq - 1000.0) < 20.0])

        def level_at(hz):
            return float(10.0 * np.log10(np.mean(power[np.abs(freq - hz) <= max(0.25, 0.02 * hz)]) / ref))

        cells = []
        for p in points:
            if p < 0.48 * rate:
                value = level_at(p)
                cells.append(f'{value:7.2f}')
                if p in (40, 100, 5000, 10000, 15000) and drive <= 12:
                    flat.append(abs(value))
            else:
                cells.append('      -')
        say(f'{rate:6d} {drive:5d} {"高质量" if hq else "实时"} | ' + ' '.join(cells))
        if rate == 48000 and not hq:
            grid = np.geomspace(2.0, 100.0, 400)
            levels = np.array([level_at(g) for g in grid])
            corners.append((drive, float(grid[int(np.argmax(levels > -3.0))])))
    say('低频 -3 dB 点：' + '，'.join(f'驱动 {d} 为 {c:.1f} Hz' for d, c in corners))
    check('驱动 12 dB 以内，40 Hz 到 15 kHz 起伏不超过 1 dB', max(flat) <= 1.0, f'最大 {max(flat):.2f} dB')


def test_consistency(plugin):
    section('档位、采样率和声道的一致性')
    for drive in (12.0, 24.0, 36.0):
        voice = voice_like(-18.0)
        realtime = rms(plugin.render(voice, drive=drive))
        quality = rms(plugin.render(voice, drive=drive, hq=True))
        gap = float(db(quality / realtime))
        check(f'驱动 {drive:.0f}：高质量档与实时档响度差不超过 0.5 dB', abs(gap) <= 0.5, f'{gap:+.2f} dB')
        for rate in (44100, 96000, 192000):
            other = rms(plugin.render(voice_like(-18.0, rate=rate), rate=rate, drive=drive))
            gap = float(db(other / realtime))
            check(f'驱动 {drive:.0f}：{rate} Hz 与 48 kHz 响度差不超过 1 dB', abs(gap) <= 1.0, f'{gap:+.2f} dB')

    left = tone(1000.0, -6.0)
    plugin.render(np.column_stack((left, np.zeros_like(left))), drive=36.0)
    stereo = plugin.last_stereo
    check('左声道满驱动时右声道仍为全 0', np.max(np.abs(stereo[:, 1])) == 0.0 and np.max(np.abs(stereo[:, 0])) > 0.01,
          f'右声道峰值 {np.max(np.abs(stereo[:, 1])):.2e}')


def test_trim(plugin):
    section('输出旋钮')
    x = tone(1000.0, -30.0)
    base = rms(plugin.render(x, drive=12.0, output=0.0)[RATE // 2:])
    for trim in (-12.0, -6.0, 6.0, 12.0):
        measured = float(db(rms(plugin.render(x, drive=12.0, output=trim)[RATE // 2:]) / base))
        check(f'输出 {trim:+.0f} dB', abs(measured - trim) < 0.05, f'实测 {measured:+.2f} dB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='D:/mint-build/tube')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    plugin = Plugin(args.build)
    bench_path = os.path.join(args.build, 'plugins/tube/Release/mint_tube_model_bench.dll')
    bench = ctypes.CDLL(bench_path) if os.path.exists(bench_path) else None

    test_probe(plugin)
    test_curve(bench)
    test_harmonics(plugin)
    test_alias(plugin, bench)
    test_compensation(plugin)
    test_recovery(plugin)
    test_blocking(plugin)
    test_silence(plugin)
    test_latency(plugin)
    test_response(plugin)
    test_consistency(plugin)
    test_trim(plugin)

    say()
    say('ALL PASS' if not failures else f'FAILED {len(failures)}: ' + '; '.join(failures))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
