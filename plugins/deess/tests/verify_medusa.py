#!/usr/bin/env python3
"""Medusa 核心行为验证。先按 docs/PROJECT.md 构建，再运行：

  python verify_medusa.py

全部通过时退出码为 0，并打印每项的实测数值。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal

from medusa_signals import (PLUGIN, band_attenuation, band_rms_db, change_db, freq_norm, gain_trace_db, read_wav,
                            render, sibilant_noise, speech_like, vowel, write_wav24)

FS = 48000
failures = []


def check(name, ok, detail):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    if not ok:
        failures.append(name)


def run(tmp, x, params=None, tag='out', fs=FS, block=256):
    in_path = Path(tmp) / f'{tag}_in.wav'
    out_path = Path(tmp) / f'{tag}_out.wav'
    write_wav24(in_path, x, fs)
    render(in_path, out_path, PLUGIN, params, block)
    _, i = read_wav(in_path)
    _, o = read_wav(out_path)
    return i, o


def vowel_stats(i, o, vowels, fs, settle_ms=60, guard_ms=30):
    # 尾部留 guard_ms：下一个齿音可能和元音尾巴交叠，那段不算元音稳态。
    settle = int(settle_ms * 0.001 * fs)
    guard = int(guard_ms * 0.001 * fs)
    steady = [change_db(i[:, 0], o[:, 0], a + settle, b - guard) for a, b in vowels[1:]]
    transition = [change_db(i[:, 0], o[:, 0], a, a + settle) for a, b in vowels[1:]]
    return max(steady), max(transition)


def test_default(tmp):
    print('1. 默认参数：齿音衰减与元音改变量')
    for name in ('a_male', 'a_female', 'i_bright', 'belt'):
        for rel in (-3.0, -9.0):
            x, vowels, sibs = speech_like(vowel_name=name, sib_rel_db=rel)
            i, o = run(tmp, x, tag='default')
            attn = [band_attenuation(i[:, 0], o[:, 0], s, FS) for s in sibs[1:]]
            steady, trans = vowel_stats(i, o, vowels, FS)
            ok_attn = np.median(attn) >= 3.0
            check(f'{name} 齿音相对 {rel:+.0f} dB',
                  ok_attn and steady < -40.0,
                  f'齿音衰减 {np.median(attn):.1f} dB，元音稳态改变 {steady:.1f} dB，元音起始 60 ms 改变 {trans:.1f} dB')


def test_connected(tmp):
    print('1b. 元音与齿音交叠 10 ms（连续语流）')
    for name in ('a_male', 'a_female'):
        x, vowels, sibs = speech_like(vowel_name=name, sib_rel_db=-3.0, gap_ms=-10.0)
        i, o = run(tmp, x, tag='connected')
        attn = float(np.median([band_attenuation(i[:, 0], o[:, 0], s, FS) for s in sibs[1:]]))
        steady, trans = vowel_stats(i, o, vowels, FS)
        check(name, attn >= 3.0 and steady < -40.0, f'齿音衰减 {attn:.1f} dB，元音稳态改变 {steady:.1f} dB，元音起始 60 ms 改变 {trans:.1f} dB')


def test_vowel_only(tmp):
    print('2. 纯元音（含明亮元音）：不应产生任何衰减')
    onset = np.minimum(1.0, np.arange(FS * 2) / (0.010 * FS))  # 10 ms 起音，接近真实元音
    for name in ('a_male', 'a_female', 'i_bright', 'belt'):
        for level in (-12.0, 0.0, 12.0):
            x = vowel(FS * 2, FS, name, -18.0 + level) * onset
            i, o = run(tmp, x, tag='vowel')
            diff = np.max(np.abs(o - i))
            change = change_db(i[:, 0], o[:, 0], 0, len(i))
            check(f'{name} {level:+.0f} dB', change < -60.0, f'最大差 {diff:.2e}，改变量 {change:.1f} dB')
    for name in ('a_male', 'i_bright', 'belt'):
        x = vowel(FS * 2, FS, name) * onset
        i, o = run(tmp, x, {'Strength': 1.0}, tag='vowel_max')
        check(f'{name} 强度 100%', change_db(i[:, 0], o[:, 0], 0, len(i)) < -60.0,
              f'改变量 {change_db(i[:, 0], o[:, 0], 0, len(i)):.1f} dB')


def test_level_tracking(tmp):
    print('3. 输入电平变化 ±12 dB：自适应应保持同样的衰减')
    results = {}
    for level in (-12.0, 0.0, 12.0):
        x, vowels, sibs = speech_like(level_db=level)
        i, o = run(tmp, x, tag='level')
        results[level] = float(np.median([band_attenuation(i[:, 0], o[:, 0], s, FS) for s in sibs[1:]]))
    spread = max(results.values()) - min(results.values())
    detail = '，'.join(f'{k:+.0f} dB → {v:.2f} dB' for k, v in results.items())
    check('静态 ±12 dB', spread <= 0.5, f'{detail}，差距 {spread:.2f} dB')

    # 同一条信号中途变化 ±12 dB，看收敛。
    for step in (-12.0, 12.0):
        x, vowels, sibs = speech_like(cycles=20)
        half = len(x) // 2
        y = x.copy()
        y[half:] *= 10 ** (step / 20)
        i, o = run(tmp, y, tag='step')
        before = np.median([band_attenuation(i[:, 0], o[:, 0], s, FS) for s in sibs if s[1] < half][1:])
        after_all = [(s[0] - half, band_attenuation(i[:, 0], o[:, 0], s, FS)) for s in sibs if s[0] > half]
        settled = np.median([a for t, a in after_all if t > 2.0 * FS]) if any(t > 2.0 * FS for t, a in after_all) else float('nan')
        first = after_all[0][1]
        check(f'中途跳变 {step:+.0f} dB', abs(settled - before) <= 1.5,
              f'跳变前 {before:.2f} dB，跳变后首个齿音 {first:.2f} dB，2 秒后 {settled:.2f} dB')


def test_strength(tmp):
    print('4. 强度：0% 完全直通，往上单调')
    x, vowels, sibs = speech_like(sib_rel_db=-3.0)
    i, o = run(tmp, x, {'Strength': 0.0}, tag='s0')
    check('强度 0%', np.max(np.abs(o - i)) == 0.0, f'最大差 {np.max(np.abs(o - i)):.2e}')
    values = []
    for s in (0.25, 0.5, 0.75, 1.0):
        i, o = run(tmp, x, {'Strength': s}, tag='s')
        values.append(float(np.median([band_attenuation(i[:, 0], o[:, 0], g, FS) for g in sibs[1:]])))
    check('单调上升', all(b > a for a, b in zip(values, values[1:])), '，'.join(f'{v:.1f}' for v in values) + ' dB（25/50/75/100%）')
    check('满强度上限', values[-1] <= 24.0, f'{values[-1]:.1f} dB')


def test_frequency(tmp):
    print('5. 频率旋钮：对准时衰减更深')
    for band, centre in (((3000.0, 5000.0), 4000.0), ((8000.0, 12000.0), 9500.0)):
        x, vowels, sibs = speech_like(sib_band=band, sib_rel_db=-3.0)
        aligned = {'Frequency': freq_norm(centre)}
        far_hz = 12000.0 if centre < 6000 else 3000.0
        i, o = run(tmp, x, aligned, tag='fa')
        a = float(np.median([band_attenuation(i[:, 0], o[:, 0], g, FS, *band) for g in sibs[1:]]))
        i, o = run(tmp, x, {'Frequency': freq_norm(far_hz)}, tag='fb')
        b = float(np.median([band_attenuation(i[:, 0], o[:, 0], g, FS, *band) for g in sibs[1:]]))
        check(f'{band[0] / 1000:.0f}–{band[1] / 1000:.0f} kHz 齿音', a > b + 1.0,
              f'旋钮 {centre / 1000:.1f} kHz 衰减 {a:.1f} dB，旋钮 {far_hz / 1000:.0f} kHz 衰减 {b:.1f} dB')


def test_air_and_body(tmp):
    print('6. 不发闷：齿音段内，检测频带之外的改变量')
    # 用宽带噪声当齿音，看被削掉的频段形状。
    x, vowels, sibs = speech_like(sib_band=(1500.0, 20000.0), sib_rel_db=-1.0)
    i, o = run(tmp, x, tag='wide')
    for lo, hi, limit in ((1500.0, 2500.0, 1.5), (4000.0, 10000.0, None), (14000.0, 20000.0, 2.5)):
        value = float(np.median([band_attenuation(i[:, 0], o[:, 0], g, FS, lo, hi) for g in sibs[1:]]))
        if limit is None:
            print(f'    4–10 kHz 衰减 {value:.1f} dB')
        else:
            check(f'{lo / 1000:.1f}–{hi / 1000:.1f} kHz 衰减', value <= limit, f'{value:.2f} dB（上限 {limit} dB）')


def test_artifacts(tmp):
    print('7. 无抽吸、无咔嗒、无颗粒')
    # 抽吸：持续元音上叠齿音，元音频段（<2.5 kHz）短时电平不应被带着动。
    v = vowel(FS * 6, FS, 'a_male')
    s = sibilant_noise(FS * 6, FS, rms_db=-21.0)
    env = np.zeros(FS * 6)
    bursts = []
    for k in range(10):
        a = int((0.3 + 0.55 * k) * FS)
        n = int(0.15 * FS)
        env[a:a + n] = signal.windows.tukey(n, 0.15)
        bursts.append((a, a + n))
    x = v + s * env
    i, o = run(tmp, x, tag='pump')
    sos = signal.butter(6, 2500, 'low', fs=FS, output='sos')
    li, lo = signal.sosfilt(sos, i[:, 0]), signal.sosfilt(sos, o[:, 0])
    win = int(0.01 * FS)
    def env_db(y):
        return 10 * np.log10(np.convolve(y ** 2, np.ones(win) / win, 'same') + 1e-12)
    pump = np.abs(env_db(lo) - env_db(li))[FS // 4:-win]
    check('元音频段电平抖动', np.max(pump) <= 1.0, f'<2.5 kHz 短时电平最大变化 {np.max(pump):.2f} dB')

    # 咔嗒：输入先砍掉 12 kHz 以上，衰减开合如果有台阶，差信号就会在 18 kHz 以上露头。
    x, vowels, sibs = speech_like(sib_rel_db=0.0)
    x = signal.sosfiltfilt(signal.butter(12, 12000, 'low', fs=FS, output='sos'), x)
    i, o = run(tmp, x, {'Strength': 0.8}, tag='click')
    sos = signal.butter(12, 18000, 'high', fs=FS, output='sos')
    peak_in = 20 * np.log10(np.max(np.abs(signal.sosfilt(sos, i[:, 0]))) + 1e-12)
    peak_delta = 20 * np.log10(np.max(np.abs(signal.sosfilt(sos, o[:, 0] - i[:, 0]))) + 1e-12)
    check('差信号 >18 kHz 峰值', peak_delta < -80.0, f'{peak_delta:.1f} dBFS（输入同频段 {peak_in:.1f} dBFS）')

    # 增益变化率：陡边的 6.5 kHz 正弦爆发是最坏情况，同步解调后得到干净的增益曲线。
    n = FS * 2
    t = np.arange(n) / FS
    tone = np.zeros(n)
    tone[int(0.8 * FS):int(1.3 * FS)] = 10 ** (-20 / 20)
    tone *= np.sin(2 * np.pi * 6500 * t)
    x = np.concatenate([vowel(FS // 2, FS, 'a_male'), tone])
    i, o = run(tmp, x, tag='rate')
    sos = signal.butter(4, 1500, 'low', fs=FS, output='sos')
    carrier = np.exp(-2j * np.pi * 6500 * np.arange(len(x)) / FS)
    di = np.abs(signal.sosfilt(sos, i[:, 0] * carrier))
    do = np.abs(signal.sosfilt(sos, o[:, 0] * carrier))
    start = FS // 2 + int(0.8 * FS)
    stop = FS // 2 + int(1.3 * FS)
    rel = 20 * np.log10((do[start + 96:stop - 48] + 1e-12) / (di[start + 96:stop - 48] + 1e-12))
    rate = np.max(np.abs(np.diff(rel))) * FS / 1000.0
    settle_gain = np.median(rel[len(rel) // 2:])
    overshoot = np.min(rel) - settle_gain
    check('增益变化率', rate < 12.0, f'最大 {rate:.2f} dB/ms，稳态 {settle_gain:.1f} dB，过冲 {overshoot:.2f} dB')

    # 颗粒：持续齿音下增益曲线的起伏。
    v = vowel(FS // 2, FS, 'a_male')
    s = sibilant_noise(FS * 2, FS, rms_db=-19.0)
    x = np.concatenate([v, s])
    i, o = run(tmp, x, tag='grain')
    t, trace = gain_trace_db(i[:, 0], o[:, 0], FS)
    seg = trace[(t > 0.8) & (t < 2.3)]
    ripple = float(np.std(seg))
    check('持续齿音增益起伏', ripple <= 1.0, f'平均 {np.mean(seg):.2f} dB，标准差 {ripple:.2f} dB')


def test_monitor(tmp):
    print('8. 监听齿音：输出 + 监听 = 输入')
    x, vowels, sibs = speech_like(sib_rel_db=-3.0)
    i, o = run(tmp, x, tag='mon_a')
    i2, m = run(tmp, x, {'Monitor': 1.0}, tag='mon_b')
    residual = np.max(np.abs(i - o - m))
    check('零点', residual < 1e-5, f'输入 - 输出 - 监听 的最大值 {residual:.2e}')
    vowel_energy = np.sqrt(np.mean(m[vowels[3][0] + int(0.3 * FS):vowels[3][1], 0] ** 2))
    sib_energy = np.sqrt(np.mean(m[sibs[3][0]:sibs[3][1], 0] ** 2))
    check('元音段安静', 20 * np.log10(vowel_energy + 1e-12) < 20 * np.log10(sib_energy + 1e-12) - 40.0,
          f'齿音段 {20 * np.log10(sib_energy + 1e-12):.1f} dB，元音稳态段 {20 * np.log10(vowel_energy + 1e-12):.1f} dB')


def test_lookahead(tmp):
    print('9. Look-ahead：上报延迟、对齐、起音捕捉')
    x, vowels, sibs = speech_like(sib_rel_db=-3.0)
    i0, o0 = run(tmp, x, tag='la0')
    for ms in (2.5, 5.0, 10.0):
        in_path = Path(tmp) / 'la_in.wav'
        out_path = Path(tmp) / 'la_out.wav'
        write_wav24(in_path, x, FS)
        text = render(in_path, out_path, PLUGIN, {'Look-ahead': ms / 10.0})
        _, o = read_wav(out_path)
        _, i = read_wav(in_path)
        expected = int(round(ms * 0.001 * FS))
        reported = int(text.split('reported latency')[-1].strip())
        check(f'{ms} ms 上报', reported == expected, f'上报 {reported} 采样，期望 {expected}')
        # 对齐：元音稳态段应与输入逐采样接近。
        a, b = vowels[3]
        settle = int(0.3 * FS)
        change = change_db(i[:, 0], o[:, 0], a + settle, b)
        corr = signal.correlate(o[a:b, 0], i[a:b, 0], 'full', method='fft')
        lag = int(np.argmax(corr)) - (b - a - 1)
        check(f'{ms} ms 对齐', lag == 0 and change < -40.0, f'互相关峰值位置 {lag}，元音稳态改变 {change:.1f} dB')
        first = [band_attenuation(i[:, 0], o[:, 0], s, FS, skip_ms=0, tail_ms=125) for s in sibs[1:]]
        first0 = [band_attenuation(i0[:, 0], o0[:, 0], s, FS, skip_ms=0, tail_ms=125) for s in sibs[1:]]
        print(f'    齿音前 25 ms 衰减：无 Look-ahead {np.median(first0):.1f} dB，{ms} ms {np.median(first):.1f} dB')

    # 旁通时延迟不变，输出仍与输入对齐且逐采样相同。
    write_wav24(Path(tmp) / 'by_in.wav', x, FS)
    text = render(Path(tmp) / 'by_in.wav', Path(tmp) / 'by_out.wav', PLUGIN, {'Look-ahead': 0.5, 'Bypass': 1.0})
    _, i = read_wav(Path(tmp) / 'by_in.wav')
    _, o = read_wav(Path(tmp) / 'by_out.wav')
    check('旁通时保持延迟', np.max(np.abs(i - o)) < 1e-6, f'{text.split("reported")[-1].strip()}，与输入最大差 {np.max(np.abs(i - o)):.2e}')


def test_invariance(tmp):
    print('10. 缓冲区与采样率')
    x, vowels, sibs = speech_like(sib_rel_db=-3.0, cycles=3)
    i, ref = run(tmp, x, tag='b256')
    for block in (1, 32, 1024):
        i2, o = run(tmp, x, tag='bn', block=block)
        check(f'缓冲区 {block}', np.max(np.abs(o - ref)) < 2e-6, f'与缓冲区 256 的最大差 {np.max(np.abs(o - ref)):.2e}')
    for fs in (44100, 96000, 192000):
        x2, vowels2, sibs2 = speech_like(fs=fs, sib_rel_db=-3.0, cycles=4)
        i, o = run(tmp, x2, tag='fs', fs=fs)
        attn = float(np.median([band_attenuation(i[:, 0], o[:, 0], s, fs) for s in sibs2[1:]]))
        steady, _ = vowel_stats(i, o, vowels2, fs)
        check(f'{fs} Hz', attn >= 3.0 and steady < -40.0, f'齿音衰减 {attn:.1f} dB，元音稳态改变 {steady:.1f} dB')


def test_stereo_silence(tmp):
    print('11. 立体声联动与静音')
    n = FS * 2
    l = sibilant_noise(n, FS, rms_db=-20.0)
    r = l * 0.25
    v = vowel(FS // 2, FS, 'a_male')
    x = np.stack([np.concatenate([v, l]), np.concatenate([v, r])], axis=1)
    i, o = run(tmp, x, tag='stereo')
    t, gl = gain_trace_db(i[:, 0], o[:, 0], FS)
    t, gr = gain_trace_db(i[:, 1], o[:, 1], FS)
    seg = (t > 0.7) & (t < 2.4)
    check('左右增益相同', np.max(np.abs(gl[seg] - gr[seg])) < 0.2, f'最大差 {np.max(np.abs(gl[seg] - gr[seg])):.3f} dB，平均衰减 {np.mean(gl[seg]):.2f} dB')

    silence = np.zeros((FS * 2, 2))
    i, o = run(tmp, silence, tag='silence')
    check('静音', np.all(o == 0.0), f'最大输出 {np.max(np.abs(o)):.2e}')
    hiss = sibilant_noise(FS * 2, FS, rms_db=-80.0)
    i, o = run(tmp, hiss, tag='hiss')
    check('-80 dBFS 底噪不动作', change_db(i[:, 0], o[:, 0], 0, len(i)) < -60.0, f'改变量 {change_db(i[:, 0], o[:, 0], 0, len(i)):.1f} dB')


def main():
    if not PLUGIN.exists():
        print(f'找不到 {PLUGIN}，先构建 mint_deess_VST3')
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        for test in (test_default, test_connected, test_vowel_only, test_level_tracking, test_strength, test_frequency,
                     test_air_and_body, test_artifacts, test_monitor, test_lookahead, test_invariance,
                     test_stereo_silence):
            test(tmp)
    print(f'\n{"全部通过" if not failures else "未通过: " + ", ".join(failures)}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
