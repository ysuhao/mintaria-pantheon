"""与对标混响的黑盒脉冲响应对比。只经 mint_probe 喂信号、读输出。
对标插件的路径从环境变量 ELYSIUM_REFERENCE_VST3 取，见 NOTES.md。

对标插件自带模拟底噪，单样本脉冲的能量会被淹没，所以两边都用指数扫频反卷积取脉冲响应。
对标插件的 Decay 显示的是 0–1 的相对量，用二分法按实测 1 kHz RT60 对齐到目标秒数。

  python test_compare.py [目标秒数]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common
import ir_metrics as m

WORK = common.workdir()
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 2.2
IR_SECONDS = max(5.0, TARGET * 1.6)
SWEEP_SECONDS = 6.0

X, INVERSE = m.sweep(common.RATE, SWEEP_SECONDS)
STIMULUS = os.path.join(WORK, f"cmp_sweep_{IR_SECONDS:.1f}.wav")
if not os.path.exists(STIMULUS):
    padded = np.zeros((len(X) + int(IR_SECONDS * common.RATE), 2), dtype=np.float32)
    padded[:len(X), 0] = X
    padded[:len(X), 1] = X
    m.write_wav(STIMULUS, common.RATE, padded)

# 对标插件的通用设定：全湿、无预延迟、高通最低、无动态处理、亮度和低频偏移居中、阻尼 50 %。
# 它的 Damping 为 0 时 1 kHz 衰减被压在 2 s 左右，所以两边都取 50 %。
REF_BASE = {
    "Dry/Wet": 1.0,
    "Reverb Level": 0.5,
    "Brightness": 0.5,
    "Drive": 0.0,
    "HPF Cutoff Frequency": 0.0,
    "Stereo Width": 0.5,
    "PreDelay": 0.0,
    "Distance": 0.0,
    "Damping": 0.5,
    "Bass Offset": 0.5,
    "Crossover": 0.5,
    "Dynamic On/Off": 0.0,
    "Tremolo Amount": 0.0,
}

# 对标插件的八种算法。归一化值与算法名的对应由 elysium_param_text 查得，记在 NOTES.md。
REF_ALGORITHMS = {
    "A1": 0.0625,
    "A2": 0.1875,
    "A3": 0.3125,
    "A4": 0.4375,
    "A5": 0.5625,
    "A6": 0.6875,
    "A7": 0.8125,
    "A8": 0.9375,
}


def to_ir(path):
    rate, recorded = m.read_wav(path)
    return rate, m.deconvolve(recorded, INVERSE, len(X), IR_SECONDS, rate)


def ref_render(name, sets):
    out = os.path.join(WORK, f"ref_{name}.wav")
    common.render(common.REFERENCE, STIMULUS, out, sets)
    return to_ir(out)


def ref_align(label, algorithm):
    """二分 Decay，让 1 kHz T30 落在目标上。"""
    low, high = 0.0, 1.0
    best = None
    for step in range(10):
        decay = 0.5 * (low + high)
        sets = dict(REF_BASE, Algorithm=algorithm, Decay=decay)
        rate, ir = ref_render(f"{label.replace(' ', '')}_{step}", sets)
        rt = m.band_rt(ir, rate, 1000)
        if best is None or (np.isfinite(rt) and abs(rt - TARGET) < abs(best[3] - TARGET)):
            best = (decay, rate, ir, rt)
        if np.isfinite(rt) and abs(rt - TARGET) / TARGET < 0.02:
            break
        if np.isfinite(rt) and rt > TARGET:
            high = decay
        else:
            low = decay
    return best


def elysium(name, **settings):
    out = os.path.join(WORK, f"ely_{name}.wav")
    common.render(common.PLUGIN, STIMULUS, out, common.elysium_sets(**settings))
    return to_ir(out)


def tone_tail_width(plugin, sets, name):
    """1 kHz 正弦响 1.5 秒后停，取停止后 0.1 秒起 1 秒的尾音，量 -20 dB 谱宽。纯线性不调制时等于窗的主瓣宽度。"""
    rate = common.RATE
    n = int(rate * 3.5)
    t = np.arange(n) / rate
    x = 0.25 * np.sin(2.0 * np.pi * 1000.0 * t)
    x[int(1.5 * rate):] = 0.0
    x = np.stack([x, x], axis=1)
    path = os.path.join(WORK, "cmp_tone.wav")
    if not os.path.exists(path):
        m.write_wav(path, rate, x)
    out = os.path.join(WORK, f"tone_{name}.wav")
    common.render(plugin, path, out, sets)
    _, y = m.read_wav(out)
    seg = y[int(1.6 * rate):int(2.6 * rate), 0]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    f = np.fft.rfftfreq(8 * len(seg), 1.0 / rate)
    keep = (f > 800.0) & (f < 1200.0)
    db = 20.0 * np.log10(np.maximum(spectrum[keep], 1e-12))
    db -= np.max(db)
    above = f[keep][db > -20.0]
    return float(above[-1] - above[0]) if len(above) else float("nan")


def row(name, ir, rate):
    left = ir[:, 0]
    late_start = max(0.25, 0.5 * TARGET)
    late = m.spectral_roughness(left, rate, late_start, 1.0)
    early = m.spectral_roughness(left, rate, 0.05, 0.5)
    comb, lag = m.comb_peak(left, rate)
    return (f"{name:<26}{m.band_rt(ir, rate, 1000):>7.2f}{1000 * m.onset(left, rate):>7.1f}{1000 * m.mixing_time(left, rate):>7.0f}"
            f"{early['std']:>6.1f}{early['peak']:>6.1f}{late['std']:>6.1f}{late['peak']:>6.1f}"
            f"{m.persistence(left, rate, late_start, late_start + 0.5):>6.2f}{comb:>7.3f}{lag:>7.1f}{m.stereo_correlation(ir, rate):>6.2f}{m.energy_db(ir):>7.1f}")


def main():
    print(f"target 1 kHz RT60 {TARGET} s; all wet, no pre-delay; IR from a {SWEEP_SECONDS:.0f} s exponential sweep at -12 dBFS")
    print(f"{'':<26}{'T30':>7}{'on':>7}{'mix':>7}{'eStd':>6}{'ePk':>6}{'lStd':>6}{'lPk':>6}{'pers':>6}{'comb':>7}{'lag':>7}{'IACC':>6}{'EdB':>7}")

    with ThreadPoolExecutor(4) as pool:
        ref_futures = {label: pool.submit(ref_align, label, value) for label, value in REF_ALGORITHMS.items()}
        ely_futures = {}
        for algorithm in common.ALGORITHMS:
            for character in common.CHARACTERS:
                ely_futures[f"{algorithm} {character}"] = pool.submit(
                    elysium, f"{algorithm}_{character}", time=TARGET, algorithm=algorithm, character=character, damping=50.0, mod=30.0)

    results = {}
    for label, future in ref_futures.items():
        decay, rate, ir, rt = future.result()
        results[f"Ref {label}"] = (rate, ir)
        print(row(f"Ref {label} d={decay:.3f}", ir, rate))
    for label, future in ely_futures.items():
        rate, ir = future.result()
        results[f"Ely {label}"] = (rate, ir)
        print(row(f"Ely {label}", ir, rate))

    print("\nband energy re 1 kHz, dB")
    print(f"{'':<26}" + "".join(f"{c:>7}" for c in m.OCTAVES))
    for name, (rate, ir) in results.items():
        levels = m.band_levels(ir[:, 0], rate)
        print(f"{name:<26}" + "".join(f"{levels[c]:>7.1f}" for c in m.OCTAVES))

    print("\nT30 by octave")
    print(f"{'':<26}" + "".join(f"{c:>7}" for c in m.OCTAVES))
    for name, (rate, ir) in results.items():
        print(f"{name:<26}" + "".join(f"{m.band_rt(ir, rate, c):>7.2f}" for c in m.OCTAVES))

    print("\n1 kHz tone tail, -20 dB spectral width in Hz (modulation spread; about 2 Hz is the analysis window alone)")
    for label, future in ref_futures.items():
        decay, _, _, _ = future.result()
        sets = dict(REF_BASE, Algorithm=REF_ALGORITHMS[label], Decay=decay)
        print(f"Ref {label:<23}{tone_tail_width(common.REFERENCE, sets, 'ref_' + label):>7.1f}")
    for algorithm in common.ALGORITHMS:
        for mod in (0.0, 30.0, 60.0, 100.0):
            sets = common.elysium_sets(time=TARGET, algorithm=algorithm, damping=50.0, mod=mod)
            print(f"Ely {algorithm} mod {mod:<9.0f}{tone_tail_width(common.PLUGIN, sets, f'ely_{algorithm}_{mod:.0f}'):>9.1f}")

    print("\nnormalised echo density at time, ms")
    marks = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    print(f"{'':<26}" + "".join(f"{c:>6}" for c in marks))
    for name, (rate, ir) in results.items():
        t, ned = m.echo_density(ir[:, 0], rate)
        print(f"{name:<26}" + "".join(f"{np.interp(c * 0.001, t, ned):>6.2f}" for c in marks))


if __name__ == "__main__":
    main()
