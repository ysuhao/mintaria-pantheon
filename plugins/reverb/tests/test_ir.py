"""通过 mint_probe 渲染真实 VST3 的脉冲响应并测量。

  python test_ir.py rt        各算法各时间的 1 kHz RT60 误差
  python test_ir.py bands     分频段衰减：低频、中频、分频点、高频阻尼
  python test_ir.py quality   回声密度、频谱起伏、自相关、声道相关、能量
  python test_ir.py chars     三种特性的带宽、RT60 和能量
  python test_ir.py tail      getTailLengthSeconds 的公式与实测尾长
  python test_ir.py rates     各采样率的 RT60
  python test_ir.py stereo    单边输入的交叉能量
  python test_ir.py block1    单采样缓冲区渲染
  python test_ir.py latency   默认设定下干声和湿声的起始位置

环境变量 ELYSIUM_TEST_DIR 指定中间文件目录。
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common
import ir_metrics as m

WORK = common.workdir()
PARALLEL = 4
IMPULSE_LOCK = threading.Lock()


def impulse_file(seconds, rate=common.RATE):
    path = os.path.join(WORK, f"imp_{rate}_{seconds:.1f}.wav")
    with IMPULSE_LOCK:
        if not os.path.exists(path):
            m.write_wav(path, rate, m.impulse(rate, seconds, 0.5))
    return path


def render_ir(name, seconds, rate=common.RATE, block=256, **settings):
    out = os.path.join(WORK, f"{name}.wav")
    common.render(common.PLUGIN, impulse_file(seconds, rate), out, common.elysium_sets(**settings), block=block)
    r, ir = m.read_wav(out)
    return r, ir / 0.5


def batch(jobs):
    with ThreadPoolExecutor(PARALLEL) as pool:
        return list(pool.map(lambda j: j[0](*j[1], **j[2]), jobs))


def test_rt():
    times = [0.3, 0.6, 1.0, 2.2, 5.0, 10.0, 20.0]
    jobs = []
    for algorithm in common.ALGORITHMS:
        for t in times:
            jobs.append((render_ir, (f"rt_{algorithm}_{t}", max(2.0, t * 1.3) + 0.5), dict(time=t, algorithm=algorithm)))
    results = batch(jobs)
    print("1 kHz octave T30 (T20 below 0.6 s); damping 50 %, modulation 30 %, modern character")
    print(f"{'algo':<9}" + "".join(f"{t:>8.1f}s err%" for t in times))
    i = 0
    worst = 0.0
    for algorithm in common.ALGORITHMS:
        row = f"{algorithm:<9}"
        for t in times:
            rate, ir = results[i]
            i += 1
            measured = m.band_rt(ir, rate, 1000, -5.0, -25.0 if t < 0.6 else -35.0)
            err = 100.0 * (measured - t) / t
            worst = max(worst, abs(err))
            row += f"{measured:>7.2f} {err:>+6.1f}"
        print(row)
    print(f"worst error {worst:.1f} %")


def test_bands():
    cases = [
        dict(time=2.0),
        dict(time=2.0, damping=0.0),
        dict(time=2.0, damping=100.0),
        dict(time=2.0, damping=0.0, low=2.0, crossover=300.0),
        dict(time=2.0, damping=0.0, low=0.5, crossover=300.0),
        dict(time=2.0, damping=0.0, mid=2.0, crossover=300.0),
        dict(time=2.0, damping=0.0, low=2.0, crossover=1000.0),
    ]
    jobs = []
    for algorithm in common.ALGORITHMS:
        for k, case in enumerate(cases):
            jobs.append((render_ir, (f"bands_{algorithm}_{k}", 8.0), dict(algorithm=algorithm, **case)))
    results = batch(jobs)
    i = 0
    header = "".join(f"{c:>7}" for c in m.OCTAVES)
    for algorithm in common.ALGORITHMS:
        print(f"{algorithm}: T30 by octave{'':<20}{header}")
        for case in cases:
            rate, ir = results[i]
            i += 1
            label = ", ".join(f"{k}={v}" for k, v in case.items() if k != "time")
            print(f"  {label:<36}" + "".join(f"{m.band_rt(ir, rate, c):>7.2f}" for c in m.OCTAVES))


def quality_row(ir, rate, t, name):
    left = ir[:, 0]
    density = m.mixing_time(left, rate)
    early = m.spectral_roughness(left, rate, 0.05, 0.5)
    late_start = max(0.25, 0.5 * t)
    late = m.spectral_roughness(left, rate, late_start, 1.0)
    persist = m.persistence(left, rate, late_start, late_start + 0.5)
    comb, lag = m.comb_peak(left, rate)
    return (f"{name:<22}{1000 * density:>8.0f}{1000 * m.onset(left, rate):>8.1f}{early['std']:>7.1f}{early['peak']:>7.1f}"
            f"{late['std']:>7.1f}{late['peak']:>7.1f}{late['peak_hz']:>7.0f}{persist:>7.2f}{comb:>7.3f}{lag:>7.1f}"
            f"{m.stereo_correlation(ir, rate):>7.2f}{m.energy_db(ir):>7.1f}")


QUALITY_HEADER = (f"{'':<22}{'mix ms':>8}{'on ms':>8}{'eStd':>7}{'ePk':>7}{'lStd':>7}{'lPk':>7}{'lPkHz':>7}{'pers':>7}"
                  f"{'comb':>7}{'lagms':>7}{'IACC':>7}{'EdB':>7}")


def test_quality():
    print("mix = 90 % echo-density time; on = onset; e/l = early/late narrow-band spectrum (std and peak in dB re local median);")
    print("pers = peak persistence; comb = envelope-flattened autocorrelation peak; ideal diffuse tail: std 5.6, peak ~11-13, pers ~0, comb < 0.05")
    print(QUALITY_HEADER)
    times = [1.0, 2.2, 5.0, 20.0]
    jobs = []
    names = []
    for algorithm in common.ALGORITHMS:
        for t in times:
            for mod in (30.0, 0.0):
                if mod == 0.0 and t != 2.2:
                    continue
                name = f"{algorithm} T{t} mod{mod:.0f}"
                names.append((name, t))
                jobs.append((render_ir, (f"q_{algorithm}_{t}_{mod:.0f}", max(4.0, 1.2 * t)), dict(time=t, algorithm=algorithm, mod=mod)))
    results = batch(jobs)
    for (name, t), (rate, ir) in zip(names, results):
        print(quality_row(ir, rate, t, name))


def test_chars():
    jobs = []
    for algorithm in common.ALGORITHMS:
        for character in common.CHARACTERS:
            jobs.append((render_ir, (f"c_{algorithm}_{character}", 6.0), dict(time=2.2, algorithm=algorithm, character=character, damping=0.0)))
    results = batch(jobs)
    i = 0
    print(f"{'':<20}{'T30 1k':>8}{'T20 4k':>8}{'E dB':>8}" + "".join(f"{c:>8}" for c in m.OCTAVES))
    for algorithm in common.ALGORITHMS:
        for character in common.CHARACTERS:
            rate, ir = results[i]
            i += 1
            levels = m.band_levels(ir[:, 0], rate)
            print(f"{algorithm + ' ' + character:<20}{m.band_rt(ir, rate, 1000):>8.2f}{m.band_rt(ir, rate, 4000, -5, -25):>8.2f}{m.energy_db(ir):>8.1f}"
                  + "".join(f"{levels[c]:>8.1f}" for c in m.OCTAVES))


def test_tail():
    print("time from impulse until the wet level stays below -90 dB re its peak, vs the value the plugin reports")
    cases = [dict(time=0.3), dict(time=2.2), dict(time=2.2, predelay=100.0), dict(time=2.2, low=2.5, crossover=400.0), dict(time=8.0, mid=2.0)]
    jobs = []
    for algorithm in common.ALGORITHMS:
        for k, case in enumerate(cases):
            longest = case["time"] * max(case.get("low", 1.0), case.get("mid", 1.0))
            jobs.append((render_ir, (f"tail_{algorithm}_{k}", 1.5 * longest + 2.5), dict(algorithm=algorithm, damping=0.0, **case)))
    results = batch(jobs)
    i = 0
    for algorithm in common.ALGORITHMS:
        for case in cases:
            rate, ir = results[i]
            i += 1
            longest = case["time"] * max(case.get("low", 1.0), case.get("mid", 1.0))
            reported = 0.001 * case.get("predelay", 0.0) + 1.5 * longest + 0.1
            env = np.max(np.abs(ir), axis=1)
            above = np.nonzero(env >= np.max(env) * 10 ** (-90.0 / 20.0))[0]
            measured = above[-1] / rate
            flag = "OK" if measured <= reported else "SHORT"
            print(f"{algorithm:<9}{str(case):<44} measured {measured:6.2f} s  reported {reported:6.2f} s  {flag}")


def test_rates():
    rates = [44100, 48000, 88200, 96000, 192000]
    jobs = []
    for algorithm in common.ALGORITHMS:
        for rate in rates:
            jobs.append((render_ir, (f"rate_{algorithm}_{rate}", 5.0, rate), dict(time=2.2, algorithm=algorithm)))
    results = batch(jobs)
    i = 0
    print(f"{'':<9}" + "".join(f"{r:>9}" for r in rates) + "   (1 kHz T30, target 2.20 s)")
    for algorithm in common.ALGORITHMS:
        row = f"{algorithm:<9}"
        for _ in rates:
            rate, ir = results[i]
            i += 1
            row += f"{m.band_rt(ir, rate, 1000):>9.2f}"
        print(row)


def test_stereo():
    print("impulse into the left input only; energy of each output, dB")
    seconds = 5.0
    path = os.path.join(WORK, "imp_left.wav")
    with IMPULSE_LOCK:
        if not os.path.exists(path):
            x = m.impulse(common.RATE, seconds, 0.5)
            x[:, 1] = 0.0
            m.write_wav(path, common.RATE, x)

    def job(algorithm):
        out = os.path.join(WORK, f"left_{algorithm}.wav")
        common.render(common.PLUGIN, path, out, common.elysium_sets(algorithm=algorithm))
        return m.read_wav(out)

    with ThreadPoolExecutor(PARALLEL) as pool:
        results = list(pool.map(job, common.ALGORITHMS))
    for algorithm, (rate, ir) in zip(common.ALGORITHMS, results):
        energy = 10.0 * np.log10(np.sum(ir * ir, axis=0) / 0.25)
        print(f"{algorithm:<9}L {energy[0]:6.1f}   R {energy[1]:6.1f}   R output peak in the first 5 ms {20 * np.log10(np.max(np.abs(ir[:int(0.005 * rate), 1])) + 1e-12):7.1f}")


def test_block1():
    print("mint_probe render through the VST3 with a 1 sample buffer against 256 samples")
    for algorithm in common.ALGORITHMS:
        rate, one = render_ir(f"b1_{algorithm}", 4.0, block=1, algorithm=algorithm)
        rate, big = render_ir(f"b256_{algorithm}", 4.0, block=256, algorithm=algorithm)
        print(f"{algorithm:<9}finite {bool(np.all(np.isfinite(one)))}  T30 block 1 {m.band_rt(one, rate, 1000):.2f}  block 256 {m.band_rt(big, rate, 1000):.2f}  energy {m.energy_db(one):.2f} / {m.energy_db(big):.2f} dB")


def test_latency():
    print("default mix 25 %: first non-zero output sample for an impulse at sample 0, and the dry gain")
    for algorithm in common.ALGORITHMS:
        out = os.path.join(WORK, f"lat_{algorithm}.wav")
        sets = common.elysium_sets(algorithm=algorithm, mix=25.0, character="24bit")
        common.render(common.PLUGIN, impulse_file(1.0), out, sets)
        rate, y = m.read_wav(out)
        print(f"{algorithm:<9}first non-zero sample {int(np.argmax(np.abs(y[:, 0]) > 0))}  output[0] / input[0] {y[0, 0] / 0.5:.4f}")


if __name__ == "__main__":
    tests = dict(rt=test_rt, bands=test_bands, quality=test_quality, chars=test_chars, tail=test_tail, rates=test_rates,
                 stereo=test_stereo, block1=test_block1, latency=test_latency)
    for key in (sys.argv[1:] or list(tests)):
        print(f"\n===== {key} =====")
        tests[key]()
