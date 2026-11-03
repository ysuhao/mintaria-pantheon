"""幅频验证：mint_probe render 脉冲响应算出的幅频，与设计目标（模拟原型）比较。

输出：
  1. 每种类型、每个采样率、每个中心频率的最大绝对误差（目标 > -60 dB 的频点，20 Hz 到 0.98 奈奎斯特）。
  2. 高频钟形和普通双线性变换的对照。
  3. 不同采样率之间在 20 kHz 以下的曲线差异。
"""
import sys

import numpy as np

import iris_common as ic

RATES = [44100, 48000, 96000, 192000]
FREQS = [100, 1000, 5000, 10000, 15000, 18000, 20000]

CASES = []
for gain, q in [(12, 1.0), (-12, 1.0), (6, 0.5), (12, 4.0), (-6, 8.0)]:
    CASES.append(dict(kind="bell", gain=gain, q=q))
for kind in ("lowShelf", "highShelf", "tilt"):
    for gain in (6, -12):
        CASES.append(dict(kind=kind, gain=gain, q=1.0))
for slope in (6, 12, 24, 48, 96):
    CASES.append(dict(kind="lowCut", slope=slope, q=1.0))
    CASES.append(dict(kind="highCut", slope=slope, q=1.0))
CASES.append(dict(kind="notch", q=2.0))
CASES.append(dict(kind="notch", q=8.0))


def measure(job):
    rate, f0, case = job
    name = "resp_{}_{}_{}_{}_{}_{}".format(rate, f0, case["kind"], case.get("gain", 0), case.get("q", 1), case.get("slope", 12))
    sets = ic.band_sets(1, freq=float(f0), **{k: v for k, v in case.items()})
    freqs, measured = ic.impulse_response_db(rate, sets, name)
    target = ic.analog_target_db(freqs, case["kind"], f0, gain=case.get("gain", 0.0), q=case.get("q", 1.0), slope=case.get("slope", 12))
    upper = 0.98 * rate / 2
    mask = (freqs >= 20) & (freqs <= upper) & (target > -60)
    err = np.abs(measured - target)
    audible = (freqs >= 20) & (freqs <= 20000) & (target > -60)
    return dict(rate=rate, f0=f0, case=case, maxerr=float(err[mask].max()) if mask.any() else 0.0,
                audible=float(err[audible].max()) if audible.any() else 0.0,
                freqs=freqs, measured=measured)


def label(case):
    if case["kind"] in ("lowCut", "highCut"):
        return f'{case["kind"]} {case["slope"]} dB/oct'
    if case["kind"] == "notch":
        return f'notch Q{case["q"]}'
    return f'{case["kind"]} {case["gain"]:+d} dB Q{case["q"]}'


def probe_agreement():
    # 抽查：mint_probe render 和 mint_eq_bench render 的响应在 -40 dB 以上一致。
    sets = ic.band_sets(1, kind="bell", freq=15000.0, gain=12.0, q=1.0)
    freqs, a = ic.impulse_response_db(48000, sets, "agree_bench")
    _, b = ic.impulse_response_db(48000, sets, "agree_probe", use_probe=True)
    mask = (freqs >= 20) & (freqs <= 23500)
    print(f"mint_probe render 与 mint_eq_bench render 的幅频最大差 {np.abs(a - b)[mask].max():.4f} dB（48 kHz，15 kHz 钟形 +12 dB Q1）")


def main():
    probe_agreement()
    jobs = [(rate, f0, case) for rate in RATES for f0 in FREQS for case in CASES if f0 < rate * 0.49]
    print(f"rendering {len(jobs)} impulse responses ...", flush=True)
    results = ic.parallel(measure, jobs, workers=4)

    print("\n== 最大误差 dB（20 Hz 到 min(20 kHz, 0.98 奈奎斯特)，目标 > -60 dB，表头是中心频率） ==")
    print("类型                     采样率   " + " ".join(f"{f:>7}" for f in FREQS))
    worst = 0.0
    for case in CASES:
        for rate in RATES:
            row = []
            for f0 in FREQS:
                r = [x for x in results if x["rate"] == rate and x["f0"] == f0 and x["case"] is case]
                value = r[0]["audible"] if r else float("nan")
                row.append(f"{value:7.2f}")
                if r:
                    worst = max(worst, value)
            print(f"{label(case):24} {rate:6}   " + " ".join(row))
    print(f"全表最大误差 {worst:.2f} dB")

    print("\n== 高频钟形：Iris 与普通双线性变换（对照）的最大误差 dB，20 Hz 到 0.98 奈奎斯特 ==")
    print("采样率 中心频率 增益/Q        Iris   双线性")
    for rate in (44100, 48000, 96000):
        for f0 in (10000, 15000, 18000, 20000):
            for gain, q in ((12, 1.0), (-12, 1.0), (12, 4.0)):
                case = next(c for c in CASES if c["kind"] == "bell" and c["gain"] == gain and c["q"] == q)
                r = next(x for x in results if x["rate"] == rate and x["f0"] == f0 and x["case"] is case)
                freqs = r["freqs"]
                mask = (freqs >= 20) & (freqs <= 0.98 * rate / 2)
                target = ic.analog_target_db(freqs, "bell", f0, gain=gain, q=q)
                baseline = ic.blt_bell_db(freqs[mask], f0, gain, q, rate)
                print(f"{rate:6} {f0:8} {gain:+3d} dB Q{q:<4} {r['maxerr']:6.2f} {np.abs(baseline - target[mask]).max():8.2f}")

    print("\n== 采样率一致性：与 96 kHz 的曲线相比，20 Hz 到 20 kHz 的最大差 dB（目标 > -60 dB） ==")
    print("类型                     采样率   " + " ".join(f"{f:>7}" for f in FREQS))
    probe = np.geomspace(20, 20000, 800)
    for case in CASES:
        reference = {x["f0"]: x for x in results if x["rate"] == 96000 and x["case"] is case}
        for rate in (44100, 48000, 192000):
            row = []
            for f0 in FREQS:
                r = [x for x in results if x["rate"] == rate and x["f0"] == f0 and x["case"] is case]
                if not r:
                    row.append("    nan")
                    continue
                a = np.interp(probe, r[0]["freqs"], r[0]["measured"])
                b = np.interp(probe, reference[f0]["freqs"], reference[f0]["measured"])
                target = ic.analog_target_db(probe, case["kind"], f0, gain=case.get("gain", 0.0), q=case.get("q", 1.0), slope=case.get("slope", 12))
                row.append(f"{np.abs(a - b)[target > -60].max():7.2f}")
            print(f"{label(case):24} {rate:6}   " + " ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
