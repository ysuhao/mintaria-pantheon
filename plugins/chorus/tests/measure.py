#!/usr/bin/env python3
"""Sirens 的可重复验证。全部信号都是生成的，插件走 mint_probe render 和 sirens_switch_host。

python plugins/chorus/tests/measure.py --build D:/mint-build/chorus [--ref <对标.vst3>] --work <临时目录> [--only modes,mono,...]

默认 48 kHz。设环境变量 SIRENS_TEST_RATE 可换采样率，用来核对调制参数不随采样率变化：
SIRENS_TEST_RATE=96000 python plugins/chorus/tests/measure.py --build ... --work ... --only modes

退出码非 0 表示有硬性指标不达标。对标路径从命令行传入，代码里不出现其名称。
"""
import argparse
import itertools
import os
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

FS = int(os.environ.get("SIRENS_TEST_RATE", "48000"))
BAND = (100.0, 8000.0)
MODES = ["Mode I", "Mode II", "Mode III", "Mode IV"]
# 任务书的硬性门槛，来源写在 NOTES.md。
MAX_PITCH_CENTS = 10.0
MAX_CLICK_HF_DBFS = -90.0
DEFAULT_MIX = 0.40


def read_wav(path):
    rate, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float64) / 2 ** 31
    data = data.astype(np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return rate, data


def write_wav(path, left, right):
    wavfile.write(path, FS, np.stack([left, right], 1).astype(np.float32))


class Rig:
    def __init__(self, args):
        self.probe = os.path.join(args.build, "tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe")
        self.plugin = os.path.join(args.build, "plugins/chorus/mint_chorus_artefacts/Release/VST3/Mintaria Sirens.vst3")
        self.host = os.path.join(args.build, "plugins/chorus/sirens_switch_host_artefacts/Release/sirens_switch_host.exe")
        self.ref = args.ref
        self.work = args.work
        os.makedirs(self.work, exist_ok=True)

    def path(self, name):
        return os.path.join(self.work, name)

    def render(self, plugin, in_path, sets, tag="out", block=256):
        out = self.path(tag + ".wav")
        cmd = [self.probe, "render", plugin, in_path, out, "--block", str(block)]
        for key, value in sets.items():
            cmd += ["--set", f"{key}={value}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if "rendered" not in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
        return read_wav(out)[1]

    def sirens(self, in_path, bits, mix=DEFAULT_MIX, compat=0, noise=0, block=256):
        sets = {name: (bits >> i) & 1 for i, name in enumerate(MODES)}
        sets.update({"Mix": mix, "Mono Compat": compat, "Noise": noise})
        return self.render(self.plugin, in_path, sets, block=block)

    def reference(self, in_path, mode, mode4=0, mix=0.5):
        # 用参数序号：2 是前三个模式的互斥枚举，3 是第四个模式的独立开关，4 是干湿。
        sets = {"2": mode, "3": mode4, "4": mix}
        return self.render(self.ref, in_path, sets)


def combo_name(bits):
    roman = ["I", "II", "III", "IV"]
    names = [roman[i] for i in range(4) if (bits >> i) & 1]
    return "+".join(names) if names else "关"


def delay_trajectory(wet, dry, seg=FS // 20, max_lag=FS // 40):
    """短帧互相关求出湿声延迟随时间的轨迹，返回 ms。"""
    points = []
    n = len(dry)
    for k in range(0, n - seg - max_lag, seg):
        a = dry[k:k + seg]
        b = wet[k:k + seg + max_lag]
        c = signal.correlate(b, a, "valid")
        c[:FS // 2000] = 0.0
        i = int(np.argmax(np.abs(c)))
        d = 0.0
        if 0 < i < len(c) - 1:
            d = 0.5 * (abs(c[i - 1]) - abs(c[i + 1])) / (abs(c[i - 1]) - 2 * abs(c[i]) + abs(c[i + 1]) + 1e-20)
        points.append((i + d) / FS * 1000.0)
    return np.array(points)


def line_pitch(wet, freq=5000.0, frame=FS // 4):
    """追踪主峰频率，读延迟线本身的音高偏移（音分）。全湿下左右声道可能混有另一条线，它较弱，不影响主峰。"""
    window = signal.windows.hann(frame)
    pad = frame * 8
    freqs = np.fft.rfftfreq(pad, 1.0 / FS)
    cents = []
    for k in range(FS, len(wet) - frame - FS, frame // 2):
        spec = np.abs(np.fft.rfft(wet[k:k + frame] * window, n=pad))
        i = int(np.argmax(spec))
        d = 0.5 * (spec[i - 1] - spec[i + 1]) / (spec[i - 1] - 2 * spec[i] + spec[i + 1])
        cents.append(1200.0 * np.log2((freqs[i] + d * FS / pad) / freq))
    cents = np.array(cents)
    spectrum = np.abs(np.fft.rfft(cents - cents.mean(), n=len(cents) * 8))
    rates = np.fft.rfftfreq(len(cents) * 8, (frame // 2) / FS)
    return cents, rates[int(np.argmax(spectrum))]


def composite_wobble(y, freq=1000.0):
    """实际输出的瞬时音高起伏：单音的 Hilbert 瞬时频率。含干声时这才是听到的结果。"""
    phase = np.unwrap(np.angle(signal.hilbert(y[FS:-FS])))
    inst = np.diff(phase) * FS / (2 * np.pi)
    sos = signal.butter(2, 20.0, "lp", fs=FS, output="sos")
    inst = signal.sosfiltfilt(sos, inst)[FS // 2:-FS // 2]
    return 1200.0 * np.log2(np.maximum(inst, 1.0) / freq)


def measure_modes(rig, plugin_kind="sirens"):
    """每个模式的调制深度、速率和音高偏移。全湿，噪声关。"""
    n = FS * 24
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n) * 0.1
    noise_path = rig.path("noise24.wav")
    write_wav(noise_path, x, x)
    t = np.arange(n) / FS
    tone = 0.25 * np.sin(2 * np.pi * 5000.0 * t)
    tone_path = rig.path("tone5k.wav")
    write_wav(tone_path, tone, tone)
    low = 0.25 * np.sin(2 * np.pi * 1000.0 * t)
    low_path = rig.path("tone1k.wav")
    write_wav(low_path, low, low)

    rows = []
    for i in range(4):
        wet = rig.sirens(noise_path, 1 << i, mix=1.0)
        left, right = delay_trajectory(wet[:, 0], x), delay_trajectory(wet[:, 1], x)
        wet_tone = rig.sirens(tone_path, 1 << i, mix=1.0)
        cents_l, rate = line_pitch(wet_tone[:, 0])
        cents_r, _ = line_pitch(wet_tone[:, 1])
        mixed = rig.sirens(low_path, 1 << i)
        wobble = composite_wobble(mixed[:, 0])
        rows.append(dict(mode=i + 1, l_min=left.min(), l_max=left.max(), r_min=right.min(), r_max=right.max(),
                         rate=rate, cents_l=np.percentile(np.abs(cents_l), 95), cents_r=np.percentile(np.abs(cents_r), 95),
                         cents_mean=0.5 * (cents_l.mean() + cents_r.mean()), wobble=np.percentile(np.abs(wobble), 99)))
    return rows


def mono_transfer(mono, x, frame=FS // 2, hop=FS // 4, nperseg=2048):
    """把折叠结果对干声求短时传输函数，0.5 s 一帧、用 1/6 倍频程平滑前的原始值。"""
    frames = []
    for k in range(FS, len(x) - frame, hop):
        f, pxy = signal.csd(x[k:k + frame], mono[k:k + frame], FS, nperseg=nperseg, noverlap=nperseg // 2)
        _, pxx = signal.welch(x[k:k + frame], FS, nperseg=nperseg, noverlap=nperseg // 2)
        band = (f >= BAND[0]) & (f <= BAND[1])
        frames.append(20 * np.log10(np.abs(pxy[band]) / pxx[band] + 1e-9))
    return np.array(frames)


def mono_stats(frames):
    spread = frames.max(axis=1) - frames.min(axis=1)
    return dict(p2p_median=float(np.median(spread)), p2p_max=float(spread.max()), deepest=float(frames.min()),
                highest=float(frames.max()))


def width_stats(stereo, x):
    """宽度三个数：S/M 比、宽带互相关、低频段（200-1000 Hz）互相关最大值。后者最能反映宽度塌陷。"""
    start = FS
    l, r = stereo[start:, 0], stereo[start:, 1]
    mid, side = 0.5 * (l + r), 0.5 * (l - r)
    sm = 10 * np.log10((np.mean(side ** 2) + 1e-20) / (np.mean(mid ** 2) + 1e-20))
    corr = float(np.corrcoef(l, r)[0, 1])
    sos = signal.butter(4, [200, 1000], "bp", fs=FS, output="sos")
    lf, rf = signal.sosfilt(sos, l), signal.sosfilt(sos, r)
    seg = FS // 10
    series = []
    for k in range(0, len(lf) - seg, seg):
        a, b = lf[k:k + seg], rf[k:k + seg]
        series.append(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-20))
    series = np.array(series)
    return dict(sm=float(sm), corr=corr, band_mean=float(series.mean()), band_max=float(series.max()),
                band_min=float(series.min()))


def out_vs_in(stereo, x):
    return 20 * np.log10(np.sqrt(np.mean(stereo[FS:] ** 2)) / np.sqrt(np.mean(x ** 2)))


def click_score(y, marks, rate=FS):
    """切换瞬间的高频冲击。输入是 1 kHz 单音、湿声带限在 10.5 kHz，12 kHz 以上本该没有东西，有就是不连续。"""
    sos = signal.butter(8, 12000, "hp", fs=rate, output="sos")
    hf = np.abs(signal.sosfilt(sos, y))
    win = int(0.04 * rate)
    return 20 * np.log10(max(hf[m:m + win].max() for m in marks) + 1e-12)


def run_clicks(rig, failures):
    print("\n== 切换咔嗒（1 kHz 正弦，全部 240 个有序跳变）")
    # 正弦的两阶差分很平，切换造成的不连续最显眼。
    t = np.arange(FS * 40) / FS
    tone = 0.25 * np.sin(2 * np.pi * 1000.0 * t)
    in_path = rig.path("switch_in.wav")
    write_wav(in_path, tone, tone)
    for block, mix, compat in ((256, DEFAULT_MIX, 0), (256, 1.0, 0), (64, DEFAULT_MIX, 1), (1024, DEFAULT_MIX, 0)):
        out = rig.path("switch_out.wav")
        cmd = [rig.host, rig.plugin, "sequence", in_path, out, "--block", str(block), "--interval-ms", "150",
               "--set", f"Mix={mix}", "--set", f"Mono Compat={compat}", "--set", "Noise=0"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if "rendered" not in res.stdout:
            raise RuntimeError(res.stdout + res.stderr)
        _, y = read_wav(out)
        marks = np.loadtxt(out + ".txt", dtype=int)
        worst = max(click_score(y[:, ch], marks) for ch in range(2))
        ok = worst < MAX_CLICK_HF_DBFS
        print(f"block {block:5d} mix {mix:.2f} 兼容 {compat}: {len(marks)} 次切换，切换窗 12 kHz 以上峰值 {worst:7.1f} dBFS {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"click block {block} mix {mix}")

        if block == 256 and mix == DEFAULT_MIX:
            # 探测器自检：在一次切换处人为加 0.1 dB 的瞬时增益跳变，必须能被抓到。
            spiked = y[:, 0].copy()
            spiked[marks[len(marks) // 2]:] *= 10 ** (0.1 / 20)
            detected = click_score(spiked, marks)
            print(f"探测器自检（注入 0.1 dB 瞬时跳变）：{detected:7.1f} dBFS {'OK' if detected > MAX_CLICK_HF_DBFS else 'FAIL'}")
            if detected <= MAX_CLICK_HF_DBFS:
                failures.append("click detector insensitive")

    for name in ("Mono Compat", "Mix"):
        y, marks = render_toggle(rig, in_path, name, ["--set", "Noise=0"])
        score = max(click_score(y[:, ch], marks) for ch in range(2))
        ok = score < MAX_CLICK_HF_DBFS
        print(f"反复开关 {name}（噪声关）: {len(marks)} 次，{score:7.1f} dBFS {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"click toggle {name}")

    # 噪声本身就占 12 kHz 以上，开关噪声只能和噪声常开时比，不能多出冲击。
    y, marks = render_toggle(rig, in_path, "Noise", [])
    steady = rig.sirens(in_path, 0b0101, noise=1)
    score = max(click_score(y[:, ch], marks) for ch in range(2))
    base = max(click_score(steady[:, ch], marks) for ch in range(2))
    ok = score <= base + 1.0
    print(f"反复开关 Noise: {len(marks)} 次，{score:7.1f} dBFS，噪声常开时同窗 {base:7.1f} dBFS {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("click toggle Noise")


def render_toggle(rig, in_path, name, extra):
    out = rig.path("toggle_out.wav")
    res = subprocess.run([rig.host, rig.plugin, "toggle", in_path, out, name, "--interval-ms", "150", "--set", "Mode III=1"] + extra,
                         capture_output=True, text=True, timeout=600)
    if "rendered" not in res.stdout:
        raise RuntimeError(res.stdout + res.stderr)
    return read_wav(out)[1], np.loadtxt(out + ".txt", dtype=int)


def run_stress(rig, failures):
    print("\n== 边界（44.1–192 kHz、块长 1/7/64/2048、单/立体声、静音、极小值）")
    res = subprocess.run([rig.host, rig.plugin, "stress"], capture_output=True, text=True, timeout=1800)
    print(res.stdout.strip())
    if res.returncode != 0:
        failures.append("stress")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--ref", default="")
    parser.add_argument("--work", required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    rig = Rig(args)
    failures = []
    only = set(args.only.split(",")) if args.only else None

    def want(name):
        return only is None or name in only

    if want("modes"):
        print("== 各模式的调制（全湿、噪声关）")
        print("模式 | 左线 ms | 右线 ms | 速率 Hz | 线音高偏移(95 分位) 左/右 音分 | 平均偏移 音分 | 默认干湿下合成音高起伏(99 分位) 音分")
        for row in measure_modes(rig):
            print(f"{row['mode']:>3} | {row['l_min']:5.2f}–{row['l_max']:5.2f} | {row['r_min']:5.2f}–{row['r_max']:5.2f} | {row['rate']:.3f} | "
                  f"{row['cents_l']:5.2f}/{row['cents_r']:5.2f} | {row['cents_mean']:+.3f} | {row['wobble']:5.2f}")
            if max(row["cents_l"], row["cents_r"]) > MAX_PITCH_CENTS:
                failures.append(f"pitch mode {row['mode']}")

    n = FS * 20
    x = np.random.default_rng(5).standard_normal(n) * 0.1
    noise_path = rig.path("noise20.wav")
    write_wav(noise_path, x, x)

    if want("mono"):
        print("\n== 单声道折叠（白噪声，(L+R)/2 对干声，100 Hz–8 kHz，每 0.5 s 一帧）")
        print("组合 | 兼容 | 峰谷差 中位/最大 dB | 最深谷 dB | 最高峰 dB")
        for bits in range(1, 16):
            for compat in (0, 1):
                stereo = rig.sirens(noise_path, bits, compat=compat)
                frames = mono_transfer(0.5 * (stereo[:, 0] + stereo[:, 1]), x)
                s = mono_stats(frames)
                print(f"{combo_name(bits):14s} | {compat} | {s['p2p_median']:5.2f}/{s['p2p_max']:5.2f} | {s['deepest']:6.2f} | {s['highest']:+5.2f}")
                if compat == 1 and s["p2p_max"] > 1.0:
                    failures.append(f"compat mono comb {combo_name(bits)}")

        # 兼容模式的 L+R 必须严格等于干声。
        stereo = rig.sirens(noise_path, 15, compat=1)
        dry = rig.sirens(noise_path, 15, mix=0.0)
        diff = 0.5 * (stereo[:, 0] + stereo[:, 1])
        gain = np.dot(diff, x) / np.dot(x, x)
        residual = diff - gain * x
        db = 20 * np.log10(np.sqrt(np.mean(residual[FS:] ** 2)) / (gain * x.std()))
        print(f"兼容模式 L+R 减去干声的残差：{db:.1f} dB（干声增益 {gain:.3f}）")
        if db > -60:
            failures.append("compat residual")
        dry_err = 20 * np.log10(np.sqrt(np.mean((dry[:, 0] - x) ** 2)) / x.std() + 1e-12)
        print(f"干湿 0% 与输入的差：{dry_err:.1f} dB")
        if dry_err > -80:
            failures.append("mix 0 not dry")

    if want("bypass"):
        print("\n== 全部抬起应等于直通")
        off = rig.sirens(noise_path, 0, noise=1)
        err = 20 * np.log10(np.sqrt(np.mean((off - x[:, None]) ** 2)) / x.std() + 1e-12)
        print(f"与输入的差：{err:.1f} dB")
        if err > -80:
            failures.append("all off not transparent")

    if want("width"):
        print("\n== 立体声宽度（白噪声，干湿 40%）")
        print("组合 | 兼容 | S/M dB | 宽带相关 | 200-1k 相关 平均/最大 | 输出/输入 dB")
        for bits in (1, 2, 4, 8, 3, 9, 15):
            for compat in (0, 1):
                stereo = rig.sirens(noise_path, bits, compat=compat)
                w = width_stats(stereo, x)
                print(f"{combo_name(bits):14s} | {compat} | {w['sm']:6.2f} | {w['corr']:.3f} | {w['band_mean']:.3f}/{w['band_max']:.3f} | {out_vs_in(stereo, x):+.2f}")

    if want("spectrum"):
        print("\n== 湿声频响（全湿）")
        stereo = rig.sirens(noise_path, 1, mix=1.0)
        f, pxx = signal.welch(x, FS, nperseg=8192)
        _, pyy = signal.welch(stereo[:, 0], FS, nperseg=8192)
        h = 10 * np.log10(pyy / pxx)
        pts = (50, 90, 150, 300, 1000, 3000, 6000, 9000, 10500, 12000, 14000, 18000)
        print(" ".join(f"{p}:{h[np.argmin(abs(f - p))]:+.1f}" for p in pts))

    if want("noise"):
        print("\n== 噪声（1 kHz 正弦 -18 dBFS，I 模式全湿，陷波后读底噪）")
        t = np.arange(FS * 8) / FS
        tone = 10 ** (-18 / 20) * np.sin(2 * np.pi * 1000.0 * t)
        path = rig.path("noise_tone.wav")
        write_wav(path, tone, tone)

        def floor_db(y):
            # 只看 4–9 kHz：离基波足够远，又在 BBD 带限之内。
            sos = signal.butter(10, [4000, 9000], "bp", fs=FS, output="sos")
            r = signal.sosfilt(sos, y[:, 0])[FS:]
            return 20 * np.log10(np.sqrt(np.mean(r ** 2)) + 1e-12)

        on, off = rig.sirens(path, 1, mix=1.0, noise=1), rig.sirens(path, 1, mix=1.0, noise=0)
        print(f"4–9 kHz 残余：噪声开 {floor_db(on):.1f} dBFS，噪声关 {floor_db(off):.1f} dBFS")
        if floor_db(on) - floor_db(off) < 3:
            failures.append("noise switch has no effect")
        zero_path = rig.path("zero.wav")
        write_wav(zero_path, t * 0, t * 0)
        silent = rig.sirens(zero_path, 15, noise=1)
        print(f"静音输入的输出峰值：{np.max(np.abs(silent)):.3e}")
        if np.max(np.abs(silent)) != 0.0:
            failures.append("silence not silent")

    if want("clicks"):
        run_clicks(rig, failures)

    if want("stress"):
        run_stress(rig, failures)

    if want("ref") and rig.ref:
        print("\n== 对标黑盒（同一信号）")
        print("设置 | S/M dB | 宽带相关 | 200-1k 相关 平均/最大 | 折叠峰谷差 中位/最大 | 最深谷 | 输出/输入 dB")
        t = np.arange(FS * 24) / FS
        tone5k = rig.path("ref_tone5k.wav")
        write_wav(tone5k, 0.25 * np.sin(2 * np.pi * 5000.0 * t), 0.25 * np.sin(2 * np.pi * 5000.0 * t))
        tone1k = rig.path("ref_tone1k.wav")
        write_wav(tone1k, 0.25 * np.sin(2 * np.pi * 1000.0 * t), 0.25 * np.sin(2 * np.pi * 1000.0 * t))
        for i, value in enumerate((0.34, 0.67, 1.0)):
            cents, rate = line_pitch(rig.reference(tone5k, value, 0, mix=1.0)[:, 0])
            wobble = composite_wobble(rig.reference(tone1k, value, 0)[:, 0])
            print(f"对标 {i + 1}：速率 {rate:.3f} Hz，线音高偏移 95 分位 {np.percentile(np.abs(cents), 95):.2f} 音分，默认干湿合成起伏 99 分位 {np.percentile(np.abs(wobble), 99):.2f} 音分")
        for i, value in enumerate((0.34, 0.67, 1.0)):
            for mode4 in (0, 1):
                stereo = rig.reference(noise_path, value, mode4)
                w = width_stats(stereo, x)
                s = mono_stats(mono_transfer(0.5 * (stereo[:, 0] + stereo[:, 1]), x))
                print(f"对标 {i + 1}{'+4' if mode4 else '  '} | {w['sm']:6.2f} | {w['corr']:.3f} | {w['band_mean']:.3f}/{w['band_max']:.3f} | "
                      f"{s['p2p_median']:5.2f}/{s['p2p_max']:5.2f} | {s['deepest']:6.2f} | {out_vs_in(stereo, x):+.2f}")
        for bits in (1, 2, 4, 8, 15):
            stereo = rig.sirens(noise_path, bits)
            w = width_stats(stereo, x)
            s = mono_stats(mono_transfer(0.5 * (stereo[:, 0] + stereo[:, 1]), x))
            print(f"Sirens {combo_name(bits):8s} | {w['sm']:6.2f} | {w['corr']:.3f} | {w['band_mean']:.3f}/{w['band_max']:.3f} | "
                  f"{s['p2p_median']:5.2f}/{s['p2p_max']:5.2f} | {s['deepest']:6.2f} | {out_vs_in(stereo, x):+.2f}")

    print("\n结果：" + ("全部通过" if not failures else "不达标 " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
