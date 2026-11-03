"""多频段 CPU 占用。用 mint_eq_bench cpu 跑真实的 processBlock，单线程，占实时百分比（12 轮的最小值和中位数）。"""
import iris_common as ic


def bells(count, **kw):
    sets = []
    for b in range(count):
        sets += ic.band_sets(b + 1, kind="bell", freq=60.0 * (200.0 / 60.0) ** (b / max(1, count - 1)) * (1 + b), gain=3.0 if b % 2 else -3.0, q=1.5, **kw)
    return sets


def cuts(count, slope):
    sets = []
    for b in range(count):
        sets += ic.band_sets(b + 1, kind="lowCut", freq=20.0 + 5.0 * b, slope=slope, q=1.0)
    return sets


CASES = [
    ("无频段", []),
    ("1 钟形", bells(1)),
    ("12 钟形", bells(12)),
    ("24 钟形", bells(24)),
    ("24 钟形 中/侧", bells(24, stereo="mid")),
    ("12 钟形 动态", bells(12, dynamic=True, thresh=-40.0, rng=-6.0)),
    ("24 钟形 动态", bells(24, dynamic=True, thresh=-40.0, rng=-6.0)),
    ("4 个 96 dB/oct 切除", cuts(4, 96)),
    ("24 个 96 dB/oct 切除（最坏）", cuts(24, 96)),
]


def main():
    print(ic.run([ic.BENCH, "design"]))
    for rate, block in ((48000, 128), (48000, 32), (96000, 128), (192000, 256)):
        print(f"== {rate} Hz，缓冲区 {block} ==")
        for name, sets in CASES:
            out = ic.run([ic.BENCH, "cpu", ic.PLUGIN, "--rate", str(rate), "--block", str(block)] + ic.set_args(sets))
            print(f"{name:32} {out.strip().split(' at ')[0]}", flush=True)


if __name__ == "__main__":
    main()
