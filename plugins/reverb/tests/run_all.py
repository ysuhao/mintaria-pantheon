"""复跑全部验证，结果打印到标准输出。先按 NOTES.md 构建好这几个目标：
mint_probe、mint_reverb_VST3、elysium_dsp_bench、elysium_param_text。

  python run_all.py [--long]

--long 把稳定性测试从 1 分钟加到 30 分钟（模拟时长，不是墙钟时长）。
设了环境变量 ELYSIUM_REFERENCE_VST3 才会跑对标对比。
"""
import os
import subprocess
import sys

import common

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args, env=None):
    print(f"\n##### {' '.join(str(a) for a in args)}", flush=True)
    return subprocess.run([str(a) for a in args], env=env).returncode


def main():
    failures = 0
    work = common.workdir()
    minutes = "30" if "--long" in sys.argv else "1"

    failures += run(common.PROBE, "check", common.PLUGIN, "--expect-latency", "0") != 0
    run(common.PROBE, "params", common.PLUGIN)
    run(common.PROBE, "shot", common.PLUGIN, os.path.join(work, "ui_main.png"))
    run(common.PROBE, "shot", common.PLUGIN, os.path.join(work, "ui_advanced.png"), env=dict(os.environ, ELYSIUM_SHOW_ADVANCED="1"))

    for parameter, steps in (("Time", 4), ("Pre-Delay", 1), ("Low Decay", 1), ("Mix", 1)):
        run(common.PARAM_TEXT, common.PLUGIN, parameter, steps)

    for mode in (("stability", minutes), ("clicks",), ("blocks",), ("silence",), ("converter",), ("cpu",)):
        failures += run(common.BENCH, *mode) != 0

    failures += run(sys.executable, os.path.join(HERE, "test_ir.py")) != 0

    if common.REFERENCE:
        failures += run(sys.executable, os.path.join(HERE, "test_compare.py"), "2.2") != 0
    else:
        print("\nELYSIUM_REFERENCE_VST3 not set, skipped the reference comparison")

    print(f"\nfailed steps: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
