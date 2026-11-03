"""渲染辅助：调用 mint_probe，把物理参数换算成归一化值。"""
import math
import os
import subprocess
import tempfile

BUILD = os.environ.get("MINT_BUILD", "D:/mint-build/reverb")
PROBE = f"{BUILD}/tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
PLUGIN = f"{BUILD}/plugins/reverb/mint_reverb_artefacts/Release/VST3/Mintaria Elysium.vst3"
BENCH = f"{BUILD}/plugins/reverb/Release/elysium_dsp_bench.exe"
PARAM_TEXT = f"{BUILD}/plugins/reverb/elysium_param_text_artefacts/Release/elysium_param_text.exe"
REFERENCE = os.environ.get("ELYSIUM_REFERENCE_VST3", "")
RATE = 48000

ALGORITHMS = ["hall", "plate", "room", "chamber"]
CHARACTERS = ["12bit", "24bit", "modern"]


def log_norm(value, low, high):
    return math.log(value / low) / math.log(high / low)


def predelay_norm(ms):
    # NormalisableRange::setSkewForCentre (40) on 0-250
    skew = math.log(0.5) / math.log(40.0 / 250.0)
    return (ms / 250.0) ** skew


def elysium_sets(time=2.2, predelay=0.0, mix=100.0, damping=50.0, low=1.0, mid=1.0, crossover=400.0, mod=30.0,
                 algorithm="hall", character="modern"):
    return {
        "Time": log_norm(time, 0.2, 20.0),
        "Pre-Delay": predelay_norm(predelay),
        "Mix": mix / 100.0,
        "Damping": damping / 100.0,
        "Low Decay": log_norm(low, 0.25, 2.5),
        "Mid Decay": log_norm(mid, 0.25, 2.5),
        "Crossover": log_norm(crossover, 100.0, 2000.0),
        "Mod Depth": mod / 100.0,
        "Algorithm": ALGORITHMS.index(algorithm) / 3.0,
        "Character": CHARACTERS.index(character) / 2.0,
    }


def render(plugin, input_wav, output_wav, sets=None, block=256, timeout=280):
    args = [PROBE, "render", plugin, input_wav, output_wav, "--block", str(block)]
    for key, value in (sets or {}).items():
        args += ["--set", f"{key}={value:.6f}"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or "FAIL" in result.stdout:
        raise RuntimeError(f"mint_probe failed: {result.stdout} {result.stderr}")
    return result.stdout.strip()


def workdir():
    path = os.environ.get("ELYSIUM_TEST_DIR") or os.path.join(tempfile.gettempdir(), "elysium_test")
    os.makedirs(path, exist_ok=True)
    return path
