#!/usr/bin/env python3
"""验证图手术：逐帧有状态推理必须和官方整段推理的结果一致，同时测单帧耗时。

用法：python -I verify_stateful_onnx.py <官方模型目录> <有状态模型目录>
"""
import sys
import time

import numpy as np
import onnxruntime as ort


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


class Stateful:
    def __init__(self, path):
        self.session = session(path)
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.state = {i.name: np.zeros(i.shape, np.float32) for i in self.session.get_inputs() if i.name[:2] in ("h_", "p_")}

    def run(self, feed):
        values = dict(zip(self.outputs, self.session.run(None, {**feed, **self.state})))
        for name, old in self.state.items():
            new = values[name + "_out"]
            self.state[name] = new if name[0] == "h" else np.concatenate([old[:, :, 1:], new], axis=2)
        return values


def main():
    official, stateful = sys.argv[1], sys.argv[2]
    frames = 300
    rng = np.random.default_rng(1)
    erb = (np.cumsum(rng.normal(size=(1, 1, frames, 32)), axis=2) * 0.02).astype(np.float32)
    spec = (rng.normal(size=(1, 2, frames, 96)) * 0.3).astype(np.float32)

    enc_off = session(f"{official}/enc.onnx")
    names = [o.name for o in enc_off.get_outputs()]
    out = dict(zip(names, enc_off.run(None, {"feat_erb": erb, "feat_spec": spec})))
    mask_off = session(f"{official}/erb_dec.onnx").run(None, {k: out[k] for k in ("emb", "e3", "e2", "e1", "e0")})[0]
    coefs_off = session(f"{official}/df_dec.onnx").run(None, {"emb": out["emb"], "c0": out["c0"]})[0]

    enc, erb_dec, df_dec = (Stateful(f"{stateful}/{n}.onnx") for n in ("enc", "erb_dec", "df_dec"))
    masks, coefs, times = [], [], []
    for t in range(frames):
        start = time.perf_counter()
        e = enc.run({"feat_erb": erb[:, :, t:t + 1], "feat_spec": spec[:, :, t:t + 1]})
        m = erb_dec.run({k: e[k] for k in ("emb", "e3", "e2", "e1", "e0")})
        c = df_dec.run({"emb": e["emb"], "c0": e["c0"]})
        times.append(time.perf_counter() - start)
        masks.append(m["m"])
        coefs.append(c["coefs"])

    mask_diff = np.abs(np.concatenate(masks, 2) - mask_off).max()
    coef_diff = np.abs(np.concatenate(coefs, 1) - coefs_off).max()
    times = np.array(times[5:]) * 1e3
    print(f"mask diff {mask_diff:.2e}, coef diff {coef_diff:.2e}")
    print(f"ms/frame avg {times.mean():.2f}, p99 {np.percentile(times, 99):.2f}, max {times.max():.2f}")
    if mask_diff > 1e-4 or coef_diff > 1e-4:
        print("FAIL stateful graph differs from official graph")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
