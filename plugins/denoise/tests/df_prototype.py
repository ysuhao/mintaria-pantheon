"""DfEngine.cpp 的 Python 版本，用来看内部信号和试验改动，不进产品。

和 C++ 引擎一样读 models/ 下的有状态模型。Options 全部取默认值时结果应与 C++ 一致，
analyze_singing.py 会先检查这一点。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

FFT = 960
HOP = 480
FREQS = FFT // 2 + 1
NB_ERB = 32
NB_DF = 96
DF_ORDER = 5


def erb_bands():
    freq2erb = lambda hz: 9.265 * np.log1p(hz / (24.7 * 9.265))
    erb2freq = lambda erb: 24.7 * 9.265 * (np.exp(erb / 9.265) - 1.0)
    width = 48000 / FFT
    step = freq2erb(24000.0) / NB_ERB
    bands, previous, over = [], 0, 0
    for i in range(1, NB_ERB + 1):
        bin_ = int(round(float(np.float32(erb2freq(i * step))) / width))
        count = bin_ - previous - over
        if count < 2:
            over = 2 - count
            count = 2
        else:
            over = 0
        bands.append(count)
        previous = bin_
    bands[-1] += 1
    bands[-1] -= max(0, sum(bands) - FREQS)
    return np.array(bands)


class Net:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.state = {i.name: np.zeros(i.shape, np.float32) for i in self.session.get_inputs() if i.name[:2] in ("h_", "p_")}

    def run(self, feed):
        values = dict(zip(self.outputs, self.session.run(None, {**feed, **self.state})))
        for name, old in self.state.items():
            new = values[name + "_out"]
            self.state[name] = new if name[0] == "h" else np.concatenate([old[:, :, 1:], new], axis=2)
        return values


@dataclass
class Options:
    min_db_thresh: float = -15.0     # 低于它直接给零掩码，不跑解码器
    max_db_thresh: float = 35.0      # 高于它不做处理
    norm_alpha: float = 0.99         # 特征归一化的平滑系数，对应 1 秒
    use_df: bool = True              # False 时只用 ERB 掩码


class DfPrototype:
    def __init__(self, model_dir, lookahead, options=None):
        model_dir = Path(model_dir)
        self.lookahead = lookahead
        self.options = options or Options()
        self.bands = erb_bands()
        self.band_of_bin = np.repeat(np.arange(NB_ERB), self.bands)
        self.enc = Net(model_dir / "enc.onnx")
        self.erb_dec = Net(model_dir / "erb_dec.onnx")
        self.df_dec = Net(model_dir / "df_dec.onnx")
        i = np.arange(FFT)
        s = np.sin(0.5 * np.pi * (i + 0.5) / (FFT // 2))
        self.window = np.sin(0.5 * np.pi * s * s).astype(np.float32)
        self.analysis_mem = np.zeros(HOP, np.float32)
        self.synthesis_mem = np.zeros(HOP, np.float32)
        self.buffer_y = [np.zeros(FREQS, np.complex64) for _ in range(DF_ORDER + lookahead)]
        self.buffer_x = [np.zeros(FREQS, np.complex64) for _ in range(DF_ORDER)]
        self.mean_norm = (-60.0 + np.arange(NB_ERB) * (-30.0) / (NB_ERB - 1)).astype(np.float32)
        self.unit_norm = (0.001 + np.arange(NB_DF) * (0.0001 - 0.001) / (NB_DF - 1)).astype(np.float32)
        self.skip = 0

    def process(self, audio):
        """处理单声道音频。返回输出和每帧的跟踪信息（lsnr、阶段、掩码、ERB 特征）。"""
        frames = len(audio) // HOP
        out = np.zeros(frames * HOP, np.float32)
        trace = {"lsnr": np.full(frames, np.nan), "stage": np.zeros(frames, int), "mask": np.full((frames, NB_ERB), np.nan),
                 "feat": np.zeros((frames, NB_ERB))}
        for f in range(frames):
            out[f * HOP : (f + 1) * HOP] = self._frame(audio[f * HOP : (f + 1) * HOP].astype(np.float32), f, trace)
        return out.astype(np.float64), trace

    def _frame(self, x, f, trace):
        opt = self.options
        self.skip = self.skip + 1 if np.mean(x**2) < 1e-7 else 0
        if self.skip > 5:
            trace["stage"][f] = 3
            return np.zeros(HOP, np.float32)

        frame = np.concatenate([self.analysis_mem, x]) * self.window
        self.analysis_mem = x.copy()
        spec = (np.fft.rfft(frame) / FFT).astype(np.complex64)
        self.buffer_y = self.buffer_y[1:] + [spec.copy()]
        self.buffer_x = self.buffer_x[1:] + [spec.copy()]

        power = np.bincount(self.band_of_bin, np.abs(spec) ** 2, NB_ERB) / self.bands
        db = 10.0 * np.log10(power + 1e-10)
        self.mean_norm = db * (1 - opt.norm_alpha) + self.mean_norm * opt.norm_alpha
        feat = ((db - self.mean_norm) / 40.0).astype(np.float32)
        self.unit_norm = np.abs(spec[:NB_DF]) * (1 - opt.norm_alpha) + self.unit_norm * opt.norm_alpha
        norm = spec[:NB_DF] / np.sqrt(self.unit_norm)
        feat_spec = np.stack([norm.real, norm.imag]).astype(np.float32)[None, :, None, :]
        trace["feat"][f] = feat

        e = self.enc.run({"feat_erb": feat[None, None, None, :], "feat_spec": feat_spec})
        lsnr = float(e["lsnr"].ravel()[0])
        trace["lsnr"][f] = lsnr

        apply_gains = opt.min_db_thresh <= lsnr <= opt.max_db_thresh
        apply_zeros = lsnr < opt.min_db_thresh
        target = self.buffer_y[DF_ORDER - 1]

        if apply_gains:
            mask = self.erb_dec.run({k: e[k] for k in ("emb", "e3", "e2", "e1", "e0")})["m"].ravel()
            trace["mask"][f] = mask
            target *= mask[self.band_of_bin]
            trace["stage"][f] = 1
        elif apply_zeros:
            target[:] = 0
            trace["mask"][f] = 0.0
            trace["stage"][f] = 0
        else:
            trace["stage"][f] = 2

        result = target.copy()
        if apply_gains and opt.use_df:
            coefs = self.df_dec.run({"emb": e["emb"], "c0": e["c0"]})["coefs"].ravel()
            coefs = coefs.reshape(NB_DF, DF_ORDER, 2)
            filtered = np.zeros(NB_DF, np.complex64)
            for n in range(DF_ORDER):
                filtered += self.buffer_x[n][:NB_DF] * (coefs[:, n, 0] + 1j * coefs[:, n, 1])
            result[:NB_DF] = filtered

        result = result.copy()
        result[0] = result[0].real
        result[-1] = result[-1].real
        time = np.fft.irfft(result, FFT) * FFT * self.window
        output = time[:HOP] + self.synthesis_mem
        self.synthesis_mem = time[HOP:].astype(np.float32)
        return output.astype(np.float32)
