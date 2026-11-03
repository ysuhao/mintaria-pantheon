# Mintaria Lethe 第三方组件

随插件分发的第三方内容，发售时要把这份文件和对应的许可证文本一起放进安装包。

| 组件 | 版本 | 来源 | 许可证 |
|---|---|---|---|
| DeepFilterNet3 模型权重（`models/df3`、`models/df3_ll`） | 仓库 `models/DeepFilterNet3_onnx.tar.gz` 和 `DeepFilterNet3_ll_onnx.tar.gz`，经 `tools/make_stateful_onnx.py` 改图，权重未改 | https://github.com/Rikorose/DeepFilterNet | MIT 或 Apache-2.0（仓库整体声明） |
| DeepFilterNet 的 DSP 算法（`DfEngine.cpp` 按 libDF 移植） | main，2024-10-17 | 同上 | MIT 或 Apache-2.0 |
| ONNX Runtime | 1.31.1 | https://github.com/microsoft/onnxruntime | MIT，另见随包的 `ThirdPartyNotices-onnxruntime.txt` |
| RNNoise（含 little 模型，静态链接） | 70f1d25，v0.2 之后 | https://github.com/xiph/rnnoise | BSD-3-Clause，见随包的 `LICENSE-rnnoise.txt` |
| Microsoft Visual C++ 运行库 | 构建机所带版本 | Visual Studio Redist | Microsoft 软件许可条款，可再分发 |

DeepFilterNet 版权声明：Copyright (c) 2021 Hendrik Schröter。
