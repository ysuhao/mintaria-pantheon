# Third-party components

| Component | Version | Source | License |
|---|---|---|---|
| JUCE | 8.0.12 | https://github.com/juce-framework/JUCE | AGPL-3.0 |
| VST3 SDK (inside JUCE) | as included in JUCE 8.0.12 | https://github.com/steinbergmedia/vst3sdk | MIT |

Mintaria Lethe also includes the components listed in
[plugins/denoise/THIRD_PARTY.md](plugins/denoise/THIRD_PARTY.md): DeepFilterNet3
model weights and DSP algorithms (MIT or Apache-2.0), ONNX Runtime (MIT), RNNoise
(BSD-3-Clause) and the Microsoft Visual C++ runtime. Their license texts ship
inside `Mintaria Lethe.vst3/Contents/Resources/lethe/`.

JUCE modules contain further third-party code under their own permissive
licenses. Those notices are kept in the JUCE source tree.
