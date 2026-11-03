# Mintaria Pantheon

Eleven vocal VST3 plugins by MINT RYU, built for live and real-time vocal chains. Windows x64. Interface in English, Korean and Chinese.

MINT RYU 出品的 11 款人声 VST3 插件，面向直播和实时人声。支持 Windows x64，界面有英文、韩文、中文。

MINT RYU가 만든 보컬 VST3 플러그인 11종입니다. 라이브와 실시간 보컬 체인을 위해 설계했습니다. Windows x64를 지원하며 인터페이스는 영어, 한국어, 중국어로 제공됩니다.

**[Website / 官网 / 웹사이트](https://vst.mintryu.com/)** · **[Download / 下载 / 다운로드](https://github.com/ysuhao/mintaria-pantheon/releases/latest)**

| No. | Plugin | Function | 功能 | 기능 | Default latency |
|---|---|---|---|---|---|
| I | Mintaria Janus | Gate | 门限 | 게이트 | 0 |
| II | Mintaria Lethe | Denoise | 降噪 | 노이즈 제거 | 31.67 ms (48 kHz, 256 buffer) |
| III | Mintaria Atlas | Compressor | 压缩 | 컴프레서 | 0 |
| IV | Mintaria Iris | EQ | 均衡 | EQ | 0 |
| V | Mintaria Medusa | De-esser | 去齿音 | 디에서 | 0 |
| VI | Mintaria Helios | Fader | 推子 | 페이더 | 0 |
| VII | Mintaria Elysium | Reverb | 混响 | 리버브 | 0 |
| VIII | Mintaria Echo | Delay | 延迟 | 딜레이 | 0 |
| IX | Mintaria Sirens | Chorus | 合唱 | 코러스 | 0 |
| X | Mintaria Aegis | Limiter | 限制器 | 리미터 | 1.5 ms |
| XI | Mintaria Vesta | Tube Saturation | 电子管饱和 | 진공관 새츄레이션 | 0 |

## Install / 安装 / 설치

Download the ZIP from [Releases](https://github.com/ysuhao/mintaria-pantheon/releases/latest), extract all of it, run `install.bat`, then rescan VST3 plugins in your host.

从 [Releases](https://github.com/ysuhao/mintaria-pantheon/releases/latest) 下载 ZIP，完整解压后运行 `install.bat`，再在宿主里重新扫描 VST3 插件。

[Releases](https://github.com/ysuhao/mintaria-pantheon/releases/latest)에서 ZIP을 받아 전체 압축을 해제하고 `install.bat`을 실행한 뒤, 호스트에서 VST3 플러그인을 다시 스캔하세요.

## Build

Requires Windows x64, Visual Studio 2026 with the C++ workload, CMake 3.22 or newer, and [JUCE 8.0.12](https://github.com/juce-framework/JUCE/tree/8.0.12). The Lethe build downloads pinned, hash-checked copies of ONNX Runtime and RNNoise during configuration.

```bash
git clone --depth 1 -b 8.0.12 https://github.com/juce-framework/JUCE.git
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DMINT_JUCE_DIR=JUCE
cmake --build build --config Release --parallel 4 --target mint_gate_VST3 mint_denoise_VST3 mint_comp_VST3 mint_eq_VST3 mint_deess_VST3 mint_rider_VST3 mint_reverb_VST3 mint_delay_VST3 mint_chorus_VST3 mint_limiter_VST3 mint_tube_VST3
```

Add `-DMINT_ONLY=<slug>` to configure one plugin, then build its `mint_<slug>_VST3` target. The slugs are the folder names under `plugins/`. Release bundles are written to `build/plugins/<slug>/mint_<slug>_artefacts/Release/VST3/`.

## License

[GNU AGPL-3.0-only](LICENSE) with the additional terms in [NOTICE](NOTICE). Modified versions must publish their source under the same license and use a different product name. Third-party components are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
