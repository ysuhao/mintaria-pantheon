#pragma once

#include <juce_core/juce_core.h>

#include <cstdint>
#include <cstring>
#include <memory>

// 人声估计引擎的接口。引擎只在专用工作线程上运行，可以分配内存；音频线程从不直接调用它。
namespace lethe
{

constexpr int engineRate = 48000;
constexpr int hopSize = 480;

// 按位判断，不受 /fp:fast 影响。
inline bool isFinite (float value) noexcept
{
    uint32_t bits = 0;
    std::memcpy (&bits, &value, sizeof (bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

class VoiceEngine
{
public:
    virtual ~VoiceEngine() = default;

    virtual void reset() = 0;

    // 每声道 hopSize 个采样。返回 false 表示这一帧不可用，调用方按丢帧处理。
    virtual bool process (const float* const* input, float* const* output, int numChannels) = 0;
};

enum class EngineKind { standard, lowLatency, light };
constexpr int numEngineKinds = 3;

// 在工作线程上调用。失败时返回空指针，错误写进 error。
std::unique_ptr<VoiceEngine> createEngine (EngineKind kind, int numChannels, juce::String& error);

// RNNoise 静态链接，没有外部文件，也不依赖 ONNX Runtime。
std::unique_ptr<VoiceEngine> createRnnoiseEngine (int numChannels);

// 不加载引擎也要能算延迟，所以这两个值不放在 VoiceEngine 上。
// 算法延迟：输出的第 j 个采样对应输入的第 j - delay 个采样，不含攒帧的等待。
int getEngineDelay (EngineKind kind) noexcept;
// 给工作线程留的处理时间，单位是 48 kHz 采样数。超出后走干声回退。
int getEngineBudget (EngineKind kind) noexcept;

} // namespace lethe
