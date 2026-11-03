#pragma once

#include "LetheEngine.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

// 音频线程和推理线程之间的整条通路。
//
// 所有环形缓冲都按绝对采样序号索引，两个线程只交换序号，不交换指针，也没有锁。
//
//   host 采样率              48 kHz（序号 m）                 host 采样率
//   input ──┬── 重采样 ──▶ [ input ring ] ─工作线程─▶ [ voice ring ] ── 重采样 ──▶ 人声 V
//           └─────────────── 干声延迟线 ──────────────────────────────────▶ 干声 D
//
//   输出 = 人声增益 · V + 环境增益 · (D - V)
//
// V 的序号 j 与输入序号 j - A 对应，A 是引擎的算法延迟。两个方向的重采样都用居中的 FIR，
// 时间轴用整数比例锁定，所以 V 和 D 在任何采样率下都是整数采样对齐，也不会漂移。
//
// 工作线程来不及时，音频线程提前 fadeSamples 个采样就知道，平滑切到 V := D，
// 此时输出是干声，环境音分量为 0，整个切换没有跳变。
namespace lethe
{

// 带查表和线性插值的 Kaiser 窗 sinc。h(k - f)，k 从 -halfWidth + 1 到 halfWidth。
class SincTable
{
public:
    void prepare (double cutoffNormalised, int zeroCrossings);

    int getHalfWidth() const noexcept                 { return halfWidth; }

    // 把 k 从 -halfWidth + 1 到 halfWidth 的抽头写进 out。
    void getCoefficients (double fraction, float* out) const noexcept;

    static constexpr int maxHalfWidth = 160;

private:
    static constexpr int numPhases = 512;
    int halfWidth = 0;
    std::vector<float> table;
};

class Core final : private juce::Thread
{
public:
    Core();
    ~Core() override;

    enum class EngineState { idle, loading, ready, failed };

    struct Stats
    {
        std::atomic<int64_t> framesProcessed { 0 }, framesDropped { 0 }, framesFailed { 0 };
        std::atomic<int64_t> underruns { 0 }, inferMicrosTotal { 0 };
        std::atomic<int> inferMicrosMax { 0 };
        std::array<std::atomic<int64_t>, 9> publishHistogram {}; // 输入攒满一帧到结果可读的耗时
    };

    // 非实时线程调用。不加载模型，模型在工作线程上异步加载，加载完之前输出干声。
    void prepare (double sampleRate, int maxBlockSize, int numChannels, EngineKind initialKind);
    void release();

    int getLatencySamples (EngineKind kind) const noexcept;
    EngineKind getActiveKind() const noexcept         { return activeKind.load (std::memory_order_relaxed); }

    // 音频线程调用。waitForVoice 只在离线渲染时为 true，此时可以等工作线程。
    void process (float* const* channels, int numSamples, float voiceGain, float ambienceGain,
                  EngineKind requestedKind, bool bypassed, bool waitForVoice) noexcept;

    EngineState getEngineState() const noexcept       { return engineState.load (std::memory_order_acquire); }
    bool isFallingBack() const noexcept               { return fallingBack.load (std::memory_order_relaxed); }
    juce::String getEngineError() const;

    Stats stats;

    // 测试用：工作线程处理下一帧前先睡这么久，用来确定性地触发回退。
    std::atomic<int> testWorkerStallMs { 0 };

private:
    void run() override;
    void processChunk (float* const* channels, int numSamples, float voiceGain, float ambienceGain,
                       bool bypassed, bool waitForVoice) noexcept;
    void pushInput (const float* const* channels, int numSamples) noexcept;
    void waitUntilVoiceReady (int numSamples) noexcept;
    void applySwitchFade (float* const* channels, int numSamples) noexcept;
    void resetOutputClock() noexcept;
    void beginKind (EngineKind kind) noexcept;
    bool processOneFrame (int64_t frameStart, VoiceEngine& engine);
    bool ensureEngine (EngineKind kind);
    void wake() noexcept;
    void recordPublishTime() noexcept;
    int computeLatency (EngineKind kind) const noexcept;
    int64_t voiceIndexForOutput (int64_t outputIndex) const noexcept;

    // 配置
    int64_t hostRate = 48000;
    int maxBlock = 0, channelCount = 0, fadeSamples48 = 48, fadeHost = 48;
    bool resampling = false;
    SincTable downTable, upTable;
    std::array<int, numEngineKinds> latency {};

    // 环形缓冲，大小都是 2 的幂
    std::vector<std::vector<float>> dryRing, inputRing, voiceRing;
    int64_t dryMask = 0, ringMask = 0;

    // 音频线程私有
    int64_t hostWritten = 0;              // 已写入干声线的采样数
    int64_t downIndex = 0, downRemainder = 0; // 下一个要产生的 48 kHz 输入序号，及其时间的余数
    int64_t upIndex = 0, upRemainder = 0;     // 下一个输出采样在 48 kHz 时间轴上的整数部分和余数
    int activeLatency = 0, activeDelay = 0;
    float blend = 0.0f, switchGain = 1.0f;
    int holdOff = 0;
    bool switching = false;
    EngineKind pendingKind = EngineKind::standard;
    float smoothedVoice = -1.0f, smoothedAmbience = -1.0f, gainStep = 0.001f;
    std::array<float, SincTable::maxHalfWidth * 2> tapBuffer {};

    // 线程间
    std::atomic<int64_t> inputWritten { 0 };   // input ring 已写到的序号
    std::atomic<int64_t> readPosition { 0 };   // 音频线程正在读的 voice 序号
    std::atomic<int64_t> voiceStart { 0 }, voiceEnd { 0 }; // voice ring 中连续有效的区间
    std::atomic<int> generation { 0 }, voiceGeneration { -1 };
    std::atomic<int64_t> generationStart { 0 };
    std::atomic<int64_t> lastInputTicks { 0 };
    std::atomic<EngineKind> activeKind { EngineKind::standard };
    std::atomic<EngineState> engineState { EngineState::idle };
    std::atomic<bool> fallingBack { false };
    void* wakeEvent = nullptr;

    // 工作线程私有
    std::array<std::unique_ptr<VoiceEngine>, numEngineKinds> engines;
    std::vector<std::vector<float>> frameIn, frameOut;
    std::vector<const float*> frameInPointers;
    std::vector<float*> frameOutPointers;
    mutable juce::SpinLock errorLock;
    juce::String engineError;
};

} // namespace lethe
