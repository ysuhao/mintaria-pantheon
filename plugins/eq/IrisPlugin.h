// Mintaria Iris 的处理器。频谱分析走无锁 FIFO，不进音频通路。
#pragma once

#include "IrisDsp.h"

#include <mint_core/MintCore.h>

#include <array>
#include <atomic>
#include <memory>
#include <vector>

namespace iris
{

constexpr float maxLookaheadMs = 10.0f;
constexpr double minFrequency = 10.0, maxFrequency = 30000.0;

inline juce::String bandPrefix (int band)
{
    return "b" + juce::String (band + 1).paddedLeft ('0', 2) + "_";
}

// 单生产者单消费者的采样 FIFO。音频线程写，界面线程读，满了就丢弃。
class SpectrumFifo
{
public:
    SpectrumFifo() : fifo (1 << 16), storage ((size_t) (1 << 16), 0.0f) {}

    void push (const float* data, int n) noexcept
    {
        const auto scope = fifo.write (n);
        if (scope.blockSize1 > 0)
            std::copy (data, data + scope.blockSize1, storage.begin() + scope.startIndex1);
        if (scope.blockSize2 > 0)
            std::copy (data + scope.blockSize1, data + scope.blockSize1 + scope.blockSize2, storage.begin() + scope.startIndex2);
    }

    int pull (float* dest, int capacity) noexcept
    {
        const auto scope = fifo.read (juce::jmin (capacity, fifo.getNumReady()));
        if (scope.blockSize1 > 0)
            std::copy (storage.begin() + scope.startIndex1, storage.begin() + scope.startIndex1 + scope.blockSize1, dest);
        if (scope.blockSize2 > 0)
            std::copy (storage.begin() + scope.startIndex2, storage.begin() + scope.startIndex2 + scope.blockSize2, dest + scope.blockSize1);
        return scope.blockSize1 + scope.blockSize2;
    }

private:
    juce::AbstractFifo fifo;
    std::vector<float> storage;
};

class IrisProcessor final : public mint::Processor
{
public:
    IrisProcessor();

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;

    // 从参数读取频段设置，界面和音频线程都可以调用。
    BandSettings readBand (int band) const noexcept;
    bool isBandUsed (int band) const noexcept       { return bandParams[(size_t) band].used->load (std::memory_order_relaxed) > 0.5f; }
    float getDynamicOffset (int band) const noexcept { return bands[(size_t) band].getDynamicOffset(); }
    double getRate() const noexcept                 { return currentRate.load (std::memory_order_relaxed); }

    SpectrumFifo inputSpectrum, outputSpectrum;
    std::atomic<bool> spectrumWanted { false };

private:
    struct BandParams
    {
        std::atomic<float>* used = nullptr;
        std::atomic<float>* on = nullptr;
        std::atomic<float>* type = nullptr;
        std::atomic<float>* slope = nullptr;
        std::atomic<float>* stereo = nullptr;
        std::atomic<float>* freq = nullptr;
        std::atomic<float>* gain = nullptr;
        std::atomic<float>* q = nullptr;
        std::atomic<float>* dynamic = nullptr;
        std::atomic<float>* threshold = nullptr;
        std::atomic<float>* range = nullptr;
    };

    void tick() noexcept;
    void updateLookahead() noexcept;
    void pushSpectrum (SpectrumFifo&, const double* const* audio, int channels, int n) noexcept;

    std::array<BandParams, numBands> bandParams;
    std::array<Band, numBands> bands;

    std::atomic<float>* outputParam = nullptr;
    std::atomic<float>* lookaheadParam = nullptr;
    std::atomic<float>* attackParam = nullptr;
    std::atomic<float>* releaseParam = nullptr;

    std::atomic<double> currentRate { 48000.0 };
    int tickInterval = 128, samplesToTick = 0;
    mint::LookaheadDelay lookahead;
    juce::LinearSmoothedValue<double> outputGain { 1.0 };
};

} // namespace iris
