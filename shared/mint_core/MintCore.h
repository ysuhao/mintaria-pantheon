#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include <atomic>
#include <cmath>
#include <vector>

// 所有插件共用的实时安全 DSP 工具与处理器基类。processBlock 内不分配内存、不加锁。
namespace mint
{

inline float dbToGain (float db) noexcept     { return std::pow (10.0f, db * 0.05f); }
inline float gainToDb (float gain) noexcept   { return 20.0f * std::log10 (std::max (gain, 1.0e-6f)); }

// 一阶包络，直接跟随线性电平或 dB 值。
class EnvelopeFollower
{
public:
    void prepare (double newSampleRate) noexcept          { sampleRate = newSampleRate; }
    void setTimes (float attackMs, float releaseMs) noexcept
    {
        attackCoef  = coefficient (attackMs);
        releaseCoef = coefficient (releaseMs);
    }
    void reset (float value = 0.0f) noexcept              { state = value; }

    float process (float input) noexcept
    {
        const auto coef = input > state ? attackCoef : releaseCoef;
        state = input + coef * (state - input);
        return state;
    }

private:
    float coefficient (float ms) const noexcept
    {
        if (ms <= 0.0f)
            return 0.0f;
        return std::exp (-1.0f / (0.001f * ms * (float) sampleRate));
    }

    double sampleRate = 48000.0;
    float attackCoef = 0.0f, releaseCoef = 0.0f, state = 0.0f;
};

// Look-ahead 的音频通路延迟。默认 0 采样，即直通。
class LookaheadDelay
{
public:
    void prepare (int numChannels, int maxDelaySamples)
    {
        size = maxDelaySamples + 1;
        lines.assign ((size_t) numChannels, std::vector<float> ((size_t) size, 0.0f));
        writePos = 0;
    }

    void setDelaySamples (int newDelay) noexcept          { delay = juce::jlimit (0, size - 1, newDelay); }
    int getDelaySamples() const noexcept                  { return delay; }

    void reset() noexcept
    {
        for (auto& line : lines)
            std::fill (line.begin(), line.end(), 0.0f);
        writePos = 0;
    }

    void process (juce::AudioBuffer<float>& buffer) noexcept
    {
        if (delay == 0 || lines.empty())
            return;

        const auto numChannels = juce::jmin (buffer.getNumChannels(), (int) lines.size());
        const auto numSamples = buffer.getNumSamples();
        auto pos = writePos;

        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = buffer.getWritePointer (ch);
            auto* line = lines[(size_t) ch].data();
            pos = writePos;

            for (int i = 0; i < numSamples; ++i)
            {
                line[pos] = data[i];
                auto readPos = pos - delay;
                if (readPos < 0)
                    readPos += size;
                data[i] = line[readPos];
                if (++pos == size)
                    pos = 0;
            }
        }

        writePos = pos;
    }

private:
    std::vector<std::vector<float>> lines;
    int size = 1, delay = 0, writePos = 0;
};

// 音频线程写、界面线程读的电平值，保存区间内的峰值。
class MeterSource
{
public:
    void push (float db) noexcept
    {
        auto current = value.load (std::memory_order_relaxed);
        while (db > current && ! value.compare_exchange_weak (current, db, std::memory_order_relaxed)) {}
    }

    float take (float resetTo = -120.0f) noexcept     { return value.exchange (resetTo, std::memory_order_relaxed); }

private:
    std::atomic<float> value { -120.0f };
};

// 处理器基类：立体声/单声道总线、参数树、状态存取，以及安全的延迟上报。
class Processor : public juce::AudioProcessor,
                  private juce::Timer
{
public:
    static BusesProperties stereoBuses()
    {
        return BusesProperties()
            .withInput ("Input", juce::AudioChannelSet::stereo(), true)
            .withOutput ("Output", juce::AudioChannelSet::stereo(), true);
    }

    // 带外部侧链输入。processBlock 里用 getBusBuffer (buffer, true, 1) 取侧链，它未启用时要能正常工作。
    static BusesProperties stereoBusesWithSidechain()
    {
        return stereoBuses().withInput ("Sidechain", juce::AudioChannelSet::stereo(), false);
    }

    explicit Processor (juce::AudioProcessorValueTreeState::ParameterLayout layout,
                        BusesProperties buses = stereoBuses())
        : juce::AudioProcessor (std::move (buses)),
          apvts (*this, nullptr, "PARAMS", std::move (layout))
    {
        startTimerHz (10);
    }

    juce::AudioProcessorValueTreeState apvts;

    // 可在任意线程调用。实际上报在消息线程完成，音频线程里不会通知宿主。
    void requestLatencySamples (int samples) noexcept   { requestedLatency.store (samples, std::memory_order_relaxed); }

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        const auto out = layouts.getMainOutputChannelSet();
        if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
            return false;
        return out == layouts.getMainInputChannelSet();
    }

    const juce::String getName() const override             { return JucePlugin_Name; }
    bool acceptsMidi() const override                       { return false; }
    bool producesMidi() const override                      { return false; }
    bool hasEditor() const override                         { return true; }
    double getTailLengthSeconds() const override            { return 0.0; }
    int getNumPrograms() override                           { return 1; }
    int getCurrentProgram() override                        { return 0; }
    void setCurrentProgram (int) override                   {}
    const juce::String getProgramName (int) override        { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& dest) override
    {
        if (auto xml = apvts.copyState().createXml())
            copyXmlToBinary (*xml, dest);
    }

    void setStateInformation (const void* data, int size) override
    {
        if (auto xml = getXmlFromBinary (data, size))
            if (xml->hasTagName (apvts.state.getType()))
                apvts.replaceState (juce::ValueTree::fromXml (*xml));
    }

private:
    void timerCallback() override
    {
        const auto wanted = requestedLatency.load (std::memory_order_relaxed);
        if (wanted != getLatencySamples())
            setLatencySamples (wanted);
    }

    std::atomic<int> requestedLatency { 0 };
};

} // namespace mint
