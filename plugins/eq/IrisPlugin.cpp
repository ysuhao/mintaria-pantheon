#include "IrisPlugin.h"

namespace iris
{

namespace
{
    juce::NormalisableRange<float> logRange (float low, float high)
    {
        const auto ratio = std::log (high / low);
        return { low, high,
                 [=] (float, float, float norm) { return low * std::exp (norm * ratio); },
                 [=] (float, float, float value) { return std::log (value / low) / ratio; },
                 [] (float, float, float value) { return value; } };
    }

    juce::String frequencyText (float hz)
    {
        if (hz >= 1000.0f)
            return juce::String (hz * 0.001f, 2) + " kHz";
        return (hz < 100.0f ? juce::String (hz, 1) : juce::String (juce::roundToInt (hz))) + " Hz";
    }

    juce::StringArray utf8Choices (std::initializer_list<const char*> items)
    {
        juce::StringArray out;
        for (const auto* item : items)
            out.add (juce::String::fromUTF8 (item));
        return out;
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;

        const auto decibels = [] (float v, int) { return (v > 0.0f ? "+" : "") + String (v, 1) + " dB"; };
        const auto milliseconds = [] (float v, int) { return (v < 10.0f ? String (v, 1) : String (roundToInt (v))) + " ms"; };

        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "output", 1 }, "Output",
                                                           NormalisableRange<float> (-30.0f, 30.0f, 0.01f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB").withStringFromValueFunction (decibels)));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "lookahead_ms", 1 }, "Look-ahead",
                                                           NormalisableRange<float> (0.0f, maxLookaheadMs, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms").withStringFromValueFunction (milliseconds)));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "attack_ms", 1 }, "Dynamic Attack", logRange (0.5f, 100.0f), 5.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms").withStringFromValueFunction (milliseconds)));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "release_ms", 1 }, "Dynamic Release", logRange (10.0f, 1000.0f), 120.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms").withStringFromValueFunction (milliseconds)));

        const auto types = utf8Choices ({ "钟形", "低架", "高架", "低切", "高切", "陷波", "倾斜" });
        const auto stereo = utf8Choices ({ "立体声", "左", "右", "中", "侧" });
        const auto onOff = utf8Choices ({ "旁通", "启用" });
        const auto dynamic = utf8Choices ({ "静态", "动态" });

        StringArray slopes;
        for (const auto s : slopeChoices)
            slopes.add (String (s));

        const auto frequencyRange = logRange ((float) minFrequency, (float) maxFrequency);
        const auto qRange = logRange (0.025f, 40.0f);

        for (int b = 0; b < numBands; ++b)
        {
            const auto id = bandPrefix (b);
            const auto name = "B" + String (b + 1).paddedLeft ('0', 2) + " ";
            const auto defaultFrequency = 40.0f * std::pow (10000.0f / 40.0f, (float) b / (float) (numBands - 1));

            layout.add (std::make_unique<AudioParameterBool> (ParameterID { id + "used", 1 }, name + "Used", false));
            layout.add (std::make_unique<AudioParameterChoice> (ParameterID { id + "on", 1 }, name + "On", onOff, 1));
            layout.add (std::make_unique<AudioParameterChoice> (ParameterID { id + "type", 1 }, name + "Type", types, 0));
            layout.add (std::make_unique<AudioParameterChoice> (ParameterID { id + "slope", 1 }, name + "Slope", slopes, 1));
            layout.add (std::make_unique<AudioParameterChoice> (ParameterID { id + "stereo", 1 }, name + "Stereo", stereo, 0));
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id + "freq", 1 }, name + "Freq", frequencyRange, defaultFrequency,
                                                               AudioParameterFloatAttributes().withLabel ("Hz").withStringFromValueFunction ([] (float v, int) { return frequencyText (v); })));
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id + "gain", 1 }, name + "Gain", NormalisableRange<float> (-30.0f, 30.0f, 0.01f), 0.0f,
                                                               AudioParameterFloatAttributes().withLabel ("dB").withStringFromValueFunction (decibels)));
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id + "q", 1 }, name + "Q", qRange, 1.0f,
                                                               AudioParameterFloatAttributes().withStringFromValueFunction ([] (float v, int) { return String (v, v < 10.0f ? 2 : 1); })));
            layout.add (std::make_unique<AudioParameterChoice> (ParameterID { id + "dyn", 1 }, name + "Dynamic", dynamic, 0));
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id + "thresh", 1 }, name + "Threshold", NormalisableRange<float> (-80.0f, 0.0f, 0.1f), -24.0f,
                                                               AudioParameterFloatAttributes().withLabel ("dB").withStringFromValueFunction (decibels)));
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id + "range", 1 }, name + "Range", NormalisableRange<float> (-30.0f, 30.0f, 0.1f), -6.0f,
                                                               AudioParameterFloatAttributes().withLabel ("dB").withStringFromValueFunction (decibels)));
        }

        return layout;
    }
}

IrisProcessor::IrisProcessor() : mint::Processor (createLayout())
{
    outputParam = apvts.getRawParameterValue ("output");
    lookaheadParam = apvts.getRawParameterValue ("lookahead_ms");
    attackParam = apvts.getRawParameterValue ("attack_ms");
    releaseParam = apvts.getRawParameterValue ("release_ms");

    for (int b = 0; b < numBands; ++b)
    {
        const auto id = bandPrefix (b);
        auto& p = bandParams[(size_t) b];
        p.used = apvts.getRawParameterValue (id + "used");
        p.on = apvts.getRawParameterValue (id + "on");
        p.type = apvts.getRawParameterValue (id + "type");
        p.slope = apvts.getRawParameterValue (id + "slope");
        p.stereo = apvts.getRawParameterValue (id + "stereo");
        p.freq = apvts.getRawParameterValue (id + "freq");
        p.gain = apvts.getRawParameterValue (id + "gain");
        p.q = apvts.getRawParameterValue (id + "q");
        p.dynamic = apvts.getRawParameterValue (id + "dyn");
        p.threshold = apvts.getRawParameterValue (id + "thresh");
        p.range = apvts.getRawParameterValue (id + "range");
    }
}

BandSettings IrisProcessor::readBand (int band) const noexcept
{
    const auto& p = bandParams[(size_t) band];
    const auto load = [] (const std::atomic<float>* value) { return value->load (std::memory_order_relaxed); };

    BandSettings s;
    s.active = load (p.used) > 0.5f && load (p.on) > 0.5f;
    s.type = (BandType) juce::jlimit (0, 6, (int) load (p.type));
    s.slopeIndex = juce::jlimit (0, (int) slopeChoices.size() - 1, (int) load (p.slope));
    s.stereo = (StereoMode) juce::jlimit (0, 4, (int) load (p.stereo));
    s.frequency = (double) load (p.freq);
    s.gainDb = (double) load (p.gain);
    s.q = (double) load (p.q);
    s.dynamic = load (p.dynamic) > 0.5f;
    s.thresholdDb = (double) load (p.threshold);
    s.rangeDb = (double) load (p.range);
    return s;
}

void IrisProcessor::prepareToPlay (double sampleRate, int)
{
    currentRate.store (sampleRate, std::memory_order_relaxed);

    // 控制率约 375 Hz，系数在两次控制之间逐采样线性过渡。
    tickInterval = juce::jmax (16, juce::roundToInt (sampleRate / 375.0));
    samplesToTick = 0;

    for (auto& band : bands)
        band.prepare (sampleRate, tickInterval);

    lookahead.prepare (getTotalNumOutputChannels(), (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate));
    outputGain.reset (sampleRate, 0.02);
    outputGain.setCurrentAndTargetValue ((double) mint::dbToGain (outputParam->load()));
    updateLookahead();
}

void IrisProcessor::updateLookahead() noexcept
{
    const auto samples = juce::roundToInt (lookaheadParam->load() * 0.001 * currentRate.load (std::memory_order_relaxed));
    lookahead.setDelaySamples (samples);
    requestLatencySamples (samples);
}

void IrisProcessor::tick() noexcept
{
    updateLookahead();
    outputGain.setTargetValue ((double) mint::dbToGain (outputParam->load()));

    const auto attack = (double) attackParam->load();
    const auto release = (double) releaseParam->load();
    for (int b = 0; b < numBands; ++b)
        bands[(size_t) b].tick (readBand (b), attack, release);
}

void IrisProcessor::pushSpectrum (SpectrumFifo& fifo, const double* const* audio, int channels, int n) noexcept
{
    std::array<float, maxChunk> mono;
    for (int i = 0; i < n; ++i)
        mono[(size_t) i] = (float) (channels > 1 ? 0.5 * (audio[0][i] + audio[1][i]) : audio[0][i]);
    fifo.push (mono.data(), n);
}

void IrisProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const auto channels = juce::jmin (2, buffer.getNumChannels());
    const auto total = buffer.getNumSamples();
    const auto wantSpectrum = spectrumWanted.load (std::memory_order_relaxed);

    std::array<std::array<double, maxChunk>, 2> storage;
    double* audio[2] = { storage[0].data(), storage[1].data() };

    auto position = 0;
    while (position < total)
    {
        if (samplesToTick == 0)
        {
            tick();
            samplesToTick = tickInterval;
        }

        const auto n = juce::jmin (maxChunk, samplesToTick, total - position);

        // 检测器读未延迟的输入，所以 Look-ahead 打开后增益变化会提前到达。
        for (int ch = 0; ch < channels; ++ch)
        {
            const auto* in = buffer.getReadPointer (ch, position);
            for (int i = 0; i < n; ++i)
                audio[ch][i] = (double) in[i];
        }

        for (auto& band : bands)
            band.detect (audio, channels, n);

        if (wantSpectrum)
            pushSpectrum (inputSpectrum, audio, channels, n);

        if (lookahead.getDelaySamples() > 0)
        {
            float* views[2] = { buffer.getWritePointer (0, position), channels > 1 ? buffer.getWritePointer (1, position) : nullptr };
            juce::AudioBuffer<float> view (views, channels, n);
            lookahead.process (view);
            for (int ch = 0; ch < channels; ++ch)
                for (int i = 0; i < n; ++i)
                    audio[ch][i] = (double) views[ch][i];
        }

        for (auto& band : bands)
            band.process (audio, channels, n);

        for (int i = 0; i < n; ++i)
        {
            const auto gain = outputGain.getNextValue();
            for (int ch = 0; ch < channels; ++ch)
                audio[ch][i] *= gain;
        }

        if (wantSpectrum)
            pushSpectrum (outputSpectrum, audio, channels, n);

        for (int ch = 0; ch < channels; ++ch)
        {
            auto* out = buffer.getWritePointer (ch, position);
            for (int i = 0; i < n; ++i)
                out[i] = (float) audio[ch][i];
        }

        position += n;
        samplesToTick -= n;
    }
}

} // namespace iris

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new iris::IrisProcessor();
}
