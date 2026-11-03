// 样板插件：演示共享基类、Look-ahead、电平表和界面控件的用法。新插件照此结构编写。
#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

namespace
{
    constexpr float maxLookaheadMs = 10.0f;

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "gain", 1 }, "Gain",
                                                           NormalisableRange<float> (-24.0f, 24.0f, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (v, 1) + " dB"; })));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "lookahead_ms", 1 }, "Look-ahead",
                                                           NormalisableRange<float> (0.0f, maxLookaheadMs, 0.1f), 0.0f));
        return layout;
    }

    class SampleProcessor final : public mint::Processor
    {
    public:
        SampleProcessor() : mint::Processor (createLayout())
        {
            gain = apvts.getRawParameterValue ("gain");
            lookaheadMs = apvts.getRawParameterValue ("lookahead_ms");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            currentRate = sampleRate;
            lookahead.prepare (getTotalNumOutputChannels(), (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate));
            smoothedGain.reset (sampleRate, 0.02);
            updateLookahead();
        }

        void releaseResources() override {}

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;
            updateLookahead();

            inputMeter.push (mint::gainToDb (buffer.getMagnitude (0, buffer.getNumSamples())));

            lookahead.process (buffer);

            smoothedGain.setTargetValue (mint::dbToGain (gain->load()));
            smoothedGain.applyGain (buffer, buffer.getNumSamples());
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter;
        std::atomic<float>* lookaheadMs = nullptr;

    private:
        void updateLookahead() noexcept
        {
            const auto samples = (int) std::round (lookaheadMs->load() * 0.001 * currentRate);
            lookahead.setDelaySamples (samples);
            requestLatencySamples (samples);
        }

        double currentRate = 48000.0;
        std::atomic<float>* gain = nullptr;
        mint::LookaheadDelay lookahead;
        juce::LinearSmoothedValue<float> smoothedGain { 1.0f };
    };

    class SampleEditor final : public mint::ui::EditorBase
    {
    public:
        explicit SampleEditor (SampleProcessor& p)
            : EditorBase (p, { "SAMPLE", { "Sample", "샘플", "样板" }, "0" }),
              owner (p),
              gainDial (p.apvts, "gain", { "Gain", "게인", "增益" }, true),
              meter (p.inputMeter, { "Input", "입력", "输入" })
        {
            addAndMakeVisible (gainDial);
            addAndMakeVisible (meter);
            setSize (width, height);
        }

        void resized() override
        {
            auto area = getContentBounds();
            meter.setBounds (area.removeFromLeft (40));
            gainDial.setBounds (area.withSizeKeepingCentre (224, 280));
        }

    private:
        juce::String getLookaheadText() const override
        {
            const auto ms = owner.lookaheadMs->load();
            return ms <= 0.0f ? mint::ui::tr ({ "Off · 0 ms", "끔 · 0 ms", "关闭 · 0 ms" }) : juce::String (ms, 1) + " ms";
        }

        SampleProcessor& owner;
        mint::ui::Dial gainDial;
        mint::ui::Meter meter;
    };

    juce::AudioProcessorEditor* SampleProcessor::createEditor()
    {
        return new SampleEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SampleProcessor();
}
