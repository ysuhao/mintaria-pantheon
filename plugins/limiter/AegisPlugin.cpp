// Mintaria Aegis：链路最后一级的限制器，防止直播爆音。
#include "AegisDsp.h"

#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

#include <array>

namespace
{
    constexpr float maxLookaheadMs = 10.0f;
    constexpr std::array<float, 4> lookaheadChoicesMs { 0.0f, 1.5f, 5.0f, 10.0f };
    constexpr int chunkSize = 128;
    constexpr double switchFadeSeconds = 0.003;

    // 拦截坏数据和离谱的电平，避免乘阈值增益后变成 Inf。
    inline float sanitise (float x) noexcept
    {
        return std::isfinite (x) ? juce::jlimit (-1.0e6f, 1.0e6f, x) : 0.0f;
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        const auto decibels = [] (float v, int) { return String (v, 1) + " dB"; };

        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "threshold", 1 }, "Threshold",
                                                           NormalisableRange<float> (-30.0f, 0.0f, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction (decibels)));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "ceiling", 1 }, "Ceiling",
                                                           NormalisableRange<float> (-30.0f, 0.0f, 0.1f), -1.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction (decibels)));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "release", 1 }, "Release",
                                                           NormalisableRange<float> (1.0f, 1000.0f, 0.1f, 0.3f), 60.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms")
                                                               .withStringFromValueFunction ([] (float v, int)
                                                               {
                                                                   return String (v, v < 10.0f ? 1 : 0) + " ms";
                                                               })));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "auto_release", 1 }, "Auto Release",
                                                            StringArray { String::fromUTF8 ("关"), String::fromUTF8 ("开") }, 1));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "lookahead_ms", 1 }, "Look-ahead",
                                                            StringArray { "0 ms", "1.5 ms", "5 ms", "10 ms" }, 1));
        return layout;
    }

    class AegisProcessor final : public mint::Processor
    {
    public:
        AegisProcessor() : mint::Processor (createLayout())
        {
            threshold = apvts.getRawParameterValue ("threshold");
            ceiling = apvts.getRawParameterValue ("ceiling");
            release = apvts.getRawParameterValue ("release");
            autoRelease = apvts.getRawParameterValue ("auto_release");
            lookaheadChoice = apvts.getRawParameterValue ("lookahead_ms");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            currentRate = sampleRate;
            const auto maxSamples = (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate);
            delay.prepare (getTotalNumOutputChannels(), maxSamples);
            limiter.prepare (sampleRate, maxSamples);
            limiter.setRelease (release->load(), autoRelease->load() > 0.5f);

            inputGain.reset (sampleRate, 0.02);
            inputGain.setCurrentAndTargetValue (mint::dbToGain (-threshold->load()));
            outputCeiling.reset (sampleRate, 0.01);
            outputCeiling.setCurrentAndTargetValue (mint::dbToGain (ceiling->load()));

            fadeStep = (float) (1.0 / (switchFadeSeconds * sampleRate));
            applyLookahead (wantedLookahead());
            // 默认就有延迟，宿主激活时读到的必须已是真值，不能等定时器。
            setLatencySamples (currentLookahead);
            fade = 1.0f;
            fadingOut = false;
            muteSamples = 0;
        }

        void releaseResources() override {}

        void reset() override
        {
            delay.reset();
            limiter.reset();
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;

            limiter.setRelease (release->load(), autoRelease->load() > 0.5f);
            inputGain.setTargetValue (mint::dbToGain (-threshold->load()));
            outputCeiling.setTargetValue (mint::dbToGain (ceiling->load()));

            if (wantedLookahead() != currentLookahead)
                fadingOut = true;

            const auto numChannels = juce::jmin (buffer.getNumChannels(), getTotalNumOutputChannels());
            auto inputPeak = 0.0f, minGain = 1.0f;

            for (int start = 0; start < buffer.getNumSamples(); start += chunkSize)
            {
                const auto n = juce::jmin (chunkSize, buffer.getNumSamples() - start);
                juce::AudioBuffer<float> chunk (buffer.getArrayOfWritePointers(), numChannels, start, n);

                inputPeak = juce::jmax (inputPeak, computeGains (chunk, minGain));
                delay.process (chunk);
                writeOutput (chunk);

                // 整段静音之后才切换前瞻长度，延迟线的跳变不会被听到。
                if (fadingOut && fade <= 0.0f)
                {
                    applyLookahead (wantedLookahead());
                    fadingOut = false;
                    muteSamples = currentLookahead;
                }
            }

            inputMeter.push (mint::gainToDb (inputPeak));
            // 共享电平表的衰减模式把 -24…0 映射成不亮到全亮，这里把衰减量 R 平移成 R - 24。
            if (minGain < 1.0f)
                reductionMeter.push (-mint::gainToDb (minGain) - 24.0f);
        }

        // 旁通也要保持上报的延迟，否则切换旁通时整条链路错位。
        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            delay.process (buffer);
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, reductionMeter;

    private:
        float getLookaheadMs() const noexcept
        {
            return lookaheadChoicesMs[(size_t) juce::jlimit (0, (int) lookaheadChoicesMs.size() - 1,
                                                             juce::roundToInt (lookaheadChoice->load()))];
        }

        int wantedLookahead() const noexcept    { return (int) std::round (getLookaheadMs() * 0.001 * currentRate); }

        void applyLookahead (int samples) noexcept
        {
            currentLookahead = samples;
            delay.setDelaySamples (samples);
            delay.reset();
            limiter.setLookahead (samples);
            requestLatencySamples (samples);
        }

        // 乘阈值增益（阈值降低多少，整体就抬高多少），算出每个采样的增益。返回未乘增益时的输入峰值。
        float computeGains (juce::AudioBuffer<float>& chunk, float& minGain) noexcept
        {
            const auto numChannels = chunk.getNumChannels();
            auto inputPeak = 0.0f;

            for (int i = 0; i < chunk.getNumSamples(); ++i)
            {
                const auto boost = inputGain.getNextValue();
                auto rawPeak = 0.0f;

                for (int ch = 0; ch < numChannels; ++ch)
                {
                    auto* data = chunk.getWritePointer (ch);
                    const auto x = sanitise (data[i]);
                    rawPeak = juce::jmax (rawPeak, std::abs (x));
                    data[i] = x * boost;
                }

                inputPeak = juce::jmax (inputPeak, rawPeak);
                gains[(size_t) i] = limiter.process (rawPeak * boost);
                minGain = juce::jmin (minGain, gains[(size_t) i]);
            }

            return inputPeak;
        }

        void writeOutput (juce::AudioBuffer<float>& chunk) noexcept
        {
            const auto numChannels = chunk.getNumChannels();
            const auto clipStage = currentLookahead == 0;

            for (int i = 0; i < chunk.getNumSamples(); ++i)
            {
                const auto limit = outputCeiling.getNextValue();
                const auto scale = limit * nextFade();

                for (int ch = 0; ch < numChannels; ++ch)
                {
                    auto* data = chunk.getWritePointer (ch);
                    const auto limited = data[i] * gains[(size_t) i];
                    const auto shaped = clipStage ? aegis::softClip (limited) : limited;
                    // 最后一道钳位只负责吃掉浮点舍入，保证硬上限。
                    data[i] = juce::jlimit (-limit, limit, shaped * scale);
                }
            }
        }

        float nextFade() noexcept
        {
            if (fadingOut)
                fade = juce::jmax (0.0f, fade - fadeStep);
            else if (muteSamples > 0)
                --muteSamples;
            else if (fade < 1.0f)
                fade = juce::jmin (1.0f, fade + fadeStep);
            return fade;
        }

        std::atomic<float>* threshold = nullptr;
        std::atomic<float>* ceiling = nullptr;
        std::atomic<float>* release = nullptr;
        std::atomic<float>* autoRelease = nullptr;
        std::atomic<float>* lookaheadChoice = nullptr;

        double currentRate = 48000.0;
        int currentLookahead = 0, muteSamples = 0;
        float fade = 1.0f, fadeStep = 0.01f;
        bool fadingOut = false;

        aegis::Limiter limiter;
        mint::LookaheadDelay delay;
        juce::LinearSmoothedValue<float> inputGain { 1.0f }, outputCeiling { 1.0f };
        std::array<float, chunkSize> gains {};
    };

    class AegisEditor final : public mint::ui::EditorBase
    {
    public:
        explicit AegisEditor (AegisProcessor& p)
            : EditorBase (p, { "AEGIS", { "Limiter", "리미터", "限制器" }, "X" }),
              thresholdDial (p.apvts, "threshold", { "Threshold", "스레숄드", "阈值" }, true),
              ceilingDial (p.apvts, "ceiling", { "Ceiling", "실링", "输出上限" }),
              releaseDial (p.apvts, "release", { "Release", "릴리즈", "释放" }),
              // 只改显示文字，参数的选项名（"关"/"开"、"0 ms" 等）保持原样。
              autoSwitch (p.apvts, "auto_release", { { "Off", "끔", "关闭" }, { "On", "켬", "开启" } }),
              lookaheadSwitch (p.apvts, "lookahead_ms", { "0 ms", "1.5 ms", "5 ms", "10 ms" }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              reductionMeter (p.reductionMeter, { "Reduction", "리덕션", "增益衰减" }, true),
              autoAttachment (*p.apvts.getParameter ("auto_release"), [this] (float isOn)
              {
                  // 自动释放接管时释放旋钮不起作用，调暗并禁用。
                  const auto isManual = isOn <= 0.5f;
                  releaseDial.setEnabled (isManual);
                  // 共享 Dial 的双击数值输入不看 isEnabled，鼠标命中也要一起关掉。
                  releaseDial.setInterceptsMouseClicks (isManual, isManual);
                  releaseDial.setAlpha (isManual ? 1.0f : 0.45f);
              }, nullptr)
        {
            autoAttachment.sendInitialUpdate();
            for (auto* child : std::initializer_list<juce::Component*> { &thresholdDial, &ceilingDial, &releaseDial, &autoSwitch,
                                                                         &lookaheadSwitch, &inputMeter, &reductionMeter })
                addAndMakeVisible (child);
            setSize (width, height);
        }

        void paint (juce::Graphics& g) override
        {
            EditorBase::paint (g);

            g.setColour (mint::ui::theme().inkMuted);
            g.setFont (mint::ui::bodyFont (12.0f));
            g.drawText (mint::ui::tr ({ "Auto Release", "자동 릴리즈", "自动释放" }), autoLabel, juce::Justification::centred);
            g.drawText (mint::ui::tr ({ "Look-ahead", "룩어헤드", "Look-ahead" }), lookaheadLabel, juce::Justification::centred);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeter.setBounds (area.removeFromLeft (meterWidth));
            reductionMeter.setBounds (area.removeFromRight (meterWidth));

            thresholdDial.setBounds (area.withSizeKeepingCentre (224, 280));

            const auto sideWidth = (area.getWidth() - 224) / 2;
            ceilingDial.setBounds (area.removeFromLeft (sideWidth).withSizeKeepingCentre (150, 190));

            auto releaseColumn = area.removeFromRight (sideWidth).withSizeKeepingCentre (150, 250);
            releaseDial.setBounds (releaseColumn.removeFromTop (190));
            releaseColumn.removeFromTop (8);
            autoLabel = releaseColumn.removeFromTop (16);
            autoSwitch.setBounds (releaseColumn.withSizeKeepingCentre (96, 26));

            auto footer = getFooterSlot();
            lookaheadLabel = footer.removeFromTop (16);
            lookaheadSwitch.setBounds (footer.withSizeKeepingCentre (288, 26));
        }

    private:
        // 标签最长的是 Reduction，放得下就不会被压缩。
        static constexpr int meterWidth = 64;

        mint::ui::Dial thresholdDial, ceilingDial, releaseDial;
        mint::ui::Segmented autoSwitch, lookaheadSwitch;
        mint::ui::Meter inputMeter, reductionMeter;
        juce::ParameterAttachment autoAttachment;
        juce::Rectangle<int> autoLabel, lookaheadLabel;
    };

    juce::AudioProcessorEditor* AegisProcessor::createEditor()
    {
        return new AegisEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AegisProcessor();
}
