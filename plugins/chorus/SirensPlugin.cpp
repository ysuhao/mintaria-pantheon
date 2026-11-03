// Mintaria Sirens：合唱。四个可组合的模式按钮是主角，其余只有干湿和两个小开关。
#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

#include "SirensDsp.h"

namespace
{
    const std::array<const char*, 4> modeIds { "mode1", "mode2", "mode3", "mode4" };
    const std::array<const char*, 4> modeNames { "Mode I", "Mode II", "Mode III", "Mode IV" };
    const std::array<const char*, 4> modeNumerals { "I", "II", "III", "IV" };
    // 按实测宽度排序：单开时侧中比 -10.6、-9.7、-8.7、-7.9 dB，见 NOTES.md。
    const std::array<mint::ui::Text, 4> modeDescriptions { mint::ui::Text ("Light", "가볍게", "轻"),
                                                           mint::ui::Text ("Wide", "넓게", "宽"),
                                                           mint::ui::Text ("Wider", "더 넓게", "更宽"),
                                                           mint::ui::Text ("Widest", "가장 넓게", "最宽") };

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;
        for (size_t i = 0; i < modeIds.size(); ++i)
            layout.add (std::make_unique<AudioParameterBool> (ParameterID { modeIds[i], 1 }, modeNames[i], i == 0));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "mix", 1 }, "Mix",
                                                           NormalisableRange<float> (0.0f, 100.0f, 1.0f), 40.0f,
                                                           AudioParameterFloatAttributes().withLabel ("%")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (roundToInt (v)) + " %"; })));
        layout.add (std::make_unique<AudioParameterBool> (ParameterID { "mono_compat", 1 }, "Mono Compat", false));
        layout.add (std::make_unique<AudioParameterBool> (ParameterID { "noise", 1 }, "Noise", false));
        return layout;
    }

    class SirensProcessor final : public mint::Processor
    {
    public:
        SirensProcessor() : mint::Processor (createLayout())
        {
            for (size_t i = 0; i < modeIds.size(); ++i)
                modes[i] = apvts.getRawParameterValue (modeIds[i]);
            mix = apvts.getRawParameterValue ("mix");
            monoCompat = apvts.getRawParameterValue ("mono_compat");
            noise = apvts.getRawParameterValue ("noise");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            engine.prepare (sampleRate);
        }

        void releaseResources() override {}

        void reset() override     { engine.reset(); }

        double getTailLengthSeconds() const override     { return sirens::maxDelayMs * 0.001; }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;

            const auto numSamples = buffer.getNumSamples();
            if (numSamples == 0 || buffer.getNumChannels() == 0)
                return;

            inputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));

            sirens::Engine::Targets targets;
            for (size_t i = 0; i < modes.size(); ++i)
                targets.modes[i] = modes[i]->load() > 0.5f;
            targets.mix = mix->load() * 0.01f;
            targets.monoCompat = monoCompat->load() > 0.5f;
            targets.noise = noise->load() > 0.5f;

            auto* left = buffer.getWritePointer (0);
            auto* right = getTotalNumOutputChannels() > 1 && buffer.getNumChannels() > 1 ? buffer.getWritePointer (1) : nullptr;
            engine.process (left, right, numSamples, targets);

            outputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, outputMeter;

    private:
        sirens::Engine engine;
        std::array<std::atomic<float>*, 4> modes {};
        std::atomic<float>* mix = nullptr;
        std::atomic<float>* monoCompat = nullptr;
        std::atomic<float>* noise = nullptr;
    };

    // 绑定开关参数的控件基类：点一下翻转。共享层的 Segmented 是单选，组合模式用不了。
    class LatchControl : public juce::Component
    {
    public:
        LatchControl (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, mint::ui::Text text)
            : label (std::move (text)),
              attachment (*state.getParameter (paramId), [this] (float value)
              {
                  on = value > 0.5f;
                  repaint();
              }, nullptr)
        {
            setRepaintsOnMouseActivity (true);
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
            attachment.sendInitialUpdate();
        }

        // 无障碍标题随界面语言变化，由编辑器在 languageChanged() 里调用。
        virtual void refreshTitle()
        {
            setTitle (mint::ui::tr (label));
        }

        void mouseDown (const juce::MouseEvent&) override
        {
            attachment.setValueAsCompleteGesture (on ? 0.0f : 1.0f);
        }

    protected:
        mint::ui::Text label;
        bool on = false;

    private:
        juce::ParameterAttachment attachment;
    };

    // 模式卡：可以同时按下几张。下方一行短说明，让人不试听也知道区别。
    class ModeCard final : public LatchControl
    {
    public:
        ModeCard (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const char* numeral, mint::ui::Text descriptionText)
            : LatchControl (state, paramId, numeral),
              description (std::move (descriptionText))
        {
            refreshTitle();
        }

        void refreshTitle() override
        {
            LatchControl::refreshTitle();
            setDescription (mint::ui::tr (description));
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = mint::ui::theme();
            const auto bounds = getLocalBounds().toFloat();

            g.setColour (on ? t.ink : t.plate);
            g.fillRect (bounds);

            g.setColour (isMouseOver() && ! on ? t.accent : t.ink);
            g.drawRect (bounds, 1.0f);

            auto content = getLocalBounds().reduced (14, 0);

            g.setColour (on ? t.plate : t.ink);
            g.setFont (mint::ui::headingFont (38.0f));
            g.drawText (mint::ui::tr (label), content.removeFromTop (68), juce::Justification::centred);

            // 按下时亮起的黄铜细线，从远处也能一眼看清哪几张是开的。
            g.setColour (on ? t.trim : t.trimSoft);
            g.fillRect (content.removeFromTop (2));

            g.setColour (on ? t.plate : t.inkMuted);
            g.setFont (mint::ui::bodyFont (14.0f));
            g.drawFittedText (mint::ui::tr (description), content, juce::Justification::centred, 1, 0.8f);
        }

    private:
        mint::ui::Text description;
    };

    // 独立的小开关：圆角外形加状态点，和方形的模式卡、单选的分段按钮都不像。
    class StatusSwitch final : public LatchControl
    {
    public:
        StatusSwitch (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, mint::ui::Text text)
            : LatchControl (state, paramId, std::move (text))
        {
            refreshTitle();
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = mint::ui::theme();
            const auto bounds = getLocalBounds().toFloat().reduced (0.5f);
            const auto radius = bounds.getHeight() * 0.5f;

            g.setColour (t.plate);
            g.fillRoundedRectangle (bounds, radius);

            g.setColour (isMouseOver() ? t.accent : (on ? t.ink : t.trim));
            g.drawRoundedRectangle (bounds, radius, 1.0f);

            const auto dotSize = 10.0f;
            const auto dot = juce::Rectangle<float> (dotSize, dotSize).withCentre ({ bounds.getX() + radius + 2.0f, bounds.getCentreY() });
            if (on)
            {
                g.setColour (t.accent);
                g.fillEllipse (dot);
            }
            g.setColour (on ? t.accentEnd : t.inkMuted);
            g.drawEllipse (dot, 1.0f);

            g.setColour (on ? t.ink : t.inkMuted);
            g.setFont (mint::ui::bodyFont (14.0f));
            g.drawFittedText (mint::ui::tr (label), getLocalBounds().withTrimmedLeft ((int) (radius * 2.0f) + 6).withTrimmedRight ((int) radius),
                              juce::Justification::centredLeft, 1, 0.8f);
        }
    };

    class SirensEditor final : public mint::ui::EditorBase
    {
    public:
        explicit SirensEditor (SirensProcessor& p)
            : EditorBase (p, { "SIRENS", { "Chorus", "코러스", "合唱" }, "IX" }),
              mixDial (p.apvts, "mix", { "Mix", "믹스", "混合" }, true),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              outputMeter (p.outputMeter, { "Output", "출력", "输出" }),
              compatSwitch (p.apvts, "mono_compat", { "Mono Compat", "모노 호환", "单声道兼容" }),
              noiseSwitch (p.apvts, "noise", { "Noise", "노이즈", "底噪" })
        {
            for (size_t i = 0; i < modeIds.size(); ++i)
            {
                modeCards[i] = std::make_unique<ModeCard> (p.apvts, modeIds[i], modeNumerals[i], modeDescriptions[i]);
                addAndMakeVisible (*modeCards[i]);
            }
            addAndMakeVisible (mixDial);
            addAndMakeVisible (inputMeter);
            addAndMakeVisible (outputMeter);
            addAndMakeVisible (compatSwitch);
            addAndMakeVisible (noiseSwitch);
            setSize (width, height);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeter.setBounds (area.removeFromLeft (40));
            outputMeter.setBounds (area.removeFromRight (40));
            area.reduce (24, 0);

            auto row = area.removeFromTop (106);
            const auto gap = 14;
            const auto cardWidth = (row.getWidth() - gap * 3) / 4;
            for (auto& card : modeCards)
            {
                card->setBounds (row.removeFromLeft (cardWidth));
                row.removeFromLeft (gap);
            }

            area.removeFromTop (8);
            mixDial.setBounds (area.withSizeKeepingCentre (176, area.getHeight()));

            const auto switchWidth = 172, switchGap = 16;
            auto footer = getFooterSlot().withSizeKeepingCentre (switchWidth * 2 + switchGap, 32);
            compatSwitch.setBounds (footer.removeFromLeft (switchWidth));
            noiseSwitch.setBounds (footer.removeFromRight (switchWidth));
        }

    protected:
        void languageChanged() override
        {
            for (auto& card : modeCards)
                card->refreshTitle();
            compatSwitch.refreshTitle();
            noiseSwitch.refreshTitle();
        }

    private:
        std::array<std::unique_ptr<ModeCard>, 4> modeCards;
        mint::ui::Dial mixDial;
        mint::ui::Meter inputMeter, outputMeter;
        StatusSwitch compatSwitch, noiseSwitch;
    };

    juce::AudioProcessorEditor* SirensProcessor::createEditor()
    {
        return new SirensEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SirensProcessor();
}
