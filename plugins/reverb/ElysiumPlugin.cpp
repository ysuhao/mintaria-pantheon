// Mintaria Elysium：面向人声的算法混响，默认 0 延迟。音频处理全部在 ElysiumDsp.h。
#include "ElysiumDsp.h"

#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

namespace
{
    using juce::String;

    juce::NormalisableRange<float> logRange (float low, float high)
    {
        return { low, high,
                 [] (float a, float b, float v) { return a * std::pow (b / a, v); },
                 [] (float a, float b, float x) { return std::log (x / a) / std::log (b / a); },
                 [] (float, float, float x) { return x; } };
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;

        const auto addFloat = [&layout] (const char* id, const char* name, NormalisableRange<float> range, float defaultValue,
                                         std::function<String (float, int)> text)
        {
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, name, std::move (range), defaultValue,
                                                               AudioParameterFloatAttributes().withStringFromValueFunction (std::move (text))));
        };

        const auto percent = [] (float v, int) { return String (roundToInt (v)) + " %"; };
        const auto multiple = [] (float v, int) { return "x" + String (v, 2); };

        addFloat ("time", "Time", logRange (0.2f, 20.0f), 2.2f,
                  [] (float v, int) { return String (v, v < 10.0f ? 2 : 1) + " s"; });

        NormalisableRange<float> predelayRange (0.0f, 250.0f);
        predelayRange.setSkewForCentre (40.0f);
        addFloat ("predelay", "Pre-Delay", predelayRange, 0.0f, [] (float v, int) { return String (v, v < 10.0f ? 1 : 0) + " ms"; });

        addFloat ("mix", "Mix", NormalisableRange<float> (0.0f, 100.0f), 25.0f, percent);
        addFloat ("damping", "Damping", NormalisableRange<float> (0.0f, 100.0f), 50.0f, percent);
        addFloat ("low_decay", "Low Decay", logRange (0.25f, 2.5f), 1.0f, multiple);
        addFloat ("mid_decay", "Mid Decay", logRange (0.25f, 2.5f), 1.0f, multiple);
        addFloat ("crossover", "Crossover", logRange (100.0f, 2000.0f), 400.0f, [] (float v, int) { return String (roundToInt (v)) + " Hz"; });
        addFloat ("mod_depth", "Mod Depth", NormalisableRange<float> (0.0f, 100.0f), 30.0f, percent);

        const auto u8 = [] (const char* text) { return String::fromUTF8 (text); };
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "algorithm", 1 }, "Algorithm",
                                                            StringArray { u8 ("大厅"), u8 ("板式"), u8 ("房间"), u8 ("腔体") }, 0));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "character", 1 }, "Character",
                                                            StringArray { u8 ("12 位"), u8 ("24 位"), u8 ("现代") }, 1));
        return layout;
    }

    class ElysiumProcessor final : public mint::Processor
    {
    public:
        ElysiumProcessor() : mint::Processor (createLayout())
        {
            time = apvts.getRawParameterValue ("time");
            predelay = apvts.getRawParameterValue ("predelay");
            mix = apvts.getRawParameterValue ("mix");
            damping = apvts.getRawParameterValue ("damping");
            lowDecay = apvts.getRawParameterValue ("low_decay");
            midDecay = apvts.getRawParameterValue ("mid_decay");
            crossover = apvts.getRawParameterValue ("crossover");
            modDepth = apvts.getRawParameterValue ("mod_depth");
            algorithm = apvts.getRawParameterValue ("algorithm");
            character = apvts.getRawParameterValue ("character");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            reverb.prepare (sampleRate, readParams());
        }

        void releaseResources() override {}
        void reset() override                           { reverb.clear(); }

        double getTailLengthSeconds() const override    { return elysium::tailSeconds (readParams()); }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;
            const auto numSamples = buffer.getNumSamples();
            const auto numChannels = getTotalNumOutputChannels();
            if (numSamples == 0 || numChannels == 0)
                return;

            inputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));

            reverb.setParams (readParams());
            reverb.process (buffer.getWritePointer (0), numChannels > 1 ? buffer.getWritePointer (1) : nullptr, numSamples);

            outputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, outputMeter;

    private:
        elysium::Params readParams() const noexcept
        {
            elysium::Params p;
            p.timeSeconds = time->load();
            p.predelayMs = predelay->load();
            p.mix = mix->load() * 0.01f;
            p.damping = damping->load() * 0.01f;
            p.lowMultiplier = lowDecay->load();
            p.midMultiplier = midDecay->load();
            p.crossoverHz = crossover->load();
            p.modDepth = modDepth->load() * 0.01f;
            p.algorithm = juce::roundToInt (algorithm->load());
            p.character = juce::roundToInt (character->load());
            return p;
        }

        std::atomic<float>* time = nullptr;
        std::atomic<float>* predelay = nullptr;
        std::atomic<float>* mix = nullptr;
        std::atomic<float>* damping = nullptr;
        std::atomic<float>* lowDecay = nullptr;
        std::atomic<float>* midDecay = nullptr;
        std::atomic<float>* crossover = nullptr;
        std::atomic<float>* modDepth = nullptr;
        std::atomic<float>* algorithm = nullptr;
        std::atomic<float>* character = nullptr;

        elysium::Reverb reverb;
    };

    // 展开或收起高级区的开关。共享层没有这种控件，外观跟 Segmented 的单格一致。
    class AdvancedToggle final : public juce::Component
    {
    public:
        std::function<void()> onToggle;
        bool open = false;

        AdvancedToggle()
        {
            setRepaintsOnMouseActivity (true);
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = mint::ui::theme();
            const auto cell = getLocalBounds().toFloat();
            if (open)
            {
                g.setColour (t.ink);
                g.fillRect (cell);
            }
            else if (isMouseOver())
            {
                g.setColour (t.trimSoft.withAlpha (0.45f));
                g.fillRect (cell);
            }
            g.setColour (t.trim);
            g.drawRect (cell, 1.0f);
            g.setColour (open ? t.plate : t.ink);
            g.setFont (mint::ui::bodyFont (13.0f));
            g.drawText (mint::ui::tr ({ "Advanced", "고급", "高级" }), cell, juce::Justification::centred);
        }

        void mouseDown (const juce::MouseEvent&) override
        {
            if (onToggle)
                onToggle();
        }
    };

    class ElysiumEditor final : public mint::ui::EditorBase
    {
    public:
        explicit ElysiumEditor (ElysiumProcessor& p)
            : EditorBase (p, { "ELYSIUM", { "Reverb", "리버브", "混响" }, "VII" }),
              owner (p),
              mixDial (p.apvts, "mix", { "Mix", "믹스", "混合" }, true),
              timeDial (p.apvts, "time", { "Time", "타임", "时间" }),
              predelayDial (p.apvts, "predelay", { "Pre-delay", "프리딜레이", "预延迟" }),
              dampingDial (p.apvts, "damping", { "Damping", "댐핑", "高频阻尼" }),
              lowDial (p.apvts, "low_decay", { "Low Decay", "저역 디케이", "低频衰减" }),
              midDial (p.apvts, "mid_decay", { "Mid Decay", "중역 디케이", "中频衰减" }),
              crossoverDial (p.apvts, "crossover", { "Crossover", "크로스오버", "分频点" }),
              modDial (p.apvts, "mod_depth", { "Mod Depth", "모듈레이션 뎁스", "调制深度" }),
              algorithmChoice (p.apvts, "algorithm",
                              { { "Hall", "홀", "大厅" }, { "Plate", "플레이트", "板式" }, { "Room", "룸", "房间" }, { "Chamber", "챔버", "腔体" } }),
              characterChoice (p.apvts, "character",
                                { { "12-bit", "12비트", "12 位" }, { "24-bit", "24비트", "24 位" }, { "Modern", "모던", "现代" } }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              outputMeter (p.outputMeter, { "Output", "출력", "输出" })
        {
            for (auto* c : std::initializer_list<juce::Component*> { &mixDial, &timeDial, &predelayDial, &dampingDial,
                                                                     &lowDial, &midDial, &crossoverDial, &modDial,
                                                                     &algorithmChoice, &characterChoice, &inputMeter, &outputMeter, &toggle })
                addAndMakeVisible (*c);

            // 环境变量只给 mint_probe shot 截高级区用。正常使用时开关状态跟工程一起保存。
            setAdvancedOpen ((bool) owner.apvts.state.getProperty ("advanced_open", false)
                             || juce::SystemStats::getEnvironmentVariable ("ELYSIUM_SHOW_ADVANCED", {}).isNotEmpty());
            toggle.onToggle = [this]
            {
                setAdvancedOpen (! toggle.open);
                owner.apvts.state.setProperty ("advanced_open", toggle.open, nullptr);
            };

            setSize (width, height);
        }

        void paint (juce::Graphics& g) override
        {
            EditorBase::paint (g);

            g.setColour (mint::ui::theme().inkMuted);
            g.setFont (mint::ui::bodyFont (12.0f));
            g.drawText (mint::ui::tr ({ "Algorithm", "알고리즘", "算法" }), algorithmCaption, juce::Justification::centredLeft);
            g.drawText (mint::ui::tr ({ "Character", "캐릭터", "质感" }), characterCaption, juce::Justification::centredLeft);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeter.setBounds (area.removeFromLeft (40));
            outputMeter.setBounds (area.removeFromRight (40));
            area.reduce (12, 0);

            mixDial.setBounds (area.removeFromLeft (224).withSizeKeepingCentre (224, 280));
            area.removeFromLeft (12);

            // 小旋钮的圆心和大旋钮对齐。
            const auto row = area.withSizeKeepingCentre (area.getWidth(), 180).translated (0, -23);

            auto mainRow = row;
            const auto mainWidth = mainRow.getWidth() / 3;
            timeDial.setBounds (mainRow.removeFromLeft (mainWidth));
            predelayDial.setBounds (mainRow.removeFromLeft (mainWidth));
            dampingDial.setBounds (mainRow);

            auto advancedRow = row;
            const auto advancedWidth = advancedRow.getWidth() / 4;
            lowDial.setBounds (advancedRow.removeFromLeft (advancedWidth));
            midDial.setBounds (advancedRow.removeFromLeft (advancedWidth));
            crossoverDial.setBounds (advancedRow.removeFromLeft (advancedWidth));
            modDial.setBounds (advancedRow);

            // 组标题的基线跟左边的"插件延迟"对齐。
            auto footer = getFooterSlot().withTrimmedTop (-2);
            const auto layoutChoice = [&footer] (juce::Component& control, juce::Rectangle<int>& caption, int choiceWidth)
            {
                auto slot = footer.removeFromLeft (choiceWidth);
                caption = slot.removeFromTop (16);
                control.setBounds (slot.removeFromTop (28));
                footer.removeFromLeft (8);
            };

            layoutChoice (algorithmChoice, algorithmCaption, 196);
            layoutChoice (characterChoice, characterCaption, 146);
            toggle.setBounds (footer.withTrimmedTop (16).withHeight (28));
        }

    private:
        void setAdvancedOpen (bool shouldOpen)
        {
            toggle.open = shouldOpen;
            toggle.repaint();
            for (auto* c : std::initializer_list<juce::Component*> { &timeDial, &predelayDial, &dampingDial })
                c->setVisible (! shouldOpen);
            for (auto* c : std::initializer_list<juce::Component*> { &lowDial, &midDial, &crossoverDial, &modDial })
                c->setVisible (shouldOpen);
        }

        ElysiumProcessor& owner;
        mint::ui::Dial mixDial, timeDial, predelayDial, dampingDial, lowDial, midDial, crossoverDial, modDial;
        mint::ui::Segmented algorithmChoice, characterChoice;
        mint::ui::Meter inputMeter, outputMeter;
        AdvancedToggle toggle;
        juce::Rectangle<int> algorithmCaption, characterCaption;
    };

    juce::AudioProcessorEditor* ElysiumProcessor::createEditor()
    {
        return new ElysiumEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ElysiumProcessor();
}
