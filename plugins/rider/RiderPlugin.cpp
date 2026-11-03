// Mintaria Helios：人声自动推子。把人声段的短时电平推向目标，静音和呼吸段保持增益不动。
#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

#include "RiderDsp.h"
#include "RiderUi.h"

namespace
{
    constexpr float maxLookaheadMs = 10.0f;
    constexpr int chunkSize = 512;

    juce::String utf8 (const char* text)    { return juce::String::fromUTF8 (text); }

    std::unique_ptr<juce::AudioParameterFloat> dbParam (const char* id, const char* name, float low, float high, float initial, bool showSign, float step = 0.1f)
    {
        using namespace juce;
        return std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, name,
                                                      NormalisableRange<float> (low, high, step), initial,
                                                      AudioParameterFloatAttributes().withLabel ("dB")
                                                          .withStringFromValueFunction ([showSign] (float v, int)
                                                          {
                                                              return (showSign && v > 0.0f ? "+" : "") + String (v, 1) + " dB";
                                                          }));
    }

    std::unique_ptr<juce::AudioParameterFloat> percentParam (const char* id, const char* name, float initial)
    {
        using namespace juce;
        return std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, name,
                                                      NormalisableRange<float> (0.0f, 100.0f, 1.0f), initial,
                                                      AudioParameterFloatAttributes().withLabel ("%")
                                                          .withStringFromValueFunction ([] (float v, int) { return String (juce::roundToInt (v)) + " %"; }));
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add (dbParam ("target_db", "Target", -36.0f, -6.0f, -18.0f, false));
        layout.add (dbParam ("range_up_db", "Range Up", 0.0f, 24.0f, 6.0f, true));
        layout.add (dbParam ("range_down_db", "Range Down", 0.0f, 24.0f, 6.0f, true));
        layout.add (percentParam ("speed", "Speed", 50.0f));
        layout.add (percentParam ("gate_amount", "Vocal Gate", 50.0f));
        layout.add (percentParam ("music_follow", "Music Follow", 100.0f));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "write_auto", 1 }, "Write Automation",
                                                            StringArray { utf8 ("关"), utf8 ("写入") }, 0));
        layout.add (dbParam ("ride_gain", "Ride Gain", -24.0f, 24.0f, 0.0f, true, 0.01f));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "lookahead_ms", 1 }, "Look-ahead",
                                                           NormalisableRange<float> (0.0f, maxLookaheadMs, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (v, 1) + " ms"; })));
        return layout;
    }

    // 把当前增益写到 ride_gain。只在消息线程调用宿主通知。
    class AutomationWriter final : private juce::Timer
    {
    public:
        AutomationWriter (juce::RangedAudioParameter& rideGain, std::atomic<float>& writeSwitch,
                          std::atomic<float>& currentGainDb, std::function<bool()> canWrite)
            : parameter (rideGain), enabled (writeSwitch), gain (currentGainDb), allowed (std::move (canWrite))
        {
            startTimerHz (30);
        }

        ~AutomationWriter() override
        {
            stopTimer();
            setGesture (false);
        }

    private:
        void timerCallback() override
        {
            const auto writing = enabled.load (std::memory_order_relaxed) > 0.5f && allowed();
            setGesture (writing);
            if (! writing)
                return;

            const auto value = parameter.convertTo0to1 (gain.load (std::memory_order_relaxed));
            if (std::abs (value - parameter.getValue()) > minimumStep)
                parameter.setValueNotifyingHost (value);
        }

        void setGesture (bool shouldBeOpen)
        {
            if (shouldBeOpen == gestureOpen)
                return;
            gestureOpen = shouldBeOpen;
            if (shouldBeOpen)
                parameter.beginChangeGesture();
            else
                parameter.endChangeGesture();
        }

        static constexpr float minimumStep = 0.01f / 48.0f;     // 0.01 dB

        juce::RangedAudioParameter& parameter;
        std::atomic<float>& enabled;
        std::atomic<float>& gain;
        std::function<bool()> allowed;
        bool gestureOpen = false;
    };

    class RiderProcessor final : public mint::Processor
    {
    public:
        RiderProcessor()
            : mint::Processor (createLayout(), stereoBusesWithSidechain()),
              writer (*apvts.getParameter ("ride_gain"), *apvts.getRawParameterValue ("write_auto"),
                      telemetry.gainDb, [this] { return ! isNonRealtime(); })
        {
            target = apvts.getRawParameterValue ("target_db");
            rangeUp = apvts.getRawParameterValue ("range_up_db");
            rangeDown = apvts.getRawParameterValue ("range_down_db");
            speed = apvts.getRawParameterValue ("speed");
            gateAmount = apvts.getRawParameterValue ("gate_amount");
            musicFollow = apvts.getRawParameterValue ("music_follow");
            lookaheadMs = apvts.getRawParameterValue ("lookahead_ms");
        }

        bool isBusesLayoutSupported (const BusesLayout& layouts) const override
        {
            if (! mint::Processor::isBusesLayoutSupported (layouts))
                return false;
            if (layouts.inputBuses.size() < 2)
                return true;
            const auto sidechain = layouts.getChannelSet (true, 1);
            return sidechain.isDisabled() || sidechain == juce::AudioChannelSet::mono() || sidechain == juce::AudioChannelSet::stereo();
        }

        void prepareToPlay (double sampleRate, int) override
        {
            currentRate = sampleRate;
            lookahead.prepare (getMainBusNumOutputChannels(), (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate));
            engine.applySettings (readSettings());
            engine.prepare (sampleRate);
            updateLookahead();
        }

        void releaseResources() override {}

        void reset() override
        {
            engine.reset();
            lookahead.reset();
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;
            updateLookahead();
            engine.applySettings (readSettings());

            auto mainBus = getBusBuffer (buffer, false, 0);
            const auto sidechain = readSidechain (buffer);
            const auto numChannels = mainBus.getNumChannels();
            const auto numSamples = mainBus.getNumSamples();
            if (numChannels == 0)
                return;

            inputMeter.push (mint::gainToDb (mainBus.getMagnitude (0, numSamples)));

            // 检测始终看未延迟的输入，增益加在延迟后的音频上。Look-ahead 就来自这个时间差。
            for (int start = 0; start < numSamples; start += chunkSize)
            {
                const auto count = juce::jmin (chunkSize, numSamples - start);
                const auto* left = mainBus.getReadPointer (0, start);
                const auto* right = mainBus.getReadPointer (numChannels > 1 ? 1 : 0, start);

                for (int i = 0; i < count; ++i)
                    gains[(size_t) i] = engine.processFrame (left[i], right[i], sidechainPowerAt (sidechain, start + i));

                juce::AudioBuffer<float> chunk (mainBus.getArrayOfWritePointers(), numChannels, start, count);
                lookahead.process (chunk);

                for (int ch = 0; ch < numChannels; ++ch)
                    juce::FloatVectorOperations::multiply (chunk.getWritePointer (ch), gains.data(), count);
            }

            outputMeter.push (mint::gainToDb (mainBus.getMagnitude (0, numSamples)));

            telemetry.gainDb.store (engine.getGainDb(), std::memory_order_relaxed);
            telemetry.gateOpen.store (engine.isGateOpen(), std::memory_order_relaxed);
            telemetry.effectiveTargetDb.store (engine.getEffectiveTargetDb(), std::memory_order_relaxed);
            telemetry.musicDb.store (engine.getMusicDb(), std::memory_order_relaxed);
            telemetry.sidechainConnected.store (sidechain.channels > 0, std::memory_order_relaxed);
        }

        // 旁路时仍走延迟线，保持与宿主延迟补偿对齐。
        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            auto mainBus = getBusBuffer (buffer, false, 0);
            lookahead.process (mainBus);
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, outputMeter;
        helios::RiderTelemetry telemetry;
        std::atomic<float>* lookaheadMs = nullptr;

    private:
        struct SidechainView
        {
            const float* left = nullptr;
            const float* right = nullptr;
            int channels = 0;
        };

        SidechainView readSidechain (juce::AudioBuffer<float>& buffer) const noexcept
        {
            SidechainView view;
            if (getBusCount (true) < 2 || ! getBus (true, 1)->isEnabled())
                return view;

            auto bus = getBusBuffer (buffer, true, 1);
            view.channels = bus.getNumChannels();
            if (view.channels > 0)
            {
                view.left = bus.getReadPointer (0);
                view.right = bus.getReadPointer (view.channels > 1 ? 1 : 0);
            }
            return view;
        }

        static double sidechainPowerAt (const SidechainView& view, int index) noexcept
        {
            if (view.channels == 0)
                return 0.0;
            const double l = view.left[index], r = view.right[index];
            return 0.5 * (l * l + r * r);
        }

        helios::RiderSettings readSettings() const noexcept
        {
            helios::RiderSettings s;
            s.targetDb = target->load();
            s.rangeUpDb = rangeUp->load();
            s.rangeDownDb = rangeDown->load();
            s.speed = speed->load() * 0.01f;
            s.gateAmount = gateAmount->load() * 0.01f;
            s.follow = musicFollow->load() * 0.01f;
            return s;
        }

        void updateLookahead() noexcept
        {
            const auto samples = (int) std::round (lookaheadMs->load() * 0.001 * currentRate);
            lookahead.setDelaySamples (samples);
            requestLatencySamples (samples);
        }

        double currentRate = 48000.0;
        std::atomic<float>* target = nullptr;
        std::atomic<float>* rangeUp = nullptr;
        std::atomic<float>* rangeDown = nullptr;
        std::atomic<float>* speed = nullptr;
        std::atomic<float>* gateAmount = nullptr;
        std::atomic<float>* musicFollow = nullptr;

        helios::RiderEngine engine;
        mint::LookaheadDelay lookahead;
        std::array<float, chunkSize> gains {};
        AutomationWriter writer;
    };

    class RiderEditor final : public mint::ui::EditorBase
    {
    public:
        explicit RiderEditor (RiderProcessor& p)
            : EditorBase (p, { "HELIOS", { "Fader", "페이더", "推子" }, "VI" }),
              owner (p),
              targetDial (p.apvts, "target_db", { "Target", "타겟", "目标电平" }, true),
              upDial (p.apvts, "range_up_db", { "Range Up", "레인지 업", "上推范围" }),
              downDial (p.apvts, "range_down_db", { "Range Down", "레인지 다운", "下压范围" }),
              speedDial (p.apvts, "speed", { "Speed", "스피드", "速度" }),
              gateDial (p.apvts, "gate_amount", { "Vocal Gate", "보컬 게이트", "人声门" }),
              followDial (p.apvts, "music_follow", { "Music Follow", "반주 팔로우", "伴奏跟随" }),
              lookaheadDial (p.apvts, "lookahead_ms", { "Look-ahead", "룩어헤드", "前瞻" }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              outputMeter (p.outputMeter, { "Output", "출력", "输出" }),
              gainMeter (p.telemetry, p.apvts),
              gainTrace (p.telemetry, p.apvts),
              writeSwitch (p.apvts, "write_auto", { { "Off", "끔", "关闭" }, { "Write", "기록", "写入" } }),
              viewTabs ({ { "Gain Curve", "게인 커브", "增益曲线" }, { "Advanced", "고급", "高级" } }, [this] (int index)
              {
                  owner.apvts.state.setProperty (advancedViewProperty(), index == 1, nullptr);
                  showAdvanced (index == 1);
              }),
              writeWatch (*p.apvts.getParameter ("write_auto"), [this] (float value)
              {
                  writeOn = value > 0.5f;
                  showAdvanced (advanced);
              }, nullptr)
        {
            for (auto* child : std::initializer_list<juce::Component*> { &targetDial, &upDial, &downDial, &speedDial,
                                                                         &gateDial, &followDial, &lookaheadDial,
                                                                         &inputMeter, &outputMeter, &gainMeter, &gainTrace,
                                                                         &writeSwitch, &viewTabs })
                addAndMakeVisible (child);

            advanced = (bool) p.apvts.state.getProperty (advancedViewProperty(), false);
            viewTabs.setSelected (advanced ? 1 : 0);
            writeWatch.sendInitialUpdate();
            setSize (width, height);
        }

        void paint (juce::Graphics& g) override
        {
            EditorBase::paint (g);

            g.setColour (mint::ui::theme().inkMuted);
            g.setFont (mint::ui::bodyFont (12.0f));
            if (writeSwitch.isVisible())
                g.drawFittedText (mint::ui::tr ({ "Write Automation", "오토메이션 기록", "写自动化" }), writeSwitch.getX(), getFooterSlot().getY() - 2, writeSwitch.getWidth(), 16, juce::Justification::centredLeft, 1, 0.8f);
            g.drawText (mint::ui::tr ({ "View", "보기", "视图" }), viewTabs.getX(), getFooterSlot().getY() - 2, viewTabs.getWidth(), 16, juce::Justification::centredLeft);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeter.setBounds (area.removeFromLeft (40));
            outputMeter.setBounds (area.removeFromRight (40));
            area.removeFromRight (8);
            gainMeter.setBounds (area.removeFromRight (64));
            area.removeFromRight (10);

            auto left = area.removeFromLeft (216);
            targetDial.setBounds (left.withTrimmedLeft (4));
            area.removeFromLeft (8);

            auto dialRow = area.removeFromTop (142);
            const auto dialWidth = dialRow.getWidth() / 3;
            upDial.setBounds (dialRow.removeFromLeft (dialWidth));
            downDial.setBounds (dialRow.removeFromLeft (dialWidth));
            speedDial.setBounds (dialRow);

            area.removeFromTop (6);
            gainTrace.setBounds (area);

            auto advancedRow = area;
            const auto advancedWidth = advancedRow.getWidth() / 3;
            gateDial.setBounds (advancedRow.removeFromLeft (advancedWidth));
            followDial.setBounds (advancedRow.removeFromLeft (advancedWidth));
            lookaheadDial.setBounds (advancedRow);

            auto footer = getFooterSlot();
            footer.removeFromTop (14);
            viewTabs.setBounds (footer.withSizeKeepingCentre (200, footer.getHeight()).withHeight (28));
            writeSwitch.setBounds (footer.removeFromLeft (100).withHeight (28));
        }

    private:
        // 只存在界面状态里，不是参数：跟工程和预设一起保存，但宿主看不到，也不进撤销。
        static juce::Identifier advancedViewProperty()     { return "ui_advanced_view"; }

        void showAdvanced (bool shouldShow)
        {
            advanced = shouldShow;
            gainTrace.setVisible (! advanced);
            gateDial.setVisible (advanced);
            followDial.setVisible (advanced);
            lookaheadDial.setVisible (advanced);
            // 写自动化收在高级视图里，但正在写的时候不能藏起来。
            writeSwitch.setVisible (advanced || writeOn);
            repaint (getFooterSlot().withTop (getFooterSlot().getY() - 2));
        }

        juce::String getLookaheadText() const override
        {
            const auto ms = owner.lookaheadMs->load();
            return ms <= 0.0f ? mint::ui::tr ({ "Off · 0 ms", "끔 · 0 ms", "关闭 · 0 ms" }) : juce::String (ms, 1) + " ms";
        }

        RiderProcessor& owner;
        mint::ui::Dial targetDial, upDial, downDial, speedDial, gateDial, followDial, lookaheadDial;
        mint::ui::Meter inputMeter, outputMeter;
        helios::GainMeter gainMeter;
        helios::GainTrace gainTrace;
        mint::ui::Segmented writeSwitch;
        helios::ViewTabs viewTabs;
        bool advanced = false, writeOn = false;
        juce::ParameterAttachment writeWatch;
    };

    juce::AudioProcessorEditor* RiderProcessor::createEditor()
    {
        return new RiderEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RiderProcessor();
}
