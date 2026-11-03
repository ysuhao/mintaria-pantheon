// Mintaria Lethe：人声和环境音分离式降噪。
// 核心通路和线程模型见 LetheCore.h；这里只有参数、界面和离线渲染的判断。
#include "LetheCore.h"

#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

namespace
{
    constexpr float minDb = -60.0f;

    // 旋钮拉到底就是彻底关掉，不留 -60 dB 的残余。
    float knobToGain (float db) noexcept
    {
        return db <= minDb ? 0.0f : mint::dbToGain (db);
    }

    juce::String dbText (float db)
    {
        return db <= minDb ? juce::String::fromUTF8 ("关闭") : juce::String (db, 1) + " dB";
    }

    // 旋钮下方显示的文字，绘制时取当前语言。
    juce::String dbDisplayText (float db)
    {
        return db <= minDb ? mint::ui::tr ({ "Off", "끔", "关闭" }) : juce::String (db, 1) + " dB";
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;

        const auto range = [] { return NormalisableRange<float> (minDb, 0.0f, 0.1f); };
        const auto attributes = []
        {
            return AudioParameterFloatAttributes().withLabel ("dB").withStringFromValueFunction ([] (float v, int) { return dbText (v); });
        };

        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "voice", 1 }, "Voice", range(), 0.0f, attributes()));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "ambience", 1 }, "Ambience", range(), minDb, attributes()));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "mode", 1 }, "Mode",
                                                            StringArray { String::fromUTF8 ("标准"), String::fromUTF8 ("低延迟"), String::fromUTF8 ("轻量") }, 1));
        return layout;
    }

    // 有些宿主离线渲染时不设置 non-realtime 标志，mint_probe 也不设。这种情况下工作线程
    // 追不上调用速度，全程都会走干声回退。所以看两次回调之间的空闲时间：实时宿主的空闲时间
    // 接近音频时长，离线渲染几乎为零。等工作线程的时间在回调内部，不影响判断。
    // 进出都要连续几个窗口，单次卡顿或暂停不会切换。
    class FasterThanRealtimeDetector
    {
    public:
        void prepare (double sampleRate) noexcept
        {
            rate = sampleRate;
            windowSamples = juce::jmax (1024, (int) (sampleRate * 0.02));
            samples = fastWindows = slowWindows = 0;
            lastEnd = 0;
            idleTicks = 0;
            fast = false;
        }

        void blockStarted() noexcept
        {
            if (lastEnd != 0)
                idleTicks += juce::Time::getHighResolutionTicks() - lastEnd;
        }

        void blockFinished() noexcept
        {
            lastEnd = juce::Time::getHighResolutionTicks();
        }

        bool update (int numSamples) noexcept
        {
            samples += numSamples;
            if (samples < windowSamples)
                return fast;

            const auto idle = juce::Time::highResolutionTicksToSeconds (idleTicks);
            const auto audio = (double) samples / rate;
            samples = 0;
            idleTicks = 0;

            if (idle < audio * 0.25)
            {
                slowWindows = 0;
                if (++fastWindows >= 4)
                    fast = true;
            }
            else if (idle > audio * 0.5)
            {
                fastWindows = 0;
                if (++slowWindows >= 4)
                    fast = false;
            }
            return fast;
        }

    private:
        double rate = 48000.0;
        int windowSamples = 1024, samples = 0, fastWindows = 0, slowWindows = 0;
        juce::int64 lastEnd = 0, idleTicks = 0;
        bool fast = false;
    };

    class LetheProcessor final : public mint::Processor
    {
    public:
        LetheProcessor() : mint::Processor (createLayout())
        {
            voice = apvts.getRawParameterValue ("voice");
            ambience = apvts.getRawParameterValue ("ambience");
            mode = apvts.getRawParameterValue ("mode");
        }

        void prepareToPlay (double sampleRate, int samplesPerBlock) override
        {
            const auto kind = currentKind();
            core.prepare (sampleRate, samplesPerBlock, juce::jlimit (1, 2, getMainBusNumOutputChannels()), kind);
            detector.prepare (sampleRate);

            // 延迟先同步设一次，宿主在 prepareToPlay 之后就会读。后面切档走基类的 Timer。
            const auto latency = core.getLatencySamples (kind);
            setLatencySamples (latency);
            requestLatencySamples (latency);
        }

        void releaseResources() override
        {
            core.release();
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            render (buffer, false);
        }

        // 旁路也走同一条延迟线，否则宿主的延迟补偿会让信号跳时间。
        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            render (buffer, true);
        }

        juce::AudioProcessorEditor* createEditor() override;

        // 返回三语文字，由界面在刷新时用 tr() 取当前语言。没有状态要显示时是空文字。
        mint::ui::Text getStatusText() const
        {
            using State = lethe::Core::EngineState;
            switch (core.getEngineState())
            {
                case State::failed:  return { "Model not loaded · Dry output", "모델 로드 안 됨 · 드라이 출력", "模型未加载 · 输出干声" };
                case State::loading: return { "Loading model · Dry output", "모델 로딩 중 · 드라이 출력", "模型加载中 · 输出干声" };
                case State::idle:    return {};
                case State::ready:   break;
            }
            return core.isFallingBack() ? mint::ui::Text { "Processing timed out · Dry output", "처리 시간 초과 · 드라이 출력", "处理超时 · 输出干声" }
                                        : mint::ui::Text {};
        }

        mint::MeterSource inputMeter, outputMeter;

    private:
        lethe::EngineKind currentKind() const noexcept
        {
            return (lethe::EngineKind) juce::jlimit (0, lethe::numEngineKinds - 1, (int) mode->load());
        }

        void render (juce::AudioBuffer<float>& buffer, bool bypassed) noexcept
        {
            juce::ScopedNoDenormals noDenormals;

            const auto numSamples = buffer.getNumSamples();
            if (numSamples == 0)
                return;

            inputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));

            const auto kind = currentKind();
            requestLatencySamples (core.getLatencySamples (kind));

            detector.blockStarted();
            const auto offline = detector.update (numSamples) || isNonRealtime();
            core.process (buffer.getArrayOfWritePointers(), numSamples,
                          knobToGain (voice->load()), knobToGain (ambience->load()), kind, bypassed, offline);
            detector.blockFinished();

            outputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));
        }

        lethe::Core core;
        FasterThanRealtimeDetector detector;
        std::atomic<float>* voice = nullptr;
        std::atomic<float>* ambience = nullptr;
        std::atomic<float>* mode = nullptr;
    };

    class LetheEditor final : public mint::ui::EditorBase
    {
    public:
        explicit LetheEditor (LetheProcessor& p)
            : EditorBase (p, { "LETHE", { "Denoise", "노이즈 제거", "降噪" }, "II" }),
              owner (p),
              voiceDial (p.apvts, "voice", { "Voice", "보이스", "人声" }),
              ambienceDial (p.apvts, "ambience", { "Ambience", "앰비언스", "环境音" }),
              // 显示标签单独给；参数的选项名（标准、低延迟、轻量）不变。
              modeSwitch (p.apvts, "mode", { { "Standard", "표준", "标准" }, { "Low latency", "로우 레이턴시", "低延迟" }, { "Light", "라이트", "轻量" } }),
              inputMeterView (p.inputMeter, { "Input", "입력", "输入" }),
              outputMeterView (p.outputMeter, { "Output", "출력", "输出" })
        {
            // 旋钮最低点的“关闭”是界面文字，跟随语言。参数自身的文本（宿主自动化看到的）不变。
            for (auto* dial : { &voiceDial, &ambienceDial })
                dial->slider.textFromValueFunction = [] (double v) { return dbDisplayText ((float) v); };

            addAndMakeVisible (voiceDial);
            addAndMakeVisible (ambienceDial);
            addAndMakeVisible (modeSwitch);
            addAndMakeVisible (inputMeterView);
            addAndMakeVisible (outputMeterView);

            status.setJustificationType (juce::Justification::centred);
            status.setFont (mint::ui::bodyFont (13.0f));
            status.setColour (juce::Label::textColourId, mint::ui::theme().inkMuted);
            addAndMakeVisible (status);

            setSize (width, height);
            statusTimer.startTimerHz (4);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeterView.setBounds (area.removeFromLeft (40));
            outputMeterView.setBounds (area.removeFromRight (40));

            status.setBounds (area.removeFromBottom (24));

            const auto dialWidth = juce::jmin (224, area.getWidth() / 2);
            const auto dialHeight = juce::jmin (280, area.getHeight());
            auto centre = area.withSizeKeepingCentre (dialWidth * 2, dialHeight);
            voiceDial.setBounds (centre.removeFromLeft (dialWidth));
            ambienceDial.setBounds (centre);

            modeSwitch.setBounds (getFooterSlot());
        }

    private:
        // EditorBase 已经私有继承 Timer，状态文字用一个单独的定时器。
        struct StatusTimer final : juce::Timer
        {
            explicit StatusTimer (LetheEditor& e) : editor (e) {}
            void timerCallback() override { editor.refreshStatus(); }
            LetheEditor& editor;
        };

        void refreshStatus()
        {
            status.setText (mint::ui::tr (owner.getStatusText()), juce::dontSendNotification);
        }

        // Label 自己存了文字和字体，切换语言后要重设。
        void languageChanged() override
        {
            status.setFont (mint::ui::bodyFont (13.0f));
            refreshStatus();
        }

        LetheProcessor& owner;
        mint::ui::Dial voiceDial, ambienceDial;
        mint::ui::Segmented modeSwitch;
        mint::ui::Meter inputMeterView, outputMeterView;
        juce::Label status;
        StatusTimer statusTimer { *this };
    };

    juce::AudioProcessorEditor* LetheProcessor::createEditor()
    {
        return new LetheEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new LetheProcessor();
}
