// N° VIII 延迟。音频引擎在 EchoEngine.h，这里是参数、节拍同步和界面。
#include "EchoEngine.h"

#include <mint_ui/MintUi.h>

#include <limits>

namespace
{
    juce::String zh (const char* text)      { return juce::String::fromUTF8 (text); }

    // 按实际时长从短到长排列，转动旋钮时时间单调变长。beats 以四分音符为 1。
    // label 是参数选项字符串，不能改；base 和 kind 只用于界面显示（kind：0 普通，1 三连，2 附点）。
    struct Division { const char* label; const char* base; int kind; double beats; };

    constexpr Division divisions[] = {
        { "1/32 三连", "1/32", 1, 1.0 / 12.0 },
        { "1/32", "1/32", 0, 0.125 },
        { "1/16 三连", "1/16", 1, 1.0 / 6.0 },
        { "1/32 附点", "1/32", 2, 0.1875 },
        { "1/16", "1/16", 0, 0.25 },
        { "1/8 三连", "1/8", 1, 1.0 / 3.0 },
        { "1/16 附点", "1/16", 2, 0.375 },
        { "1/8", "1/8", 0, 0.5 },
        { "1/4 三连", "1/4", 1, 2.0 / 3.0 },
        { "1/8 附点", "1/8", 2, 0.75 },
        { "1/4", "1/4", 0, 1.0 },
        { "1/2 三连", "1/2", 1, 4.0 / 3.0 },
        { "1/4 附点", "1/4", 2, 1.5 },
        { "1/2", "1/2", 0, 2.0 },
        { "1/1 三连", "1/1", 1, 8.0 / 3.0 },
        { "1/2 附点", "1/2", 2, 3.0 },
        { "1/1", "1/1", 0, 4.0 },
        { "1/1 附点", "1/1", 2, 6.0 },
    };
    constexpr int numDivisions = (int) (sizeof (divisions) / sizeof (divisions[0]));
    constexpr int defaultDivision = 10;

    // 同步模式大旋钮下的时值文字，只改显示，不动参数选项。英文 T/D，韩文用通行的音乐术语。
    juce::String divisionDisplayText (int index)
    {
        const auto& d = divisions[juce::jlimit (0, numDivisions - 1, index)];
        if (d.kind == 0)
            return d.base;

        const mint::ui::Text triplet { " T", " T", " 三连" }, dotted { " D", " D", " 附点" };
        return juce::String (d.base) + mint::ui::tr (d.kind == 1 ? triplet : dotted);
    }

    juce::String millisecondsText (float ms)
    {
        if (ms >= 1000.0f)
            return juce::String (ms * 0.001f, 3) + " s";
        // 参数步进 0.1 ms。只在有小数时才显示，300 ms 不写成 300.0 ms。
        const auto whole = std::abs (ms - std::round (ms)) < 0.05f;
        return juce::String (ms, whole ? 0 : 1) + " ms";
    }

    // 带单位的文字转回参数值。旋钮双击输入和宿主里的数值输入都走这里，
    // 默认解析只取数字，"7.0 kHz" 会变成 7 Hz，"1.200 s" 会变成 1.2 ms。
    float millisecondsFromText (const juce::String& text)
    {
        const auto trimmed = text.trim().toLowerCase();
        const auto inSeconds = trimmed.endsWith ("s") && ! trimmed.endsWith ("ms");
        return trimmed.getFloatValue() * (inSeconds ? 1000.0f : 1.0f);
    }

    float frequencyFromText (const juce::String& text)
    {
        return text.getFloatValue() * (text.containsIgnoreCase ("k") ? 1000.0f : 1.0f);
    }

    juce::String frequencyText (float hz)
    {
        return hz >= 1000.0f ? juce::String (hz * 0.001f, 1) + " kHz" : juce::String (juce::roundToInt (hz)) + " Hz";
    }

    juce::String percentText (float value)  { return juce::String (juce::roundToInt (value)) + " %"; }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        using Float = AudioParameterFloat;
        using Attributes = AudioParameterFloatAttributes;

        const auto addFloat = [] (AudioProcessorValueTreeState::ParameterLayout& layout, const char* id, const char* name,
                                  NormalisableRange<float> range, float initial, String (*text) (float),
                                  float (*parse) (const String&) = nullptr)
        {
            auto attributes = Attributes().withStringFromValueFunction ([text] (float v, int) { return text (v); });
            if (parse != nullptr)
                attributes = attributes.withValueFromStringFunction (parse);
            layout.add (std::make_unique<Float> (ParameterID { id, 1 }, name, range, initial, attributes));
        };

        const auto skewed = [] (float low, float high, float step, float centre)
        {
            NormalisableRange<float> range (low, high, step);
            range.setSkewForCentre (centre);
            return range;
        };

        StringArray divisionNames;
        for (const auto& division : divisions)
            divisionNames.add (zh (division.label));

        AudioProcessorValueTreeState::ParameterLayout layout;
        addFloat (layout, "time_ms", "Time", skewed (1.0f, 4000.0f, 0.1f, 300.0f), 300.0f, millisecondsText, millisecondsFromText);
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "time_division", 1 }, "Division", divisionNames, defaultDivision));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "time_base", 1 }, "Time Base",
                                                            StringArray { zh ("毫秒"), zh ("同步") }, 0));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "mode", 1 }, "Mode",
                                                            StringArray { zh ("立体声"), zh ("乒乓"), zh ("单声道") }, 0));
        addFloat (layout, "feedback", "Feedback", NormalisableRange<float> (0.0f, 100.0f, 0.1f), 30.0f, percentText);
        addFloat (layout, "mix", "Mix", NormalisableRange<float> (0.0f, 100.0f, 0.1f), 20.0f, percentText);
        addFloat (layout, "hi_cut", "High Cut", skewed (1000.0f, 20000.0f, 1.0f, 6000.0f), 7000.0f, frequencyText, frequencyFromText);
        addFloat (layout, "lo_cut", "Low Cut", skewed (20.0f, 2000.0f, 1.0f, 200.0f), 120.0f, frequencyText, frequencyFromText);
        addFloat (layout, "mod_depth", "Mod Depth", NormalisableRange<float> (0.0f, 100.0f, 0.1f), 0.0f, percentText);
        addFloat (layout, "mod_rate", "Mod Rate", skewed (0.1f, 10.0f, 0.01f, 1.0f), 0.6f,
                  [] (float v) { return String (v, 2) + " Hz"; });
        addFloat (layout, "drive", "Drive", NormalisableRange<float> (0.0f, 18.0f, 0.1f), 6.0f,
                  [] (float v) { return String (v, 1) + " dB"; });
        addFloat (layout, "lofi", "LoFi", NormalisableRange<float> (0.0f, 100.0f, 0.1f), 0.0f, percentText);
        addFloat (layout, "fallback_bpm", "Fallback BPM", NormalisableRange<float> (20.0f, 300.0f, 0.1f), 120.0f,
                  [] (float v) { return String (v, 1) + " BPM"; });
        return layout;
    }

    class EchoProcessor final : public mint::Processor
    {
    public:
        // 同步模式用的速度从哪里来，界面据此提示。
        enum class TempoSource { host, heldHost, fallback };

        EchoProcessor() : mint::Processor (createLayout())
        {
            timeMs = apvts.getRawParameterValue ("time_ms");
            division = apvts.getRawParameterValue ("time_division");
            timeBase = apvts.getRawParameterValue ("time_base");
            mode = apvts.getRawParameterValue ("mode");
            feedback = apvts.getRawParameterValue ("feedback");
            mix = apvts.getRawParameterValue ("mix");
            hiCut = apvts.getRawParameterValue ("hi_cut");
            loCut = apvts.getRawParameterValue ("lo_cut");
            modDepth = apvts.getRawParameterValue ("mod_depth");
            modRate = apvts.getRawParameterValue ("mod_rate");
            drive = apvts.getRawParameterValue ("drive");
            lofi = apvts.getRawParameterValue ("lofi");
            fallbackBpm = apvts.getRawParameterValue ("fallback_bpm");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            engine.prepare (sampleRate, getTotalNumOutputChannels());
            engine.setSettings (readSettings());
            engine.reset();
            prepared = true;
        }

        void releaseResources() override    { prepared = false; }

        void reset() override
        {
            if (prepared)
                engine.reset();
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;
            const auto numSamples = buffer.getNumSamples();

            inputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));

            updateTempoFromHost();
            engine.setSettings (readSettings());
            engine.process (buffer.getArrayOfWritePointers(), numSamples);

            outputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));
        }

        // 回声衰减到 -60 dB 的时间。按反馈量估算，环内滤波只会让实际尾音更短，所以这是上限。
        // 反馈接近 100% 时尾音不会自行结束，如实报无限长。
        double getTailLengthSeconds() const override
        {
            const auto gain = (double) feedback->load() * 0.01;
            if (gain >= 0.999)
                return std::numeric_limits<double>::infinity();

            const auto repeats = gain <= 0.001 ? 1.0 : 1.0 + std::ceil (std::log (0.001) / std::log (gain));
            return repeats * delaySeconds (currentBpm()) + echo::timeFadeSeconds;
        }

        juce::AudioProcessorEditor* createEditor() override;

        double currentBpm() const noexcept
        {
            const auto held = lastHostBpm.load (std::memory_order_relaxed);
            return held > 0.0 ? held : (double) fallbackBpm->load();
        }

        TempoSource currentTempoSource() const noexcept
        {
            if (lastHostBpm.load (std::memory_order_relaxed) <= 0.0)
                return TempoSource::fallback;
            return hostBpmIsLive.load (std::memory_order_relaxed) ? TempoSource::host : TempoSource::heldHost;
        }

        bool isSynced() const noexcept      { return timeBase->load() > 0.5f; }

        // 同步模式下算出的回声间隔。超过延迟线长度时对折，回声仍然落在节拍网格上。
        double delaySeconds (double bpm) const noexcept
        {
            if (! isSynced())
                return (double) timeMs->load() * 0.001;

            const auto index = juce::jlimit (0, numDivisions - 1, juce::roundToInt (division->load()));
            auto seconds = divisions[index].beats * 60.0 / juce::jmax (1.0, bpm);
            while (seconds > echo::maxDelaySeconds)
                seconds *= 0.5;
            return seconds;
        }

        bool isFolded (double bpm) const noexcept
        {
            if (! isSynced())
                return false;
            const auto index = juce::jlimit (0, numDivisions - 1, juce::roundToInt (division->load()));
            return divisions[index].beats * 60.0 / juce::jmax (1.0, bpm) > echo::maxDelaySeconds;
        }

        mint::MeterSource inputMeter, outputMeter;

        // 高级视图的开关存在 apvts.state 上，随工程保存，关掉窗口再开也不重置。不是音频参数，不进自动化。
        static juce::Identifier advancedViewProperty()      { return "advanced_view"; }
        bool isAdvancedViewOpen() const                     { return apvts.state.getProperty (advancedViewProperty(), false); }

        void setAdvancedViewOpen (bool open)
        {
            apvts.state.setProperty (advancedViewProperty(), open, nullptr);
        }

    private:
        // 宿主断开速度信息时保持最后一次的值，从未拿到过才用备用速度。
        void updateTempoFromHost() noexcept
        {
            auto live = false;
            if (auto* head = getPlayHead())
                if (auto position = head->getPosition())
                    if (auto bpm = position->getBpm())
                        if (*bpm >= 1.0 && *bpm <= 1000.0)
                        {
                            lastHostBpm.store (*bpm, std::memory_order_relaxed);
                            live = true;
                        }
            hostBpmIsLive.store (live, std::memory_order_relaxed);
        }

        echo::Settings readSettings() const noexcept
        {
            echo::Settings s;
            s.delaySeconds = delaySeconds (currentBpm());
            s.feedback = feedback->load() * 0.01f;
            s.mix = mix->load() * 0.01f;
            s.hiCutHz = hiCut->load();
            s.loCutHz = loCut->load();
            s.modDepth = modDepth->load() * 0.01f;
            s.modRateHz = modRate->load();
            s.driveDb = drive->load();
            s.lofi = lofi->load() * 0.01f;
            s.mode = (echo::Mode) juce::jlimit (0, 2, juce::roundToInt (mode->load()));
            return s;
        }

        echo::Engine engine;
        bool prepared = false;
        std::atomic<double> lastHostBpm { 0.0 };
        std::atomic<bool> hostBpmIsLive { false };

        std::atomic<float>* timeMs = nullptr;
        std::atomic<float>* division = nullptr;
        std::atomic<float>* timeBase = nullptr;
        std::atomic<float>* mode = nullptr;
        std::atomic<float>* feedback = nullptr;
        std::atomic<float>* mix = nullptr;
        std::atomic<float>* hiCut = nullptr;
        std::atomic<float>* loCut = nullptr;
        std::atomic<float>* modDepth = nullptr;
        std::atomic<float>* modRate = nullptr;
        std::atomic<float>* drive = nullptr;
        std::atomic<float>* lofi = nullptr;
        std::atomic<float>* fallbackBpm = nullptr;
    };

    // 共享层的 Segmented 只能绑定参数。主控和高级区的分页不是音频参数，所以这里画一个外观相同的。
    // 外观和悬停反馈跟随共享层的 Segmented。
    class PageSwitch final : public juce::Component
    {
    public:
        std::function<void (bool)> onChange;

        void setAdvanced (bool advanced)
        {
            selected = advanced;
            repaint();
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = mint::ui::theme();
            const auto w = (float) getWidth() * 0.5f;
            const auto h = (float) getHeight();
            const auto current = selected ? 1 : 0;
            const juce::String names[] = { mint::ui::tr ({ "Main", "메인", "主控" }), mint::ui::tr ({ "Advanced", "고급", "高级" }) };

            if (hovered >= 0 && hovered != current)
            {
                g.setColour (t.trimSoft.withAlpha (0.45f));
                g.fillRect (juce::Rectangle<float> ((float) hovered * w, 0.0f, w, h));
            }

            g.setColour (t.ink);
            g.fillRect (juce::Rectangle<float> ((float) current * w, 0.0f, w, h));

            g.setColour (t.trim);
            g.drawRect (getLocalBounds().toFloat(), 1.0f);
            g.fillRect (juce::Rectangle<float> (w - 0.5f, 0.0f, 1.0f, h));

            g.setFont (mint::ui::bodyFont (13.0f));
            for (int i = 0; i < 2; ++i)
            {
                g.setColour (i == current ? t.plate : t.ink);
                g.drawFittedText (names[i], juce::Rectangle<float> ((float) i * w, 0.0f, w, h).reduced (3.0f, 0.0f).toNearestInt(),
                                  juce::Justification::centred, 1, 0.75f);
            }
        }

        void mouseDown (const juce::MouseEvent& e) override
        {
            setAdvanced (e.x >= getWidth() / 2);
            if (onChange != nullptr)
                onChange (selected);
        }

        void mouseMove (const juce::MouseEvent& e) override     { setHovered (e.x >= getWidth() / 2 ? 1 : 0); }
        void mouseExit (const juce::MouseEvent&) override       { setHovered (-1); }

    private:
        void setHovered (int cell)
        {
            if (cell != hovered)
            {
                hovered = cell;
                repaint();
            }
        }

        bool selected = false;
        int hovered = -1;
    };

    // 同步模式下说明实际用的速度和回声间隔。没有宿主速度时，用户要能从这里看出来。
    // 毫秒模式下回声间隔就是主旋钮的数值，不重复显示。
    class TempoReadout final : public juce::Component,
                               private juce::Timer
    {
    public:
        explicit TempoReadout (EchoProcessor& p) : processor (p)
        {
            setInterceptsMouseClicks (false, false);
            refresh();
            startTimerHz (4);
        }

        void paint (juce::Graphics& g) override
        {
            if (value.isEmpty())
                return;

            const auto& t = mint::ui::theme();
            auto area = getLocalBounds();

            g.setColour (t.inkMuted);
            g.setFont (mint::ui::bodyFont (11.0f));
            g.drawText (mint::ui::tr (headingText()), area.removeFromTop (16), juce::Justification::centredLeft);

            g.setColour (t.ink);
            g.setFont (mint::ui::bodyFont (14.0f));
            g.drawText (value + (folded ? mint::ui::tr ({ " (too long, halved)", " (최대치 초과, 절반 적용)", "（超长，已对折）" }) : juce::String()),
                        area.removeFromTop (22), juce::Justification::centredLeft);
        }

    private:
        mint::ui::Text headingText() const
        {
            switch (source)
            {
                case EchoProcessor::TempoSource::host:     return { "Host tempo", "호스트 템포", "宿主速度" };
                case EchoProcessor::TempoSource::heldHost: return { "Host tempo lost, using last", "호스트 템포 신호 없음, 이전 값 유지", "宿主已停止提供速度，沿用上次" };
                case EchoProcessor::TempoSource::fallback: break;
            }
            return { "No host tempo, using fallback", "호스트 템포 없음, 대체 템포 사용", "宿主未提供速度，使用备用速度" };
        }

        void timerCallback() override   { refresh(); }

        void refresh()
        {
            const auto bpm = processor.currentBpm();
            const auto synced = processor.isSynced();
            const auto nextSource = processor.currentTempoSource();
            const auto nextValue = synced ? juce::String (bpm, 1) + " BPM  /  " + millisecondsText ((float) processor.delaySeconds (bpm) * 1000.0f)
                                          : juce::String();
            const auto nextFolded = processor.isFolded (bpm);

            if (nextSource != source || nextValue != value || nextFolded != folded)
            {
                source = nextSource;
                value = nextValue;
                folded = nextFolded;
                repaint();
            }
        }

        EchoProcessor& processor;
        EchoProcessor::TempoSource source = EchoProcessor::TempoSource::fallback;
        juce::String value;
        bool folded = false;
    };

    class EchoEditor final : public mint::ui::EditorBase,
                             private juce::ValueTree::Listener,
                             private juce::AsyncUpdater
    {
    public:
        explicit EchoEditor (EchoProcessor& p)
            : EditorBase (p, { "ECHO", { "Delay", "딜레이", "延迟" }, "VIII" }),
              owner (p),
              timeDial (p.apvts, "time_ms", { "Time", "타임", "时间" }, true),
              divisionDial (p.apvts, "time_division", { "Time", "타임", "时间" }, true),
              feedbackDial (p.apvts, "feedback", { "Feedback", "피드백", "反馈" }),
              mixDial (p.apvts, "mix", { "Mix", "믹스", "混合" }),
              hiCutDial (p.apvts, "hi_cut", { "High Cut", "하이컷", "高切" }),
              loCutDial (p.apvts, "lo_cut", { "Low Cut", "로우컷", "低切" }),
              driveDial (p.apvts, "drive", { "Drive", "드라이브", "驱动" }),
              lofiDial (p.apvts, "lofi", { "LoFi", "로파이", "低保真" }),
              depthDial (p.apvts, "mod_depth", { "Mod Depth", "모듈레이션 뎁스", "调制深度" }),
              rateDial (p.apvts, "mod_rate", { "Mod Rate", "모듈레이션 레이트", "调制速率" }),
              fallbackDial (p.apvts, "fallback_bpm", { "Fallback Tempo", "대체 템포", "备用速度" }),
              modeSwitch (p.apvts, "mode", { { "Stereo", "스테레오", "立体声" }, { "Ping-Pong", "핑퐁", "乒乓" }, { "Mono", "모노", "单声道" } }),
              timeBaseSwitch (p.apvts, "time_base", { { "ms", "ms", "ms" }, { "Sync", "싱크", "同步" } }),
              inputMeterView (p.inputMeter, { "Input", "입력", "输入" }),
              outputMeterView (p.outputMeter, { "Output", "출력", "输出" }),
              readout (p),
              timeBaseWatcher (*p.apvts.getParameter ("time_base"), [this] (float value) { synced = value > 0.5f; updateVisibility(); }, nullptr)
        {
            for (auto* child : std::initializer_list<juce::Component*> { &timeDial, &divisionDial, &feedbackDial, &mixDial,
                                                                         &modeSwitch, &timeBaseSwitch,
                                                                         &hiCutDial, &loCutDial, &driveDial, &lofiDial,
                                                                         &depthDial, &rateDial, &fallbackDial,
                                                                         &inputMeterView, &outputMeterView, &readout, &pageSwitch })
                addAndMakeVisible (*child);

            pageSwitch.setAdvanced (owner.isAdvancedViewOpen());
            pageSwitch.onChange = [this] (bool advanced)
            {
                owner.setAdvancedViewOpen (advanced);
                updateVisibility();
            };
            owner.apvts.state.addListener (this);

            // 只改显示，参数选项不变。要在 SliderAttachment 建好之后设置，它会占用这两个函数。
            // 双击输入的是显示文字，所以转回时也按显示文字找，找不到就不动。
            divisionDial.slider.textFromValueFunction = [] (double value) { return divisionDisplayText (juce::roundToInt (value)); };
            divisionDial.slider.valueFromTextFunction = [this] (const juce::String& text)
            {
                for (int i = 0; i < numDivisions; ++i)
                    if (text.trim().equalsIgnoreCase (divisionDisplayText (i)) || text.trim().equalsIgnoreCase (juce::String::fromUTF8 (divisions[i].label)))
                        return (double) i;
                return divisionDial.slider.getValue();
            };
            divisionDial.repaint();

            timeBaseWatcher.sendInitialUpdate();
            setSize (width, height);
        }

        ~EchoEditor() override
        {
            owner.apvts.state.removeListener (this);
            cancelPendingUpdate();
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeterView.setBounds (area.removeFromLeft (40));
            outputMeterView.setBounds (area.removeFromRight (40));
            area.reduce (16, 0);

            // 主控页
            auto main = area;
            const auto bigArea = main.removeFromLeft (240).withSizeKeepingCentre (224, 280);
            timeDial.setBounds (bigArea);
            divisionDial.setBounds (bigArea);

            // 旋钮是视觉重心，开关压成一排放在下面，底边和主旋钮的标签对齐。
            main.removeFromLeft (16);
            main = main.withSizeKeepingCentre (main.getWidth(), 280);
            auto switches = main.removeFromBottom (28);
            modeSwitch.setBounds (switches.removeFromLeft (204));
            switches.removeFromLeft (12);
            timeBaseSwitch.setBounds (switches);
            main.removeFromBottom (14);
            auto dials = main;
            feedbackDial.setBounds (dials.removeFromLeft (dials.getWidth() / 2).withTrimmedRight (4));
            mixDial.setBounds (dials.withTrimmedLeft (4));

            // 高级页
            const auto rowHeight = area.getHeight() / 2;
            const auto cellWidth = area.getWidth() / 4;
            auto top = area.withHeight (rowHeight);
            auto bottom = top.translated (0, rowHeight);
            for (auto* dial : { &hiCutDial, &loCutDial, &driveDial, &lofiDial })
                dial->setBounds (top.removeFromLeft (cellWidth).reduced (4, 2));
            for (auto* dial : { &depthDial, &rateDial, &fallbackDial })
                dial->setBounds (bottom.removeFromLeft (cellWidth).reduced (4, 2));

            auto footer = getFooterSlot();
            pageSwitch.setBounds (footer.removeFromRight (130).withSizeKeepingCentre (130, 28));
            footer.removeFromRight (10);
            readout.setBounds (footer.withTrimmedTop (-2));
        }

    private:
        // 分页按钮、读数和时值文字都在绘制时取当前语言，这里只要求重绘。
        void languageChanged() override
        {
            divisionDial.repaint();
            pageSwitch.repaint();
            readout.repaint();
        }

        // 高级视图的开关可能被宿主恢复工程改掉，这时调用线程不一定是消息线程。
        void valueTreePropertyChanged (juce::ValueTree&, const juce::Identifier& property) override
        {
            if (property == EchoProcessor::advancedViewProperty())
                triggerAsyncUpdate();
        }

        void valueTreeRedirected (juce::ValueTree&) override    { triggerAsyncUpdate(); }

        void handleAsyncUpdate() override
        {
            pageSwitch.setAdvanced (owner.isAdvancedViewOpen());
            updateVisibility();
        }

        void updateVisibility()
        {
            const auto advanced = owner.isAdvancedViewOpen();
            timeDial.setVisible (! advanced && ! synced);
            divisionDial.setVisible (! advanced && synced);
            for (auto* child : std::initializer_list<juce::Component*> { &feedbackDial, &mixDial, &modeSwitch, &timeBaseSwitch })
                child->setVisible (! advanced);
            for (auto* dial : { &hiCutDial, &loCutDial, &driveDial, &lofiDial, &depthDial, &rateDial, &fallbackDial })
                dial->setVisible (advanced);
        }

        EchoProcessor& owner;
        mint::ui::Dial timeDial, divisionDial, feedbackDial, mixDial;
        mint::ui::Dial hiCutDial, loCutDial, driveDial, lofiDial, depthDial, rateDial, fallbackDial;
        mint::ui::Segmented modeSwitch, timeBaseSwitch;
        mint::ui::Meter inputMeterView, outputMeterView;
        TempoReadout readout;
        PageSwitch pageSwitch;
        juce::ParameterAttachment timeBaseWatcher;
        bool synced = false;
    };

    juce::AudioProcessorEditor* EchoProcessor::createEditor()
    {
        return new EchoEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new EchoProcessor();
}
