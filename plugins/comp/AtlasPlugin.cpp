// Mintaria Atlas：面向人声的压缩器，默认 0 延迟，不开过采样。
#include "AtlasDsp.h"

#include <mint_ui/MintUi.h>

namespace
{
    constexpr float maxLookaheadMs = 10.0f;
    constexpr float declickSeconds = 0.00015f;  // 增益路径上的短平滑，磨掉启动段的折角
    constexpr float maxHoldMs = 14.0f;          // 峰值窗口上限，盖住约 71 Hz 的半周期
    constexpr float lookaheadFadeSeconds = 0.005f;
    constexpr float maxDetectorLevel = 1000.0f; // 防止宿主送来的异常大值把检测器状态撑到 Inf
    constexpr int maxChunk = 256;

    juce::String formatDb (float v, int)     { return juce::String (v, 1) + " dB"; }
    juce::String formatRatio (float v, int)  { return juce::String (v, 1) + ":1"; }
    juce::String formatMs (float v, int)
    {
        if (v >= 1000.0f)
            return juce::String (v * 0.001f, 2) + " s";
        return juce::String (v, v < 10.0f ? 1 : 0) + " ms";
    }

    juce::NormalisableRange<float> skewedRange (float low, float high, float step, float centre)
    {
        juce::NormalisableRange<float> range (low, high, step);
        range.setSkewForCentre (centre);
        return range;
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        const auto floatParam = [] (const char* id, const char* name, NormalisableRange<float> range, float value,
                                    const char* unit, String (*format) (float, int))
        {
            return std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, name, std::move (range), value,
                                                          AudioParameterFloatAttributes().withLabel (unit)
                                                              .withStringFromValueFunction (format));
        };

        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add (floatParam ("threshold_db", "Threshold", NormalisableRange<float> (-60.0f, 0.0f, 0.1f), -18.0f, "dB", formatDb));
        layout.add (floatParam ("ratio", "Ratio", skewedRange (1.0f, 20.0f, 0.1f, 4.0f), 3.0f, "", formatRatio));
        layout.add (floatParam ("attack_ms", "Attack", skewedRange (0.5f, 200.0f, 0.1f, 15.0f), 8.0f, "ms", formatMs));
        layout.add (floatParam ("release_ms", "Release", skewedRange (5.0f, 3000.0f, 1.0f, 150.0f), 120.0f, "ms", formatMs));
        layout.add (floatParam ("makeup_db", "Makeup", NormalisableRange<float> (-12.0f, 24.0f, 0.1f), 0.0f, "dB", formatDb));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "auto_release", 1 }, "Auto Release",
                                                            StringArray { String::fromUTF8 ("手动"), String::fromUTF8 ("自动") }, 1));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "behavior", 1 }, "Behavior",
                                                            StringArray { "Opto", "Electro" }, 1));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "character", 1 }, "Character",
                                                            StringArray { "Warm", "Smooth" }, 1));
        layout.add (floatParam ("lookahead_ms", "Look-ahead", NormalisableRange<float> (0.0f, maxLookaheadMs, 0.1f), 0.0f, "ms", formatMs));
        return layout;
    }

    class AtlasProcessor final : public mint::Processor
    {
    public:
        AtlasProcessor() : mint::Processor (createLayout())
        {
            threshold = apvts.getRawParameterValue ("threshold_db");
            ratio = apvts.getRawParameterValue ("ratio");
            attackMs = apvts.getRawParameterValue ("attack_ms");
            releaseMs = apvts.getRawParameterValue ("release_ms");
            makeup = apvts.getRawParameterValue ("makeup_db");
            autoRelease = apvts.getRawParameterValue ("auto_release");
            behavior = apvts.getRawParameterValue ("behavior");
            character = apvts.getRawParameterValue ("character");
            lookaheadMs = apvts.getRawParameterValue ("lookahead_ms");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            currentRate = sampleRate;
            lookahead.prepare (getTotalNumOutputChannels(), (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate));

            smoothedThreshold.reset (sampleRate, 0.02);
            smoothedRatio.reset (sampleRate, 0.02);
            smoothedMakeup.reset (sampleRate, 0.02);
            smoothedWarm.reset (sampleRate, 0.02);
            lookaheadFade.reset (sampleRate, lookaheadFadeSeconds);
            smoothedWet.reset (sampleRate, 0.02);

            crest.prepare (sampleRate);
            ceiling.prepare (sampleRate);
            for (auto& shaper : warm)
                shaper.prepare (sampleRate);

            declickCoef = std::exp (-1.0f / (declickSeconds * (float) sampleRate));
            reset();
        }

        void releaseResources() override {}

        void reset() override
        {
            lookahead.reset();
            hold.reset();
            crest.reset();
            ceiling.reset();
            for (auto& shaper : warm)
                shaper.reset();

            reductionDb = 0.0f;
            declickedDb = 0.0f;

            smoothedThreshold.setCurrentAndTargetValue (threshold->load());
            smoothedRatio.setCurrentAndTargetValue (ratio->load());
            smoothedMakeup.setCurrentAndTargetValue (makeup->load());
            smoothedWarm.setCurrentAndTargetValue (character->load() < 0.5f ? 1.0f : 0.0f);

            // 没有音频在走，直接切到目标延迟，不用淡变。
            const auto samples = lookaheadSamples();
            lookahead.setDelaySamples (samples);
            requestLatencySamples (samples);
            lookaheadFade.setCurrentAndTargetValue (1.0f);
            lookaheadRefill = 0;
            smoothedWet.setCurrentAndTargetValue (1.0f);
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            process (buffer, false);
        }

        // 旁通时检测器和延迟都继续走：延迟保持信号对齐，检测器保证恢复时增益已经就位，切换没有台阶。
        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            process (buffer, true);
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, reductionMeter;
        std::atomic<float>* lookaheadMs = nullptr;
        bool showAdvanced = false;      // 只在消息线程读写，重开窗口时停在原来的分页。

    private:
        void process (juce::AudioBuffer<float>& buffer, bool bypassed) noexcept
        {
            juce::ScopedNoDenormals noDenormals;

            const auto numChannels = juce::jmin (buffer.getNumChannels(), getTotalNumOutputChannels(), 2);
            if (numChannels == 0)
                return;

            updateTargets();
            smoothedWet.setTargetValue (bypassed ? 0.0f : 1.0f);

            for (int pos = 0; pos < buffer.getNumSamples(); pos += maxChunk)
                processChunk (buffer, pos, juce::jmin (maxChunk, buffer.getNumSamples() - pos), numChannels);
        }

        int lookaheadSamples() const noexcept
        {
            return (int) std::round (lookaheadMs->load() * 0.001 * currentRate);
        }

        void updateTargets() noexcept
        {
            smoothedThreshold.setTargetValue (threshold->load());
            smoothedRatio.setTargetValue (ratio->load());
            smoothedMakeup.setTargetValue (makeup->load());
            smoothedWarm.setTargetValue (character->load() < 0.5f ? 1.0f : 0.0f);

            const auto rate = (float) currentRate;
            const auto attackSeconds = attackMs->load() * 0.001f;
            attackCoef = std::exp (-1.0f / (juce::jmax (attackSeconds - declickSeconds, 1.0e-5f) * rate));

            useAutoRelease = autoRelease->load() >= 0.5f;

            // 峰值窗口会让释放晚开始，平均晚 3/4 个窗口。从时间常数里扣掉，让实测值贴近旋钮。
            const auto releaseSeconds = (useAutoRelease ? atlas::autoReleaseBaseMs : releaseMs->load()) * 0.001f;
            const auto holdSeconds = juce::jmin (maxHoldMs * 0.001f, 0.25f * releaseSeconds);
            hold.setBucketSamples ((int) (0.5f * holdSeconds * rate));
            releaseBaseSeconds = releaseSeconds - 0.75f * holdSeconds;

            releaseShape = behavior->load() < 0.5f ? atlas::ReleaseShape::opto : atlas::ReleaseShape::electro;

            // 延迟不能在音频走着的时候跳变，先淡出，切换，再淡入。
            const auto wanted = lookaheadSamples();
            if (wanted != lookahead.getDelaySamples() && lookaheadRefill <= 0)
                lookaheadFade.setTargetValue (0.0f);
        }

        void processChunk (juce::AudioBuffer<float>& buffer, int start, int numSamples, int numChannels) noexcept
        {
            float* const channels[2] = { buffer.getWritePointer (0, start),
                                         buffer.getWritePointer (numChannels - 1, start) };
            float reductions[maxChunk], blockPeak = 0.0f, maxReduction = 0.0f;

            // 检测器看的是未延迟的输入，所以 Look-ahead 开启后增益会提前动作。
            for (int i = 0; i < numSamples; ++i)
            {
                auto magnitude = 0.0f;
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    auto& sample = channels[ch][i];
                    if (! std::isfinite (sample))
                        sample = 0.0f;
                    magnitude = juce::jmax (magnitude, std::abs (sample));
                }
                magnitude = juce::jmin (magnitude, maxDetectorLevel);

                blockPeak = juce::jmax (blockPeak, magnitude);

                const auto thresholdDb = smoothedThreshold.getNextValue();
                const auto currentRatio = smoothedRatio.getNextValue();
                const auto autoFactor = crest.process (magnitude);
                const auto held = hold.process (magnitude);

                const auto target = atlas::computeReductionDb (mint::gainToDb (held), thresholdDb, currentRatio);

                if (target > reductionDb)
                {
                    reductionDb = target + attackCoef * (reductionDb - target);
                }
                else
                {
                    const auto tau = releaseBaseSeconds * (useAutoRelease ? autoFactor : 1.0f)
                                     * atlas::releaseShapeFactor (releaseShape, reductionDb);
                    reductionDb = target + std::exp (-1.0f / (tau * (float) currentRate)) * (reductionDb - target);
                    if (reductionDb < 1.0e-5f)
                        reductionDb = 0.0f;
                }

                declickedDb = reductionDb + declickCoef * (declickedDb - reductionDb);
                if (declickedDb < 1.0e-5f)
                    declickedDb = 0.0f;

                reductions[(size_t) i] = declickedDb;
                maxReduction = juce::jmax (maxReduction, declickedDb);
            }

            inputMeter.push (mint::gainToDb (blockPeak));

            float* const views[2] = { channels[0], channels[1] };
            juce::AudioBuffer<float> chunk (views, numChannels, numSamples);
            lookahead.process (chunk);

            auto maxLimitDb = 0.0f;

            for (int i = 0; i < numSamples; ++i)
            {
                const auto gr = reductions[(size_t) i];
                const auto gain = mint::dbToGain (-gr);
                const auto output = mint::dbToGain (smoothedMakeup.getNextValue()) * lookaheadFade.getNextValue();
                const auto warmMix = smoothedWarm.getNextValue();
                const auto wet = smoothedWet.getNextValue();
                const auto thresholdGain = mint::dbToGain (smoothedThreshold.getCurrentValue());
                const auto amount = warmMix * gr / (gr + 12.0f);

                auto peak = 0.0f;
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    const auto dry = channels[ch][i];
                    auto y = dry * gain;
                    if (warmMix > 0.0f)
                        y += warm[(size_t) ch].process (y, thresholdGain, amount);
                    else
                        warm[(size_t) ch].reset();

                    y = dry + wet * (y * output - dry);
                    channels[ch][i] = y;
                    peak = juce::jmax (peak, std::abs (y));
                }

                const auto limit = wet > 0.0f ? ceiling.process (peak) : 1.0f;
                if (limit < 1.0f)
                {
                    for (int ch = 0; ch < numChannels; ++ch)
                        channels[ch][i] *= limit;
                    maxLimitDb = juce::jmax (maxLimitDb, -mint::gainToDb (limit));
                }
            }

            // 共享电平表把 -24..0 映射成 0..满格（自上而下），所以衰减量要平移后再送。
            reductionMeter.push (juce::jmin (0.0f, smoothedWet.getCurrentValue() * (maxReduction + maxLimitDb) - 24.0f));

            advanceLookaheadSwitch (numSamples);
        }

        // 切换分三步：淡出走完后切延迟，等新延迟的缓冲填满（延迟为 0 时缓冲不写入，里面是旧数据），再淡入。
        void advanceLookaheadSwitch (int numSamples) noexcept
        {
            if (lookaheadRefill > 0)
            {
                lookaheadRefill -= numSamples;
                if (lookaheadRefill <= 0)
                    lookaheadFade.setTargetValue (1.0f);
            }
            else if (lookaheadFade.getTargetValue() == 0.0f && ! lookaheadFade.isSmoothing())
            {
                const auto wanted = lookaheadSamples();
                lookahead.setDelaySamples (wanted);
                requestLatencySamples (wanted);
                lookaheadRefill = wanted;
                if (wanted == 0)
                    lookaheadFade.setTargetValue (1.0f);
            }
        }

        double currentRate = 48000.0;
        std::atomic<float>* threshold = nullptr;
        std::atomic<float>* ratio = nullptr;
        std::atomic<float>* attackMs = nullptr;
        std::atomic<float>* releaseMs = nullptr;
        std::atomic<float>* makeup = nullptr;
        std::atomic<float>* autoRelease = nullptr;
        std::atomic<float>* behavior = nullptr;
        std::atomic<float>* character = nullptr;

        mint::LookaheadDelay lookahead;
        atlas::PeakHold hold;
        atlas::CrestRelease crest;
        atlas::WarmShaper warm[2];
        atlas::OutputCeiling ceiling;
        atlas::ReleaseShape releaseShape = atlas::ReleaseShape::electro;

        juce::LinearSmoothedValue<float> smoothedThreshold, smoothedRatio, smoothedMakeup, smoothedWarm;
        juce::LinearSmoothedValue<float> lookaheadFade { 1.0f }, smoothedWet { 1.0f };
        int lookaheadRefill = 0;

        float reductionDb = 0.0f, declickedDb = 0.0f;
        float attackCoef = 0.0f, declickCoef = 0.0f, releaseBaseSeconds = 0.1f;
        bool useAutoRelease = true;
    };

    // 共享层的 Segmented 只能绑定参数。分页不是音频参数，所以这里画一个外观相同的。
    class PageSwitch final : public juce::Component
    {
    public:
        PageSwitch()
        {
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
        }

        std::function<void (bool)> onChange;

        void setAdvanced (bool advanced)
        {
            selected = advanced ? 1 : 0;
            repaint();
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = mint::ui::theme();
            const auto w = (float) getWidth() * 0.5f;
            const auto h = (float) getHeight();
            const juce::String names[] = { mint::ui::tr ({ "Main", "메인", "主控" }), mint::ui::tr ({ "Advanced", "고급", "高级" }) };

            if (hovered >= 0 && hovered != selected)
            {
                g.setColour (t.trimSoft.withAlpha (0.45f));
                g.fillRect (juce::Rectangle<float> ((float) hovered * w, 0.0f, w, h));
            }

            g.setColour (t.ink);
            g.fillRect (juce::Rectangle<float> ((float) selected * w, 0.0f, w, h));

            g.setColour (t.trim);
            g.drawRect (getLocalBounds().toFloat(), 1.0f);
            g.fillRect (juce::Rectangle<float> (w - 0.5f, 0.0f, 1.0f, h));

            g.setFont (mint::ui::bodyFont (13.0f));
            for (int i = 0; i < 2; ++i)
            {
                g.setColour (i == selected ? t.plate : t.ink);
                g.drawFittedText (names[i], juce::Rectangle<float> ((float) i * w, 0.0f, w, h).reduced (3.0f, 0.0f).toNearestInt(),
                                  juce::Justification::centred, 1, 0.75f);
            }
        }

        void mouseDown (const juce::MouseEvent& e) override
        {
            if (! e.mods.isLeftButtonDown())
                return;
            setAdvanced (cellAt (e.x) == 1);
            if (onChange != nullptr)
                onChange (selected == 1);
        }

        void mouseMove (const juce::MouseEvent& e) override
        {
            setHovered (cellAt (e.x));
        }

        void mouseExit (const juce::MouseEvent&) override
        {
            setHovered (-1);
        }

    private:
        int cellAt (int x) const
        {
            return x >= getWidth() / 2 ? 1 : 0;
        }

        void setHovered (int cell)
        {
            if (cell != hovered)
            {
                hovered = cell;
                repaint();
            }
        }

        int selected = 0, hovered = -1;
    };

    class AtlasEditor final : public mint::ui::EditorBase
    {
    public:
        explicit AtlasEditor (AtlasProcessor& p)
            : EditorBase (p, { "ATLAS", { "Compressor", "컴프레서", "压缩" }, "III" }),
              owner (p),
              thresholdDial (p.apvts, "threshold_db", { "Threshold", "스레숄드", "阈值" }, true),
              ratioDial (p.apvts, "ratio", { "Ratio", "레이시오", "压缩比" }),
              attackDial (p.apvts, "attack_ms", { "Attack", "어택", "起音" }),
              releaseDial (p.apvts, "release_ms", { "Release", "릴리즈", "释放" }),
              makeupDial (p.apvts, "makeup_db", { "Makeup", "메이크업", "补偿增益" }),
              lookaheadDial (p.apvts, "lookahead_ms", { "Look-ahead", "룩어헤드", "前瞻" }),
              autoSwitch (p.apvts, "auto_release", { { "Manual", "수동", "手动" }, { "Auto", "자동", "自动" } }),
              behaviorSwitch (p.apvts, "behavior", { "Opto", "Electro" }),
              characterSwitch (p.apvts, "character", { "Warm", "Smooth" }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              reductionMeter (p.reductionMeter, { "Reduction", "리덕션", "增益衰减" }, true),
              autoAttachment (*p.apvts.getParameter ("auto_release"), [this] (float value) { setAutoRelease (value >= 0.5f); })
        {
            for (auto* child : std::initializer_list<juce::Component*> { &thresholdDial, &ratioDial, &attackDial, &releaseDial,
                                                                         &makeupDial, &lookaheadDial, &autoSwitch, &behaviorSwitch,
                                                                         &characterSwitch, &inputMeter, &reductionMeter, &pageSwitch })
                addAndMakeVisible (child);

            pageSwitch.setAdvanced (owner.showAdvanced);
            pageSwitch.onChange = [this] (bool advanced)
            {
                owner.showAdvanced = advanced;
                updatePage();
            };

            autoAttachment.sendInitialUpdate();
            updatePage();
            setSize (width, height);
        }

        void paint (juce::Graphics& g) override
        {
            EditorBase::paint (g);

            const auto& t = mint::ui::theme();

            if (owner.showAdvanced)
            {
                g.setColour (t.inkMuted);
                g.setFont (mint::ui::bodyFont (12.5f));
                g.drawText (mint::ui::tr ({ "Auto Release", "자동 릴리즈", "自动释放" }), autoCaption, juce::Justification::centred);
                g.drawText (mint::ui::tr ({ "Release Shape", "릴리즈 커브", "释放特性" }), behaviorCaption, juce::Justification::centred);
                g.drawText (mint::ui::tr ({ "Harmonics", "하모닉스", "谐波特性" }), characterCaption, juce::Justification::centred);
                return;
            }

            // 开关在高级页，这里说明释放旋钮为什么变暗。
            g.setColour (t.inkMuted);
            g.setFont (mint::ui::bodyFont (12.0f));
            g.drawText (mint::ui::tr ({ "Auto Release", "자동 릴리즈", "自动释放" }), autoReadout.withHeight (16), juce::Justification::centredLeft);
            g.setColour (t.ink);
            g.setFont (mint::ui::bodyFont (14.0f));
            g.drawText (autoRelease ? mint::ui::tr ({ "Auto", "자동", "自动" }) : mint::ui::tr ({ "Manual", "수동", "手动" }),
                        autoReadout.withTrimmedTop (18), juce::Justification::centredLeft);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeter.setBounds (area.removeFromLeft (80));
            reductionMeter.setBounds (area.removeFromRight (80));
            area.reduce (10, 0);

            // 主控页
            auto main = area;
            thresholdDial.setBounds (main.removeFromLeft (224).withSizeKeepingCentre (224, 280));
            main.removeFromLeft (8);

            auto top = main.removeFromTop (main.getHeight() / 2);
            const auto cell = main.getWidth() / 2;
            ratioDial.setBounds (top.removeFromLeft (cell));
            attackDial.setBounds (top);
            releaseDial.setBounds (main.removeFromLeft (cell));
            makeupDial.setBounds (main);

            // 高级页
            auto advanced = area;
            lookaheadDial.setBounds (advanced.removeFromLeft (216).withSizeKeepingCentre (160, 150));

            auto switches = advanced.withSizeKeepingCentre (240, 176);
            const auto layoutSwitch = [&switches] (juce::Component& control, juce::Rectangle<int>& caption)
            {
                caption = switches.removeFromTop (16);
                control.setBounds (switches.removeFromTop (28));
                switches.removeFromTop (22);
            };

            layoutSwitch (autoSwitch, autoCaption);
            layoutSwitch (behaviorSwitch, behaviorCaption);
            layoutSwitch (characterSwitch, characterCaption);

            auto footer = getFooterSlot();
            pageSwitch.setBounds (footer.removeFromRight (130).withSizeKeepingCentre (130, 28));
            autoReadout = footer.removeFromLeft (150).withY (426).withHeight (40);
        }

    private:
        juce::String getLookaheadText() const override
        {
            // 高级页有前瞻旋钮，不再重复读数。
            if (owner.showAdvanced)
                return {};

            const auto ms = owner.lookaheadMs->load();
            return ms <= 0.0f ? mint::ui::tr ({ "Off · 0 ms", "끔 · 0 ms", "关闭 · 0 ms" }) : juce::String (ms, 1) + " ms";
        }

        void updatePage()
        {
            const auto advanced = owner.showAdvanced;
            for (auto* child : std::initializer_list<juce::Component*> { &thresholdDial, &ratioDial, &attackDial, &releaseDial, &makeupDial })
                child->setVisible (! advanced);
            for (auto* child : std::initializer_list<juce::Component*> { &lookaheadDial, &autoSwitch, &behaviorSwitch, &characterSwitch })
                child->setVisible (advanced);
            repaint();
        }

        void setAutoRelease (bool shouldBeAuto)
        {
            autoRelease = shouldBeAuto;

            // 连鼠标也不接，双击数值输入同样关掉。
            releaseDial.setEnabled (! autoRelease);
            releaseDial.setAlpha (autoRelease ? 0.45f : 1.0f);
            releaseDial.setInterceptsMouseClicks (! autoRelease, ! autoRelease);
            repaint (autoReadout);
        }

        AtlasProcessor& owner;
        mint::ui::Dial thresholdDial, ratioDial, attackDial, releaseDial, makeupDial, lookaheadDial;
        mint::ui::Segmented autoSwitch, behaviorSwitch, characterSwitch;
        mint::ui::Meter inputMeter, reductionMeter;
        PageSwitch pageSwitch;
        juce::Rectangle<int> autoCaption, behaviorCaption, characterCaption, autoReadout;
        bool autoRelease = true;
        juce::ParameterAttachment autoAttachment;
    };

    juce::AudioProcessorEditor* AtlasProcessor::createEditor()
    {
        return new AtlasEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AtlasProcessor();
}
