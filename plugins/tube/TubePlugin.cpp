// Mintaria Vesta：电子管线路放大器式的饱和。默认 0 延迟，靠 ADAA 抗混叠；高质量档开过采样并如实上报延迟。
#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

#include "TubeModel.h"

namespace
{
    constexpr int maxChannels = 2;
    constexpr int controlInterval = 32;     // 系数和增益斜坡的更新间隔，与宿主缓冲区大小无关
    constexpr float minRecoveryMs = 10.0f, maxRecoveryMs = 2000.0f;
    constexpr double qualityFadeSeconds = 0.005;
    constexpr double maxInput = 16.0;       // +24 dBFS

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;

        const auto decibels = [] (float v, int) { return (v > 0.0f ? "+" : "") + String (v, 1) + " dB"; };

        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "drive", 1 }, "Drive",
                                                           NormalisableRange<float> (0.0f, (float) vesta::maxDriveDb, 0.1f), 12.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction (decibels)));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "output", 1 }, "Output",
                                                           NormalisableRange<float> (-12.0f, 12.0f, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction (decibels)));

        // 对数刻度：归一化值 n 对应 10 ms * 200^n。
        NormalisableRange<float> recoveryRange (minRecoveryMs, maxRecoveryMs,
                                                [] (float start, float end, float n) { return start * std::pow (end / start, n); },
                                                [] (float start, float end, float v) { return std::log (v / start) / std::log (end / start); });
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "recovery", 1 }, "Recovery", recoveryRange, 120.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms")
                                                               .withStringFromValueFunction ([] (float v, int)
                                                               {
                                                                   return v < 1000.0f ? String (roundToInt (v)) + " ms"
                                                                                      : String (v * 0.001f, 2) + " s";
                                                               })));

        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "quality", 1 }, "Quality",
                                                            StringArray { String::fromUTF8 ("实时"), String::fromUTF8 ("高质量") }, 0));
        return layout;
    }

    const vesta::CompensationTable& compensationTable()
    {
        static const vesta::CompensationTable table;
        return table;
    }

    class TubeProcessor final : public mint::Processor
    {
    public:
        TubeProcessor() : mint::Processor (createLayout())
        {
            drive = apvts.getRawParameterValue ("drive");
            output = apvts.getRawParameterValue ("output");
            recovery = apvts.getRawParameterValue ("recovery");
            quality = apvts.getRawParameterValue ("quality");
            compensationTable();
        }

        void prepareToPlay (double sampleRate, int) override
        {
            currentRate = sampleRate;

            // 过采样后的内部采样率至少 176.4 kHz。
            const size_t order = sampleRate < 88000.0 ? 2 : 1;
            oversampling = std::make_unique<juce::dsp::Oversampling<float>> ((size_t) maxChannels, order,
                                                                             juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR,
                                                                             true, true);
            oversampling->initProcessing ((size_t) controlInterval);
            hqLatency = juce::roundToInt (oversampling->getLatencyInSamples());
            bypassDelay.prepare (maxChannels, hqLatency);

            for (int ch = 0; ch < maxChannels; ++ch)
            {
                chains[(size_t) ch].prepare (vesta::voicing(), sampleRate);
                realtimeStages[(size_t) ch].prepare (sampleRate);
                hqStages[(size_t) ch].prepare (sampleRate * (double) oversampling->getOversamplingFactor());
            }

            smoothedDrive.reset (sampleRate, 0.05);
            smoothedDrive.setCurrentAndTargetValue (drive->load());
            smoothedTrim.reset (sampleRate, 0.02);
            smoothedTrim.setCurrentAndTargetValue (mint::dbToGain (output->load()));

            currentHq = wantsHq();
            fade = 1.0;
            lastRecoveryMs = -1.0f;
            applyDriveControls (smoothedDrive.getCurrentValue());
            driveGain = driveGainFor (smoothedDrive.getCurrentValue());
            compensation = compensationTable().gainFor (smoothedDrive.getCurrentValue());
            trim = smoothedTrim.getCurrentValue();

            reportLatency();
        }

        void releaseResources() override {}

        void reset() override
        {
            for (int ch = 0; ch < maxChannels; ++ch)
            {
                chains[(size_t) ch].reset();
                realtimeStages[(size_t) ch].reset();
                hqStages[(size_t) ch].reset();
            }
            bypassDelay.reset();
            if (oversampling != nullptr)
                oversampling->reset();
        }

        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            // 旁路时也保持上报的延迟，宿主的延迟补偿才不会跳。
            bypassDelay.setDelaySamples (currentHq ? hqLatency : 0);
            bypassDelay.process (buffer);
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;

            const auto numChannels = juce::jmin (buffer.getNumChannels(), getTotalNumOutputChannels(), maxChannels);
            const auto numSamples = buffer.getNumSamples();
            if (numChannels == 0 || numSamples == 0)
                return;

            for (int ch = 0; ch < numChannels; ++ch)
                inputMeter.push (mint::gainToDb (buffer.getMagnitude (ch, 0, numSamples)));

            smoothedDrive.setTargetValue (drive->load());
            smoothedTrim.setTargetValue (mint::dbToGain (output->load()));
            updateRecovery();

            juce::dsp::AudioBlock<float> whole (buffer.getArrayOfWritePointers(), (size_t) numChannels, (size_t) numSamples);

            for (int pos = 0; pos < numSamples; pos += controlInterval)
            {
                const auto n = juce::jmin (controlInterval, numSamples - pos);
                processSegment (whole.getSubBlock ((size_t) pos, (size_t) n), numChannels, n);
            }

            for (int ch = 0; ch < numChannels; ++ch)
                outputMeter.push (mint::gainToDb (buffer.getMagnitude (ch, 0, numSamples)));
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, outputMeter;

    private:
        bool wantsHq() const noexcept                       { return quality->load() >= 0.5f; }
        void reportLatency() noexcept                       { requestLatencySamples (currentHq ? hqLatency : 0); }
        static double driveGainFor (float driveDb) noexcept { return vesta::voicing().inputScale * std::pow (10.0, (double) driveDb * 0.05); }

        void updateRecovery() noexcept
        {
            const auto ms = recovery->load();
            if (juce::approximatelyEqual (ms, lastRecoveryMs))
                return;

            lastRecoveryMs = ms;
            for (int ch = 0; ch < maxChannels; ++ch)
            {
                realtimeStages[(size_t) ch].setRecoverySeconds ((double) ms * 0.001);
                hqStages[(size_t) ch].setRecoverySeconds ((double) ms * 0.001);
            }
        }

        void applyDriveControls (float driveDb) noexcept
        {
            const auto even = vesta::evenAt (vesta::voicing(), (double) driveDb);
            for (int ch = 0; ch < maxChannels; ++ch)
            {
                chains[(size_t) ch].setDrive (vesta::voicing(), (double) driveDb);
                realtimeStages[(size_t) ch].setEven (even);
                hqStages[(size_t) ch].setEven (even);
            }
        }

        // 切换档位时先淡出，换好后再淡入。两档延迟不同，直接切会有爆音。
        double advanceFade (int n) noexcept
        {
            const auto step = (double) n / (qualityFadeSeconds * currentRate);

            if (wantsHq() != currentHq)
            {
                fade = juce::jmax (0.0, fade - step);
                if (fade <= 0.0)
                {
                    currentHq = wantsHq();
                    for (int ch = 0; ch < maxChannels; ++ch)
                    {
                        realtimeStages[(size_t) ch].reset();
                        hqStages[(size_t) ch].reset();
                    }
                    oversampling->reset();
                    reportLatency();
                }
            }
            else
            {
                fade = juce::jmin (1.0, fade + step);
            }

            return fade;
        }

        void processSegment (juce::dsp::AudioBlock<float> block, int numChannels, int n) noexcept
        {
            const auto driveDb = smoothedDrive.skip (n);
            applyDriveControls (driveDb);

            const auto ramp = 1.0 / (double) n;
            const auto driveStart = driveGain, driveStep = (driveGainFor (driveDb) - driveGain) * ramp;
            const auto compStart = compensation, compStep = (compensationTable().gainFor ((double) driveDb) - compensation) * ramp;
            const auto trimEnd = (double) smoothedTrim.skip (n);
            const auto trimStart = trim, trimStep = (trimEnd - trim) * ramp;
            const auto fadeStart = fade;
            const auto fadeStep = (advanceFade (n) - fadeStart) * ramp;

            driveGain += driveStep * n;
            compensation += compStep * n;
            trim = trimEnd;

            for (int ch = 0; ch < numChannels; ++ch)
            {
                auto* data = block.getChannelPointer ((size_t) ch);
                auto& chain = chains[(size_t) ch];

                for (int i = 0; i < n; ++i)
                {
                    // 非法或离谱的输入不能进入模型，否则状态会永久变成 NaN。
                    const auto x = std::isfinite (data[i]) ? juce::jlimit (-maxInput, maxInput, (double) data[i]) : 0.0;
                    data[i] = (float) (chain.pre (x) * (driveStart + driveStep * (i + 1)));
                }
            }

            if (currentHq)
            {
                auto inner = oversampling->processSamplesUp (block);
                const auto innerSamples = (int) inner.getNumSamples();

                for (int ch = 0; ch < numChannels; ++ch)
                {
                    auto* data = inner.getChannelPointer ((size_t) ch);
                    auto& stage = hqStages[(size_t) ch];
                    for (int i = 0; i < innerSamples; ++i)
                        data[i] = (float) stage.processSample ((double) data[i]);
                }

                oversampling->processSamplesDown (block);
            }
            else
            {
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    auto* data = block.getChannelPointer ((size_t) ch);
                    auto& stage = realtimeStages[(size_t) ch];
                    for (int i = 0; i < n; ++i)
                        data[i] = (float) stage.processSample ((double) data[i]);
                }
            }

            for (int ch = 0; ch < numChannels; ++ch)
            {
                auto* data = block.getChannelPointer ((size_t) ch);
                auto& chain = chains[(size_t) ch];

                for (int i = 0; i < n; ++i)
                {
                    const auto k = (double) (i + 1);
                    const auto gain = (compStart + compStep * k) * (trimStart + trimStep * k) * (fadeStart + fadeStep * k);
                    data[i] = (float) (chain.post ((double) data[i]) * gain);
                }
            }
        }

        std::atomic<float>* drive = nullptr;
        std::atomic<float>* output = nullptr;
        std::atomic<float>* recovery = nullptr;
        std::atomic<float>* quality = nullptr;

        double currentRate = 48000.0;
        std::array<vesta::ChannelChain, maxChannels> chains;
        std::array<vesta::TubeStage, maxChannels> realtimeStages, hqStages;
        std::unique_ptr<juce::dsp::Oversampling<float>> oversampling;
        mint::LookaheadDelay bypassDelay;
        int hqLatency = 0;

        juce::LinearSmoothedValue<float> smoothedDrive { 12.0f }, smoothedTrim { 1.0f };
        double driveGain = 1.0, compensation = 1.0, trim = 1.0, fade = 1.0;
        float lastRecoveryMs = -1.0f;
        bool currentHq = false;
    };

    class TubeEditor final : public mint::ui::EditorBase
    {
    public:
        explicit TubeEditor (TubeProcessor& p)
            : EditorBase (p, { "VESTA", { "Tube Saturation", "진공관 새츄레이션", "电子管饱和" }, "XI" }),
              driveDial (p.apvts, "drive", { "Drive", "드라이브", "驱动" }, true),
              outputDial (p.apvts, "output", { "Output", "출력", "输出" }),
              recoveryDial (p.apvts, "recovery", { "Recovery", "리커버리", "恢复" }),
              qualitySwitch (p.apvts, "quality", { { "Realtime", "실시간", "实时" }, { "High Quality", "고품질", "高质量" } }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              outputMeter (p.outputMeter, { "Output", "출력", "输出" })
        {
            addAndMakeVisible (driveDial);
            addAndMakeVisible (outputDial);
            addAndMakeVisible (recoveryDial);
            addAndMakeVisible (qualitySwitch);
            addAndMakeVisible (inputMeter);
            addAndMakeVisible (outputMeter);
            languageChanged();
            setSize (width, height);
        }

        void resized() override
        {
            auto area = getContentBounds();
            inputMeter.setBounds (area.removeFromLeft (40));
            outputMeter.setBounds (area.removeFromRight (40));

            auto centre = area.withSizeKeepingCentre (224, 280);
            driveDial.setBounds (centre);

            const auto smallWidth = 140, smallHeight = 170;
            recoveryDial.setBounds (area.withRight (centre.getX()).withSizeKeepingCentre (smallWidth, smallHeight));
            outputDial.setBounds (area.withLeft (centre.getRight()).withSizeKeepingCentre (smallWidth, smallHeight));

            qualitySwitch.setBounds (getFooterSlot().withSizeKeepingCentre (220, 32));
        }

    protected:
        void languageChanged() override
        {
            recoveryDial.slider.setTooltip (mint::ui::tr ({
                "How long the sound takes to come back after a loud passage: short returns fast with a light squash, long stays squashed longer and deeper.",
                "큰 소리가 지나간 뒤 소리가 돌아오는 데 걸리는 시간입니다. 짧으면 빠르게 돌아오고 살짝 눌리며, 길면 더 오래, 더 깊게 눌립니다.",
                "响亮的声音过后，声音要多久才回到原样：调短回得快、压得浅，调长压得更久、更深。" }));
        }

    private:
        mint::ui::Dial driveDial, outputDial, recoveryDial;
        mint::ui::Segmented qualitySwitch;
        mint::ui::Meter inputMeter, outputMeter;
    };

    juce::AudioProcessorEditor* TubeProcessor::createEditor()
    {
        return new TubeEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new TubeProcessor();
}
