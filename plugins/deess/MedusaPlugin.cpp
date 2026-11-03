// Mintaria Medusa：去齿音。结构说明和选型依据见 NOTES.md。
//
// 音频通路是三段级联的减法动态钟形：y = x - (1 - g) * BP(x)。
// 增益为 1 时输出与输入逐采样相同；滤波器状态不随增益变化，所以衰减再快也不会调制滤波器。
#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

#include <array>

namespace
{
    constexpr float maxLookaheadMs = 10.0f;
    constexpr int numBands = 3;
    constexpr float bandSpread = 1.45f;
    constexpr float bandQ = 1.2f;
    constexpr int chunkSize = 256;
    constexpr int maxChannels = 2;

    // 检测器时间常数。
    constexpr float bandEnvMs = 1.5f;
    constexpr float voiceEnvMs = 8.0f;
    constexpr float voicePeakAttackMs = 0.5f;  // 要比频带包络更快，元音起音时才不会被误判成齿音
    constexpr float refRiseMs = 30.0f;
    constexpr float refFallDbPerSecond = 12.0f;
    constexpr float refGateDb = 20.0f;      // 比参考低这么多就算没有人声，参考冻结
    constexpr float floorDb = -72.0f;       // 低于此电平的频带不参与衰减
    constexpr float floorFadeDb = 6.0f;
    constexpr float kneeDb = 8.0f;
    constexpr float voicedMarginDb = 3.0f;  // 频带能量高于“此刻低频能量 - 该值”才可能算齿音
    constexpr float releaseMs = 40.0f;
    constexpr float voicedReleaseMs = 15.0f;   // 元音回来时要尽快放开，不碰元音
    constexpr float gainSmoothMs = 0.4f;
    constexpr float frequencySmoothMs = 30.0f;
    constexpr float monitorFadeMs = 5.0f;

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "strength", 1 }, "Strength",
                                                           NormalisableRange<float> (0.0f, 100.0f, 0.1f), 50.0f,
                                                           AudioParameterFloatAttributes().withLabel ("%")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (roundToInt (v)) + " %"; })));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "frequency", 1 }, "Frequency",
                                                           NormalisableRange<float> (3000.0f, 12000.0f, 1.0f, 0.5f), 6500.0f,
                                                           AudioParameterFloatAttributes().withLabel ("Hz")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (v * 0.001f, 1) + " kHz"; })));
        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "monitor", 1 }, "Monitor",
                                                            StringArray { String::fromUTF8 ("正常输出"), String::fromUTF8 ("监听齿音") }, 0));
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "lookahead_ms", 1 }, "Look-ahead",
                                                           NormalisableRange<float> (0.0f, maxLookaheadMs, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (v, 1) + " ms"; })));
        return layout;
    }

    // TPT 状态变量滤波器。系数和状态分开，这样多个通道可以共用一组系数。
    struct SvfCoefs
    {
        float a1 = 1.0f, a2 = 0.0f, a3 = 0.0f, k = 1.0f;

        void set (float frequency, float q, double sampleRate) noexcept
        {
            const auto g = std::tan (juce::MathConstants<float>::pi * juce::jmin (frequency, 0.45f * (float) sampleRate) / (float) sampleRate);
            k = 1.0f / q;
            a1 = 1.0f / (1.0f + g * (g + k));
            a2 = g * a1;
            a3 = g * a2;
        }
    };

    struct Svf
    {
        float ic1 = 0.0f, ic2 = 0.0f, low = 0.0f, band = 0.0f;

        void tick (float x, const SvfCoefs& c) noexcept
        {
            const auto v3 = x - ic2;
            band = c.a1 * ic1 + c.a2 * v3;
            low = ic2 + c.a2 * ic1 + c.a3 * v3;
            ic1 = 2.0f * band - ic1;
            ic2 = 2.0f * low - ic2;
        }

        // 中心频率处增益为 1 的二阶带通。
        float tickBand (float x, const SvfCoefs& c) noexcept    { tick (x, c); return c.k * band; }
        float tickLow (float x, const SvfCoefs& c) noexcept     { tick (x, c); return low; }
        float tickHigh (float x, const SvfCoefs& c) noexcept    { tick (x, c); return x - c.k * band - low; }

        void reset() noexcept                                   { ic1 = ic2 = low = band = 0.0f; }
        bool isFinite() const noexcept                          { return std::isfinite (ic1) && std::isfinite (ic2); }
    };

    inline float powerToDb (float power) noexcept
    {
        return 10.0f * std::log10 (std::max (power, 1.0e-13f));
    }

    inline float softKnee (float over) noexcept
    {
        if (over <= -0.5f * kneeDb)
            return 0.0f;
        if (over >= 0.5f * kneeDb)
            return over;
        const auto t = over + 0.5f * kneeDb;
        return t * t / (2.0f * kneeDb);
    }

    class MedusaProcessor final : public mint::Processor
    {
    public:
        MedusaProcessor() : mint::Processor (createLayout())
        {
            strength = apvts.getRawParameterValue ("strength");
            frequency = apvts.getRawParameterValue ("frequency");
            monitor = apvts.getRawParameterValue ("monitor");
            lookaheadMs = apvts.getRawParameterValue ("lookahead_ms");
        }

        void prepareToPlay (double sampleRate, int) override
        {
            currentRate = sampleRate;
            lookahead.prepare (juce::jmax (1, getTotalNumOutputChannels()), (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate));

            controlInterval = juce::jmax (1, (int) std::round (sampleRate / 12000.0));
            const auto controlRate = sampleRate / (double) controlInterval;
            bandEnvCoef = onePole (bandEnvMs, sampleRate);
            voiceEnvCoef = onePole (voiceEnvMs, sampleRate);
            voicePeakCoef = onePole (voicePeakAttackMs, sampleRate);
            refRiseCoef = onePole (refRiseMs, controlRate);
            refFallPerTick = refFallDbPerSecond / (float) controlRate;
            slowReleaseCoef = onePole (releaseMs, sampleRate);
            voicedReleaseCoef = onePole (voicedReleaseMs, sampleRate);
            releaseCoef = slowReleaseCoef;
            gainSmoothCoef = onePole (gainSmoothMs, sampleRate);
            frequencyCoef = onePole (frequencySmoothMs, controlRate);
            monitorCoef = onePole (monitorFadeMs, sampleRate);

            voiceHighCoefs.set (120.0f, 0.707f, sampleRate);
            reset();
            updateLookahead();
        }

        void releaseResources() override {}

        void reset() override
        {
            for (auto& ch : channels)
            {
                for (auto& f : ch.audio)    f.reset();
                for (auto& f : ch.detect)   f.reset();
                ch.preHigh.reset();
                ch.voiceHigh.reset();
                ch.voiceLow.reset();
                ch.voiceLow2.reset();
                ch.bandEnv.fill (0.0f);
                ch.voiceEnv = ch.voicePeak = 0.0f;
            }

            targetGain.fill (1.0f);
            slowGain.fill (1.0f);
            gain.fill (1.0f);
            referenceDb = -120.0f;
            controlCountdown = 0;
            smoothedLogFrequency = std::log2 (frequency->load());
            lastFrequency = -1.0f;
            monitorMix = monitor->load() > 0.5f ? 1.0f : 0.0f;
            lookahead.reset();
            lookahead.setDelaySamples (wantedDelaySamples());
            switchState = SwitchState::running;
            switchGain = 1.0f;
            setFilters (frequency->load());
        }

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;
            updateLookahead();

            const auto numSamples = buffer.getNumSamples();
            const auto numChannels = juce::jmin (buffer.getNumChannels(), getTotalNumOutputChannels(), maxChannels);
            if (numChannels <= 0 || numSamples <= 0)
                return;

            inputMeter.push (mint::gainToDb (buffer.getMagnitude (0, numSamples)));

            for (int start = 0; start < numSamples; start += chunkSize)
                processChunk (buffer, start, juce::jmin (chunkSize, numSamples - start), numChannels);

            // 上游偶发的 NaN 不能把滤波器永久卡死。
            for (int ch = 0; ch < numChannels; ++ch)
                if (! channels[(size_t) ch].audio[0].isFinite() || ! channels[(size_t) ch].detect[0].isFinite())
                {
                    reset();
                    break;
                }
        }

        // 旁通时也要保留 Look-ahead 的延迟，否则开关旁通时音频会跳。
        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            juce::ScopedNoDenormals noDenormals;
            updateLookahead();
            if (lookahead.getDelaySamples() != wantedDelaySamples())
            {
                lookahead.setDelaySamples (wantedDelaySamples());
                lookahead.reset();
                switchState = SwitchState::running;
                switchGain = 1.0f;
            }
            lookahead.process (buffer);
        }

        juce::AudioProcessorEditor* createEditor() override;

        mint::MeterSource inputMeter, reductionMeter;
        std::atomic<float>* lookaheadMs = nullptr;

    private:
        struct Channel
        {
            std::array<Svf, numBands> audio, detect;
            Svf preHigh, voiceHigh, voiceLow, voiceLow2;
            std::array<float, numBands> bandEnv {};
            float voiceEnv = 0.0f, voicePeak = 0.0f;
        };

        static float onePole (float ms, double rate) noexcept
        {
            return std::exp (-1.0f / (0.001f * ms * (float) rate));
        }

        void processChunk (juce::AudioBuffer<float>& buffer, int start, int n, int numChannels) noexcept
        {
            detectAndSmooth (buffer, start, n, numChannels);
            advanceDelaySwitch (n);

            // 检测在延迟之前，音频在延迟之后，Look-ahead 打开时衰减会提前到位。
            juce::AudioBuffer<float> view (buffer.getArrayOfWritePointers(), numChannels, start, n);
            lookahead.process (view);

            const auto wantMonitor = monitor->load() > 0.5f ? 1.0f : 0.0f;
            auto mixAtStart = monitorMix;

            for (int ch = 0; ch < numChannels; ++ch)
            {
                auto* data = buffer.getWritePointer (ch, start);
                auto& state = channels[(size_t) ch];
                auto mix = mixAtStart;

                for (int i = 0; i < n; ++i)
                {
                    const auto dry = data[i];
                    auto wet = dry;

                    for (int b = 0; b < numBands; ++b)
                    {
                        const auto band = state.audio[(size_t) b].tickBand (wet, audioCoefs[(size_t) b]);
                        wet -= (1.0f - gainTrace[(size_t) b][(size_t) i]) * band;
                    }

                    mix = wantMonitor + monitorCoef * (mix - wantMonitor);
                    // 监听时输出 dry - wet，正好是被削掉的部分。
                    data[i] = wet + mix * (dry - 2.0f * wet);

                    if (switchState != SwitchState::running)
                        data[i] *= switchTrace[(size_t) i];
                }

                monitorMix = mix;
            }

            if (std::abs (monitorMix - wantMonitor) < 1.0e-5f)
                monitorMix = wantMonitor;
        }

        // LookaheadDelay 在 0 延迟时不写延迟线，直接改延迟会读到旧音频。
        // 运行中改 Look-ahead 要淡出，清空，等延迟线填满，再淡入。只在块边界切换。
        void advanceDelaySwitch (int n) noexcept
        {
            const auto wanted = wantedDelaySamples();

            // 放在块开头判断：这样淡入最后一块的增益曲线也会被用上。
            if (switchState == SwitchState::fadingIn && switchGain >= 1.0f)
                switchState = SwitchState::running;

            if (switchState == SwitchState::running && lookahead.getDelaySamples() != wanted)
                switchState = SwitchState::fadingOut;

            if (switchState == SwitchState::fadingOut && switchGain <= 0.0f)
            {
                lookahead.setDelaySamples (wanted);
                lookahead.reset();
                switchWaitLeft = wanted;
                switchState = SwitchState::waiting;
            }

            if (switchState == SwitchState::waiting && switchWaitLeft <= 0)
                switchState = lookahead.getDelaySamples() == wanted ? SwitchState::fadingIn : SwitchState::fadingOut;

            if (switchState == SwitchState::running)
                return;

            const auto step = 1.0f / (monitorFadeMs * 0.001f * (float) currentRate);
            for (int i = 0; i < n; ++i)
            {
                if (switchState == SwitchState::fadingOut)
                    switchGain = std::max (0.0f, switchGain - step);
                else if (switchState == SwitchState::waiting)
                    --switchWaitLeft;
                else
                    switchGain = std::min (1.0f, switchGain + step);

                switchTrace[(size_t) i] = switchGain;
            }
        }

        int wantedDelaySamples() const noexcept
        {
            return (int) std::round (lookaheadMs->load() * 0.001 * currentRate);
        }

        void detectAndSmooth (const juce::AudioBuffer<float>& buffer, int start, int n, int numChannels) noexcept
        {
            std::array<const float*, maxChannels> input {};
            for (int ch = 0; ch < numChannels; ++ch)
                input[(size_t) ch] = buffer.getReadPointer (ch, start);

            for (int i = 0; i < n; ++i)
            {
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    const auto x = input[(size_t) ch][i];
                    auto& state = channels[(size_t) ch];

                    // 带通的低频裙边只有 -12 dB/oct，元音的基频和低次谐波会漏进检测。先过一道高通。
                    const auto high = state.preHigh.tickHigh (x, preHighCoefs);
                    for (int b = 0; b < numBands; ++b)
                    {
                        const auto band = state.detect[(size_t) b].tickBand (high, detectCoefs[(size_t) b]);
                        auto& env = state.bandEnv[(size_t) b];
                        env = band * band + bandEnvCoef * (env - band * band);
                    }

                    const auto voice = state.voiceLow2.tickLow (state.voiceLow.tickLow (state.voiceHigh.tickHigh (x, voiceHighCoefs), voiceLowCoefs), voiceLowCoefs);
                    const auto voicePower = voice * voice;
                    state.voiceEnv = voicePower + voiceEnvCoef * (state.voiceEnv - voicePower);
                    state.voicePeak = voicePower + (voicePower > state.voicePeak ? voicePeakCoef : voiceEnvCoef) * (state.voicePeak - voicePower);
                }

                if (--controlCountdown <= 0)
                {
                    controlCountdown = controlInterval;
                    updateControl (numChannels);
                }

                for (size_t b = 0; b < (size_t) numBands; ++b)
                {
                    // 目标衰减由两级平滑得到，第一级非对称决定起落，第二级固定，去掉折角。
                    const auto target = targetGain[b];
                    const auto coef = target < slowGain[b] ? attackCoef : releaseCoef;
                    slowGain[b] = target + coef * (slowGain[b] - target);
                    gain[b] = slowGain[b] + gainSmoothCoef * (gain[b] - slowGain[b]);

                    // 衰减结束后回到精确的 1，元音段逐采样不变。
                    if (target >= 1.0f && 1.0f - gain[b] < 1.0e-5f)
                        slowGain[b] = gain[b] = 1.0f;

                    gainTrace[b][(size_t) i] = gain[b];
                }
            }
        }

        // 约 12 kHz 控制率：算参考电平、自适应阈值和目标衰减。全部在 dB 域做差，输入电平整体变化时结果不变。
        void updateControl (int numChannels) noexcept
        {
            auto logFrequency = std::log2 (frequency->load());
            smoothedLogFrequency = logFrequency + frequencyCoef * (smoothedLogFrequency - logFrequency);
            const auto f0 = std::pow (2.0f, smoothedLogFrequency);
            if (std::abs (f0 - lastFrequency) > 0.5f)
                setFilters (f0);

            auto voicePower = 0.0f, voicePeak = 0.0f;
            std::array<float, numBands> bandPower {};
            for (int ch = 0; ch < numChannels; ++ch)
            {
                const auto& state = channels[(size_t) ch];
                voicePower = std::max (voicePower, state.voiceEnv);
                voicePeak = std::max (voicePeak, state.voicePeak);
                for (size_t b = 0; b < (size_t) numBands; ++b)
                    bandPower[b] = std::max (bandPower[b], state.bandEnv[b]);
            }

            const auto voiceDb = powerToDb (voicePower);

            // 参考电平跟随人声本体：上升快，下降慢，没有人声（齿音段、气口、静音）时冻结。
            if (voiceDb > referenceDb)
                referenceDb = voiceDb + refRiseCoef * (std::max (referenceDb, voiceDb - 24.0f) - voiceDb);
            else if (voiceDb > referenceDb - refGateDb)
                referenceDb -= refFallPerTick;

            const auto s = juce::jlimit (0.0f, 1.0f, strength->load() * 0.01f);
            const auto targetAboveRef = -4.5f - 21.5f * s;
            const auto slope = 0.5f + 0.25f * s;
            const auto limitDb = 16.0f * std::pow (s, 0.7f);
            const auto voicePeakDb = powerToDb (voicePeak);
            const auto voicedThreshold = voicePeakDb - voicedMarginDb;

            auto deepestDb = 0.0f, loudestBandDb = -130.0f;
            for (size_t b = 0; b < (size_t) numBands; ++b)
            {
                const auto bandDb = powerToDb (bandPower[b]);
                loudestBandDb = std::max (loudestBandDb, bandDb);

                // 阈值取两者较高者：齿音要同时超过“人声本体下方的目标”和“此刻的低频能量”。
                // 后者保证明亮的元音不被当成齿音。
                const auto threshold = std::max (referenceDb + targetAboveRef, voicedThreshold);
                const auto fade = juce::jlimit (0.0f, 1.0f, (bandDb - floorDb) / floorFadeDb);
                const auto reductionDb = std::min (slope * softKnee (bandDb - threshold), limitDb) * fade;

                targetGain[b] = reductionDb <= 0.0f ? 1.0f : mint::dbToGain (-reductionDb);
                deepestDb = std::max (deepestDb, -mint::gainToDb (gain[b]));
            }

            releaseCoef = voicePeakDb > loudestBandDb ? voicedReleaseCoef : slowReleaseCoef;

            // Meter 的衰减模式按“值越大点亮越多”，范围 -24..0，所以这里推的是 衰减量 - 24。
            if (deepestDb > 0.05f)
                reductionMeter.push (deepestDb - 24.0f);
        }

        void setFilters (float f0) noexcept
        {
            lastFrequency = f0;
            const std::array<float, numBands> centres { f0 / bandSpread, f0, f0 * bandSpread };
            for (size_t b = 0; b < (size_t) numBands; ++b)
            {
                audioCoefs[b].set (centres[b], bandQ, currentRate);
                detectCoefs[b].set (centres[b], bandQ, currentRate);
            }
            preHighCoefs.set (centres[0] * 0.6f, 0.707f, currentRate);
            voiceLowCoefs.set (juce::jmin (3000.0f, centres[0] * 0.6f), 0.707f, currentRate);
        }

        void updateLookahead() noexcept
        {
            const auto ms = lookaheadMs->load();
            requestLatencySamples (wantedDelaySamples());

            // 有 Look-ahead 时起音可以更柔和，同时仍在齿音到达前到位。
            attackCoef = onePole (juce::jlimit (1.0f, 4.0f, ms * 0.4f), currentRate);
        }

        enum class SwitchState { running, fadingOut, waiting, fadingIn };

        double currentRate = 48000.0;
        std::atomic<float>* strength = nullptr;
        std::atomic<float>* frequency = nullptr;
        std::atomic<float>* monitor = nullptr;

        mint::LookaheadDelay lookahead;
        std::array<Channel, maxChannels> channels;
        std::array<SvfCoefs, numBands> audioCoefs, detectCoefs;
        SvfCoefs preHighCoefs, voiceHighCoefs, voiceLowCoefs;

        std::array<float, numBands> targetGain {}, slowGain {}, gain {};
        std::array<std::array<float, chunkSize>, numBands> gainTrace {};
        std::array<float, chunkSize> switchTrace {};
        SwitchState switchState = SwitchState::running;
        float switchGain = 1.0f;
        int switchWaitLeft = 0;

        float bandEnvCoef = 0.0f, voiceEnvCoef = 0.0f, voicePeakCoef = 0.0f, refRiseCoef = 0.0f, refFallPerTick = 0.0f;
        float attackCoef = 0.0f, releaseCoef = 0.0f, slowReleaseCoef = 0.0f, voicedReleaseCoef = 0.0f, gainSmoothCoef = 0.0f, frequencyCoef = 0.0f, monitorCoef = 0.0f;
        float referenceDb = -120.0f, smoothedLogFrequency = 12.0f, lastFrequency = -1.0f, monitorMix = 0.0f;
        int controlInterval = 4, controlCountdown = 0;
    };

    class MedusaEditor final : public mint::ui::EditorBase
    {
    public:
        explicit MedusaEditor (MedusaProcessor& p)
            : EditorBase (p, { "MEDUSA", { "De-esser", "디에서", "去齿音" }, "V" }),
              strengthDial (p.apvts, "strength", { "Strength", "강도", "强度" }, true),
              frequencyDial (p.apvts, "frequency", { "Frequency", "주파수", "频率" }),
              lookaheadDial (p.apvts, "lookahead_ms", { "Look-ahead", "룩어헤드", "前瞻" }),
              monitorSwitch (p.apvts, "monitor", { { "Normal output", "기본 출력", "正常输出" }, { "Hear sibilance", "치찰음 듣기", "监听齿音" } }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              reductionMeter (p.reductionMeter, { "Reduction", "리덕션", "增益衰减" }, true)
        {
            addAndMakeVisible (strengthDial);
            addAndMakeVisible (frequencyDial);
            addAndMakeVisible (lookaheadDial);
            addAndMakeVisible (monitorSwitch);
            addAndMakeVisible (inputMeter);
            addAndMakeVisible (reductionMeter);
            setSize (width, height);
        }

        void resized() override
        {
            auto area = getContentBounds();
            // 表要放得下最长的标签（Reduction、增益衰减），不能靠缩字。
            inputMeter.setBounds (area.removeFromLeft (64));
            reductionMeter.setBounds (area.removeFromRight (64));

            // 强度居中，频率和前瞻左右对称，三个旋钮的圆心在同一高度。
            constexpr int bigWidth = 224, bigHeight = 280, smallWidth = 150, smallHeight = 170;
            const auto sideWidth = (area.getWidth() - bigWidth) / 2;
            const auto big = area.withSizeKeepingCentre (bigWidth, bigHeight);
            // 圆心高度：大旋钮 (280 - 46) / 2，小旋钮 (170 - 40) / 2。
            const auto smallTop = big.getY() + 117 - 65;

            strengthDial.setBounds (big);
            frequencyDial.setBounds (area.removeFromLeft (sideWidth).withSizeKeepingCentre (smallWidth, smallHeight).withY (smallTop));
            lookaheadDial.setBounds (area.removeFromRight (sideWidth).withSizeKeepingCentre (smallWidth, smallHeight).withY (smallTop));
            monitorSwitch.setBounds (getFooterSlot().withSizeKeepingCentre (240, 34));
        }

    private:
        mint::ui::Dial strengthDial, frequencyDial, lookaheadDial;
        mint::ui::Segmented monitorSwitch;
        mint::ui::Meter inputMeter, reductionMeter;
    };

    juce::AudioProcessorEditor* MedusaProcessor::createEditor()
    {
        return new MedusaEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MedusaProcessor();
}
