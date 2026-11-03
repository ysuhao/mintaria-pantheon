// Mintaria Janus：面向人声的下行扩展器。人声停顿时像推子一样平滑压低，不直接切断。
//
// 音频通路只有“延迟（默认 0）→ 增益”，检测通路的滤波不进音频，所以 Look-ahead 关闭时延迟严格为 0，
// 增益为 0 dB 时输出与输入逐采样相同。
//
// 增益的计算分四步：
//   1. 检测：高通和高架滤波后取短时 RMS，按正弦校准（正弦的读数等于峰值）。
//      噪声的读数比峰值低十几 dB，房间底噪才压得下去，单个尖峰也打不开。再加峰值保持和回落。
//   2. 状态：高于阈值即打开；跌破“阈值 - 迟滞”并走完保持时间才关闭。迟滞和保持负责不抖动。
//   3. 目标：打开时 0 dB。关闭时按膝宽从 0 过渡到满衰减，膝宽以下正好等于“衰减量”。
//      膝内看的是慢速平滑后的电平，否则颤音和混响尾巴会被扩展比放大成明显的音量起伏。
//   4. 轨迹：dB 域限速再过一个一阶平滑，一阶导连续，开合都不会有咔嗒声。
#include <mint_core/MintCore.h>
#include <mint_ui/MintUi.h>

#include <array>

namespace
{
    constexpr float maxLookaheadMs = 10.0f;
    constexpr int maxChunk = 256;
    constexpr int maxDetectorChannels = 2;

    // 增益轨迹末端的一阶平滑。Look-ahead 的开启斜坡要把它的滞后算进去。
    constexpr float gainSmoothMs = 0.3f;
    constexpr float lookaheadRateLimitDbPerMs = 40.0f;
    constexpr float creepRatio = 0.25f;
    constexpr float kneeLevelSmoothMs = 100.0f;
    constexpr float detectorAverageMs = 5.0f;
    constexpr float bypassDbPerMs = 4.0f;
    constexpr float delaySwitchFadeMs = 3.0f;

    enum Mode { modeVocal, modeSpeech, modeInstrument, modeDuck };

    struct ModeTuning
    {
        float highPassHz;       // 检测通路高通，挡开喷麦和脚步低频
        float shelfDb;          // 检测通路 4 kHz 高架，让弱的清辅音也能打开
        float attackDbPerMs;    // 打开速度（闪避模式下是压低速度）
        float holdMs;           // 跌破关闭点后再保持多久
        float hysteresisDb;     // 关闭点比阈值低多少
        float kneeDb;           // 关闭后从阈值到满衰减的跨度
        float peakHoldMs;       // 检测峰值保持，盖住低频波形残留的纹波
        float detectorFallMs;   // 峰值保持结束后的回落时间常数
    };

    constexpr std::array<ModeTuning, 4> tunings { {
        { 100.0f, 3.0f, 20.0f,  60.0f, 4.0f, 12.0f, 12.0f, 25.0f },   // 人声：尾音长，保持适中
        { 120.0f, 6.0f, 30.0f, 120.0f, 5.0f, 12.0f, 12.0f, 25.0f },   // 语音：词间停顿不放下去，清辅音最弱
        {  40.0f, 0.0f, 10.0f,  30.0f, 3.0f, 10.0f, 25.0f, 25.0f },   // 乐器：低频多，瞬态干净
        {  80.0f, 0.0f,  3.0f, 150.0f, 4.0f,  1.0f, 12.0f, 25.0f },   // 闪避：侧链一响就压低，停了才回来
    } };

    // 检测通路用的双二阶，系数在 prepare 里算好，音频线程只切换。
    struct BiquadCoefficients
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
    };

    struct Biquad
    {
        double z1 = 0.0, z2 = 0.0;

        double process (double in, const BiquadCoefficients& c) noexcept
        {
            const auto out = c.b0 * in + z1;
            z1 = c.b1 * in - c.a1 * out + z2;
            z2 = c.b2 * in - c.a2 * out;
            return out;
        }
    };

    BiquadCoefficients makeHighPass (double rate, double hz) noexcept
    {
        constexpr double q = 0.7071;
        const auto w = juce::MathConstants<double>::twoPi * hz / rate;
        const auto alpha = std::sin (w) / (2.0 * q);
        const auto cosW = std::cos (w);
        const auto a0 = 1.0 + alpha;
        return { (1.0 + cosW) * 0.5 / a0, -(1.0 + cosW) / a0, (1.0 + cosW) * 0.5 / a0, -2.0 * cosW / a0, (1.0 - alpha) / a0 };
    }

    BiquadCoefficients makeHighShelf (double rate, double hz, double db) noexcept
    {
        const auto a = std::pow (10.0, db / 40.0);
        const auto w = juce::MathConstants<double>::twoPi * juce::jmin (hz, rate * 0.45) / rate;
        const auto cosW = std::cos (w);
        const auto alpha = std::sin (w) * 0.5 * std::sqrt (2.0);
        const auto root = 2.0 * std::sqrt (a) * alpha;
        const auto a0 = (a + 1.0) - (a - 1.0) * cosW + root;
        return { a * ((a + 1.0) + (a - 1.0) * cosW + root) / a0,
                 -2.0 * a * ((a - 1.0) + (a + 1.0) * cosW) / a0,
                 a * ((a + 1.0) + (a - 1.0) * cosW - root) / a0,
                 2.0 * ((a - 1.0) - (a + 1.0) * cosW) / a0,
                 ((a + 1.0) - (a - 1.0) * cosW - root) / a0 };
    }

    juce::String utf8 (const char* text)
    {
        return juce::String::fromUTF8 (text);
    }

    juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;

        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "threshold", 1 }, "Threshold",
                                                           NormalisableRange<float> (-80.0f, 0.0f, 0.1f), -40.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (v, 1) + " dB"; })));

        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "range", 1 }, "Range",
                                                           NormalisableRange<float> (0.0f, 60.0f, 0.1f), 18.0f,
                                                           AudioParameterFloatAttributes().withLabel ("dB")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (v, 1) + " dB"; })));

        NormalisableRange<float> releaseRange (20.0f, 2000.0f, 1.0f);
        releaseRange.setSkewForCentre (200.0f);
        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "release", 1 }, "Release", releaseRange, 500.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms")
                                                               .withStringFromValueFunction ([] (float v, int) { return String (roundToInt (v)) + " ms"; })));

        layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "mode", 1 }, "Mode",
                                                            StringArray { utf8 ("人声"), utf8 ("语音"), utf8 ("乐器"), utf8 ("闪避") }, 0));

        layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "lookahead_ms", 1 }, "Look-ahead",
                                                           NormalisableRange<float> (0.0f, maxLookaheadMs, 0.1f), 0.0f,
                                                           AudioParameterFloatAttributes().withLabel ("ms")
                                                               .withStringFromValueFunction ([] (float v, int) { return v <= 0.0f ? mint::ui::tr ({ "Off", "끔", "关闭" }) : String (v, 1) + " ms"; })));
        return layout;
    }

    enum class SidechainState { unavailable, silent, active };

    class GateProcessor final : public mint::Processor
    {
    public:
        GateProcessor() : mint::Processor (createLayout(), stereoBusesWithSidechain())
        {
            threshold = apvts.getRawParameterValue ("threshold");
            range = apvts.getRawParameterValue ("range");
            release = apvts.getRawParameterValue ("release");
            mode = apvts.getRawParameterValue ("mode");
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

            for (size_t m = 0; m < tunings.size(); ++m)
            {
                highPass[m] = makeHighPass (sampleRate, tunings[m].highPassHz);
                highShelf[m] = makeHighShelf (sampleRate, 4000.0, tunings[m].shelfDb);
            }

            gainSmoothCoef = 1.0f - std::exp (-1.0f / (gainSmoothMs * 0.001f * (float) sampleRate));
            delayFadeStep = 1.0f / juce::jmax (1.0f, delaySwitchFadeMs * 0.001f * (float) sampleRate);

            smoothedThreshold.reset (sampleRate, 0.02);
            smoothedRange.reset (sampleRate, 0.02);
            smoothedRelease.reset (sampleRate, 0.02);

            lookahead.prepare (getTotalNumOutputChannels(), (int) std::ceil (maxLookaheadMs * 0.001 * sampleRate));

            resetState();
        }

        void reset() override
        {
            resetState();
        }

        void releaseResources() override {}

        void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            process (buffer, false);
        }

        void processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
        {
            process (buffer, true);
        }

        juce::AudioProcessorEditor* createEditor() override;

        Mode getMode() const noexcept                       { return (Mode) juce::jlimit (0, 3, (int) mode->load()); }
        SidechainState getSidechainState() const noexcept   { return sidechainState.load (std::memory_order_relaxed); }

        mint::MeterSource inputMeter, sidechainMeter, reductionMeter;
        std::atomic<float>* lookaheadMs = nullptr;

    private:
        void resetState() noexcept
        {
            const auto delay = wantedDelaySamples();
            lookahead.reset();
            lookahead.setDelaySamples (delay);
            requestLatencySamples (delay);
            delaySwitch = DelaySwitch::idle;
            delaySwitchGain = 1.0f;
            delayMuteSamples = 0;

            for (auto& channel : detectorFilters)
                channel = {};

            averagePower = 0.0f;
            envelope = 0.0f;
            envelopeHold = 0;
            wasDuck = getMode() == modeDuck;
            isActive = ! wasDuck;
            holdCounter = wasDuck ? 0 : (int) (tunings[(size_t) getMode()].holdMs * 0.001 * currentRate);
            kneeLevelDb = 0.0f;
            gainDb = 0.0f;
            smoothDb = 0.0f;
            sidechainQuietSamples = 0;

            smoothedThreshold.setCurrentAndTargetValue (threshold->load());
            smoothedRange.setCurrentAndTargetValue (range->load());
            smoothedRelease.setCurrentAndTargetValue (release->load());
        }

        int wantedDelaySamples() const noexcept
        {
            return (int) std::round (lookaheadMs->load() * 0.001 * currentRate);
        }

        void process (juce::AudioBuffer<float>& buffer, bool bypassed) noexcept
        {
            juce::ScopedNoDenormals noDenormals;

            auto mainBus = getBusBuffer (buffer, true, 0);
            const auto numChannels = mainBus.getNumChannels();
            const auto numSamples = mainBus.getNumSamples();

            // 侧链未启用时取到的是 0 声道缓冲区，下面按“没有侧链”处理，不读它。
            auto* sidechainBus = getBus (true, 1);
            const auto hasSidechainBus = sidechainBus != nullptr && sidechainBus->isEnabled();
            auto sidechain = hasSidechainBus ? getBusBuffer (buffer, true, 1) : juce::AudioBuffer<float>();

            // 输入表要在处理之前取值，增益和延迟都是就地改写缓冲区的。
            inputMeter.push (mint::gainToDb (mainBus.getMagnitude (0, numSamples)));
            sidechainMeter.push (hasSidechainBus ? mint::gainToDb (sidechain.getMagnitude (0, numSamples)) : -120.0f);

            updateDelaySwitch();
            trackSidechain (sidechain, hasSidechainBus, numSamples);

            auto* const* channels = mainBus.getArrayOfWritePointers();

            for (int pos = 0; pos < numSamples;)
            {
                updateDelaySwitch();

                auto n = juce::jmin (numSamples - pos, maxChunk);

                // 淡出走完的那一采样正好落在分段末尾，下一段才切换延迟。
                if (delaySwitch == DelaySwitch::fadingOut)
                    n = juce::jmin (n, juce::jmax (1, (int) std::ceil (delaySwitchGain / delayFadeStep)));
                else if (delaySwitch == DelaySwitch::muted)
                    n = juce::jmin (n, juce::jmax (1, delayMuteSamples));

                computeGains (channels, numChannels, sidechain, pos, n, bypassed);

                juce::AudioBuffer<float> segment (channels, numChannels, pos, n);
                lookahead.process (segment);

                for (int i = 0; i < n; ++i)
                {
                    gains[(size_t) i] *= nextDelaySwitchGain();

                    for (int ch = 0; ch < numChannels; ++ch)
                        channels[ch][pos + i] *= gains[(size_t) i];
                }

                pos += n;
            }

            // Meter 的衰减表约定：-24 不亮，0 全亮。所以推的是“-24 + 衰减量”，超过 24 dB 就满格。
            reductionMeter.push (peakReduction - 24.0f);
            peakReduction = 0.0f;
        }

        void trackSidechain (const juce::AudioBuffer<float>& sidechain, bool hasBus, int numSamples) noexcept
        {
            if (! hasBus)
            {
                sidechainState.store (SidechainState::unavailable, std::memory_order_relaxed);
                return;
            }

            if (sidechain.getMagnitude (0, numSamples) > 1.0e-5f)
                sidechainQuietSamples = 0;
            else
                sidechainQuietSamples = juce::jmin (sidechainQuietSamples + numSamples, (int) currentRate);

            sidechainState.store (sidechainQuietSamples >= (int) (currentRate * 0.5) ? SidechainState::silent : SidechainState::active,
                                  std::memory_order_relaxed);
        }

        void updateDelaySwitch() noexcept
        {
            const auto wanted = wantedDelaySamples();

            if (delaySwitch == DelaySwitch::idle && wanted != lookahead.getDelaySamples())
                delaySwitch = DelaySwitch::fadingOut;

            if (delaySwitch == DelaySwitch::fadingOut && delaySwitchGain <= 0.0f)
            {
                // LookaheadDelay 在 0 延迟时不写入，里面还是旧音频。清空后静音走完延迟长度，再淡入。
                if (lookahead.getDelaySamples() == 0)
                {
                    lookahead.reset();
                    delayMuteSamples = wanted;
                }

                lookahead.setDelaySamples (wanted);
                requestLatencySamples (wanted);
                delaySwitch = delayMuteSamples > 0 ? DelaySwitch::muted : DelaySwitch::fadingIn;
            }

            if (delaySwitch == DelaySwitch::muted && delayMuteSamples <= 0)
                delaySwitch = DelaySwitch::fadingIn;
        }

        float nextDelaySwitchGain() noexcept
        {
            switch (delaySwitch)
            {
                case DelaySwitch::idle:
                    return 1.0f;
                case DelaySwitch::fadingOut:
                    delaySwitchGain = juce::jmax (0.0f, delaySwitchGain - delayFadeStep);
                    return delaySwitchGain;
                case DelaySwitch::muted:
                    --delayMuteSamples;
                    return 0.0f;
                case DelaySwitch::fadingIn:
                    delaySwitchGain += delayFadeStep;
                    if (delaySwitchGain >= 1.0f)
                    {
                        delaySwitchGain = 1.0f;
                        delaySwitch = DelaySwitch::idle;
                    }
                    return delaySwitchGain;
            }
            return 1.0f;
        }

        // 算一段增益。channels 是还没进延迟线的主输入，闪避模式改看侧链。
        void computeGains (const float* const* channels, int numChannels, const juce::AudioBuffer<float>& sidechain,
                           int pos, int n, bool bypassed) noexcept
        {
            const auto currentMode = getMode();
            const auto& tuning = tunings[(size_t) currentMode];
            const auto& hp = highPass[(size_t) currentMode];
            const auto& shelf = highShelf[(size_t) currentMode];
            const auto samplesPerMs = (float) currentRate * 0.001f;
            const auto isDuck = currentMode == modeDuck;

            const auto holdSamples = (int) (tuning.holdMs * samplesPerMs);

            // 闪避和内部模式的“激活”含义相反，检测信号也不同，切换时从直通状态重新开始，不带旧状态。
            // 滤波器也要清：里面留着主信号，换到静音的侧链时余振会越过阈值，误触发闪避。
            // 回到内部模式时给满保持时间，检测器刚清零，不能因此立刻判定为低于阈值。
            if (isDuck != wasDuck)
            {
                wasDuck = isDuck;
                detectorFilters = {};
                averagePower = 0.0f;
                envelope = 0.0f;
                envelopeHold = 0;
                holdCounter = isDuck ? 0 : holdSamples;
                isActive = ! isDuck;
            }

            smoothedThreshold.setTargetValue (threshold->load());
            smoothedRange.setTargetValue (range->load());
            smoothedRelease.setTargetValue (release->load());

            const auto peakHoldSamples = (int) (tuning.peakHoldMs * samplesPerMs);
            const auto envelopeFall = std::exp (-1.0f / (tuning.detectorFallMs * samplesPerMs));
            const auto kneeLevelCoef = 1.0f - std::exp (-1.0f / (kneeLevelSmoothMs * samplesPerMs));
            const auto attackStep = attackDbPerSample (tuning, samplesPerMs);
            const auto averageCoef = 1.0f - std::exp (-1.0f / (detectorAverageMs * samplesPerMs));

            const auto detectorChannels = juce::jmin (maxDetectorChannels, isDuck ? sidechain.getNumChannels() : numChannels);

            for (int i = 0; i < n; ++i)
            {
                const auto thresholdDb = smoothedThreshold.getNextValue();
                const auto rangeDb = smoothedRange.getNextValue();
                const auto releaseStep = juce::jmax (rangeDb, 1.0f) / (smoothedRelease.getNextValue() * samplesPerMs);

                // 1. 检测
                auto power = 0.0f;
                for (int ch = 0; ch < detectorChannels; ++ch)
                {
                    const auto in = isDuck ? sidechain.getSample (ch, pos + i) : channels[ch][pos + i];
                    auto& filters = detectorFilters[(size_t) ch];
                    const auto out = (float) filters.shelf.process (filters.highPass.process ((double) in, hp), shelf);
                    power = juce::jmax (power, out * out);
                }

                averagePower += (power - averagePower) * averageCoef;
                const auto level = std::sqrt (2.0f * averagePower);

                if (level >= envelope)
                {
                    envelope = level;
                    envelopeHold = peakHoldSamples;
                }
                else if (envelopeHold > 0)
                {
                    --envelopeHold;
                }
                else
                {
                    envelope *= envelopeFall;
                }

                const auto levelDb = mint::gainToDb (envelope);

                // 2. 状态
                if (levelDb >= thresholdDb)
                {
                    isActive = true;
                    holdCounter = holdSamples;
                }
                else if (isActive)
                {
                    if (levelDb >= thresholdDb - tuning.hysteresisDb)
                        holdCounter = holdSamples;
                    else if (holdCounter > 0)
                        --holdCounter;
                    else
                    {
                        isActive = false;
                        kneeLevelDb = levelDb;
                    }
                }

                kneeLevelDb += (levelDb - kneeLevelDb) * kneeLevelCoef;

                // 3. 目标
                auto target = 0.0f;
                if (! bypassed)
                {
                    if (isDuck)
                    {
                        target = isActive ? -rangeDb : 0.0f;
                    }
                    else if (! isActive)
                    {
                        const auto t = juce::jlimit (0.0f, 1.0f, (thresholdDb - kneeLevelDb) / tuning.kneeDb);
                        target = -rangeDb * t * t * (3.0f - 2.0f * t);
                    }
                }

                // 4. 轨迹
                auto upStep = releaseStep;
                auto downStep = releaseStep;
                if (bypassed)
                    upStep = bypassDbPerMs / samplesPerMs;
                else if (isDuck)
                    downStep = attackStep;
                else
                    upStep = isActive ? attackStep : releaseStep * creepRatio;

                gainDb += juce::jlimit (-downStep, upStep, target - gainDb);
                smoothDb += (gainDb - smoothDb) * gainSmoothCoef;

                peakReduction = juce::jmax (peakReduction, -smoothDb);

                // 完全打开时严格等于 1，输出与输入逐采样相同。
                gains[(size_t) i] = smoothDb > -1.0e-4f ? 1.0f : std::exp (smoothDb * 0.11512925f);
            }

            // 持续静音时不留 denormal。
            if (averagePower < 1.0e-18f)
                averagePower = 0.0f;
            if (envelope < 1.0e-9f)
                envelope = 0.0f;
        }

        // 开启斜坡。开了 Look-ahead 就用满这段预读：信号到达输出时增益已经打开，又不比需要的更陡。
        float attackDbPerSample (const ModeTuning& tuning, float samplesPerMs) const noexcept
        {
            const auto delayMs = (float) lookahead.getDelaySamples() / samplesPerMs;
            const auto rampMs = delayMs - 4.0f * gainSmoothMs;

            if (rampMs > 0.0f && getMode() != modeDuck)
                return juce::jmin (juce::jmax (range->load(), 1.0f) / rampMs, lookaheadRateLimitDbPerMs) / samplesPerMs;

            return tuning.attackDbPerMs / samplesPerMs;
        }

        enum class DelaySwitch { idle, fadingOut, muted, fadingIn };

        struct DetectorFilters
        {
            Biquad highPass, shelf;
        };

        double currentRate = 48000.0;

        std::atomic<float>* threshold = nullptr;
        std::atomic<float>* range = nullptr;
        std::atomic<float>* release = nullptr;
        std::atomic<float>* mode = nullptr;

        juce::LinearSmoothedValue<float> smoothedThreshold, smoothedRange, smoothedRelease;

        std::array<BiquadCoefficients, tunings.size()> highPass, highShelf;
        std::array<DetectorFilters, maxDetectorChannels> detectorFilters;

        mint::LookaheadDelay lookahead;
        DelaySwitch delaySwitch = DelaySwitch::idle;
        float delaySwitchGain = 1.0f, delayFadeStep = 1.0f;
        int delayMuteSamples = 0;

        float averagePower = 0.0f, envelope = 0.0f, kneeLevelDb = 0.0f;
        int envelopeHold = 0;
        bool isActive = true, wasDuck = false;
        int holdCounter = 0;
        float gainDb = 0.0f, smoothDb = 0.0f, gainSmoothCoef = 1.0f;
        float peakReduction = 0.0f;
        std::array<float, maxChunk> gains {};

        int sidechainQuietSamples = 0;
        std::atomic<SidechainState> sidechainState { SidechainState::unavailable };
    };

    // 闪避模式下说明侧链状态，同时把输入表换成侧链表，这样阈值对着的就是真正的检测信号。
    class ModeWatcher final : public juce::Component,
                              private juce::Timer
    {
    public:
        ModeWatcher (GateProcessor& p, juce::Component& inputView, juce::Component& sidechainView)
            : owner (p), input (inputView), sidechain (sidechainView)
        {
            setInterceptsMouseClicks (false, false);
            timerCallback();
            startTimerHz (8);
        }

        ~ModeWatcher() override
        {
            stopTimer();
        }

        void paint (juce::Graphics& g) override
        {
            g.setColour (mint::ui::theme().inkMuted);
            g.setFont (mint::ui::bodyFont (11.0f));
            // 文字在绘制时按当前语言取，切换语言后重绘即可更新。
            g.drawText (mint::ui::tr (statusText()), getLocalBounds(), juce::Justification::centred);
        }

    private:
        void timerCallback() override
        {
            const auto isDuck = owner.getMode() == modeDuck;
            input.setVisible (! isDuck);
            sidechain.setVisible (isDuck);

            const auto next = isDuck ? (int) owner.getSidechainState() : -1;

            if (next != shown)
            {
                shown = next;
                repaint();
            }
        }

        mint::ui::Text statusText() const
        {
            switch (shown)
            {
                case (int) SidechainState::unavailable: return { "No sidechain · not ducking", "사이드체인 미연결 · 더킹 안 함", "侧链未接 · 不闪避" };
                case (int) SidechainState::silent:      return { "Sidechain silent · not ducking", "사이드체인 신호 없음 · 더킹 안 함", "侧链无信号 · 不闪避" };
                case (int) SidechainState::active:      return { "Sidechain active", "사이드체인 신호 있음", "侧链有信号" };
                default:                                return {};
            }
        }

        GateProcessor& owner;
        juce::Component& input;
        juce::Component& sidechain;
        int shown = -2;   // -1 无提示，否则是 SidechainState
    };

    class GateEditor final : public mint::ui::EditorBase
    {
    public:
        explicit GateEditor (GateProcessor& p)
            : EditorBase (p, { "JANUS", { "Gate", "게이트", "门限" }, "I" }),
              thresholdDial (p.apvts, "threshold", { "Threshold", "스레숄드", "阈值" }, true),
              rangeDial (p.apvts, "range", { "Range", "레인지", "范围" }),
              releaseDial (p.apvts, "release", { "Release", "릴리즈", "释放" }),
              lookaheadDial (p.apvts, "lookahead_ms", { "Look-ahead", "룩어헤드", "前瞻" }),
              modeSelector (p.apvts, "mode", { { "Vocal", "보컬", "人声" }, { "Speech", "음성", "语音" },
                                               { "Instrument", "악기", "乐器" }, { "Duck", "더킹", "闪避" } }),
              inputMeter (p.inputMeter, { "Input", "입력", "输入" }),
              sidechainMeter (p.sidechainMeter, { "Sidechain", "사이드체인", "侧链" }),
              reductionMeter (p.reductionMeter, { "Reduction", "리덕션", "增益衰减" }, true),
              watcher (p, inputMeter, sidechainMeter)
        {
            addAndMakeVisible (thresholdDial);
            addAndMakeVisible (rangeDial);
            addAndMakeVisible (releaseDial);
            addAndMakeVisible (lookaheadDial);
            addAndMakeVisible (modeSelector);
            addChildComponent (inputMeter);
            addChildComponent (sidechainMeter);
            addAndMakeVisible (reductionMeter);
            addAndMakeVisible (watcher);
            setSize (width, height);
        }

        void resized() override
        {
            auto area = getContentBounds();

            // 表的灯柱宽度不变，两侧各放宽 12 px 给标签，韩文“사이드체인”才放得下。
            const auto inputArea = area.removeFromLeft (40).expanded (12, 0);
            inputMeter.setBounds (inputArea);
            sidechainMeter.setBounds (inputArea);
            reductionMeter.setBounds (area.removeFromRight (40).expanded (12, 0));

            area.reduce (24, 0);
            auto left = area.removeFromLeft (150);
            auto right = area.removeFromRight (150);

            thresholdDial.setBounds (area.withSizeKeepingCentre (224, 280));

            // 左边一个大旋钮，对着右边两个小旋钮，视觉重量才相当。大旋钮和主控的旋钮同高。
            const auto knobCentreY = thresholdDial.getY() + (thresholdDial.getHeight() - 46) / 2;
            rangeDial.setBounds (left.withSizeKeepingCentre (150, 176).withY (knobCentreY - 68));

            // 右边两个收拢，上下都不贴边，上下沿和主控对齐。
            auto rightGroup = right.withSizeKeepingCentre (130, 270);
            releaseDial.setBounds (rightGroup.removeFromTop (130));
            lookaheadDial.setBounds (rightGroup.removeFromBottom (130));

            auto footer = getFooterSlot();
            modeSelector.setBounds (footer.removeFromTop (30));
            watcher.setBounds (footer);
        }

    private:
        mint::ui::Dial thresholdDial, rangeDial, releaseDial, lookaheadDial;
        mint::ui::Segmented modeSelector;
        mint::ui::Meter inputMeter, sidechainMeter, reductionMeter;
        ModeWatcher watcher;
    };

    juce::AudioProcessorEditor* GateProcessor::createEditor()
    {
        return new GateEditor (*this);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new GateProcessor();
}
