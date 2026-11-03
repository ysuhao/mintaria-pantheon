#pragma once

#include <mint_core/MintCore.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// 延迟的音频引擎，不依赖宿主和界面。一个采样一个采样地处理，结果与缓冲区大小无关。
//
// 反馈环（每个声道一条）：
//   延迟线读头 -> 饱和 -> 低保真 -> 高切 -> 低切 -> 取湿声 -> 环内带限 -> 反馈量 -> 写回延迟线
// 高切和低切在环内，每绕一圈就再滤一次。环内带限只作用于回到延迟线的那一路，
// 第一次回声不受影响，之后每次回声都比上一次暗。
namespace echo
{

enum class Mode { stereo, pingPong, mono };

constexpr double maxDelaySeconds = 8.0;
constexpr double maxModDepthMs = 5.0;
constexpr double timeFadeSeconds = 0.04;
constexpr double routeFadeSeconds = 0.03;
constexpr double smoothingSeconds = 0.02;
constexpr float loopBandlimitHz = 6000.0f;
constexpr int controlInterval = 8;

struct Settings
{
    double delaySeconds = 0.3;
    float feedback = 0.3f;      // 0..1
    float mix = 0.2f;           // 0..1
    float hiCutHz = 7000.0f;
    float loCutHz = 120.0f;
    float modDepth = 0.0f;      // 0..1，对应 0..maxModDepthMs
    float modRateHz = 0.6f;
    float driveDb = 6.0f;
    float lofi = 0.0f;          // 0..1
    Mode mode = Mode::stereo;
};

// 双折线干湿：50% 以下干声保持原样，只把回声加上去；50% 以上才开始压干声。
inline float dryGainForMix (float mix) noexcept     { return std::min (1.0f, 2.0f * (1.0f - mix)); }
inline float wetGainForMix (float mix) noexcept     { return std::min (1.0f, 2.0f * mix); }

class Engine
{
public:
    void prepare (double newSampleRate, int newNumChannels)
    {
        sampleRate = newSampleRate;
        numChannels = juce::jlimit (1, 2, newNumChannels);

        const auto wanted = (int) std::ceil (maxDelaySeconds * sampleRate + 2.0 * maxModDepthMs * 0.001 * sampleRate) + 16;
        lineSize = juce::nextPowerOfTwo (wanted);
        lineMask = lineSize - 1;
        for (auto& line : lines)
            line.assign ((size_t) lineSize, 0.0f);

        feedbackSmooth.reset (sampleRate, smoothingSeconds);
        mixSmooth.reset (sampleRate, smoothingSeconds);
        modDepthSmooth.reset (sampleRate, smoothingSeconds);
        lofiSmooth.reset (sampleRate, smoothingSeconds);
        driveSmooth.reset (sampleRate, smoothingSeconds);
        hiCutSmooth.reset (sampleRate, smoothingSeconds);
        loCutSmooth.reset (sampleRate, smoothingSeconds);

        fadeStep = 1.0 / (timeFadeSeconds * sampleRate);
        routeFadeStep = 1.0f / (float) (routeFadeSeconds * sampleRate);
        bandlimitGain = onePoleGain (loopBandlimitHz);

        reset();
    }

    // 清掉尾音，所有平滑量直接跳到当前设置。
    void reset() noexcept
    {
        for (auto& line : lines)
            std::fill (line.begin(), line.end(), 0.0f);
        for (auto& state : channels)
            state = {};
        writePos = 0;
        lfoPhase = 0.0;
        fadePos = 1.0;

        feedbackSmooth.setCurrentAndTargetValue (settings.feedback);
        mixSmooth.setCurrentAndTargetValue (settings.mix);
        modDepthSmooth.setCurrentAndTargetValue (settings.modDepth);
        lofiSmooth.setCurrentAndTargetValue (settings.lofi);
        driveSmooth.setCurrentAndTargetValue (settings.driveDb);
        hiCutSmooth.setCurrentAndTargetValue (settings.hiCutHz);
        loCutSmooth.setCurrentAndTargetValue (settings.loCutHz);

        targetDelay = activeDelay = previousDelay = delayToSamples (settings.delaySeconds);
        routeFrom = routeTo = route = routeFor (effectiveMode());
        routePos = 1.0f;
        controlCountdown = 0;
        controlsCurrent = false;
    }

    // 只在音频线程、process 之前调用。
    void setSettings (const Settings& next) noexcept
    {
        settings = next;
        feedbackSmooth.setTargetValue (next.feedback);
        mixSmooth.setTargetValue (next.mix);
        modDepthSmooth.setTargetValue (next.modDepth);
        lofiSmooth.setTargetValue (next.lofi);
        driveSmooth.setTargetValue (next.driveDb);
        hiCutSmooth.setTargetValue (next.hiCutHz);
        loCutSmooth.setTargetValue (next.loCutHz);
        targetDelay = delayToSamples (next.delaySeconds);

        const auto wanted = routeFor (effectiveMode());
        if (wanted != routeTo)
        {
            routeFrom = route;
            routeTo = wanted;
            routePos = 0.0f;
        }
    }

    void process (float* const* audio, int numSamples) noexcept
    {
        auto* left = audio[0];
        auto* right = numChannels > 1 ? audio[1] : nullptr;
        const auto lfoStep = (double) settings.modRateHz / sampleRate;

        for (int i = 0; i < numSamples; ++i)
        {
            if (controlCountdown <= 0)
            {
                updateControls();
                controlCountdown = controlInterval;
            }
            --controlCountdown;

            const auto feedback = feedbackSmooth.getNextValue();
            const auto mix = mixSmooth.getNextValue();
            const auto depthSamples = modDepthSmooth.getNextValue() * maxModDepthMs * 0.001 * sampleRate;

            if (routePos < 1.0f)
            {
                routePos = std::min (1.0f, routePos + routeFadeStep);
                route = Route::lerp (routeFrom, routeTo, routePos);
            }

            if (fadePos >= 1.0 && targetDelay != activeDelay)
            {
                previousDelay = activeDelay;
                activeDelay = targetDelay;
                fadePos = 0.0;
            }

            double modLeft = 0.0, modRight = 0.0;
            if (depthSamples > 0.0)
            {
                modLeft = depthSamples * std::sin (juce::MathConstants<double>::twoPi * lfoPhase);
                modRight = depthSamples * std::cos (juce::MathConstants<double>::twoPi * lfoPhase);
            }
            lfoPhase += lfoStep;
            if (lfoPhase >= 1.0)
                lfoPhase -= 1.0;

            const auto wetLeft = channels[0].processLoop (readHeads (0, modLeft), control);
            const auto wetRight = channels[1].processLoop (readHeads (1, modRight), control);

            if (fadePos < 1.0)
                fadePos += fadeStep;

            const auto inLeft = sanitise (left[i]);
            const auto inRight = right != nullptr ? sanitise (right[i]) : inLeft;

            const auto backLeft = channels[0].bandlimit (wetLeft, bandlimitGain) * feedback;
            const auto backRight = channels[1].bandlimit (wetRight, bandlimitGain) * feedback;

            lines[0][(size_t) writePos] = route.inLL * inLeft + route.inLR * inRight
                                        + route.backLL * backLeft + route.backLR * backRight;
            lines[1][(size_t) writePos] = route.inRL * inLeft + route.inRR * inRight
                                        + route.backRL * backLeft + route.backRR * backRight;
            writePos = (writePos + 1) & lineMask;

            const auto dry = dryGainForMix (mix);
            const auto wet = wetGainForMix (mix);
            left[i] = dry * inLeft + wet * (route.outLL * wetLeft + route.outLR * wetRight);
            if (right != nullptr)
                right[i] = dry * inRight + wet * (route.outRL * wetLeft + route.outRR * wetRight);
        }
    }

private:
    // 输入、反馈、输出三个 2×2 矩阵。切模式时对矩阵做线性淡变，不会有爆音；
    // 每行系数之和不大于 1，淡变途中环路增益也不会超过单位增益。
    struct Route
    {
        float inLL, inLR, inRL, inRR;
        float backLL, backLR, backRL, backRR;
        float outLL, outLR, outRL, outRR;

        bool operator!= (const Route& other) const noexcept
        {
            return inLL != other.inLL || inLR != other.inLR || inRL != other.inRL || inRR != other.inRR
                || backLL != other.backLL || backLR != other.backLR || backRL != other.backRL || backRR != other.backRR
                || outLL != other.outLL || outLR != other.outLR || outRL != other.outRL || outRR != other.outRR;
        }

        static Route lerp (const Route& a, const Route& b, float t) noexcept
        {
            const auto mid = [t] (float x, float y) { return x + (y - x) * t; };
            return { mid (a.inLL, b.inLL), mid (a.inLR, b.inLR), mid (a.inRL, b.inRL), mid (a.inRR, b.inRR),
                     mid (a.backLL, b.backLL), mid (a.backLR, b.backLR), mid (a.backRL, b.backRL), mid (a.backRR, b.backRR),
                     mid (a.outLL, b.outLL), mid (a.outLR, b.outLR), mid (a.outRL, b.outRL), mid (a.outRR, b.outRR) };
        }
    };

    static Route routeFor (Mode mode) noexcept
    {
        switch (mode)
        {
            case Mode::pingPong: // 输入混成单声道进左线，第一次回声在左，之后左右交替。
                return { 0.5f, 0.5f, 0.0f, 0.0f,
                         0.0f, 1.0f, 1.0f, 0.0f,
                         1.0f, 0.0f, 0.0f, 1.0f };
            case Mode::mono:     // 只用左线，同一路回声送到两边。
                return { 0.5f, 0.5f, 0.0f, 0.0f,
                         1.0f, 0.0f, 0.0f, 0.0f,
                         1.0f, 0.0f, 1.0f, 0.0f };
            case Mode::stereo:
            default:
                return { 1.0f, 0.0f, 0.0f, 1.0f,
                         1.0f, 0.0f, 0.0f, 1.0f,
                         1.0f, 0.0f, 0.0f, 1.0f };
        }
    }

    struct ControlValues
    {
        float a1Hi = 1.0f, a2Hi = 0.0f, a3Hi = 0.0f;
        float a1Lo = 1.0f, a2Lo = 0.0f, a3Lo = 0.0f, gLo = 0.0f;
        float driveGain = 1.0f, inverseDrive = 1.0f, bias = 0.0f, biasOffset = 0.0f;
        float lofiWet = 0.0f, lofiStep = 1.0f, lofiQuant = 0.0f;
        float a1Pre = 1.0f, a2Pre = 0.0f, a3Pre = 0.0f;
    };

    // 二阶 TPT 状态变量滤波器的系数，巴特沃斯阻尼。
    static constexpr float svfDamping = 1.41421356f;
    static constexpr float saturationCeiling = 1.5f;

    static void svfCoefficients (float g, float& a1, float& a2, float& a3) noexcept
    {
        a1 = 1.0f / (1.0f + g * (g + svfDamping));
        a2 = g * a1;
        a3 = g * a2;
    }

    struct ChannelState
    {
        float hi1 = 0.0f, hi2 = 0.0f, lo1 = 0.0f, lo2 = 0.0f, pre1 = 0.0f, pre2 = 0.0f;
        float band = 0.0f, hold = 0.0f, holdPhase = 1.0f;

        float processLoop (float x, const ControlValues& c) noexcept
        {
            const auto driven = c.driveGain * x + c.bias;
            auto y = (saturationCeiling * std::tanh (driven / saturationCeiling) - c.biasOffset) * c.inverseDrive;

            if (c.lofiWet > 0.0f)
            {
                const auto narrowed = lowpass (y, pre1, pre2, c.a1Pre, c.a2Pre, c.a3Pre);
                holdPhase += c.lofiStep;
                if (holdPhase >= 1.0f)
                {
                    holdPhase -= 1.0f;
                    hold = std::round (narrowed * c.lofiQuant) / c.lofiQuant;
                }
                y += (hold - y) * c.lofiWet;
            }

            y = lowpass (y, hi1, hi2, c.a1Hi, c.a2Hi, c.a3Hi);
            return y - lowpassBandForHighpass (y, c);
        }

        float bandlimit (float x, float gain) noexcept
        {
            const auto v = (x - band) * gain;
            const auto out = v + band;
            band = out + v;
            return out;
        }

    private:
        static float lowpass (float x, float& s1, float& s2, float a1, float a2, float a3) noexcept
        {
            const auto v3 = x - s2;
            const auto v1 = a1 * s1 + a2 * v3;
            const auto v2 = s2 + a2 * s1 + a3 * v3;
            s1 = 2.0f * v1 - s1;
            s2 = 2.0f * v2 - s2;
            return v2;
        }

        // 同一个状态变量滤波器的高通输出写成 x 减去这个量，x - k*v1 - v2。
        float lowpassBandForHighpass (float x, const ControlValues& c) noexcept
        {
            const auto v3 = x - lo2;
            const auto v1 = c.a1Lo * lo1 + c.a2Lo * v3;
            const auto v2 = lo2 + c.a2Lo * lo1 + c.a3Lo * v3;
            lo1 = 2.0f * v1 - lo1;
            lo2 = 2.0f * v2 - lo2;
            return svfDamping * v1 + v2;
        }
    };

    // 只在音色参数还在平滑时才重算。滤波器系数跳变的步子小才不会有阶梯噪声，所以间隔要短。
    void updateControls() noexcept
    {
        if (controlsCurrent && ! hiCutSmooth.isSmoothing() && ! loCutSmooth.isSmoothing()
                            && ! driveSmooth.isSmoothing() && ! lofiSmooth.isSmoothing())
            return;
        controlsCurrent = true;

        hiCutSmooth.skip (controlInterval);
        loCutSmooth.skip (controlInterval);
        driveSmooth.skip (controlInterval);
        lofiSmooth.skip (controlInterval);

        svfCoefficients (prewarp (hiCutSmooth.getCurrentValue()), control.a1Hi, control.a2Hi, control.a3Hi);
        svfCoefficients (prewarp (loCutSmooth.getCurrentValue()), control.a1Lo, control.a2Lo, control.a3Lo);

        const auto driveDb = driveSmooth.getCurrentValue();
        control.driveGain = mint::dbToGain (driveDb);
        control.inverseDrive = 1.0f / control.driveGain;
        // 少量直流偏置带来偶次谐波，偏置本身减掉，不会往环里灌直流。
        control.bias = 0.08f * driveDb / 18.0f;
        control.biasOffset = saturationCeiling * std::tanh (control.bias / saturationCeiling);

        const auto lofi = lofiSmooth.getCurrentValue();
        control.lofiWet = std::min (1.0f, lofi * 4.0f);
        if (control.lofiWet > 0.0f)
        {
            // 最低 8 位、4 kHz，相当于早期数字延迟。位数再低，人声的安静段和尾音会被量化成静音。
            const auto holdRate = 16000.0f * std::pow (4000.0f / 16000.0f, lofi);
            control.lofiStep = std::min (1.0f, holdRate / (float) sampleRate);
            control.lofiQuant = std::pow (2.0f, 11.0f - 4.0f * lofi);
            svfCoefficients (prewarp (holdRate * 0.5f), control.a1Pre, control.a2Pre, control.a3Pre);
        }
    }

    float prewarp (float hz) const noexcept
    {
        const auto limited = std::min (hz, 0.49f * (float) sampleRate);
        return std::tan (juce::MathConstants<float>::pi * limited / (float) sampleRate);
    }

    float onePoleGain (float hz) const noexcept
    {
        const auto g = prewarp (hz);
        return g / (1.0f + g);
    }

    Mode effectiveMode() const noexcept     { return numChannels == 1 ? Mode::mono : settings.mode; }

    // 静态延迟取整到采样，不做小数插值，回声不会因为插值额外变暗，误差不超过半个采样。
    double delayToSamples (double seconds) const noexcept
    {
        return juce::jlimit (8.0, maxDelaySeconds * sampleRate, std::round (seconds * sampleRate));
    }

    float readHeads (int channel, double mod) const noexcept
    {
        const auto current = readLine (channel, modulated (activeDelay, mod));
        if (fadePos >= 1.0)
            return current;

        // 等增益淡变：两头系数之和恒为 1，不会抬高环路增益。
        const auto t = (float) fadePos;
        const auto h = t * t * (3.0f - 2.0f * t);
        return current * h + readLine (channel, modulated (previousDelay, mod)) * (1.0f - h);
    }

    static double modulated (double base, double mod) noexcept
    {
        // 调制量不超过基础延迟的一半，极短延迟时读头也不会碰到写头。
        return std::max (3.0, base + juce::jlimit (-0.5 * base, 0.5 * base, mod));
    }

    float readLine (int channel, double distance) const noexcept
    {
        const auto* line = lines[(size_t) channel].data();
        const auto whole = (int) std::floor (distance);
        const auto frac = (float) (distance - whole);
        const auto newest = writePos - whole;

        if (frac <= 0.0f)
            return line[newest & lineMask];

        // 三阶 Hermite。读头在 newest 和 newest-1 之间，frac 从 newest 往旧数据方向数。
        const auto xp1 = line[(newest + 1) & lineMask];
        const auto x0 = line[newest & lineMask];
        const auto xm1 = line[(newest - 1) & lineMask];
        const auto xm2 = line[(newest - 2) & lineMask];
        const auto c1 = 0.5f * (xm1 - xp1);
        const auto c2 = xp1 - 2.5f * x0 + 2.0f * xm1 - 0.5f * xm2;
        const auto c3 = 0.5f * (xm2 - xp1) + 1.5f * (x0 - xm1);
        return ((c3 * frac + c2) * frac + c1) * frac + x0;
    }

    static float sanitise (float x) noexcept        { return std::isfinite (x) ? x : 0.0f; }

    double sampleRate = 48000.0;
    int numChannels = 2;
    Settings settings;

    std::array<std::vector<float>, 2> lines;
    int lineSize = 1, lineMask = 0, writePos = 0;

    std::array<ChannelState, 2> channels;
    ControlValues control;
    float bandlimitGain = 1.0f;
    int controlCountdown = 0;
    bool controlsCurrent = false;

    double targetDelay = 0.0, activeDelay = 0.0, previousDelay = 0.0;
    double fadePos = 1.0, fadeStep = 0.0;
    double lfoPhase = 0.0;

    Route routeFrom {}, routeTo {}, route {};
    float routePos = 1.0f, routeFadeStep = 1.0f;

    juce::LinearSmoothedValue<float> feedbackSmooth, mixSmooth, modDepthSmooth, lofiSmooth, driveSmooth;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative> hiCutSmooth { 7000.0f }, loCutSmooth { 120.0f };
};

} // namespace echo
