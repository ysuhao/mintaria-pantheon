// 频段引擎：把频段设置变成状态变量滤波器级联，处理系数平滑、淡入淡出和动态检测。
// 除 prepare 外全部实时安全。
#pragma once

#include "IrisFilterDesign.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace iris
{

constexpr int numBands = 24;
constexpr int maxSections = 8;
constexpr int maxChunk = 64;

enum class BandType { bell, lowShelf, highShelf, lowCut, highCut, notch, tilt };
enum class StereoMode { stereo, left, right, mid, side };

constexpr std::array<int, 9> slopeChoices { 6, 12, 18, 24, 30, 36, 48, 72, 96 };

inline bool typeHasGain (BandType type) noexcept
{
    return type == BandType::bell || type == BandType::lowShelf || type == BandType::highShelf || type == BandType::tilt;
}

inline bool typeIsCut (BandType type) noexcept
{
    return type == BandType::lowCut || type == BandType::highCut;
}

struct BandSettings
{
    bool active = false;
    BandType type = BandType::bell;
    int slopeIndex = 1;
    StereoMode stereo = StereoMode::stereo;
    double frequency = 1000.0, gainDb = 0.0, q = 1.0;
    bool dynamic = false;
    double thresholdDb = -24.0, rangeDb = -6.0;
};

// 面板上的 Q 换算成模拟原型的 Q。默认 Q = 1 时切除是平坦的巴特沃斯，架式没有过冲。
// 架式的响应要缓一些，面板的整个范围 0.025 到 40 对应原型的约 0.12 到 4。
inline double prototypeQ (BandType type, double q) noexcept
{
    constexpr double neutral = 0.70710678118654752;
    if (type == BandType::lowShelf || type == BandType::highShelf || type == BandType::tilt)
        return neutral * std::pow (q, 0.47);
    return neutral * q;
}

// 按斜率展开成巴特沃斯级联，用户的 Q 只放大最尖的那一节。返回节数。
inline int designBand (const BandSettings& s, double gainDb, double sampleRate, std::array<SectionDesign, maxSections>& out) noexcept
{
    const auto q0 = prototypeQ (s.type, s.q);

    switch (s.type)
    {
        case BandType::bell:      out[0] = designSection (SectionShape::bell, s.frequency, gainDb, q0, sampleRate); return 1;
        case BandType::lowShelf:  out[0] = designSection (SectionShape::lowShelf, s.frequency, gainDb, q0, sampleRate); return 1;
        case BandType::highShelf: out[0] = designSection (SectionShape::highShelf, s.frequency, gainDb, q0, sampleRate); return 1;
        case BandType::tilt:      out[0] = designSection (SectionShape::tilt, s.frequency, gainDb, q0, sampleRate); return 1;
        case BandType::notch:     out[0] = designSection (SectionShape::notch, s.frequency, 0.0, q0, sampleRate); return 1;
        case BandType::lowCut:
        case BandType::highCut:
            break;
    }

    const auto high = s.type == BandType::lowCut;
    const auto order = slopeChoices[(size_t) s.slopeIndex] / 6;
    const auto pairs = order / 2;
    auto count = 0;

    if (order % 2 == 1)
        out[(size_t) count++] = designSection (high ? SectionShape::highPass1 : SectionShape::lowPass1, s.frequency, 0.0, 0.7071, sampleRate);

    for (int k = 1; k <= pairs; ++k)
    {
        auto q = 0.5 / std::cos (pi * (2.0 * k - 1.0) / (2.0 * order));
        if (k == pairs)
            q *= s.q;
        out[(size_t) count++] = designSection (high ? SectionShape::highPass2 : SectionShape::lowPass2, s.frequency, 0.0, q, sampleRate);
    }
    return count;
}

// 一个频段所有节叠加后的幅度，界面画曲线时用。
inline double bandMagnitudeDb (const std::array<SectionDesign, maxSections>& sections, int count, double frequency, double sampleRate) noexcept
{
    auto sum = 0.0;
    for (int i = 0; i < count; ++i)
        sum += 10.0 * std::log10 (std::max (sections[(size_t) i].magnitudeSquared (frequency, sampleRate), 1.0e-30));
    return sum;
}

struct SvfCoefficients
{
    double a1 = 0.0, a2 = 0.0, a3 = 0.0, m0 = 1.0, m1 = 0.0, m2 = 0.0;
};

inline SvfCoefficients toCoefficients (const SectionDesign& d) noexcept
{
    SvfCoefficients c;
    c.a1 = 1.0 / (1.0 + d.g * (d.g + d.k));
    c.a2 = d.g * c.a1;
    c.a3 = d.g * c.a2;
    c.m0 = d.mHp;
    c.m1 = d.mBp - d.k * d.mHp;
    c.m2 = d.mLp - d.mHp;
    return c;
}

struct SvfState
{
    double ic1 = 0.0, ic2 = 0.0;

    void clear() noexcept                   { ic1 = ic2 = 0.0; }
    bool isFinite() const noexcept          { return std::isfinite (ic1) && std::isfinite (ic2); }
};

template <bool ramping>
inline void runSection (SvfState& state, SvfCoefficients c, const SvfCoefficients& delta, double* x, int n) noexcept
{
    auto ic1 = state.ic1, ic2 = state.ic2;
    for (int i = 0; i < n; ++i)
    {
        const auto in = x[i];
        const auto v3 = in - ic2;
        const auto v1 = c.a1 * ic1 + c.a2 * v3;
        const auto v2 = ic2 + c.a2 * ic1 + c.a3 * v3;
        ic1 = 2.0 * v1 - ic1;
        ic2 = 2.0 * v2 - ic2;
        x[i] = c.m0 * in + c.m1 * v1 + c.m2 * v2;

        if constexpr (ramping)
        {
            c.a1 += delta.a1; c.a2 += delta.a2; c.a3 += delta.a3;
            c.m0 += delta.m0; c.m1 += delta.m1; c.m2 += delta.m2;
        }
    }
    state.ic1 = ic1;
    state.ic2 = ic2;
}

// 动态检测器：频段限制后的一阶包络。只进不出音频通路，不增加延迟。
class Detector
{
public:
    void prepare (double rate) noexcept                { sampleRate = rate; reset(); }
    void reset() noexcept                              { state = {}; envelope = 0.0; }

    void configure (BandType type, double frequency, double q, double attackMs, double releaseMs) noexcept
    {
        g = std::tan (pi * std::min (frequency, 0.49 * sampleRate) / sampleRate);
        k = 1.0 / std::clamp (q, 0.5, 8.0);
        a1 = 1.0 / (1.0 + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
        kind = type;
        attack = std::exp (-1.0 / (0.001 * attackMs * sampleRate));
        release = std::exp (-1.0 / (0.001 * releaseMs * sampleRate));
    }

    // a、b 是要检测的声道，b 可以为空。
    void process (const double* a, const double* b, int n) noexcept
    {
        for (int i = 0; i < n; ++i)
        {
            auto level = std::abs (filter (state[0], a[i]));
            if (b != nullptr)
                level = std::max (level, std::abs (filter (state[1], b[i])));
            const auto coef = level > envelope ? attack : release;
            envelope = level + coef * (envelope - level);
        }
    }

    double levelDb() const noexcept                    { return 20.0 * std::log10 (std::max (envelope, 1.0e-6)); }

private:
    double filter (SvfState& s, double in) const noexcept
    {
        const auto v3 = in - s.ic2;
        const auto v1 = a1 * s.ic1 + a2 * v3;
        const auto v2 = s.ic2 + a2 * s.ic1 + a3 * v3;
        s.ic1 = 2.0 * v1 - s.ic1;
        s.ic2 = 2.0 * v2 - s.ic2;

        switch (kind)
        {
            case BandType::lowShelf:   return v2;
            case BandType::highShelf:  return in - k * v1 - v2;
            case BandType::tilt:       return in;
            case BandType::bell:
            case BandType::lowCut:
            case BandType::highCut:
            case BandType::notch:
                break;
        }
        return k * v1;
    }

    double sampleRate = 48000.0, g = 0.1, k = 1.0, a1 = 1.0, a2 = 0.0, a3 = 0.0;
    double attack = 0.0, release = 0.0, envelope = 0.0;
    BandType kind = BandType::bell;
    std::array<SvfState, 2> state {};
};

// 动态增益量：阈值以上每 2 dB 增益移动 1 dB，到范围为止。至少留 6 dB 的过渡。
inline double dynamicOffsetDb (double levelDb, double thresholdDb, double rangeDb) noexcept
{
    const auto excess = levelDb - thresholdDb;
    if (excess <= 0.0)
        return 0.0;
    const auto span = std::max (6.0, 2.0 * std::abs (rangeDb));
    return rangeDb * std::min (1.0, excess / span);
}

class Band
{
public:
    void prepare (double rate, int interval) noexcept
    {
        sampleRate = rate;
        tickInterval = interval;
        detector.prepare (rate);
        smoothing = 1.0 - std::exp (-((double) interval / rate) / 0.02);
        offsetSmoothing = 1.0 - std::exp (-((double) interval / rate) / 0.004);
        fadeStep = 1.0 / (3.0 * (double) interval);
        reset();
    }

    void reset() noexcept
    {
        running = false;
        mix = 0.0;
        mixStep = 0.0;
        for (auto& channel : states)
            for (auto& s : channel)
                s.clear();
        detector.reset();
        currentOffset.store (0.0f, std::memory_order_relaxed);
    }

    // 每个控制周期调用一次，从音频线程调用。
    void tick (const BandSettings& wanted, double attackMs, double releaseMs) noexcept
    {
        const auto structureChanged = wanted.type != applied.type || wanted.slopeIndex != applied.slopeIndex || wanted.stereo != applied.stereo;

        if (mixStep < 0.0 && mix <= 0.0)
        {
            mix = 0.0;
            mixStep = 0.0;
            running = false;
        }

        if (! running)
        {
            if (wanted.active)
                start (wanted, attackMs, releaseMs);
            return;
        }

        if ((! wanted.active || structureChanged) && mixStep >= 0.0)
            mixStep = -fadeStep;

        if (mixStep > 0.0 && mix >= 1.0)
        {
            mix = 1.0;
            mixStep = 0.0;
        }

        for (int s = 0; s < numSections; ++s)
        {
            coefficients[(size_t) s] = targets[(size_t) s];
            deltas[(size_t) s] = {};
        }
        ramping = false;

        if (structureChanged)
            return;

        // 频率和 Q 在对数域平滑，增益在 dB 域平滑。
        const auto logF = std::log (wanted.frequency);
        const auto logQ = std::log (wanted.q);
        smoothedLogF += smoothing * (logF - smoothedLogF);
        smoothedLogQ += smoothing * (logQ - smoothedLogQ);
        smoothedGain += smoothing * (wanted.gainDb - smoothedGain);
        if (std::abs (logF - smoothedLogF) < 1.0e-5)    smoothedLogF = logF;
        if (std::abs (logQ - smoothedLogQ) < 1.0e-5)    smoothedLogQ = logQ;
        if (std::abs (wanted.gainDb - smoothedGain) < 1.0e-4) smoothedGain = wanted.gainDb;

        applied = wanted;
        applied.frequency = std::exp (smoothedLogF);
        applied.q = std::exp (smoothedLogQ);

        const auto dynamicOn = wanted.dynamic && typeHasGain (wanted.type);
        auto offset = 0.0;
        if (dynamicOn)
        {
            detector.configure (wanted.type, applied.frequency, applied.q, attackMs, releaseMs);
            offset = dynamicOffsetDb (detector.levelDb(), wanted.thresholdDb, wanted.rangeDb);
        }
        detecting = dynamicOn;

        // 再过一级短平滑，改阈值、范围或关掉动态时增益不会在一个控制周期内跳完。
        smoothedOffset += offsetSmoothing * (offset - smoothedOffset);
        if (std::abs (offset - smoothedOffset) < 1.0e-4)
            smoothedOffset = offset;
        currentOffset.store ((float) smoothedOffset, std::memory_order_relaxed);

        const auto gain = smoothedGain + smoothedOffset;
        const auto moved = std::abs (smoothedLogF - designedLogF) > 1.0e-4
                        || std::abs (smoothedLogQ - designedLogQ) > 1.0e-3
                        || std::abs (gain - designedGain) > 2.0e-3;

        if (moved)
        {
            redesign (gain);
            ramping = true;
            const auto scale = 1.0 / (double) tickInterval;
            for (int s = 0; s < numSections; ++s)
            {
                const auto& from = coefficients[(size_t) s];
                const auto& to = targets[(size_t) s];
                deltas[(size_t) s] = { (to.a1 - from.a1) * scale, (to.a2 - from.a2) * scale, (to.a3 - from.a3) * scale,
                                       (to.m0 - from.m0) * scale, (to.m1 - from.m1) * scale, (to.m2 - from.m2) * scale };
            }
        }

        for (auto& channel : states)
            for (int s = 0; s < numSections; ++s)
                if (! channel[(size_t) s].isFinite())
                    channel[(size_t) s].clear();
    }

    // 检测器读取未延迟的输入。input 是声道指针，channels 为 1 或 2。
    void detect (const double* const* input, int channels, int n) noexcept
    {
        if (! detecting || ! running)
            return;

        const auto* a = input[0];
        const auto* b = channels > 1 ? input[1] : nullptr;
        std::array<double, maxChunk> derived;

        switch (applied.stereo)
        {
            case StereoMode::stereo:
                detector.process (a, b, n);
                break;
            case StereoMode::left:
                detector.process (a, nullptr, n);
                break;
            case StereoMode::right:
                if (b != nullptr)
                    detector.process (b, nullptr, n);
                break;
            case StereoMode::mid:
            case StereoMode::side:
                if (b == nullptr)
                {
                    if (applied.stereo == StereoMode::mid)
                        detector.process (a, nullptr, n);
                    break;
                }
                for (int i = 0; i < n; ++i)
                    derived[(size_t) i] = 0.5 * (applied.stereo == StereoMode::mid ? a[i] + b[i] : a[i] - b[i]);
                detector.process (derived.data(), nullptr, n);
                break;
        }
    }

    void process (double* const* audio, int channels, int n) noexcept
    {
        if (! running)
            return;

        const auto fading = mixStep != 0.0 || mix < 1.0;
        std::array<std::array<double, maxChunk>, 2> dry;
        if (fading)
            for (int ch = 0; ch < channels; ++ch)
                std::copy (audio[ch], audio[ch] + n, dry[(size_t) ch].begin());

        auto* left = audio[0];
        auto* right = channels > 1 ? audio[1] : nullptr;

        switch (applied.stereo)
        {
            case StereoMode::stereo:
                runChannels (left, right, n);
                break;
            case StereoMode::left:
                runChannels (left, nullptr, n);
                break;
            case StereoMode::right:
                if (right != nullptr)
                    runChannels (nullptr, right, n);
                break;
            case StereoMode::mid:
            case StereoMode::side:
                if (right == nullptr)
                {
                    if (applied.stereo == StereoMode::mid)
                        runChannels (left, nullptr, n);
                }
                else
                {
                    // 左右临时换成中、侧，处理完再换回来。
                    for (int i = 0; i < n; ++i)
                    {
                        const auto m = 0.5 * (left[i] + right[i]);
                        const auto s = 0.5 * (left[i] - right[i]);
                        left[i] = m;
                        right[i] = s;
                    }
                    if (applied.stereo == StereoMode::mid)
                        runChannels (left, nullptr, n);
                    else
                        runChannels (nullptr, right, n);
                    for (int i = 0; i < n; ++i)
                    {
                        const auto m = left[i], s = right[i];
                        left[i] = m + s;
                        right[i] = m - s;
                    }
                }
                break;
        }

        advanceCoefficients (n);

        if (fading)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                auto amount = mix;
                auto* out = audio[ch];
                const auto* d = dry[(size_t) ch].data();
                for (int i = 0; i < n; ++i)
                {
                    amount = std::clamp (amount + mixStep, 0.0, 1.0);
                    const auto shaped = amount * amount * (3.0 - 2.0 * amount);
                    out[i] = d[i] + shaped * (out[i] - d[i]);
                }
            }
            mix = std::clamp (mix + mixStep * n, 0.0, 1.0);
        }
    }

    bool isRunning() const noexcept                { return running; }
    float getDynamicOffset() const noexcept        { return currentOffset.load (std::memory_order_relaxed); }

private:
    void start (const BandSettings& wanted, double attackMs, double releaseMs) noexcept
    {
        applied = wanted;
        smoothedLogF = std::log (wanted.frequency);
        smoothedLogQ = std::log (wanted.q);
        smoothedGain = wanted.gainDb;
        smoothedOffset = 0.0;

        detecting = wanted.dynamic && typeHasGain (wanted.type);
        if (detecting)
            detector.configure (wanted.type, wanted.frequency, wanted.q, attackMs, releaseMs);
        detector.reset();

        redesign (wanted.gainDb);
        for (int s = 0; s < numSections; ++s)
        {
            coefficients[(size_t) s] = targets[(size_t) s];
            deltas[(size_t) s] = {};
        }
        ramping = false;

        for (auto& channel : states)
            for (auto& s : channel)
                s.clear();

        running = true;
        mix = 0.0;
        mixStep = fadeStep;
    }

    void redesign (double gain) noexcept
    {
        std::array<SectionDesign, maxSections> designs;
        numSections = designBand (applied, gain, sampleRate, designs);
        for (int s = 0; s < numSections; ++s)
            targets[(size_t) s] = toCoefficients (designs[(size_t) s]);
        designedLogF = smoothedLogF;
        designedLogQ = smoothedLogQ;
        designedGain = gain;
    }

    void runChannels (double* a, double* b, int n) noexcept
    {
        for (int s = 0; s < numSections; ++s)
        {
            const auto& c = coefficients[(size_t) s];
            const auto& d = deltas[(size_t) s];
            if (ramping)
            {
                if (a != nullptr)  runSection<true> (states[0][(size_t) s], c, d, a, n);
                if (b != nullptr)  runSection<true> (states[1][(size_t) s], c, d, b, n);
            }
            else
            {
                if (a != nullptr)  runSection<false> (states[0][(size_t) s], c, d, a, n);
                if (b != nullptr)  runSection<false> (states[1][(size_t) s], c, d, b, n);
            }
        }
    }

    void advanceCoefficients (int n) noexcept
    {
        if (! ramping)
            return;
        const auto m = (double) n;
        for (int s = 0; s < numSections; ++s)
        {
            auto& c = coefficients[(size_t) s];
            const auto& d = deltas[(size_t) s];
            c.a1 += d.a1 * m; c.a2 += d.a2 * m; c.a3 += d.a3 * m;
            c.m0 += d.m0 * m; c.m1 += d.m1 * m; c.m2 += d.m2 * m;
        }
    }

    double sampleRate = 48000.0, smoothing = 0.1, offsetSmoothing = 0.5, fadeStep = 0.001;
    int tickInterval = 128;

    BandSettings applied;
    double smoothedLogF = 0.0, smoothedLogQ = 0.0, smoothedGain = 0.0, smoothedOffset = 0.0;
    double designedLogF = 0.0, designedLogQ = 0.0, designedGain = 0.0;

    int numSections = 0;
    std::array<SvfCoefficients, maxSections> coefficients {}, targets {}, deltas {};
    std::array<std::array<SvfState, maxSections>, 2> states {};
    bool ramping = false, running = false, detecting = false;
    double mix = 0.0, mixStep = 0.0;

    Detector detector;
    std::atomic<float> currentOffset { 0.0f };
};

} // namespace iris
