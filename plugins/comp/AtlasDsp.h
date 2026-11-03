// Atlas 的压缩内核。不依赖界面，全部在 prepare 时分配，处理时不分配内存。
//
// 增益计算器和分支式包络的写法来自 Giannoulis, Massberg, Reiss,
// "Digital Dynamic Range Compressor Design", JAES 2012。
// 自动释放的波峰因数思路来自同一作者的 "Parameter Automation in a Dynamic Range Compressor", JAES 2013。
#pragma once

#include <mint_core/MintCore.h>

#include <algorithm>
#include <cmath>

namespace atlas
{

constexpr float kneeWidthDb = 12.0f;
constexpr float ceilingDb = -0.3f;

// 自动档的释放基准。界面上释放旋钮此时是禁用的，所以不能再受旋钮影响。取旋钮默认值，默认设置下的声音不变。
constexpr float autoReleaseBaseMs = 120.0f;

// 静态曲线，返回增益衰减量（dB，非负）。拐点是对称的二次软拐点。
inline float computeReductionDb (float levelDb, float thresholdDb, float ratio) noexcept
{
    const auto over = levelDb - thresholdDb;
    const auto slope = 1.0f - 1.0f / ratio;

    if (over <= -0.5f * kneeWidthDb)
        return 0.0f;
    if (over >= 0.5f * kneeWidthDb)
        return slope * over;

    const auto x = over + 0.5f * kneeWidthDb;
    return slope * x * x / (2.0f * kneeWidthDb);
}

enum class ReleaseShape { opto, electro };

// 释放时间常数相对旋钮值的倍数，取决于当前的衰减深度。
// 两条曲线都归一化到：从 12 dB 衰减回落 63%（到 4.41 dB）正好用掉旋钮设定的时间，差别在于前后的走法。
//   Opto：衰减深时走得快，接近 0 时越来越慢，留一条长尾。
//   Electro：衰减深时偏慢，接近 0 时加快，干净收尾。
inline float releaseShapeFactor (ReleaseShape shape, float reductionDb) noexcept
{
    if (shape == ReleaseShape::opto)
    {
        constexpr float norm = 0.6995f, maxFactor = 6.0f;
        return norm * std::min (maxFactor, 1.0f + 3.0f / std::max (reductionDb, 0.01f));
    }

    constexpr float norm = 0.948f;
    return norm * (0.35f + reductionDb / (reductionDb + 3.0f));
}

// 无延迟的滑动峰值。用两个相接的桶近似滑动窗口，上升立即跟随，下降最多滞后一个窗口。
// 窗口盖住人声基频的半周期，低频正弦的整流纹波就不会变成增益调制。
class PeakHold
{
public:
    void reset() noexcept                         { current = previous = 0.0f; counter = 0; }
    void setBucketSamples (int samples) noexcept  { bucketSamples = std::max (1, samples); }

    float process (float magnitude) noexcept
    {
        current = std::max (current, magnitude);
        const auto held = std::max (current, previous);

        if (++counter >= bucketSamples)
        {
            previous = current;
            current = 0.0f;
            counter = 0;
        }

        return held;
    }

private:
    float current = 0.0f, previous = 0.0f;
    int counter = 0, bucketSamples = 1;
};

// 自动释放：瞬态（波峰因数高）放得快，持续的声音（波峰因数低）放得慢。
class CrestRelease
{
public:
    void prepare (double sampleRate) noexcept
    {
        coef = std::exp (-1.0f / (crestWindowSeconds * (float) sampleRate));
        reset();
    }

    void reset() noexcept     { peakSquared = rmsSquared = 0.0f; }

    float process (float magnitude) noexcept
    {
        const auto squared = magnitude * magnitude;
        peakSquared = std::max (squared, coef * peakSquared);
        rmsSquared = squared + coef * (rmsSquared - squared);
        return factor();
    }

private:
    // 正弦（3 dB）放慢到 2 倍，15 dB 以上的瞬态加快到 0.25 倍，中间按对数插值。
    float factor() const noexcept
    {
        if (rmsSquared < 1.0e-12f)
            return 1.0f;

        const auto crestDb = 10.0f * std::log10 (std::max (1.0f, peakSquared / rmsSquared));
        const auto position = juce::jlimit (0.0f, 1.0f, (crestDb - 3.0f) / 12.0f);
        return std::pow (2.0f, 1.0f - 3.0f * position);
    }

    static constexpr float crestWindowSeconds = 0.2f;
    float coef = 0.0f, peakSquared = 0.0f, rmsSquared = 0.0f;
};

// Warm 档的谐波支路。只拿低频部分过多项式，高频不产生谐波，不用过采样也不会混叠。
// 信号按阈值电平归一化，谐波量只由衰减深度决定，不受阈值设多低的影响。
class WarmShaper
{
public:
    void prepare (double sampleRate) noexcept
    {
        lowpassCoef = 1.0f - std::exp (-juce::MathConstants<float>::twoPi * lowpassHz / (float) sampleRate);
        dcCoef = 1.0f - std::exp (-juce::MathConstants<float>::twoPi * dcHz / (float) sampleRate);
        reset();
    }

    void reset() noexcept     { low = 0.0f; dc = 0.0f; }

    // 返回要叠加到信号上的谐波。amount 是 0 到 1 的深度。
    float process (float input, float thresholdGain, float amount) noexcept
    {
        low += lowpassCoef * (input - low);

        // 除以 (1 + n²) 让大幅突发时谐波有界。
        const auto n = low / thresholdGain;
        const auto limiter = 1.0f / (1.0f + 0.25f * n * n);
        const auto second = secondLevel * n * n * limiter;
        const auto third = thirdLevel * n * n * n * limiter;

        dc += dcCoef * (second - dc);

        return amount * thresholdGain * ((second - dc) + third);
    }

private:
    static constexpr float lowpassHz = 1200.0f, dcHz = 20.0f;
    static constexpr float secondLevel = 0.018f, thirdLevel = 0.006f;
    float lowpassCoef = 0.0f, dcCoef = 0.0f, low = 0.0f, dc = 0.0f;
};

// 输出保护：瞬时启动的峰值限制，只在输出越过上限时才动作，越线之前逐比特直通。
class OutputCeiling
{
public:
    void prepare (double sampleRate) noexcept
    {
        releaseCoef = std::exp (-1.0f / (0.05f * (float) sampleRate));
        ceiling = mint::dbToGain (ceilingDb);
        reset();
    }

    void reset() noexcept     { gain = 1.0f; }

    float process (float peak) noexcept
    {
        const auto wanted = peak > ceiling ? ceiling / peak : 1.0f;

        gain = wanted < gain ? wanted : wanted + releaseCoef * (gain - wanted);

        // 回到 1 时吸附，之后的信号完全不被碰到。
        if (gain > 0.99999f)
            gain = 1.0f;
        return gain;
    }

private:
    float releaseCoef = 0.0f, ceiling = 1.0f, gain = 1.0f;
};

} // namespace atlas
