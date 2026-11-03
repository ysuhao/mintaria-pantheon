#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Aegis 的增益计算内核，只依赖标准库。所有内存都在 prepare 里分配。
//
// 有两条路径，由 Look-ahead 是否为 0 决定：
//   0 延迟：瞬时启动的增益计算 + 末级软削波（softClip）。
//   前瞻：  滑动最大值 + 释放 + 两级滑动平均，增益在峰值到达前平滑地降好。
namespace aegis
{

// 软削波拐点（-1 dB）。低于拐点的信号线性通过。
constexpr float kneeLevel = 0.891251f;
// 二次曲线从拐点起斜率线性减到 0，在 catchLevel 处恰好到 1 且斜率为 0，没有折角。
constexpr float catchLevel = 2.0f - kneeLevel;

// 把 [0, catchLevel] 映射到 [0, 1]。任何输入的输出绝对值都不超过 1。
inline float softClip (float u) noexcept
{
    const auto a = std::abs (u);
    if (a <= kneeLevel)
        return u;

    const auto over = a - kneeLevel;
    const auto shaped = a >= catchLevel ? 1.0f
                                        : kneeLevel + over - over * over / (2.0f * (catchLevel - kneeLevel));
    return u < 0.0f ? -shaped : shaped;
}

// 滑动窗口最大值（单调队列），均摊 O(1)。窗口长度可以逐采样变化。
class SlidingMax
{
public:
    void prepare (int maxWindow)
    {
        capacity = maxWindow + 2;
        values.assign ((size_t) capacity, 0.0f);
        stamps.assign ((size_t) capacity, 0u);
        reset();
    }

    void reset() noexcept
    {
        head = count = 0;
        now = 0u;
    }

    // 返回最近 window + 1 个采样（含当前）的最大值。
    float push (float value, int window) noexcept
    {
        while (count > 0 && values[(size_t) index (count - 1)] <= value)
            --count;

        const auto slot = (size_t) index (count);
        values[slot] = value;
        stamps[slot] = now;
        ++count;

        while (now - stamps[(size_t) head] > (std::uint32_t) window)
        {
            head = head + 1 == capacity ? 0 : head + 1;
            --count;
        }

        ++now;
        return values[(size_t) head];
    }

private:
    int index (int offset) const noexcept
    {
        const auto i = head + offset;
        return i >= capacity ? i - capacity : i;
    }

    std::vector<float> values;
    std::vector<std::uint32_t> stamps;
    int capacity = 2, head = 0, count = 0;
    std::uint32_t now = 0u;
};

// 滑动平均。累加和每绕一圈重算一次，不会长期漂移。
class BoxFilter
{
public:
    void prepare (int maxLength)
    {
        ring.assign ((size_t) std::max (1, maxLength), 1.0f);
        setLength (1);
    }

    void setLength (int newLength) noexcept
    {
        length = std::clamp (newLength, 1, (int) ring.size());
        reset();
    }

    // 窗口全部填 1，也就是不衰减。
    void reset() noexcept
    {
        std::fill (ring.begin(), ring.begin() + length, 1.0f);
        sum = (double) length;
        pos = 0;
    }

    float process (float value) noexcept
    {
        sum += (double) value - (double) ring[(size_t) pos];
        ring[(size_t) pos] = value;

        if (++pos == length)
        {
            pos = 0;
            sum = 0.0;
            for (int i = 0; i < length; ++i)
                sum += (double) ring[(size_t) i];
        }

        return (float) (sum / (double) length);
    }

private:
    std::vector<float> ring;
    double sum = 1.0;
    int length = 1, pos = 0;
};

// 输入是当前采样的链接峰值（已乘阈值增益），输出是当前要施加的增益。
// 前瞻模式下这个增益对应 lookahead 个采样之前的音频。
// 释放在线性增益上做一阶恢复，整条路径不需要逐采样的 log 和 pow。
class Limiter
{
public:
    void prepare (double newSampleRate, int maxLookahead)
    {
        sampleRate = newSampleRate;
        peaks.prepare (std::max (maxLookahead, (int) std::ceil (maxHoldMs * 0.001 * sampleRate)));
        attackCoef = onePole (zeroLatencyAttackMs);
        slowUpCoef = onePole (autoSustainMs);
        slowDownCoef = onePole (autoSlowMs);
        boxA.prepare (maxLookahead + 1);
        boxB.prepare (maxLookahead + 1);
        setLookahead (0);
    }

    void setLookahead (int samples) noexcept
    {
        lookahead = samples;
        // 两级窗长之和减 1 正好是 lookahead + 1，这是不过冲证明的前提（见 NOTES）。
        boxA.setLength (samples / 2 + 1);
        boxB.setLength (samples - samples / 2 + 1);
        reset();
    }

    void reset() noexcept
    {
        peaks.reset();
        boxA.reset();
        boxB.reset();
        gainFast = gainSlow = 1.0f;
    }

    // 释放时间 ms 指恢复到 90% 所需的时间，所以时间常数是 ms / ln 10。自动释放时不看旋钮。
    void setRelease (float ms, bool automatic) noexcept
    {
        isAutomatic = automatic;
        fastCoef = onePole (automatic ? autoFastMs : ms / 2.302585f);
        hold = (int) std::round ((automatic ? maxHoldMs : std::min (ms, maxHoldMs)) * 0.001f * (float) sampleRate);
    }

    float process (float peak) noexcept
    {
        return lookahead > 0 ? processLookahead (peak) : processZeroLatency (peak);
    }

private:
    float processZeroLatency (float peak) noexcept
    {
        const auto held = peaks.push (peak, hold);
        const auto target = held > kneeLevel ? kneeLevel / held : 1.0f;

        // 持续段由这条 0.1 ms 的攻击压到拐点以下，信号走软削波的线性区，稳态失真很低。
        gainFast += (target < gainFast ? attackCoef : fastCoef) * (target - gainFast);
        updateSlow (target);

        auto gain = std::min (gainFast, gainSlow);

        // 瞬时启动：这一采样如果会超过软削波的终点，立刻把增益压到位并保持。
        if (peak * gain > catchLevel)
        {
            gain = catchLevel / peak;
            gainFast = gain;
        }

        return gain;
    }

    float processLookahead (float peak) noexcept
    {
        // 窗口至少要盖住 lookahead + 1 个采样。
        const auto held = peaks.push (peak, std::max (lookahead, hold));
        const auto target = held > 1.0f ? 1.0f / held : 1.0f;

        gainFast = target < gainFast ? target : gainFast + fastCoef * (target - gainFast);
        updateSlow (target);

        return boxB.process (boxA.process (std::min (gainFast, gainSlow)));
    }

    // 程序相关释放：持续压缩时 gainSlow 慢慢降到位，松开后慢速恢复；短促的峰值拉不动它，恢复就快。
    void updateSlow (float target) noexcept
    {
        if (isAutomatic)
            gainSlow += (target < gainSlow ? slowUpCoef : slowDownCoef) * (target - gainSlow);
        else
            gainSlow = 1.0f;
    }

    float onePole (float ms) const noexcept             { return 1.0f - std::exp (-1.0f / (0.001f * ms * (float) sampleRate)); }

    static constexpr float zeroLatencyAttackMs = 0.1f;
    static constexpr float maxHoldMs = 10.0f;       // 覆盖 50 Hz 全波整流后的周期，低频稳态不被释放逐周期调制。
    static constexpr float autoFastMs = 5.0f;
    static constexpr float autoSlowMs = 215.0f;
    static constexpr float autoSustainMs = 150.0f;

    double sampleRate = 48000.0;
    SlidingMax peaks;
    BoxFilter boxA, boxB;
    int lookahead = 0, hold = 0;
    bool isAutomatic = true;
    float attackCoef = 1.0f, fastCoef = 1.0f, slowUpCoef = 1.0f, slowDownCoef = 1.0f;
    float gainFast = 1.0f, gainSlow = 1.0f;
};

} // namespace aegis
