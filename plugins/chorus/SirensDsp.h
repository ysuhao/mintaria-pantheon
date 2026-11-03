// Sirens 的音频引擎：四组常开的反相 BBD 式延迟线并联，按钮只控制每组的湿声增益，所以任意组合切换都没有延迟跳变。
#pragma once

#include <juce_core/juce_core.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sirens
{

// 三角波调制下延迟线的变化率恒定，音高偏移也就恒定，这是"没有音高摇晃"的来源。
struct ModeSpec
{
    float baseMs;       // 两条线之间的中心
    float depthMs;      // 三角波幅度，每条线的摆幅是它的两倍
    float rateHz;
    float weight;       // 强度，决定湿声占比
    bool swapSides;     // 交换长短两条线所在的声道，组合时左右平衡
    float startPhase;   // 让组合的各组 LFO 错开
};

// 强度依次增加：I→IV 的音高偏移实测约 ±2.7、±4.3、±5.1、±6.4 音分。
inline constexpr std::array<ModeSpec, 4> modeSpecs {{
    { 9.5f, 1.6f, 0.23f, 0.62f, false, 0.00f },
    { 8.5f, 2.4f, 0.25f, 0.74f, true,  0.31f },
    { 8.0f, 1.5f, 0.47f, 0.87f, false, 0.57f },
    { 7.5f, 1.7f, 0.53f, 1.00f, true,  0.83f },
}};

inline constexpr float lineGapMs = 0.5f;        // 两条线的最小距离的一半，两线永不重合，宽度不会周期性塌陷
inline constexpr float wetMidAmount = 0.6f;     // 湿声的中置分量决定折叠后的梳状，侧向分量决定宽度。1 是左右各走各的，0 是纯侧向
inline constexpr float maxDelayMs = 20.0f;
inline constexpr double highPassHz = 90.0;
inline constexpr double bandLimitHz = 10500.0;
inline constexpr float noiseRelativeDb = -58.0f;

// 尖角只占三角波很小一段，把音高转向变成一小段连续的滑动，避免 LFO 拐点处的音高阶跃。
inline float roundedTriangle (float phase) noexcept
{
    constexpr float corner = 0.08f;
    constexpr float peak = 1.0f - corner * 0.5f;

    const auto t = 4.0f * std::abs (phase - 0.5f) - 1.0f;
    const auto u = std::abs (t);
    if (u <= 1.0f - corner)
        return t / peak;

    const auto rest = 1.0f - u;
    const auto y = (1.0f - corner) + (corner * corner - rest * rest) / (2.0f * corner);
    return std::copysign (y / peak, t);
}

// 双精度 TDF-II 二阶节，低截止频率和高采样率下也稳定。
class Biquad
{
public:
    void setLowPass (double fc, double q, double rate) noexcept
    {
        const auto k = std::tan (juce::MathConstants<double>::pi * fc / rate);
        const auto norm = 1.0 / (1.0 + k / q + k * k);
        b0 = k * k * norm;
        b1 = 2.0 * b0;
        b2 = b0;
        a1 = 2.0 * (k * k - 1.0) * norm;
        a2 = (1.0 - k / q + k * k) * norm;
    }

    void setHighPass (double fc, double q, double rate) noexcept
    {
        const auto k = std::tan (juce::MathConstants<double>::pi * fc / rate);
        const auto norm = 1.0 / (1.0 + k / q + k * k);
        b0 = norm;
        b1 = -2.0 * b0;
        b2 = b0;
        a1 = 2.0 * (k * k - 1.0) * norm;
        a2 = (1.0 - k / q + k * k) * norm;
    }

    void reset() noexcept     { z1 = z2 = 0.0; }

    double process (double x) noexcept
    {
        const auto y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

private:
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0, z1 = 0.0, z2 = 0.0;
};

// BBD 的带限：延迟前后各一个四阶巴特沃斯，相当于抗混叠滤波和重建滤波。
class BandLimit
{
public:
    void prepare (double rate) noexcept
    {
        const auto fc = std::min (bandLimitHz, rate * 0.45);
        a.setLowPass (fc, 0.54119610, rate);
        b.setLowPass (fc, 1.30656296, rate);
    }

    void reset() noexcept     { a.reset(); b.reset(); }

    double process (double x) noexcept  { return b.process (a.process (x)); }

private:
    Biquad a, b;
};

// 两级一阶串联。单级在目标变化的那个采样有斜率折点，串联后增益曲线的斜率也是连续的，切换时不产生高频。
class Smoother
{
public:
    void snap (float value) noexcept      { first = current = value; }

    float next (float target, float coef) noexcept
    {
        first += (target - first) * coef;
        current += (first - current) * coef;
        if (target <= 0.0f && current < 1.0e-6f)
            snap (0.0f);
        return current;
    }

    float get() const noexcept            { return current; }

private:
    float first = 0.0f, current = 0.0f;
};

class Engine
{
public:
    struct Targets
    {
        std::array<bool, 4> modes {};
        float mix = 0.4f;       // 0..1
        bool monoCompat = false;
        bool noise = false;
    };

    void prepare (double newRate)
    {
        rate = newRate;
        const auto wanted = (int) std::ceil (maxDelayMs * 0.001 * rate) + 8;
        size = juce::nextPowerOfTwo (wanted);
        mask = size - 1;
        for (auto& ring : rings)
            ring.assign ((size_t) size, 0.0f);

        smoothCoef = 1.0f - std::exp (-1.0f / (0.01f * (float) rate));
        envAttack = 1.0f - std::exp (-1.0f / (0.005f * (float) rate));
        envRelease = 1.0f - std::exp (-1.0f / (0.08f * (float) rate));
        noiseScale = std::pow (10.0f, noiseRelativeDb * 0.05f) * std::sqrt (3.0f);
        for (size_t ch = 0; ch < 2; ++ch)
        {
            inputHighPass[ch].setHighPass (highPassHz, 0.70710678, rate);
            inputBand[ch].prepare (rate);
            outputBand[ch].prepare (rate);
        }
        for (size_t i = 0; i < modeSpecs.size(); ++i)
            phaseStep[i] = modeSpecs[i].rateHz / (float) rate;

        reset();
    }

    void reset() noexcept
    {
        for (auto& ring : rings)
            std::fill (ring.begin(), ring.end(), 0.0f);
        for (size_t ch = 0; ch < 2; ++ch)
        {
            inputHighPass[ch].reset();
            inputBand[ch].reset();
            outputBand[ch].reset();
            envelope[ch] = 0.0f;
        }
        for (size_t i = 0; i < modeSpecs.size(); ++i)
            phases[i] = modeSpecs[i].startPhase;
        writePos = 0;
        noiseState = 0x9E3779B9u;
        hasState = false;
    }

    // 输入在 left/right 里，就地写回输出。单声道时 right 为 nullptr。
    void process (float* left, float* right, int numSamples, const Targets& targets) noexcept
    {
        if (! hasState)
        {
            // 刚加载时直接落到目标值，不要有一段渐入。
            for (size_t i = 0; i < modeOn.size(); ++i)
                modeOn[i].snap (targets.modes[i] ? 1.0f : 0.0f);
            mixNow.snap (targets.mix);
            compatNow.snap (targets.monoCompat ? 1.0f : 0.0f);
            noiseNow.snap (targets.noise ? 1.0f : 0.0f);
            hasState = true;
        }

        const auto compatTarget = targets.monoCompat ? 1.0f : 0.0f;
        const auto noiseTarget = targets.noise ? 1.0f : 0.0f;
        const auto msToSamples = 0.001f * (float) rate;

        for (int n = 0; n < numSamples; ++n)
        {
            float energy = 0.0f, quiet = 1.0f, inactive = 1.0f;
            for (size_t i = 0; i < modeOn.size(); ++i)
            {
                const auto on = modeOn[i].next (targets.modes[i] ? 1.0f : 0.0f, smoothCoef);
                const auto scaled = on * modeSpecs[i].weight;
                energy += scaled * scaled;
                quiet *= 1.0f - scaled;
                inactive *= 1.0f - on;
            }
            const auto mixAmount = mixNow.next (targets.mix, smoothCoef);
            const auto compatAmount = compatNow.next (compatTarget, smoothCoef);
            noiseNow.next (noiseTarget, smoothCoef);

            // 用乘积而不是 min() 把强度限制在 0..1 内，切换途中没有折点。单个模式的强度就是它的 weight，叠加时再往 1 靠。
            const auto weight = 1.0f - quiet;
            const auto active = 1.0f - inactive;
            const auto voiceNorm = 1.0f / std::max (std::sqrt (energy), 1.0e-6f);
            // 干湿 100% 总是全湿；中间值随模式强度变化；全部抬起时等于直通。
            const auto angle = juce::MathConstants<float>::halfPi * mixAmount
                               * (weight + (1.0f - weight) * mixAmount) * active;
            const auto dryGain = std::cos (angle);
            const auto wetGain = std::sin (angle);

            const auto inL = left[n];
            const auto inR = right != nullptr ? right[n] : inL;

            writeInput (0, inL);
            writeInput (1, inR);

            float wetL = 0.0f, wetR = 0.0f;
            for (size_t i = 0; i < modeSpecs.size(); ++i)
            {
                auto& phase = phases[i];
                phase += phaseStep[i];
                if (phase >= 1.0f)
                    phase -= 1.0f;

                const auto on = modeOn[i].get();
                if (on <= 0.0f)
                    continue;

                const auto& spec = modeSpecs[i];
                const auto swing = spec.depthMs * (1.0f + roundedTriangle (phase));
                const auto longLine = (spec.baseMs + lineGapMs + swing) * msToSamples;
                const auto shortLine = (spec.baseMs - lineGapMs - swing) * msToSamples;
                const auto voice = on * spec.weight * voiceNorm;

                wetL += voice * read (0, spec.swapSides ? shortLine : longLine);
                wetR += voice * read (1, spec.swapSides ? longLine : shortLine);
            }

            wetL = (float) outputBand[0].process (wetL);
            wetR = (float) outputBand[1].process (wetR);

            if (right != nullptr)
            {
                // 中置和侧向按等功率分配。兼容模式下中置归零，L+R 只剩干声，折叠后完全没有梳状。
                const auto midAmount = wetMidAmount * (1.0f - compatAmount);
                const auto sideAmount = std::sqrt (2.0f - midAmount * midAmount);
                const auto mid = 0.5f * (wetL + wetR) * midAmount;
                const auto side = 0.5f * (wetL - wetR) * sideAmount;
                left[n] = dryGain * inL + wetGain * (mid + side);
                right[n] = dryGain * inR + wetGain * (mid - side);
            }
            else
            {
                left[n] = dryGain * inL + wetGain * (wetL + wetR) * juce::MathConstants<float>::sqrt2 * 0.5f;
            }

            writePos = (writePos + 1) & mask;
        }
    }

private:
    void writeInput (size_t ch, float in) noexcept
    {
        // 噪声跟随信号包络，像压扩器一样：有声音时才有，静音时输出也是真静音。
        const auto level = std::abs (in);
        envelope[ch] += (level - envelope[ch]) * (level > envelope[ch] ? envAttack : envRelease);
        if (envelope[ch] < 1.0e-9f)
            envelope[ch] = 0.0f;

        const auto noise = nextNoise() * envelope[ch] * noiseScale * noiseNow.get();
        const auto filtered = inputBand[ch].process (inputHighPass[ch].process ((double) in + (double) noise));
        rings[ch][(size_t) writePos] = (float) filtered;
    }

    float read (size_t ch, float delaySamples) const noexcept
    {
        const auto position = (float) writePos - delaySamples;
        const auto whole = (int) std::floor (position);
        const auto frac = position - (float) whole;
        const auto* ring = rings[ch].data();

        // 4 点 Hermite，读头每个采样都在动，线性插值的高频损失会随调制起伏。
        const auto xm1 = ring[(size_t) ((whole - 1) & mask)];
        const auto x0  = ring[(size_t) (whole & mask)];
        const auto x1  = ring[(size_t) ((whole + 1) & mask)];
        const auto x2  = ring[(size_t) ((whole + 2) & mask)];
        const auto c = 0.5f * (x1 - xm1);
        const auto v = x0 - x1;
        const auto w = c + v;
        const auto a = w + v + 0.5f * (x2 - x0);
        const auto b = w + a;
        return ((a * frac - b) * frac + c) * frac + x0;
    }

    float nextNoise() noexcept
    {
        noiseState ^= noiseState << 13;
        noiseState ^= noiseState >> 17;
        noiseState ^= noiseState << 5;
        return (float) (noiseState >> 1) * (2.0f / 2147483648.0f) - 1.0f;
    }

    double rate = 48000.0;
    int size = 0, mask = 0, writePos = 0;
    std::array<std::vector<float>, 2> rings;
    std::array<Biquad, 2> inputHighPass;
    std::array<BandLimit, 2> inputBand, outputBand;
    std::array<float, 2> envelope {};
    std::array<float, 4> phases {}, phaseStep {};
    std::array<Smoother, 4> modeOn;
    Smoother mixNow, compatNow, noiseNow;
    float smoothCoef = 1.0f, envAttack = 1.0f, envRelease = 1.0f, noiseScale = 0.0f;
    juce::uint32 noiseState = 0x9E3779B9u;
    bool hasState = false;
};

} // namespace sirens
