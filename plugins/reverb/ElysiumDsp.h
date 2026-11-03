#pragma once

// Elysium 的全部音频处理。只依赖标准库，方便 tests/ 下的自检程序直接调用。
// 四种算法的结构、延迟长度和系数都是这里自行设计的，来源见 NOTES.md。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace elysium
{

constexpr float pi = 3.14159265358979f;
constexpr int controlInterval = 32;     // 系数更新周期，也是内部处理块的上限。
constexpr int numAlgorithms = 4;
constexpr int numCharacters = 3;

enum Algorithm { hall, plate, room, chamber };
enum Character { vintage12, vintage24, modern };

struct Params
{
    float timeSeconds = 2.2f;
    float predelayMs = 0.0f;
    float mix = 0.25f;              // 0–1
    float damping = 0.5f;           // 0–1
    float lowMultiplier = 1.0f;     // 低频段衰减时间相对于混响时间的倍数
    float midMultiplier = 1.0f;     // 中频段衰减时间相对于混响时间的倍数
    float crossoverHz = 400.0f;
    float modDepth = 0.3f;          // 0–1
    int algorithm = hall;
    int character = vintage24;
};

// 从参数估算尾音衰减到 -90 dB 所需的时间，宿主据此决定停止播放后还要处理多久。
inline double tailSeconds (const Params& p) noexcept
{
    if (p.mix <= 0.0f)
        return 0.0;

    const auto longest = p.timeSeconds * std::max (p.lowMultiplier, p.midMultiplier);
    return 0.001 * (double) p.predelayMs + 1.5 * (double) longest + 0.1;
}

//==============================================================================
inline int nextPrime (int n) noexcept
{
    n = std::max (n, 3) | 1;
    for (;; n += 2)
    {
        auto isPrime = true;
        for (int d = 3; d * d <= n && isPrime; d += 2)
            isPrime = n % d != 0;
        if (isPrime)
            return n;
    }
}

inline int primeSamples (float ms, double sampleRate) noexcept
{
    return nextPrime ((int) std::lround (0.001 * (double) ms * sampleRate));
}

// 按控制周期更新目标、逐采样走线性斜坡的系数。回路里的增益和读取位置都不能阶跃。
class Ramped
{
public:
    void setTarget (float target, bool instant) noexcept
    {
        const auto delta = target - value;
        if (instant || std::abs (delta) < 1.0e-5f)
        {
            value = target;
            step = 0.0f;
        }
        else
        {
            step = delta / (float) controlInterval;
        }
    }

    float next() noexcept                       { value += step; return value; }

private:
    float value = 1.0f, step = 0.0f;
};

class DelayLine
{
public:
    void prepare (int maxDelay)
    {
        auto size = 16;
        while (size < maxDelay + 8)
            size <<= 1;
        buffer.assign ((size_t) size, 0.0f);
        mask = size - 1;
        pos = 0;
    }

    void clear() noexcept                       { std::fill (buffer.begin(), buffer.end(), 0.0f); pos = 0; }

    // 在同一采样里先读后写：read (d) 取到的是 d 个采样之前写入的值，d 至少为 1。
    float read (int d) const noexcept           { return buffer[(size_t) ((pos - d) & mask)]; }
    void write (float x) noexcept               { buffer[(size_t) pos] = x; pos = (pos + 1) & mask; }

    // 三阶拉格朗日插值，读点落在中间区间内时幅度响应不超过 1，放进反馈回路不会增加能量。d 至少为 2。
    float readCubic (float d) const noexcept
    {
        const auto i = (int) d;
        const auto f = d - (float) i;
        const auto a = (f + 1.0f) * f;
        const auto b = (f - 1.0f) * (f - 2.0f);
        return read (i - 1) * (-f * b / 6.0f)
             + read (i)     * ((f + 1.0f) * b * 0.5f)
             + read (i + 1) * (-a * (f - 2.0f) * 0.5f)
             + read (i + 2) * (a * (f - 1.0f) / 6.0f);
    }

private:
    std::vector<float> buffer;
    int mask = 0, pos = 0;
};

class Allpass
{
public:
    void prepare (float ms, float coefficient, double sampleRate)
    {
        delay = primeSamples (ms, sampleRate);
        g = coefficient;
        line.prepare (delay);
    }

    void clear() noexcept                       { line.clear(); }
    int length() const noexcept                 { return delay; }

    // 放进反馈回路的全通，内部延迟也要跟着衰减，否则设定很短的时间时尾音由全通自己的振铃决定。
    void setLoss (float slopeDbPerSample, bool instant) noexcept
    {
        loss.setTarget (std::pow (10.0f, -slopeDbPerSample * (float) delay * 0.05f), instant);
    }

    float process (float x) noexcept
    {
        const auto z = loss.next() * line.read (delay);
        const auto v = x + g * z;
        line.write (v);
        return z - g * v;
    }

private:
    DelayLine line;
    int delay = 3;
    float g = 0.5f;
    Ramped loss;
};

// 周期带随机抖动的慢速调制。相位连续，所以换速率不会产生跳变。
class Lfo
{
public:
    void prepare (float hz, double sampleRate, uint32_t seed) noexcept
    {
        baseIncrement = 2.0f * hz / (float) sampleRate;
        increment = baseIncrement;
        phase = -1.0f + (float) (seed % 1000u) * 0.002f;
        state = seed * 2654435761u + 1u;
    }

    float next() noexcept
    {
        phase += increment;
        if (phase >= 1.0f)
        {
            phase -= 2.0f;
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            increment = baseIncrement * (0.75f + 0.5f * (float) (state >> 16) / 65535.0f);
        }
        return 4.0f * phase * (1.0f - std::abs (phase));
    }

private:
    float phase = 0.0f, increment = 0.0f, baseIncrement = 0.0f;
    uint32_t state = 1;
};

//==============================================================================
// 三段衰减的目标，单位是 dB/秒。回路里每段延迟按自身时长换算成增益，所有极点的半径才一致。
struct DecaySlopes
{
    float low = 30.0f, mid = 30.0f, high = 60.0f;
    float lowCoef = 0.05f, highCoef = 0.3f;
    double sampleRate = 48000.0;

    // 全通内部只能承担与频率无关的那部分衰减，取低、中频里较慢的一个，剩下的交给 DecayFilter。
    float flat() const noexcept                 { return std::min (low, mid); }
    float flatPerSample() const noexcept        { return flat() / (float) sampleRate; }
};

class DecayFilter
{
public:
    void clear() noexcept                       { low = 0.0f; high = 0.0f; }

    // total 是这个滤波器负责的整段回路时长，allpass 是其中已由全通内部承担的部分。
    void setDuration (double total, double allpass) noexcept
    {
        totalSeconds = (float) total;
        allpassSeconds = (float) allpass;
    }

    void update (const DecaySlopes& s, bool instant) noexcept
    {
        const auto insideAllpass = s.flat() * allpassSeconds;
        lowGain.setTarget (std::pow (10.0f, -(s.low * totalSeconds - insideAllpass) * 0.05f), instant);
        midGain.setTarget (std::pow (10.0f, -(s.mid * totalSeconds - insideAllpass) * 0.05f), instant);
        highGain.setTarget (std::pow (10.0f, -(s.high - s.mid) * totalSeconds * 0.05f), instant);
    }

    float process (float x, const DecaySlopes& s) noexcept
    {
        low += s.lowCoef * (x - low);
        const auto y = lowGain.next() * low + midGain.next() * (x - low);
        high += s.highCoef * (y - high);
        return high + highGain.next() * (y - high);
    }

private:
    float low = 0.0f, high = 0.0f, totalSeconds = 0.05f, allpassSeconds = 0.0f;
    Ramped lowGain, midGain, highGain;
};

//==============================================================================
class AlgorithmEngine
{
public:
    virtual ~AlgorithmEngine() = default;

    virtual void prepare (double sampleRate) = 0;
    virtual void clear() noexcept = 0;
    virtual void setDecay (const DecaySlopes&, bool instant) noexcept = 0;
    virtual void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept = 0;

    struct Voicing
    {
        float dampingHz;        // 高频阻尼开始起作用的频率
        float dampingBias;      // 阻尼量的算法差异，大于 1 更暗
        float segmentSeconds;   // 回路里一段衰减的典型时长
    };

    virtual Voicing voicing() const noexcept = 0;

    void setModDepth (float depth, bool instant) noexcept   { modSamples.setTarget (depth * modRangeSamples, instant); }

protected:
    // 调制后的读取延迟。偏移量恒为正，所以插值点不会碰到写指针。
    static float modulated (int delay, Lfo& lfo, float depth) noexcept  { return (float) delay + depth * (lfo.next() + 1.0f) + 2.0f; }
    int maxModulatedDelay (int delay) const noexcept                    { return delay + (int) (2.0f * modRangeSamples) + 8; }

    Ramped modSamples;
    float modRangeSamples = 0.0f;
};

//==============================================================================
// 大厅：输入扩散后进 8 路反馈延迟网络，哈达玛矩阵混合，每路带一节全通，读取点全部调制。
class HallEngine final : public AlgorithmEngine
{
public:
    void prepare (double sampleRate) override
    {
        static constexpr std::array<float, numInput> diffL { 1.3f, 3.1f, 4.9f, 7.3f, 11.9f }, diffR { 1.7f, 2.9f, 5.3f, 6.7f, 10.3f };
        static constexpr std::array<float, numInput> diffG { 0.72f, 0.68f, 0.60f, 0.55f, 0.5f };
        static constexpr std::array<float, size> lineMs { 31.1f, 37.3f, 43.7f, 49.1f, 57.3f, 63.7f, 71.9f, 83.3f };
        static constexpr std::array<float, size> apMs { 5.9f, 8.3f, 11.3f, 7.1f, 13.7f, 9.7f, 6.7f, 12.1f };
        static constexpr std::array<float, size> lfoHz { 0.31f, 0.43f, 0.57f, 0.67f, 0.79f, 0.89f, 1.03f, 1.13f };
        static constexpr std::array<float, size> tapMs { 11.3f, 23.9f, 17.9f, 31.7f, 41.3f, 29.3f, 53.9f, 37.1f };

        modRangeSamples = (float) (0.00135 * sampleRate);

        for (size_t i = 0; i < numInput; ++i)
        {
            inputL[i].prepare (diffL[i], diffG[i], sampleRate);
            inputR[i].prepare (diffR[i], diffG[i], sampleRate);
        }

        for (size_t i = 0; i < size; ++i)
        {
            delay[i] = primeSamples (lineMs[i], sampleRate);
            lines[i].prepare (maxModulatedDelay (delay[i]));
            loopAp[i].prepare (apMs[i], (i & 1) != 0 ? -0.5f : 0.5f, sampleRate);
            decay[i].setDuration ((double) (delay[i] + loopAp[i].length()) / sampleRate, (double) loopAp[i].length() / sampleRate);
            lfo[i].prepare (lfoHz[i], sampleRate, (uint32_t) i * 137u + 11u);
            tap[i] = primeSamples (tapMs[i], sampleRate);
        }
    }

    void clear() noexcept override
    {
        for (auto& a : inputL) a.clear();
        for (auto& a : inputR) a.clear();
        for (auto& a : loopAp) a.clear();
        for (auto& l : lines)  l.clear();
        for (auto& d : decay)  d.clear();
    }

    void setDecay (const DecaySlopes& s, bool instant) noexcept override
    {
        slopes = s;
        for (auto& d : decay)
            d.update (slopes, instant);
        for (auto& a : loopAp)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : inputL)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : inputR)
            a.setLoss (slopes.flatPerSample(), instant);
    }

    Voicing voicing() const noexcept override   { return { 3000.0f, 1.0f, 0.065f }; }

    void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept override
    {
        for (int s = 0; s < n; ++s)
        {
            const auto depth = modSamples.next();
            auto dl = inL[s], dr = inR[s];
            for (auto& a : inputL) dl = a.process (dl);
            for (auto& a : inputR) dr = a.process (dr);

            std::array<float, size> y;
            for (size_t i = 0; i < size; ++i)
                y[i] = decay[i].process (lines[i].readCubic (modulated (delay[i], lfo[i], depth)), slopes);

            const auto lateL = 0.3f * (y[0] - y[1] + y[2] + y[3] - y[4] + y[5] - y[6] - y[7]);
            const auto lateR = 0.3f * (y[0] + y[1] - y[2] + y[3] + y[4] + y[5] - y[6] + y[7]);
            const auto tapsL = lines[0].read (tap[0]) - lines[2].read (tap[2]) + lines[4].read (tap[4]) - lines[6].read (tap[6]);
            const auto tapsR = lines[1].read (tap[1]) - lines[3].read (tap[3]) + lines[5].read (tap[5]) - lines[7].read (tap[7]);

            outL[s] = 0.35f * dl + lateL + 0.2f * tapsL;
            outR[s] = 0.35f * dr + lateR + 0.2f * tapsR;

            // 八阶哈达玛矩阵的快速算法：三层蝶形再归一化。
            for (size_t stride = 1; stride < size; stride <<= 1)
                for (size_t i = 0; i < size; i += stride << 1)
                    for (size_t j = i; j < i + stride; ++j)
                    {
                        const auto a = y[j], b = y[j + stride];
                        y[j] = a + b;
                        y[j + stride] = a - b;
                    }

            const auto injectL = 0.5f * dl, injectR = 0.5f * dr;
            y[0] = 0.35355339f * y[0] + injectL;
            y[1] = 0.35355339f * y[1] + injectR;
            y[2] = 0.35355339f * y[2] - injectL;
            y[3] = 0.35355339f * y[3] - injectR;
            y[4] = 0.35355339f * y[4] + injectL;
            y[5] = 0.35355339f * y[5] + injectR;
            y[6] = 0.35355339f * y[6] - injectL;
            y[7] = 0.35355339f * y[7] - injectR;

            for (size_t i = 0; i < size; ++i)
                lines[i].write (loopAp[i].process (y[i]));
        }
    }

private:
    static constexpr size_t size = 8, numInput = 5;

    std::array<Allpass, numInput> inputL, inputR;
    std::array<DelayLine, size> lines;
    std::array<Allpass, size> loopAp;
    std::array<DecayFilter, size> decay;
    std::array<Lfo, size> lfo;
    std::array<int, size> delay {}, tap {};
    DecaySlopes slopes;
};

//==============================================================================
// 板式：一条环，四节首尾相接。每节是调制全通、延迟、衰减、全通、延迟，声音从环的多处抽头取出。
class PlateEngine final : public AlgorithmEngine
{
public:
    void prepare (double sampleRate) override
    {
        static constexpr std::array<float, numInput> diffL { 1.1f, 1.9f, 3.7f, 5.3f, 8.9f, 13.1f }, diffR { 1.3f, 2.3f, 3.1f, 5.9f, 7.9f, 14.3f };
        static constexpr std::array<float, numInput> diffG { 0.74f, 0.70f, 0.62f, 0.58f, 0.55f, 0.5f };
        static constexpr std::array<float, branches> modMs { 17.3f, 23.9f, 19.7f, 27.1f };
        static constexpr std::array<float, branches> aMs { 61.3f, 47.9f, 71.9f, 53.1f };
        static constexpr std::array<float, branches> apMs { 13.1f, 19.3f, 11.3f, 16.7f };
        static constexpr std::array<float, branches> bMs { 31.7f, 43.1f, 37.9f, 29.3f };
        static constexpr std::array<float, branches> lfoHz { 0.53f, 0.71f, 0.83f, 0.97f };
        static constexpr std::array<float, branches> tapAMsL { 3.7f, 23.1f, 41.3f, 11.9f }, tapBMsL { 19.9f, 11.7f, 7.1f, 25.3f };
        static constexpr std::array<float, branches> tapAMsR { 19.3f, 7.7f, 4.9f, 27.3f }, tapBMsR { 5.9f, 31.1f, 17.1f, 13.7f };

        modRangeSamples = (float) (0.0005 * sampleRate);

        for (size_t i = 0; i < numInput; ++i)
        {
            inputL[i].prepare (diffL[i], diffG[i], sampleRate);
            inputR[i].prepare (diffR[i], diffG[i], sampleRate);
        }

        for (size_t i = 0; i < branches; ++i)
        {
            modDelay[i] = primeSamples (modMs[i], sampleRate);
            modLine[i].prepare (maxModulatedDelay (modDelay[i]));
            aDelay[i] = primeSamples (aMs[i], sampleRate);
            aLine[i].prepare (aDelay[i]);
            ap[i].prepare (apMs[i], (i & 1) != 0 ? -0.5f : 0.5f, sampleRate);
            bDelay[i] = primeSamples (bMs[i], sampleRate);
            bLine[i].prepare (bDelay[i]);
            lfo[i].prepare (lfoHz[i], sampleRate, (uint32_t) i * 211u + 29u);
            decay[i].setDuration ((double) (modDelay[i] + aDelay[i] + ap[i].length() + bDelay[i]) / sampleRate,
                                  (double) (modDelay[i] + ap[i].length()) / sampleRate);
            tapAL[i] = primeSamples (tapAMsL[i], sampleRate);
            tapBL[i] = primeSamples (tapBMsL[i], sampleRate);
            tapAR[i] = primeSamples (tapAMsR[i], sampleRate);
            tapBR[i] = primeSamples (tapBMsR[i], sampleRate);
        }
    }

    void clear() noexcept override
    {
        for (auto& a : inputL) a.clear();
        for (auto& a : inputR) a.clear();
        for (auto& l : modLine) l.clear();
        for (auto& l : aLine) l.clear();
        for (auto& a : ap) a.clear();
        for (auto& l : bLine) l.clear();
        for (auto& d : decay) d.clear();
        carry = 0.0f;
    }

    void setDecay (const DecaySlopes& s, bool instant) noexcept override
    {
        slopes = s;
        for (auto& d : decay)
            d.update (slopes, instant);
        for (auto& a : ap)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : inputL)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : inputR)
            a.setLoss (slopes.flatPerSample(), instant);
        for (size_t i = 0; i < branches; ++i)
            modLoss[i].setTarget (std::pow (10.0f, -slopes.flatPerSample() * (float) modDelay[i] * 0.05f), instant);
    }

    Voicing voicing() const noexcept override   { return { 4500.0f, 0.75f, 0.13f }; }

    void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept override
    {
        for (int s = 0; s < n; ++s)
        {
            const auto depth = modSamples.next();
            auto dl = inL[s], dr = inR[s];
            for (auto& a : inputL) dl = a.process (dl);
            for (auto& a : inputR) dr = a.process (dr);

            // 每条延迟只取一个抽头，同一条线上取两次会形成固定滞后的梳状。
            outL[s] = 0.3f * dl
                    + 0.55f * (aLine[0].read (tapAL[0]) - bLine[0].read (tapBL[0]) + aLine[1].read (tapAL[1]) - bLine[1].read (tapBL[1])
                               + aLine[2].read (tapAL[2]) - bLine[2].read (tapBL[2]) + aLine[3].read (tapAL[3]) - bLine[3].read (tapBL[3]));
            outR[s] = 0.3f * dr
                    + 0.55f * (aLine[2].read (tapAR[2]) - bLine[2].read (tapBR[2]) + aLine[3].read (tapAR[3]) - bLine[3].read (tapBR[3])
                               - aLine[0].read (tapAR[0]) + bLine[0].read (tapBR[0]) - aLine[1].read (tapAR[1]) + bLine[1].read (tapBR[1]));

            auto x = carry;
            for (size_t i = 0; i < branches; ++i)
            {
                if (i == 0) x += 0.6f * dl;
                if (i == 2) x += 0.6f * dr;

                // 调制全通的系数正负交替，各节群延迟的低谷错开，模态分布更均匀。
                const auto g = (i & 1) != 0 ? 0.68f : -0.68f;
                const auto z = modLoss[i].next() * modLine[i].readCubic (modulated (modDelay[i], lfo[i], depth));
                const auto v = x + g * z;
                modLine[i].write (v);
                const auto a = z - g * v;

                const auto fromA = decay[i].process (aLine[i].read (aDelay[i]), slopes);
                aLine[i].write (a);

                const auto fromAp = ap[i].process (fromA);
                x = bLine[i].read (bDelay[i]);
                bLine[i].write (fromAp);
            }
            carry = x;
        }
    }

private:
    static constexpr size_t branches = 4, numInput = 6;

    std::array<Allpass, numInput> inputL, inputR;
    std::array<DelayLine, branches> modLine, aLine, bLine;
    std::array<Allpass, branches> ap;
    std::array<DecayFilter, branches> decay;
    std::array<Lfo, branches> lfo;
    std::array<int, branches> modDelay {}, aDelay {}, bDelay {};
    std::array<int, branches> tapAL {}, tapBL {}, tapAR {}, tapBR {};
    std::array<Ramped, branches> modLoss;
    DecaySlopes slopes;
    float carry = 0.0f;
};

//==============================================================================
// 房间：多抽头早期反射加 6 路短延迟网络。矩阵用六阶会议矩阵，对角线为 0，没有自反馈，短延迟也不容易成梳状。
class RoomEngine final : public AlgorithmEngine
{
public:
    void prepare (double sampleRate) override
    {
        static constexpr std::array<float, numEarly> earlyMsL { 4.3f, 8.9f, 12.7f, 17.1f, 23.9f, 31.3f, 37.9f };
        static constexpr std::array<float, numEarly> earlyMsR { 5.1f, 9.7f, 14.3f, 19.1f, 26.3f, 33.7f, 41.1f };
        static constexpr std::array<float, size> lineMs { 19.3f, 23.9f, 28.1f, 31.7f, 35.9f, 41.3f };
        static constexpr std::array<float, size> apMs { 3.1f, 4.3f, 5.3f, 2.9f, 6.1f, 3.7f };
        static constexpr std::array<float, size> ap2Ms { 11.3f, 7.9f, 13.7f, 9.1f, 15.1f, 8.3f };
        static constexpr std::array<float, size> lfoHz { 0.47f, 0.59f, 0.73f, 0.83f, 0.97f, 1.09f };

        modRangeSamples = (float) (0.0004 * sampleRate);

        for (size_t i = 0; i < numEarly; ++i)
        {
            earlyL[i] = primeSamples (earlyMsL[i], sampleRate);
            earlyR[i] = primeSamples (earlyMsR[i], sampleRate);
        }
        earlyLineL.prepare (earlyL.back());
        earlyLineR.prepare (earlyR.back());

        feedL[0].prepare (1.3f, 0.65f, sampleRate);
        feedL[1].prepare (2.3f, 0.6f, sampleRate);
        feedL[2].prepare (3.7f, 0.55f, sampleRate);
        feedR[0].prepare (1.1f, 0.65f, sampleRate);
        feedR[1].prepare (2.9f, 0.6f, sampleRate);
        feedR[2].prepare (3.3f, 0.55f, sampleRate);

        for (size_t i = 0; i < size; ++i)
        {
            delay[i] = primeSamples (lineMs[i], sampleRate);
            lines[i].prepare (maxModulatedDelay (delay[i]));
            loopAp[i].prepare (apMs[i], (i & 1) != 0 ? 0.45f : -0.45f, sampleRate);
            loopAp2[i].prepare (ap2Ms[i], (i & 1) != 0 ? -0.5f : 0.5f, sampleRate);
            const auto allpass = (double) (loopAp[i].length() + loopAp2[i].length()) / sampleRate;
            decay[i].setDuration ((double) delay[i] / sampleRate + allpass, allpass);
            lfo[i].prepare (lfoHz[i], sampleRate, (uint32_t) i * 173u + 41u);
        }
    }

    void clear() noexcept override
    {
        earlyLineL.clear();
        earlyLineR.clear();
        for (auto& a : feedL) a.clear();
        for (auto& a : feedR) a.clear();
        for (auto& a : loopAp) a.clear();
        for (auto& a : loopAp2) a.clear();
        for (auto& l : lines)  l.clear();
        for (auto& d : decay)  d.clear();
    }

    void setDecay (const DecaySlopes& s, bool instant) noexcept override
    {
        slopes = s;
        for (auto& d : decay)
            d.update (slopes, instant);
        for (auto& a : loopAp)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : loopAp2)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : feedL)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : feedR)
            a.setLoss (slopes.flatPerSample(), instant);
    }

    Voicing voicing() const noexcept override   { return { 2500.0f, 1.2f, 0.045f }; }

    void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept override
    {
        static constexpr std::array<float, numEarly> earlyGain { 0.80f, 0.68f, -0.58f, 0.50f, -0.42f, 0.34f, -0.28f };

        for (int s = 0; s < n; ++s)
        {
            const auto depth = modSamples.next();
            auto eL = 0.0f, eR = 0.0f;
            for (size_t i = 0; i < numEarly; ++i)
            {
                eL += earlyGain[i] * earlyLineL.read (earlyL[i]);
                eR += earlyGain[i] * earlyLineR.read (earlyR[i]);
            }
            earlyLineL.write (inL[s]);
            earlyLineR.write (inR[s]);

            // 早期反射只进输出。它的梳状响应如果也进回路，整段尾音都会带上这个染色。
            auto dl = inL[s], dr = inR[s];
            for (auto& a : feedL) dl = a.process (dl);
            for (auto& a : feedR) dr = a.process (dr);

            std::array<float, size> y;
            for (size_t i = 0; i < size; ++i)
                y[i] = decay[i].process (lines[i].readCubic (modulated (delay[i], lfo[i], depth)), slopes);

            outL[s] = 0.45f * eL + 0.42f * (y[0] - y[1] + y[2] + y[3] - y[4] - y[5]);
            outR[s] = 0.45f * eR + 0.42f * (y[0] + y[1] - y[2] + y[3] + y[4] - y[5]);

            const std::array<float, size> mixed {
                y[1] + y[2] + y[3] + y[4] + y[5],
                y[0] + y[2] - y[3] - y[4] + y[5],
                y[0] + y[1] + y[3] - y[4] - y[5],
                y[0] - y[1] + y[2] + y[4] - y[5],
                y[0] - y[1] - y[2] + y[3] + y[5],
                y[0] + y[1] - y[2] - y[3] + y[4]
            };

            static constexpr float scale = 0.4472136f;
            lines[0].write (loopAp2[0].process (loopAp[0].process (scale * mixed[0] + dl)));
            lines[1].write (loopAp2[1].process (loopAp[1].process (scale * mixed[1] + dr)));
            lines[2].write (loopAp2[2].process (loopAp[2].process (scale * mixed[2] - dl)));
            lines[3].write (loopAp2[3].process (loopAp[3].process (scale * mixed[3] - dr)));
            lines[4].write (loopAp2[4].process (loopAp[4].process (scale * mixed[4] + dl)));
            lines[5].write (loopAp2[5].process (loopAp[5].process (scale * mixed[5] + dr)));
        }
    }

private:
    static constexpr size_t size = 6, numEarly = 7;

    DelayLine earlyLineL, earlyLineR;
    std::array<int, numEarly> earlyL {}, earlyR {};
    std::array<Allpass, 3> feedL, feedR;
    std::array<DelayLine, size> lines;
    std::array<Allpass, size> loopAp, loopAp2;
    std::array<DecayFilter, size> decay;
    std::array<Lfo, size> lfo;
    std::array<int, size> delay {};
    DecaySlopes slopes;
};

//==============================================================================
// 腔体：一条环，六节嵌套全通串联。起音比大厅柔和，能量在环内逐节扩散。
class ChamberEngine final : public AlgorithmEngine
{
public:
    void prepare (double sampleRate) override
    {
        static constexpr std::array<float, numInput> diffL { 1.7f, 3.3f, 6.1f, 9.7f }, diffR { 1.9f, 3.7f, 5.3f, 11.3f };
        static constexpr std::array<float, numInput> diffG { 0.7f, 0.64f, 0.58f, 0.5f };
        static constexpr std::array<float, sections> outerMs { 19.7f, 23.3f, 29.9f, 17.9f, 26.1f, 31.3f };
        static constexpr std::array<float, sections> innerMs { 5.3f, 7.9f, 4.7f, 9.1f, 6.1f, 3.7f };
        static constexpr std::array<float, sections> lineMs { 23.1f, 37.3f, 19.1f, 41.9f, 29.7f, 33.1f };
        static constexpr std::array<float, sections> lfoHz { 0.37f, 0.49f, 0.61f, 0.77f, 0.91f, 1.07f };
        static constexpr std::array<float, sections> tapMsL { 2.9f, 11.9f, 7.7f, 17.3f, 5.3f, 16.1f };
        static constexpr std::array<float, sections> tapMsR { 14.3f, 4.7f, 12.7f, 3.1f, 13.3f, 9.1f };

        modRangeSamples = (float) (0.0007 * sampleRate);

        for (size_t i = 0; i < numInput; ++i)
        {
            inputL[i].prepare (diffL[i], diffG[i], sampleRate);
            inputR[i].prepare (diffR[i], diffG[i], sampleRate);
        }

        for (size_t i = 0; i < sections; ++i)
        {
            outerDelay[i] = primeSamples (outerMs[i], sampleRate);
            outerLine[i].prepare (maxModulatedDelay (outerDelay[i]));
            innerDelay[i] = primeSamples (innerMs[i], sampleRate);
            innerLine[i].prepare (innerDelay[i]);
            delay[i] = primeSamples (lineMs[i], sampleRate);
            line[i].prepare (delay[i]);
            lfo[i].prepare (lfoHz[i], sampleRate, (uint32_t) i * 97u + 53u);
            decay[i].setDuration ((double) (outerDelay[i] + innerDelay[i] + delay[i]) / sampleRate,
                                  (double) (outerDelay[i] + innerDelay[i]) / sampleRate);
            tapL[i] = primeSamples (tapMsL[i], sampleRate);
            tapR[i] = primeSamples (tapMsR[i], sampleRate);
        }
    }

    void clear() noexcept override
    {
        for (auto& a : inputL) a.clear();
        for (auto& a : inputR) a.clear();
        for (auto& l : outerLine) l.clear();
        for (auto& l : innerLine) l.clear();
        for (auto& l : line) l.clear();
        for (auto& d : decay) d.clear();
        carry = 0.0f;
    }

    void setDecay (const DecaySlopes& s, bool instant) noexcept override
    {
        slopes = s;
        for (auto& d : decay)
            d.update (slopes, instant);
        for (size_t i = 0; i < sections; ++i)
        {
            outerLoss[i].setTarget (std::pow (10.0f, -slopes.flatPerSample() * (float) outerDelay[i] * 0.05f), instant);
            innerLoss[i].setTarget (std::pow (10.0f, -slopes.flatPerSample() * (float) innerDelay[i] * 0.05f), instant);
        }
        for (auto& a : inputL)
            a.setLoss (slopes.flatPerSample(), instant);
        for (auto& a : inputR)
            a.setLoss (slopes.flatPerSample(), instant);
    }

    Voicing voicing() const noexcept override   { return { 3000.0f, 1.1f, 0.065f }; }

    void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept override
    {
        for (int s = 0; s < n; ++s)
        {
            const auto depth = modSamples.next();
            auto dl = inL[s], dr = inR[s];
            for (auto& a : inputL) dl = a.process (dl);
            for (auto& a : inputR) dr = a.process (dr);

            outL[s] = 0.42f * (line[0].read (tapL[0]) - line[1].read (tapL[1]) + line[2].read (tapL[2])
                               - line[3].read (tapL[3]) + line[4].read (tapL[4]) - line[5].read (tapL[5]));
            outR[s] = 0.42f * (line[3].read (tapR[3]) - line[4].read (tapR[4]) + line[5].read (tapR[5])
                               - line[0].read (tapR[0]) + line[1].read (tapR[1]) - line[2].read (tapR[2]));

            auto x = carry;
            for (size_t i = 0; i < sections; ++i)
            {
                if (i == 0) x += 0.7f * dl;
                if (i == 3) x += 0.7f * dr;

                // 外层全通的延迟路径里套一个内层全通。系数逐节换号，群延迟的低谷不重合。
                const auto outerG = (i & 1) != 0 ? -0.6f : 0.6f;
                const auto innerG = (i & 2) != 0 ? -0.4f : 0.4f;
                const auto zOuter = outerLoss[i].next() * outerLine[i].readCubic (modulated (outerDelay[i], lfo[i], depth));
                const auto vOuter = x + outerG * zOuter;
                const auto zInner = innerLoss[i].next() * innerLine[i].read (innerDelay[i]);
                const auto vInner = vOuter + innerG * zInner;
                innerLine[i].write (vInner);
                outerLine[i].write (zInner - innerG * vInner);
                const auto nested = zOuter - outerG * vOuter;

                x = decay[i].process (line[i].read (delay[i]), slopes);
                line[i].write (nested);
            }
            carry = x;
        }
    }

private:
    static constexpr size_t sections = 6, numInput = 4;

    std::array<Allpass, numInput> inputL, inputR;
    std::array<DelayLine, sections> outerLine, innerLine, line;
    std::array<DecayFilter, sections> decay;
    std::array<Lfo, sections> lfo;
    std::array<int, sections> outerDelay {}, innerDelay {}, delay {}, tapL {}, tapR {};
    std::array<Ramped, sections> outerLoss, innerLoss;
    DecaySlopes slopes;
    float carry = 0.0f;
};

//==============================================================================
struct Biquad
{
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    std::array<float, 2> z1 {}, z2 {};

    void setLowpass (float hz, float q, double sampleRate) noexcept
    {
        const auto w = 2.0 * 3.141592653589793 * std::min ((double) hz, 0.45 * sampleRate) / sampleRate;
        const auto alpha = std::sin (w) / (2.0 * (double) q);
        const auto cosw = std::cos (w);
        const auto a0 = 1.0 + alpha;
        b0 = (float) ((1.0 - cosw) * 0.5 / a0);
        b1 = (float) ((1.0 - cosw) / a0);
        b2 = b0;
        a1 = (float) (-2.0 * cosw / a0);
        a2 = (float) ((1.0 - alpha) / a0);
    }

    float process (float x, size_t ch) noexcept
    {
        const auto y = b0 * x + z1[ch];
        z1[ch] = b1 * x - a1 * y + z2[ch];
        z2[ch] = b2 * x - a2 * y;
        return y;
    }

    void clear() noexcept                       { z1 = {}; z2 = {}; }
};

// 转换器染色。现代档完全不处理；24 位档只有带限和 24 位量化；
// 12 位档再加一对互补的压扩，把量化放在压缩和扩展之间。压扩用同一增益收放，
// 所以平均电平和尾音斜率不变，只有量化噪声跟着信号起伏。信号足够小时量化直接归零，静音不出噪声。
class Converter
{
public:
    void prepare (Character c, double sampleRate) noexcept
    {
        character = c;
        const auto hz = c == vintage12 ? 10500.0f : 15000.0f;
        lowpassA.setLowpass (hz, 0.5412f, sampleRate);
        lowpassB.setLowpass (hz, 1.3066f, sampleRate);
        envelopeRelease = std::exp (-1.0f / (0.08f * (float) sampleRate));
        clear();
    }

    void clear() noexcept
    {
        lowpassA.clear();
        lowpassB.clear();
        envelope = {};
    }

    void process (float* left, float* right, int n) noexcept
    {
        if (character == modern)
            return;

        for (int i = 0; i < n; ++i)
        {
            left[i] = processSample (left[i], 0);
            right[i] = processSample (right[i], 1);
        }
    }

private:
    static float quantise (float x, float steps) noexcept
    {
        return (float) (int) (x * steps + std::copysign (0.5f, x)) / steps;
    }

    float processSample (float x, size_t ch) noexcept
    {
        x = lowpassB.process (lowpassA.process (x, ch), ch);

        if (character == vintage24)
            return quantise (std::clamp (x, -4.0f, 4.0f), 8388607.0f);

        envelope[ch] = std::max (std::abs (x), envelope[ch] * envelopeRelease);
        const auto gain = 1.0f / std::sqrt (std::max (envelope[ch], 2.5e-4f));
        return quantise (std::clamp (x * gain, -1.0f, 1.0f), 2047.0f) / gain;
    }

    Character character = modern;
    Biquad lowpassA, lowpassB;
    std::array<float, 2> envelope {};
    float envelopeRelease = 0.999f;
};

//==============================================================================
// 预延迟固定用整数采样读取。改设定时在两个读取点之间交叉淡化，不产生变调和跳变。
class Predelay
{
public:
    void prepare (double sampleRate)
    {
        rate = sampleRate;
        const auto maxSamples = (int) std::ceil (0.25 * sampleRate) + 2;
        left.prepare (maxSamples);
        right.prepare (maxSamples);
        fadeStep = 1.0f / (float) (0.02 * sampleRate);
        clear();
    }

    void clear() noexcept
    {
        left.clear();
        right.clear();
        fade = 1.0f;
    }

    void snapTo (float ms) noexcept             { current = next = wanted = toSamples (ms); fade = 1.0f; }
    void setTarget (float ms) noexcept          { wanted = toSamples (ms); }

    void process (float& l, float& r) noexcept
    {
        if (fade >= 1.0f && wanted != current)
        {
            next = wanted;
            fade = 0.0f;
        }

        const auto oldL = current == 0 ? l : left.read (current);
        const auto oldR = current == 0 ? r : right.read (current);
        const auto newL = next == 0 ? l : left.read (next);
        const auto newR = next == 0 ? r : right.read (next);
        left.write (l);
        right.write (r);

        if (fade < 1.0f)
        {
            l = oldL + fade * (newL - oldL);
            r = oldR + fade * (newR - oldR);
            fade += fadeStep;
            if (fade >= 1.0f)
                current = next;
        }
        else
        {
            l = oldL;
            r = oldR;
        }
    }

private:
    int toSamples (float ms) const noexcept     { return (int) std::lround (0.001 * (double) ms * rate); }

    DelayLine left, right;
    double rate = 48000.0;
    int current = 0, next = 0, wanted = 0;
    float fade = 1.0f, fadeStep = 0.001f;
};

//==============================================================================
// 整机：干声直通，湿声逐采样经过预延迟、转换器、算法、转换器后混合。没有任何块缓冲，所以延迟恒为 0。
class Reverb
{
public:
    Reverb()
    {
        engines[hall] = std::make_unique<HallEngine>();
        engines[plate] = std::make_unique<PlateEngine>();
        engines[room] = std::make_unique<RoomEngine>();
        engines[chamber] = std::make_unique<ChamberEngine>();
    }

    void prepare (double newRate, const Params& initial)
    {
        sampleRate = newRate;

        for (auto& e : engines)
            e->prepare (sampleRate);

        for (int c = 0; c < numCharacters; ++c)
        {
            converterIn[(size_t) c].prepare ((Character) c, sampleRate);
            converterOut[(size_t) c].prepare ((Character) c, sampleRate);
        }

        predelay.prepare (sampleRate);
        dcCoef = std::min (2.0f * pi * 8.0f / (float) sampleRate, 0.1f);
        smoothCoef = 1.0f - std::exp (-(float) controlInterval / (0.02f * (float) sampleRate));

        target = initial;
        snapToTarget();
        clear();
    }

    void clear() noexcept
    {
        for (auto& e : engines)
            e->clear();
        for (auto& c : converterIn)  c.clear();
        for (auto& c : converterOut) c.clear();
        predelay.clear();
        predelay.snapTo (target.predelayMs);
        dcL = dcR = 0.0f;

        for (auto& s : slots)
            s = {};
        auto& slot = slots[(size_t) currentAlgorithm];
        slot.inputGain = 1.0f;
        slot.outputGain = 1.0f;
        slot.running = true;
        controlCountdown = 0;
        characterFade = 1.0f;
        previousCharacter = currentCharacter;
    }

    void setParams (const Params& p) noexcept   { target = p; }

    // left 和 right 就地处理。单声道时 right 传 nullptr。
    void process (float* left, float* right, int numSamples) noexcept
    {
        while (numSamples > 0)
        {
            if (controlCountdown <= 0)
            {
                updateControl();
                controlCountdown = controlInterval;
            }

            const auto n = std::min (numSamples, controlCountdown);
            processChunk (left, right, n);

            left += n;
            if (right != nullptr)
                right += n;
            numSamples -= n;
            controlCountdown -= n;
        }
    }

private:
    struct Slot
    {
        float inputGain = 0.0f, outputGain = 0.0f;
        float silentSamples = 0.0f;
        bool running = false;
    };

    static float sanitise (float x) noexcept    { return std::abs (x) < 1.0e30f ? x : 0.0f; }

    // 湿声只在超过满刻度后才进入软限，上限 +6 dB。线性区内的脉冲响应不受影响。
    static float softLimit (float x) noexcept
    {
        const auto a = std::abs (x);
        if (a <= 1.0f)
            return x;
        return std::copysign (2.0f - std::exp (1.0f - a), x);
    }

    void snapToTarget() noexcept
    {
        smoothLogTime = std::log (target.timeSeconds);
        smoothLow = std::log (target.lowMultiplier);
        smoothMid = std::log (target.midMultiplier);
        smoothDamping = target.damping;
        smoothLogCross = std::log (target.crossoverHz);
        smoothMod = target.modDepth;
        smoothMix = target.mix;
        currentAlgorithm = std::clamp (target.algorithm, 0, numAlgorithms - 1);
        currentCharacter = previousCharacter = std::clamp (target.character, 0, numCharacters - 1);
        predelay.snapTo (target.predelayMs);
        for (size_t a = 0; a < (size_t) numAlgorithms; ++a)
            applyDecay (a, true);
        decayRamping = false;
        updateMixTargets();
        dryGain = dryNext = dryTarget;
        wetGain = wetNext = wetTarget;
        dryStep = wetStep = 0.0f;
    }

    void updateMixTargets() noexcept
    {
        const auto m = std::clamp (smoothMix, 0.0f, 1.0f);
        dryTarget = m >= 1.0f ? 0.0f : (m <= 0.0f ? 1.0f : std::cos (0.5f * pi * m));
        const auto wet = m <= 0.0f ? 0.0f : std::sin (0.5f * pi * m);
        // 能量随混响时间线性增长。这里补回一半，长短设定之间的响度差不至于过大。
        wetTarget = wet * std::pow (2.0f / std::exp (smoothLogTime + smoothMid), 0.25f);
    }

    void applyDecay (size_t index, bool instant) noexcept
    {
        auto& engine = *engines[index];
        const auto voicing = engine.voicing();
        const auto midTime = std::exp (smoothLogTime + smoothMid);
        const auto lowTime = std::exp (smoothLogTime + smoothLow);
        const auto ratio = std::pow (10.0f, -voicing.dampingBias * smoothDamping);

        // 高频搁架的过渡带会漏到中频。迭代解出中频斜率，让参考频率处的总衰减等于设定值。
        const auto wantedSlope = 60.0f / midTime;
        const auto n = referenceHz / voicing.dampingHz;
        auto midSlope = wantedSlope;
        for (int i = 0; i < 3; ++i)
        {
            const auto g = std::pow (10.0f, -midSlope * (1.0f / ratio - 1.0f) * voicing.segmentSeconds * 0.05f);
            const auto leakDb = 10.0f * std::log10 ((1.0f + g * g * n * n) / (1.0f + n * n));
            midSlope = std::max (0.25f * wantedSlope, wantedSlope + leakDb / voicing.segmentSeconds);
        }

        DecaySlopes slopes;
        slopes.mid = midSlope;
        slopes.low = 60.0f / lowTime;
        slopes.high = slopes.mid / ratio;
        slopes.lowCoef = 1.0f - std::exp (-2.0f * pi * std::exp (smoothLogCross) / (float) sampleRate);
        slopes.highCoef = 1.0f - std::exp (-2.0f * pi * voicing.dampingHz / (float) sampleRate);
        slopes.sampleRate = sampleRate;
        engine.setDecay (slopes, instant);
        engine.setModDepth (smoothMod, instant);
    }

    bool isIdle (const Slot& slot) const noexcept   { return slot.silentSamples > (float) sampleRate * 0.25f; }

    bool smooth (float& value, float goal, float threshold) const noexcept
    {
        const auto delta = goal - value;
        if (delta == 0.0f)
            return false;
        value = std::abs (delta) < threshold ? goal : value + smoothCoef * delta;
        return true;
    }

    void updateControl() noexcept
    {
        auto changed = false;
        changed |= smooth (smoothLogTime, std::log (target.timeSeconds), 1.0e-3f);
        changed |= smooth (smoothLow, std::log (target.lowMultiplier), 1.0e-3f);
        changed |= smooth (smoothMid, std::log (target.midMultiplier), 1.0e-3f);
        changed |= smooth (smoothDamping, target.damping, 1.0e-3f);
        changed |= smooth (smoothLogCross, std::log (target.crossoverHz), 1.0e-3f);
        changed |= smooth (smoothMod, target.modDepth, 1.0e-3f);
        const auto mixChanged = smooth (smoothMix, target.mix, 1.0e-3f);

        const auto wantedAlgorithm = std::clamp (target.algorithm, 0, numAlgorithms - 1);
        if (wantedAlgorithm != currentAlgorithm)
        {
            currentAlgorithm = wantedAlgorithm;
            auto& slot = slots[(size_t) wantedAlgorithm];
            if (! slot.running)
            {
                // 新算法从空回路开始，它的湿声是连续长出来的，不需要淡入。旧算法的尾音继续走完淡出。
                engines[(size_t) wantedAlgorithm]->clear();
                slot = {};
                slot.running = true;
                slot.outputGain = 1.0f;
                applyDecay ((size_t) wantedAlgorithm, true);
            }
        }

        const auto wantedCharacter = std::clamp (target.character, 0, numCharacters - 1);
        if (wantedCharacter != currentCharacter && characterFade >= 1.0f)
        {
            previousCharacter = currentCharacter;
            currentCharacter = wantedCharacter;
            converterIn[(size_t) currentCharacter].clear();
            converterOut[(size_t) currentCharacter].clear();
            characterFade = 0.0f;
        }

        // 系数停止变化后再走一拍，斜坡才会落到目标上并停下。
        if (changed || decayRamping)
            for (size_t a = 0; a < (size_t) numAlgorithms; ++a)
                if (slots[a].running)
                    applyDecay (a, isIdle (slots[a]));
        decayRamping = changed;

        predelay.setTarget (target.predelayMs);

        if (changed || mixChanged)
            updateMixTargets();

        dryGain = dryNext;
        wetGain = wetNext;
        dryNext = dryTarget;
        wetNext = wetTarget;
        dryStep = (dryNext - dryGain) / (float) controlInterval;
        wetStep = (wetNext - wetGain) / (float) controlInterval;
    }

    void processChunk (float* left, float* right, int n) noexcept
    {
        std::array<float, controlInterval> inL, inR, outL, outR, sumL, sumR;
        const auto mono = right == nullptr;

        auto inputPeak = 0.0f;
        for (size_t i = 0; i < (size_t) n; ++i)
        {
            auto l = sanitise (left[i]);
            auto r = mono ? l : sanitise (right[i]);
            predelay.process (l, r);

            // 极低频阻断，防止直流在长尾设定下堆积。
            dcL += dcCoef * (l - dcL);
            dcR += dcCoef * (r - dcR);
            inL[i] = l - dcL;
            inR[i] = r - dcR;
            inputPeak = std::max ({ inputPeak, std::abs (inL[i]), std::abs (inR[i]) });
        }

        runConverter (converterIn, inL.data(), inR.data(), n);

        std::fill_n (sumL.begin(), n, 0.0f);
        std::fill_n (sumR.begin(), n, 0.0f);

        const auto inputRamp = (float) n / (float) (0.005 * sampleRate);
        const auto outputRamp = (float) n / (float) (fadeSeconds * sampleRate);

        for (size_t a = 0; a < (size_t) numAlgorithms; ++a)
        {
            auto& slot = slots[a];
            if (! slot.running)
                continue;

            const auto active = (int) a == currentAlgorithm;

            // 输入和尾音都已静音时不再运算，直播中空闲段几乎不占 CPU。
            if (active && inputPeak < silenceThreshold && isIdle (slot))
                continue;

            const auto inputStart = slot.inputGain;
            const auto outputStart = slot.outputGain;
            const auto inputEnd = active ? std::min (1.0f, inputStart + inputRamp) : std::max (0.0f, inputStart - inputRamp);
            const auto outputEnd = active ? std::min (1.0f, outputStart + outputRamp) : std::max (0.0f, outputStart - outputRamp);

            std::array<float, controlInterval> rampedL, rampedR;
            for (size_t i = 0; i < (size_t) n; ++i)
            {
                const auto t = (float) (i + 1) / (float) n;
                const auto g = inputStart + t * (inputEnd - inputStart);
                rampedL[i] = g * inL[i];
                rampedR[i] = g * inR[i];
            }

            engines[a]->process (rampedL.data(), rampedR.data(), outL.data(), outR.data(), n);

            const auto scale = outputScale[a];
            auto outputPeak = 0.0f;
            for (size_t i = 0; i < (size_t) n; ++i)
            {
                const auto t = (float) (i + 1) / (float) n;
                const auto linear = outputStart + t * (outputEnd - outputStart);
                const auto g = scale * linear * linear * (3.0f - 2.0f * linear);
                outputPeak = std::max ({ outputPeak, std::abs (outL[i]), std::abs (outR[i]) });
                sumL[i] += g * outL[i];
                sumR[i] += g * outR[i];
            }

            slot.inputGain = inputEnd;
            slot.outputGain = outputEnd;
            slot.silentSamples = (inputPeak < silenceThreshold && outputPeak < silenceThreshold) ? slot.silentSamples + (float) n : 0.0f;

            if (! active && outputEnd <= 0.0f)
            {
                slot = {};
                engines[a]->clear();
            }

            // 任何异常状态都不让它进到输出，也不让它留在回路里。
            if (! (outputPeak < 1.0e6f))
            {
                engines[a]->clear();
                std::fill_n (sumL.begin(), n, 0.0f);
                std::fill_n (sumR.begin(), n, 0.0f);
            }
        }

        runConverter (converterOut, sumL.data(), sumR.data(), n);

        for (size_t i = 0; i < (size_t) n; ++i)
        {
            dryGain += dryStep;
            wetGain += wetStep;
            left[i] = dryGain * sanitise (left[i]) + softLimit (wetGain * sumL[i]);
            if (! mono)
                right[i] = dryGain * sanitise (right[i]) + softLimit (wetGain * sumR[i]);
        }

        if (characterFade < 1.0f)
        {
            characterFade += (float) n / (float) (characterFadeSeconds * sampleRate);
            if (characterFade >= 1.0f)
            {
                characterFade = 1.0f;
                converterIn[(size_t) previousCharacter].clear();
                converterOut[(size_t) previousCharacter].clear();
                previousCharacter = currentCharacter;
            }
        }
    }

    void runConverter (std::array<Converter, numCharacters>& bank, float* l, float* r, int n) noexcept
    {
        if (characterFade >= 1.0f)
        {
            bank[(size_t) currentCharacter].process (l, r, n);
            return;
        }

        std::array<float, controlInterval> oldL, oldR;
        std::copy_n (l, n, oldL.begin());
        std::copy_n (r, n, oldR.begin());
        bank[(size_t) previousCharacter].process (oldL.data(), oldR.data(), n);
        bank[(size_t) currentCharacter].process (l, r, n);

        const auto step = (float) (1.0 / (characterFadeSeconds * sampleRate));
        for (int i = 0; i < n; ++i)
        {
            const auto t = std::min (1.0f, characterFade + step * (float) (i + 1));
            l[i] = oldL[(size_t) i] + t * (l[i] - oldL[(size_t) i]);
            r[i] = oldR[(size_t) i] + t * (r[i] - oldR[(size_t) i]);
        }
    }

    static constexpr float silenceThreshold = 1.0e-7f;
    static constexpr float referenceHz = 1000.0f;
    static constexpr double fadeSeconds = 0.25, characterFadeSeconds = 0.03;
    // 让四种算法在 2.2 秒、阻尼 50 % 时脉冲响应总能量都约为 -3 dB，切算法时响度不跳。
    static constexpr std::array<float, numAlgorithms> outputScale { 1.29f, 1.19f, 0.56f, 1.46f };

    std::array<std::unique_ptr<AlgorithmEngine>, numAlgorithms> engines;
    std::array<Slot, numAlgorithms> slots;
    std::array<Converter, numCharacters> converterIn, converterOut;
    Predelay predelay;

    Params target;
    double sampleRate = 48000.0;
    int controlCountdown = 0;
    bool decayRamping = false;
    int currentAlgorithm = hall, currentCharacter = vintage24, previousCharacter = vintage24;
    float characterFade = 1.0f, smoothCoef = 0.03f;
    float smoothLogTime = 0.0f, smoothLow = 0.0f, smoothMid = 0.0f, smoothDamping = 0.0f, smoothLogCross = 0.0f, smoothMod = 0.0f, smoothMix = 0.0f;
    float dryGain = 1.0f, wetGain = 0.0f, dryStep = 0.0f, wetStep = 0.0f;
    float dryTarget = 1.0f, wetTarget = 0.0f, dryNext = 1.0f, wetNext = 0.0f;
    float dcL = 0.0f, dcR = 0.0f, dcCoef = 0.001f;
};

} // namespace elysium
