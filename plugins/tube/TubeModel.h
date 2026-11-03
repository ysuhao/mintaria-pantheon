// 电子管线路放大器的声音模型，不依赖 JUCE，方便单独测量。
//
// 信号路径（括号内是对应的硬件部分）：
//   输入 -> 输入变压器 -> 驱动增益 -> TubeStage -> 隔直 -> 输出变压器 -> 补偿
//
// 公开资料里的电路是开环的五极管输入级接反馈内的三极管级，染色主要来自输入级过载。
// 所以 TubeStage 是一条开环传输曲线，反馈的影响体现在两处：
// 驱动越大，反馈越少，二次项越强（evenAt），闭环带宽和低频转折也随之变化。
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace vesta
{

constexpr double pi = 3.14159265358979323846;
constexpr double maxDriveDb = 36.0;

// 所有音色常数集中在这里。
struct Voicing
{
    double inputScale = 0.5;        // 0 dBFS、0 dB 驱动时进入曲线的峰值

    double posKnee = 0.35, posWidth = 1.10;     // 正半周：栅流限幅，先到顶
    double negKnee = 0.60, negWidth = 1.50;     // 负半周：截止，余量约为正半周的 1.5 倍
    double railSlope = 0.02;                    // 限幅后残留的斜率

    double evenBase = 0.022;        // 0 dB 驱动时的二次项
    double evenPerDb = 0.025;       // 二次项随驱动增长：每 dB 的十进制指数
    double evenMax = 0.22;

    double biasThreshold = 0.50;    // 超过它开始有栅流
    double biasRatio = 0.70;        // 超出部分有多少变成偏置偏移
    double biasMax = 6.0;           // 偏置偏移上限，防止长时间堵塞
    double biasAttackSeconds = 0.003;

    double compensationPole = 0.90; // 补偿 ADAA 半采样平均的后级滤波器极点

    double inputCornerHz = 4.0;
    double outputCornerMinHz = 4.0, outputCornerMaxHz = 8.0;    // 驱动越大低频转折越高
    double dcBlockMinHz = 3.0, dcBlockMaxHz = 10.0;             // 同时收紧非对称削波整流出的低频
    double inputCoreFlux = 0.010, outputCoreFlux = 0.007;       // 铁芯饱和磁通（伏秒）

    double inputLowpassHz = 100000.0, outputLowpassHz = 100000.0;
    double loopLowpassMinHz = 40000.0, loopLowpassMaxHz = 100000.0;     // 反馈越少闭环带宽越窄
};

inline const Voicing& voicing()
{
    static const Voicing v;
    return v;
}

inline double driveAmount (double driveDb) noexcept     { return std::min (1.0, std::max (0.0, driveDb / maxDriveDb)); }

inline double evenAt (const Voicing& v, double driveDb) noexcept
{
    return std::min (v.evenMax, v.evenBase * std::pow (10.0, v.evenPerDb * driveDb));
}

// 非对称传输曲线 f(w) = s(w) + even * s(w)^2 及其原函数，全部是多项式闭式。
// s 在拐点内严格线性，拐点段斜率用 smoothstep 过渡，因此二阶连续。
// 只要 2 * even * |s| < 1，f 就单调。
class TubeCurve
{
public:
    explicit TubeCurve (const Voicing& voicingToUse) noexcept
        : pos (voicingToUse.posKnee, voicingToUse.posWidth, voicingToUse.railSlope),
          neg (voicingToUse.negKnee, voicingToUse.negWidth, voicingToUse.railSlope) {}

    struct Result { double s, first, second; };     // s(w)、s 的原函数、s^2 的原函数

    Result evaluate (double w) const noexcept
    {
        if (w >= 0.0)
            return pos.evaluate (w);

        const auto r = neg.evaluate (-w);
        return { -r.s, r.first, -r.second };
    }

    double shape (double w) const noexcept      { return w >= 0.0 ? pos.shape (w) : -neg.shape (-w); }

private:
    struct Side
    {
        Side (double kneeIn, double widthIn, double slopeIn) noexcept
            : knee (kneeIn), width (widthIn), slope (slopeIn), bend (1.0 - slopeIn)
        {
            top = knee + width * (1.0 - 0.5 * bend);
            firstAtTop = 0.5 * knee * knee + knee * width + width * width * (0.5 - 0.15 * bend);
            secondAtTop = kneeSecond (1.0);
        }

        double shape (double x) const noexcept
        {
            if (x <= knee)
                return x;

            const auto t = (x - knee) / width;
            if (t < 1.0)
                return knee + width * polynomial (t);

            return top + slope * (x - knee - width);
        }

        Result evaluate (double x) const noexcept
        {
            if (x <= knee)
                return { x, 0.5 * x * x, x * x * x / 3.0 };

            const auto t = (x - knee) / width;
            if (t < 1.0)
            {
                const auto t2 = t * t;
                return { knee + width * polynomial (t),
                         0.5 * knee * knee + knee * width * t + width * width * t2 * (0.5 - bend * t2 * (0.25 - 0.1 * t)),
                         kneeSecond (t) };
            }

            const auto d = x - knee - width;
            return { top + slope * d,
                     firstAtTop + d * (top + 0.5 * slope * d),
                     secondAtTop + d * (top * top + d * (top * slope + slope * slope * d / 3.0)) };
        }

        double polynomial (double t) const noexcept
        {
            return t - bend * t * t * t * (1.0 - 0.5 * t);
        }

        // 从 0 积分到拐点段 t 处的 s^2。
        double kneeSecond (double t) const noexcept
        {
            const auto m = bend;
            const auto t2 = t * t;
            const auto t3 = t2 * t;
            const auto intP = t2 * (0.5 - m * t2 * (0.25 - 0.1 * t));
            const auto intP2 = t3 * (1.0 / 3.0 + t2 * (-0.4 * m + t * (m / 6.0 + m * m * t * (1.0 / 7.0 + t * (-0.125 + t / 36.0)))));
            return knee * knee * knee / 3.0 + width * (knee * knee * t + 2.0 * knee * width * intP + width * width * intP2);
        }

        double knee, width, slope, bend;
        double top = 0.0, firstAtTop = 0.0, secondAtTop = 0.0;
    };

    Side pos, neg;
};

// 一阶抗混叠（ADAA）的电子管级，带栅流引起的偏置偏移。
//
// ADAA 等于对连续波形取相邻两点的平均，线性部分会多一个 (1 + z^-1) / 2。
// 这里用一个极点接近 -1 的后级滤波器抵消，幅度和半采样延迟一起补回来，总延迟仍为 0。
class TubeStage
{
public:
    explicit TubeStage (const Voicing& voicingToUse = voicing()) noexcept
        : voice (voicingToUse), curve (voicingToUse) {}

    void prepare (double sampleRate) noexcept
    {
        rate = sampleRate;
        attack = coefficient (voice.biasAttackSeconds);
        reset();
    }

    void reset() noexcept
    {
        bias = 0.0;
        previousW = 0.0;
        previousFirst = previousSecond = 0.0;
        compensationState = 0.0;
    }

    void setRecoverySeconds (double seconds) noexcept   { recovery = coefficient (seconds); }
    void setEven (double newEven) noexcept              { even = newEven; }
    double getBias() const noexcept                     { return bias; }

    double processSample (double u) noexcept
    {
        const auto excess = u - voice.biasThreshold;
        const auto target = excess > 0.0 ? voice.biasMax * (1.0 - std::exp (-voice.biasRatio * excess / voice.biasMax)) : 0.0;
        bias += (target > bias ? attack : recovery) * (target - bias);

        const auto w = u - bias;
        const auto now = curve.evaluate (w);
        const auto delta = w - previousW;

        double averaged;
        if (std::abs (delta) > 1.0e-5)
        {
            averaged = ((now.first - previousFirst) + even * (now.second - previousSecond)) / delta;
        }
        else
        {
            const auto s = curve.shape (0.5 * (w + previousW));
            averaged = s + even * s * s;
        }

        previousW = w;
        previousFirst = now.first;
        previousSecond = now.second;

        // 减去静态工作点，静音时输出严格为 0，偏置恢复也不产生低频冲击。
        const auto rest = curve.shape (-bias);
        const auto y = averaged - (rest + even * rest * rest);

        const auto a = voice.compensationPole;
        compensationState = (1.0 + a) * y - a * compensationState;
        return compensationState;
    }

private:
    double coefficient (double seconds) const noexcept  { return 1.0 - std::exp (-1.0 / (seconds * rate)); }

    const Voicing& voice;
    TubeCurve curve;
    double rate = 48000.0;
    double attack = 1.0, recovery = 0.001, even = 0.0;
    double bias = 0.0, previousW = 0.0, previousFirst = 0.0, previousSecond = 0.0, compensationState = 0.0;
};

// 变压器的低频饱和：一阶高通的电感换成随磁通饱和的非线性电感。
// 磁通按隐式梯形积分，牛顿迭代求解。频率越低、电平越高，磁通越大，失真集中在低频。
class SaturatingTransformer
{
public:
    void prepare (double sampleRate, double coreFlux) noexcept
    {
        rate = sampleRate;
        saturationFlux = coreFlux;
        setCorner (5.0);
        reset();
    }

    void reset() noexcept   { state = 0.0; }

    void setCorner (double hz) noexcept
    {
        integrator = std::tan (pi * hz / rate);
        const auto scaled = 2.0 * pi * hz * saturationFlux;
        curvature = 1.0 / (scaled * scaled);
    }

    double processSample (double x) noexcept
    {
        const auto rhs = state + integrator * x;
        auto phi = state;

        for (int i = 0; i < 3; ++i)
        {
            const auto phi2 = phi * phi;
            const auto current = phi * (1.0 + curvature * phi2);
            const auto slope = 1.0 + 3.0 * curvature * phi2;
            phi -= (phi + integrator * current - rhs) / (1.0 + integrator * slope);
        }

        const auto output = x - phi * (1.0 + curvature * phi * phi);
        state = phi + integrator * output;
        return output;
    }

private:
    double rate = 48000.0, saturationFlux = 0.01, integrator = 0.0, curvature = 0.0, state = 0.0;
};

// 一阶低通。Nyquist 处的增益与模拟原型一致，转折接近或高于 Nyquist 时也不会在 20 kHz 过早下垂。
class AnalogMatchedLowpass
{
public:
    void setCorner (double hz, double sampleRate) noexcept
    {
        pole = std::exp (-2.0 * pi * hz / sampleRate);
        const auto ratio = 0.5 * sampleRate / hz;
        const auto atNyquist = 1.0 / std::sqrt (1.0 + ratio * ratio);
        b0 = 0.5 * ((1.0 - pole) + atNyquist * (1.0 + pole));
        b1 = 0.5 * ((1.0 - pole) - atNyquist * (1.0 + pole));
    }

    void reset() noexcept   { previousInput = previousOutput = 0.0; }

    double processSample (double x) noexcept
    {
        previousOutput = b0 * x + b1 * previousInput + pole * previousOutput;
        previousInput = x;
        return previousOutput;
    }

private:
    double pole = 0.0, b0 = 1.0, b1 = 0.0, previousInput = 0.0, previousOutput = 0.0;
};

class DcBlock
{
public:
    void setCorner (double hz, double sampleRate) noexcept  { pole = std::exp (-2.0 * pi * hz / sampleRate); }
    void reset() noexcept                                   { previousInput = previousOutput = 0.0; }

    double processSample (double x) noexcept
    {
        previousOutput = x - previousInput + pole * previousOutput;
        previousInput = x;
        return previousOutput;
    }

private:
    double pole = 0.999, previousInput = 0.0, previousOutput = 0.0;
};

// 驱动到补偿增益的查表。用人声式的参考信号（-18 dBFS RMS）实际跑一遍电子管级得到，
// 音色常数改动后不会失准。这是静态增益，不随信号变化，不会产生抽吸。
class CompensationTable
{
public:
    explicit CompensationTable (const Voicing& v = voicing())
    {
        constexpr double rate = 48000.0;
        constexpr int length = 24000;
        std::vector<float> reference ((size_t) length);

        // 带共振峰的谐波加音节包络，峰值因数约 15 dB，接近未经处理的人声。
        auto sum = 0.0;
        for (int n = 0; n < length; ++n)
        {
            const auto t = (double) n / rate;
            auto voiced = 0.0;
            for (int h = 1; h <= 24; ++h)
            {
                const auto hz = 165.0 * h;
                const auto formant = 1.0 / (1.0 + std::pow ((hz - 500.0) / 120.0, 2.0))
                                   + 0.5 / (1.0 + std::pow ((hz - 1500.0) / 200.0, 2.0))
                                   + 0.25 / (1.0 + std::pow ((hz - 2800.0) / 300.0, 2.0));
                voiced += std::sin (2.0 * pi * hz * t + 0.9 * h * h) * (0.25 / h + formant);
            }
            const auto syllable = std::pow (std::max (0.0, std::sin (pi * 3.0 * t)), 1.5);
            reference[(size_t) n] = (float) (voiced * syllable);
            sum += voiced * syllable * voiced * syllable;
        }

        const auto scale = referenceRms / std::sqrt (sum / length);

        for (size_t i = 0; i < gains.size(); ++i)
        {
            const auto driveDb = maxDriveDb * (double) i / (double) (gains.size() - 1);
            const auto drive = v.inputScale * std::pow (10.0, driveDb / 20.0);

            TubeStage stage (v);
            stage.prepare (rate);
            stage.setRecoverySeconds (0.12);
            stage.setEven (evenAt (v, driveDb));

            auto mean = 0.0, energy = 0.0;
            for (int n = 0; n < length; ++n)
            {
                const auto y = stage.processSample (drive * scale * reference[(size_t) n]);
                mean += y;
                energy += y * y;
            }
            mean /= length;
            gains[i] = referenceRms / std::sqrt (std::max (1.0e-12, energy / length - mean * mean));
        }
    }

    // 线性插值。
    double gainFor (double driveDb) const noexcept
    {
        const auto position = driveAmount (driveDb) * (double) (gains.size() - 1);
        const auto index = std::min ((size_t) position, gains.size() - 2);
        const auto fraction = position - (double) index;
        return gains[index] + fraction * (gains[index + 1] - gains[index]);
    }

    static constexpr double referenceRms = 0.125892541;  // -18 dBFS

private:
    std::array<double, 19> gains {};
};

// 单声道的基础采样率部分。电子管级由外面驱动，因为高质量模式要在过采样域里跑它。
struct ChannelChain
{
    void prepare (const Voicing& v, double sampleRate) noexcept
    {
        rate = sampleRate;
        inputTransformer.prepare (rate, v.inputCoreFlux);
        inputTransformer.setCorner (v.inputCornerHz);
        outputTransformer.prepare (rate, v.outputCoreFlux);
        inputLowpass.setCorner (v.inputLowpassHz, rate);
        outputLowpass.setCorner (v.outputLowpassHz, rate);
        reset();
    }

    void reset() noexcept
    {
        inputTransformer.reset();
        outputTransformer.reset();
        inputLowpass.reset();
        loopLowpass.reset();
        outputLowpass.reset();
        dcBlock.reset();
    }

    void setDrive (const Voicing& v, double driveDb) noexcept
    {
        const auto amount = driveAmount (driveDb);
        outputTransformer.setCorner (v.outputCornerMinHz + amount * (v.outputCornerMaxHz - v.outputCornerMinHz));
        dcBlock.setCorner (v.dcBlockMinHz + amount * (v.dcBlockMaxHz - v.dcBlockMinHz), rate);
        loopLowpass.setCorner (v.loopLowpassMaxHz * std::pow (v.loopLowpassMinHz / v.loopLowpassMaxHz, amount), rate);
    }

    double pre (double x) noexcept      { return inputLowpass.processSample (inputTransformer.processSample (x)); }

    double post (double y) noexcept
    {
        return outputLowpass.processSample (outputTransformer.processSample (dcBlock.processSample (loopLowpass.processSample (y))));
    }

    double rate = 48000.0;
    SaturatingTransformer inputTransformer, outputTransformer;
    AnalogMatchedLowpass inputLowpass, loopLowpass, outputLowpass;
    DcBlock dcBlock;
};

} // namespace vesta
