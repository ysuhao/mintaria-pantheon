// 滤波器设计。每个二阶节都从模拟原型出发，先做频率预畸变（把模拟频率映射到 tan(w/2) 轴上），
// 再在高频段做校正：在预畸变轴上对对数幅度跑固定次数的 Levenberg-Marquardt 拟合，让数字响应贴近模拟
// 原型一直到奈奎斯特。这样接近奈奎斯特的钟形不会像普通双线性变换那样收窄。
//
// 结果用 TPT 状态变量滤波器的参数 (g, k, mHp, mBp, mLp) 表示，拖动时可以直接平滑。
// 全程不分配内存，可以在音频线程调用。
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace iris
{

constexpr double pi = 3.14159265358979323846;

enum class SectionShape { bell, lowShelf, highShelf, tilt, highPass2, lowPass2, highPass1, lowPass1, notch };

// H(s') = (mHp s'^2 + mBp s' + mLp) / (s'^2 + k s' + 1)，s' = j tan(w/2) / g。
struct SectionDesign
{
    double g = 0.1, k = 1.0, mHp = 1.0, mBp = 1.0, mLp = 1.0;

    double magnitudeSquared (double frequency, double sampleRate) const noexcept
    {
        const auto w = pi * std::min (frequency / sampleRate, 0.49999);
        const auto o = std::tan (w) / g;
        const auto v = o * o;
        const auto nr = mLp - mHp * v;
        const auto ni = mBp * o;
        const auto dr = 1.0 - v;
        const auto di = k * o;
        return (nr * nr + ni * ni) / std::max (dr * dr + di * di, 1.0e-300);
    }
};

// 模拟原型的分子、分母系数，按 s^2, s, 1 的顺序，s 已对中心频率归一化。
struct AnalogBiquad
{
    double n2, n1, n0, d2, d1, d0;

    double magnitudeSquared (double x) const noexcept
    {
        const auto v = x * x;
        const auto nr = n0 - n2 * v;
        const auto ni = n1 * x;
        const auto dr = d0 - d2 * v;
        const auto di = d1 * x;
        return (nr * nr + ni * ni) / std::max (dr * dr + di * di, 1.0e-300);
    }
};

inline bool isReciprocalShape (SectionShape shape) noexcept
{
    return shape == SectionShape::bell || shape == SectionShape::lowShelf
        || shape == SectionShape::highShelf || shape == SectionShape::tilt;
}

// 设计目标，也是 tests/ 里校验用的基准曲线。提升和衰减互为倒数。
inline AnalogBiquad analogPrototype (SectionShape shape, double gainDb, double q) noexcept
{
    // 架式和倾斜的 Q 太高会变成不受控的共振，这里收窄范围。
    if (shape == SectionShape::lowShelf || shape == SectionShape::highShelf || shape == SectionShape::tilt)
        q = std::clamp (q, 0.1, 4.0);

    switch (shape)
    {
        case SectionShape::bell:
        {
            const auto a = std::pow (10.0, gainDb / 40.0);
            return { 1.0, a / q, 1.0, 1.0, 1.0 / (a * q), 1.0 };
        }
        case SectionShape::lowShelf:
        {
            const auto a = std::pow (10.0, gainDb / 40.0);
            const auto r = std::sqrt (a) / q;
            return { a, a * r, a * a, a, r, 1.0 };
        }
        case SectionShape::highShelf:
        {
            const auto a = std::pow (10.0, gainDb / 40.0);
            const auto r = std::sqrt (a) / q;
            return { a * a, a * r, a, 1.0, r, a };
        }
        case SectionShape::tilt:
        {
            // 中心频率处 0 dB，两端分别为 -gain 和 +gain。
            const auto a = std::pow (10.0, gainDb / 20.0);
            const auto r = std::sqrt (a) / q;
            return { a, r, 1.0, 1.0, r, a };
        }
        case SectionShape::highPass2:  return { 1.0, 0.0, 0.0, 1.0, 1.0 / q, 1.0 };
        case SectionShape::lowPass2:   return { 0.0, 0.0, 1.0, 1.0, 1.0 / q, 1.0 };
        case SectionShape::highPass1:  return { 0.0, 1.0, 0.0, 0.0, 1.0, 1.0 };
        case SectionShape::lowPass1:   return { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0 };
        case SectionShape::notch:      return { 1.0, 0.0, 1.0, 1.0, 1.0 / q, 1.0 };
    }
    return { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
}

namespace detail
{
    constexpr double dbPerNeper = 4.342944819032518; // 10 / ln 10
    constexpr int maxGridPoints = 32;
    constexpr int gainShapeIterations = 5;   // 钟形、架式、倾斜，动态频段每个控制周期都要跑，5 次已经收敛
    constexpr int cutShapeIterations = 12;   // 切除和陷波收敛慢，但只在拖动时才跑
    constexpr double maxFrequencyRatio = 0.49;           // 设计频率的上限，占采样率的比例
    constexpr double uncorrectedBelowRatio = 1.0 / 48.0; // 低于这个比例时预畸变的误差已可忽略，跳过拟合

    enum class NumeratorForm { full, highPass2, highPass1, lowPass1, notch };

    struct Fit
    {
        double lnG, lnWz, lnQz, lnWp, lnQp;
    };

    struct Grid
    {
        int size = 0;
        std::array<double, maxGridPoints> v, targetDb, weight;
    };

    inline double warped (double frequency, double sampleRate, double reference) noexcept
    {
        return std::tan (pi * std::min (frequency, 0.495 * sampleRate) / sampleRate) / reference;
    }

    struct Evaluation
    {
        double cost;
        std::array<double, maxGridPoints> residual;
        std::array<std::array<double, 5>, maxGridPoints> jacobian;
    };

    inline void evaluate (const Grid& grid, const Fit& p, NumeratorForm form, double vz, Evaluation& out) noexcept
    {
        const auto wz2 = std::exp (-2.0 * p.lnWz);
        const auto iqz2 = std::exp (-2.0 * p.lnQz);
        const auto wp2 = std::exp (-2.0 * p.lnWp);
        const auto iqp2 = std::exp (-2.0 * p.lnQp);

        out.cost = 0.0;
        for (int i = 0; i < grid.size; ++i)
        {
            const auto v = grid.v[(size_t) i];
            const auto x = v * wp2;
            const auto oneMinusX = 1.0 - x;
            const auto d = oneMinusX * oneMinusX + x * iqp2;

            double n = 1.0, dnWz = 0.0, dnQz = 0.0;
            switch (form)
            {
                case NumeratorForm::full:
                {
                    const auto y = v * wz2;
                    const auto oneMinusY = 1.0 - y;
                    n = oneMinusY * oneMinusY + y * iqz2;
                    dnWz = 4.0 * oneMinusY * y - 2.0 * y * iqz2;
                    dnQz = -2.0 * y * iqz2;
                    break;
                }
                case NumeratorForm::highPass2:
                    n = v * v;
                    break;
                case NumeratorForm::highPass1:
                    n = v * (1.0 + v * wz2);
                    dnWz = -2.0 * v * v * wz2;
                    break;
                case NumeratorForm::lowPass1:
                    n = 1.0 + v * wz2;
                    dnWz = -2.0 * v * wz2;
                    break;
                case NumeratorForm::notch:
                    n = (v - vz) * (v - vz);
                    break;
            }

            n = std::max (n, 1.0e-30);
            const auto w = grid.weight[(size_t) i];
            const auto level = dbPerNeper * (p.lnG + std::log (n) - std::log (d));
            const auto r = (level - grid.targetDb[(size_t) i]) * w;
            out.residual[(size_t) i] = r;
            out.cost += r * r;

            const auto scale = dbPerNeper * w;
            auto& row = out.jacobian[(size_t) i];
            row[0] = scale;
            row[1] = scale * dnWz / n;
            row[2] = scale * dnQz / n;
            row[3] = -scale * (4.0 * oneMinusX * x - 2.0 * x * iqp2) / d;
            row[4] = scale * 2.0 * x * iqp2 / d;
        }
    }

    // 带主元的高斯消元，只解活动参数那几维。解放在最后一列。
    template <int size>
    inline bool solve (std::array<std::array<double, size + 1>, size>& m) noexcept
    {
        for (int col = 0; col < size; ++col)
        {
            auto pivot = col;
            for (int row = col + 1; row < size; ++row)
                if (std::abs (m[(size_t) row][(size_t) col]) > std::abs (m[(size_t) pivot][(size_t) col]))
                    pivot = row;
            if (std::abs (m[(size_t) pivot][(size_t) col]) < 1.0e-30)
                return false;
            std::swap (m[(size_t) col], m[(size_t) pivot]);

            for (int row = col + 1; row < size; ++row)
            {
                const auto f = m[(size_t) row][(size_t) col] / m[(size_t) col][(size_t) col];
                for (int c = col; c <= size; ++c)
                    m[(size_t) row][(size_t) c] -= f * m[(size_t) col][(size_t) c];
            }
        }

        for (int row = size - 1; row >= 0; --row)
        {
            auto value = m[(size_t) row][(size_t) size];
            for (int c = row + 1; c < size; ++c)
                value -= m[(size_t) row][(size_t) c] * m[(size_t) c][(size_t) size];
            m[(size_t) row][(size_t) size] = value / m[(size_t) row][(size_t) row];
        }
        return true;
    }

    template <int active>
    inline void refine (const Grid& grid, Fit& fit, NumeratorForm form, double vz, const std::array<int, (size_t) active>& indices, int iterations) noexcept
    {
        Evaluation evaluations[2];
        auto* current = &evaluations[0];
        auto* trial = &evaluations[1];
        evaluate (grid, fit, form, vz, *current);
        auto lambda = 0.01;

        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            std::array<std::array<double, (size_t) active + 1>, (size_t) active> m {};
            for (int i = 0; i < grid.size; ++i)
            {
                const auto& row = current->jacobian[(size_t) i];
                const auto r = current->residual[(size_t) i];
                for (int a = 0; a < active; ++a)
                {
                    const auto ja = row[(size_t) indices[(size_t) a]];
                    for (int b = a; b < active; ++b)
                        m[(size_t) a][(size_t) b] += ja * row[(size_t) indices[(size_t) b]];
                    m[(size_t) a][(size_t) active] -= ja * r;
                }
            }

            for (int a = 0; a < active; ++a)
            {
                for (int b = 0; b < a; ++b)
                    m[(size_t) a][(size_t) b] = m[(size_t) b][(size_t) a];
                m[(size_t) a][(size_t) a] += lambda * (m[(size_t) a][(size_t) a] + 1.0e-6);
            }

            if (! solve<active> (m))
                return;

            std::array<double, 5> values { fit.lnG, fit.lnWz, fit.lnQz, fit.lnWp, fit.lnQp };
            for (int a = 0; a < active; ++a)
                values[(size_t) indices[(size_t) a]] += std::clamp (m[(size_t) a][(size_t) active], -1.5, 1.5);
            const Fit candidate { values[0], values[1], values[2], values[3], values[4] };

            evaluate (grid, candidate, form, vz, *trial);
            if (trial->cost < current->cost)
            {
                fit = candidate;
                std::swap (current, trial);
                lambda = std::max (lambda * 0.3, 1.0e-6);
            }
            else
            {
                lambda = std::min (lambda * 8.0, 1.0e3);
            }
        }
    }

    inline void buildGrid (Grid& grid, const AnalogBiquad& prototype, SectionShape shape, double f0, double q, double sampleRate, double reference) noexcept
    {
        std::array<double, maxGridPoints> points;
        auto count = 0;
        const auto add = [&] (double f)
        {
            if (f > 0.5 && f < 0.97 * 0.5 * sampleRate && count < maxGridPoints - 9)
                points[(size_t) count++] = f;
        };

        if (shape == SectionShape::bell || shape == SectionShape::notch)
        {
            const auto b = 0.5 / std::max (q, 0.1);
            for (const auto scale : { 0.25, 0.5, 1.0, 2.0, 4.0 })
            {
                const auto offset = std::sqrt (1.0 + b * b * scale * scale);
                add (f0 * (offset - b * scale));
                add (f0 * (offset + b * scale));
            }
            add (f0);
            for (const auto x : { 0.125, 0.25, 0.5, 2.0, 4.0, 8.0 })
                add (f0 * x);
        }
        else
        {
            for (const auto x : { 1.0 / 16.0, 0.125, 0.25, 0.4, 2.0 / 3.0, 1.0, 1.5, 2.5, 4.0, 8.0, 16.0 })
                add (f0 * x);

            // 高 Q 时共振峰很窄，要在峰附近多放几个点。
            if (q > 1.0)
            {
                const auto b = 0.5 / q;
                for (const auto scale : { 0.5, 1.0, 2.0, 4.0 })
                {
                    const auto offset = std::sqrt (1.0 + b * b * scale * scale);
                    add (f0 * (offset - b * scale));
                    add (f0 * (offset + b * scale));
                }
            }
        }

        for (const auto x : { 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97, 0.9995 })
            points[(size_t) count++] = x * 0.5 * sampleRate;
        points[(size_t) count++] = std::min (1.0, f0 * 1.0e-3);

        const auto weighStopband = ! isReciprocalShape (shape);
        grid.size = count;
        for (int i = 0; i < count; ++i)
        {
            const auto f = points[(size_t) i];
            const auto o = std::tan (pi * f / sampleRate) / reference;
            const auto target = dbPerNeper * std::log (std::max (prototype.magnitudeSquared (f / f0), 1.0e-9));
            grid.v[(size_t) i] = o * o;
            grid.targetDb[(size_t) i] = target;
            grid.weight[(size_t) i] = weighStopband && target < -40.0 ? 0.05 : 1.0;
        }
    }
}

// 设计一个二阶节。gainDb 只对钟形、架式和倾斜有意义。
inline SectionDesign designSection (SectionShape shape, double frequency, double gainDb, double q, double sampleRate) noexcept
{
    using namespace detail;

    const auto f0 = std::min (frequency, maxFrequencyRatio * sampleRate);
    const auto w0 = 2.0 * pi * f0 / sampleRate;
    const auto reference = std::tan (0.5 * std::min (w0, 0.9 * pi));

    if (isReciprocalShape (shape) && std::abs (gainDb) < 1.0e-3)
        return { reference, 1.0 / q, 1.0, 1.0 / q, 1.0 };

    const auto invert = isReciprocalShape (shape) && gainDb < 0.0;
    const auto proto = analogPrototype (shape, std::abs (gainDb), q);

    // 初值：把模拟零极点预畸变后放到 tan(w/2) 轴上。
    Fit fit {};
    auto form = NumeratorForm::full;
    auto vz = 0.0;
    auto top = f0;

    const auto nyquistSide = warped (0.49 * sampleRate, sampleRate, reference);

    if (proto.d2 == 0.0)
    {
        // 一阶节：第二个实极点放到奈奎斯特附近，并被同位置的零点抵消。
        const auto w1 = warped (f0, sampleRate, reference);
        fit.lnWp = 0.5 * std::log (w1 * nyquistSide);
        fit.lnQp = fit.lnWp - std::log (w1 + nyquistSide);
        fit.lnWz = std::log (nyquistSide);
        form = shape == SectionShape::highPass1 ? NumeratorForm::highPass1 : NumeratorForm::lowPass1;
        fit.lnG = shape == SectionShape::highPass1 ? -2.0 * std::log (w1) : 0.0;
    }
    else
    {
        const auto fp = f0 * std::sqrt (proto.d0 / proto.d2);
        fit.lnWp = std::log (warped (fp, sampleRate, reference));
        fit.lnQp = std::log (std::sqrt (proto.d0 * proto.d2) / proto.d1);
        top = std::max (top, fp);

        if (shape == SectionShape::highPass2)
        {
            form = NumeratorForm::highPass2;
            fit.lnG = -4.0 * fit.lnWp;
        }
        else if (shape == SectionShape::notch)
        {
            form = NumeratorForm::notch;
            const auto o = std::tan (0.5 * std::min (w0, 0.995 * pi)) / reference;
            vz = o * o;
            fit.lnG = -2.0 * std::log (vz);
        }
        else if (shape == SectionShape::lowPass2)
        {
            // 没有模拟零点。数字侧需要一对高频零点，初值让奈奎斯特处的增益正好等于模拟原型。
            const auto atNyquist = proto.magnitudeSquared (0.49975 * sampleRate / f0);
            fit.lnWz = fit.lnWp - 0.25 * std::log (std::max (atNyquist, 1.0e-12));
            fit.lnQz = std::log (0.7071);
        }
        else
        {
            const auto fz = f0 * std::sqrt (proto.n0 / proto.n2);
            fit.lnWz = std::log (warped (fz, sampleRate, reference));
            fit.lnQz = std::log (std::sqrt (proto.n0 * proto.n2) / proto.n1);
            fit.lnG = 2.0 * std::log (proto.n0 / proto.d0);
            top = std::max (top, fz);
        }
    }

    // 低通一直做校正：双线性变换会在奈奎斯特放一个零点，高频衰减比模拟原型多。
    const auto isLowPass = shape == SectionShape::lowPass1 || shape == SectionShape::lowPass2;

    if (isLowPass || top >= uncorrectedBelowRatio * sampleRate)
    {
        Grid grid;
        buildGrid (grid, proto, shape, f0, q, sampleRate, reference);

        switch (form)
        {
            case NumeratorForm::full:
                refine<5> (grid, fit, form, vz, std::array<int, 5> { 0, 1, 2, 3, 4 }, isLowPass ? cutShapeIterations : gainShapeIterations);
                break;
            case NumeratorForm::highPass1:
            case NumeratorForm::lowPass1:
                refine<4> (grid, fit, form, vz, std::array<int, 4> { 0, 1, 3, 4 }, cutShapeIterations);
                break;
            case NumeratorForm::highPass2:
            case NumeratorForm::notch:
                refine<3> (grid, fit, form, vz, std::array<int, 3> { 0, 3, 4 }, cutShapeIterations);
                break;
        }
    }

    // 拟合结果换算成状态变量滤波器的参数。所有因子都在左半平面，所以稳定且最小相位。
    SectionDesign out;
    const auto rootG = std::exp (0.5 * fit.lnG);

    if (invert)
    {
        // 衰减取提升的倒数：原来的零点变成极点。
        const auto wp = std::exp (fit.lnWz);
        const auto wz = std::exp (fit.lnWp);
        out.g = reference * wp;
        out.k = std::exp (-fit.lnQz);
        out.mLp = 1.0 / rootG;
        out.mBp = out.mLp * (wp / wz) * std::exp (-fit.lnQp);
        out.mHp = out.mLp * (wp / wz) * (wp / wz);
        return out;
    }

    const auto wp = std::exp (fit.lnWp);
    out.g = reference * wp;
    out.k = std::exp (-fit.lnQp);

    switch (form)
    {
        case NumeratorForm::full:
        {
            const auto wz = std::exp (fit.lnWz);
            out.mLp = rootG;
            out.mBp = rootG * (wp / wz) * std::exp (-fit.lnQz);
            out.mHp = rootG * (wp / wz) * (wp / wz);
            break;
        }
        case NumeratorForm::highPass2:
            out.mLp = 0.0;
            out.mBp = 0.0;
            out.mHp = rootG * wp * wp;
            break;
        case NumeratorForm::highPass1:
            out.mLp = 0.0;
            out.mBp = rootG * wp;
            out.mHp = rootG * std::exp (-fit.lnWz) * wp * wp;
            break;
        case NumeratorForm::lowPass1:
            out.mLp = rootG;
            out.mBp = rootG * std::exp (-fit.lnWz) * wp;
            out.mHp = 0.0;
            break;
        case NumeratorForm::notch:
            out.mLp = rootG * vz;
            out.mBp = 0.0;
            out.mHp = rootG * wp * wp;
            break;
    }
    return out;
}

} // namespace iris
