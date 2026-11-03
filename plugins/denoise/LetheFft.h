#pragma once

#include <cmath>
#include <complex>
#include <vector>

// juce::dsp::FFT 只支持 2 的幂，模型的窗长是 960 = 2^6·3·5，所以这里放一个小的混合基 FFT。
// 不做归一化，正反变换的系数和 libDF 用的 realfft 一致。plan 之后不分配内存。
namespace lethe
{

class MixedRadixFft
{
public:
    using Complex = std::complex<float>;

    explicit MixedRadixFft (int size) : n (size), twiddles ((size_t) size), scratch ((size_t) size), work ((size_t) size)
    {
        const auto twoPi = 6.283185307179586476925286766559;
        for (int i = 0; i < n; ++i)
            twiddles[(size_t) i] = Complex ((float) std::cos (twoPi * i / n), (float) -std::sin (twoPi * i / n));

        auto remaining = n;
        for (const auto p : { 5, 3, 4, 2 })
            while (remaining % p == 0)
            {
                factors.push_back (p);
                remaining /= p;
            }
    }

    int size() const noexcept { return n; }

    // 正变换：指数为 -j。out 长度为 size()。
    void forward (const Complex* in, Complex* out) noexcept
    {
        recurse (out, in, 1, 1, 0);
    }

    // 反变换：指数为 +j，不除以 N。in 可以和 out 不同，也不会被改写。
    void inverse (const Complex* in, Complex* out) noexcept
    {
        for (int i = 0; i < n; ++i)
            work[(size_t) i] = std::conj (in[i]);
        recurse (out, work.data(), 1, 1, 0);
        for (int i = 0; i < n; ++i)
            out[i] = std::conj (out[i]);
    }

private:
    // 按时间抽取。stride 是输入的跨度，twiddleStep 是这一层旋转因子的跨度。
    void recurse (Complex* out, const Complex* in, int stride, int twiddleStep, int level) noexcept
    {
        const auto p = factors[(size_t) level];
        const auto m = n / (twiddleStep * p);

        if (m == 1)
        {
            for (int q = 0; q < p; ++q)
                out[q] = in[q * stride];
        }
        else
        {
            for (int q = 0; q < p; ++q)
                recurse (out + q * m, in + q * stride, stride * p, twiddleStep * p, level + 1);
        }

        auto* s = scratch.data();
        const auto rootStep = twiddleStep * m;
        for (int u = 0; u < m; ++u)
        {
            for (int q = 0; q < p; ++q)
                s[q] = out[u + q * m] * twiddles[(size_t) ((q * u * twiddleStep) % n)];

            for (int q = 0; q < p; ++q)
            {
                auto sum = s[0];
                for (int r = 1; r < p; ++r)
                    sum += s[r] * twiddles[(size_t) ((r * q * rootStep) % n)];
                out[u + q * m] = sum;
            }
        }
    }

    int n;
    std::vector<Complex> twiddles, scratch, work;
    std::vector<int> factors;
};

} // namespace lethe
