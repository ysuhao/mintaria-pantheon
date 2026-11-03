// Elysium DSP 自检：不经过宿主，直接驱动 ElysiumDsp.h。
//   elysium_dsp_bench stability [minutes]   长时间运行和极端参数
//   elysium_dsp_bench clicks                调整时间、算法、特性、预延迟、干湿时的跳变
//   elysium_dsp_bench blocks                任意缓冲区大小结果一致
//   elysium_dsp_bench cpu                   单实例 CPU 占用
//   elysium_dsp_bench silence               静音输入的 denormal、自激和空闲开销
//   elysium_dsp_bench converter             三种特性的带宽和信噪失真比
// 退出码非 0 表示有项目不通过。
#include "ElysiumDsp.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <xmmintrin.h>

using namespace elysium;

namespace
{
    const char* const algorithmNames[] { "hall", "plate", "room", "chamber" };
    const char* const characterNames[] { "12bit", "24bit", "modern" };

    struct DenormalGuard
    {
        unsigned int saved = _mm_getcsr();
        DenormalGuard()  { _mm_setcsr (saved | 0x8040u); }
        ~DenormalGuard() { _mm_setcsr (saved); }
    };

    struct Stats
    {
        double peak = 0.0;
        bool bad = false;
        long long subnormal = 0;

        void add (float x)
        {
            if (! std::isfinite (x))
                bad = true;
            const auto a = std::abs ((double) x);
            peak = std::max (peak, a);
            if (x != 0.0f && std::abs (x) < std::numeric_limits<float>::min())
                ++subnormal;
        }
    };

    double db (double x)    { return 20.0 * std::log10 (std::max (x, 1.0e-12)); }

    // 把信号和目标参数交给回调，逐块处理。
    template <typename Source, typename Control>
    Stats run (Reverb& reverb, double seconds, double rate, int block, Source&& source, Control&& control)
    {
        Stats stats;
        std::vector<float> l ((size_t) block), r ((size_t) block);
        const auto total = (long long) (seconds * rate);
        long long pos = 0;
        while (pos < total)
        {
            control (pos);
            for (int i = 0; i < block; ++i)
                source (pos + i, l[(size_t) i], r[(size_t) i]);
            reverb.process (l.data(), r.data(), block);
            for (int i = 0; i < block; ++i)
            {
                stats.add (l[(size_t) i]);
                stats.add (r[(size_t) i]);
            }
            pos += block;
        }
        return stats;
    }

    //==========================================================================
    int testStability (double minutes)
    {
        DenormalGuard guard;
        auto failures = 0;
        std::printf ("-- long run: 3 rounds of noise burst, 0 dB sine, silence; worst-case long decay settings\n");

        for (int a = 0; a < numAlgorithms; ++a)
        {
            Params p;
            p.timeSeconds = 20.0f;
            p.lowMultiplier = 2.5f;
            p.midMultiplier = 2.5f;
            p.damping = 0.0f;
            p.modDepth = 1.0f;
            p.mix = 1.0f;
            p.algorithm = a;
            p.character = modern;

            Reverb reverb;
            const double rate = 48000.0;
            reverb.prepare (rate, p);
            std::mt19937 rng (1234u + (unsigned) a);
            std::uniform_real_distribution<float> noise (-1.0f, 1.0f);

            // 长时间喂信号，再看满量程噪声下的电平和停止输入后的电平是否单调收敛。
            const auto stats = run (reverb, minutes * 60.0, rate, 256,
                [&] (long long n, float& l, float& r)
                {
                    const auto phase = (double) n / rate;
                    const auto section = (int) std::fmod (phase, 30.0) / 10;
                    if (section == 0)      { l = noise (rng); r = noise (rng); }
                    else if (section == 1) { l = r = (float) std::sin (2.0 * 3.141592653589793 * 440.0 * phase); }
                    else                   { l = r = 0.0f; }
                },
                [] (long long) {});

            // 最后一段的衰减斜率：连续两个 5 秒窗的电平必须下降。
            std::vector<float> l (256), r (256);
            auto windowPeak = [&] (double seconds)
            {
                double peak = 0.0;
                for (long long k = 0; k < (long long) (seconds * rate) / 256; ++k)
                {
                    std::fill (l.begin(), l.end(), 0.0f);
                    std::fill (r.begin(), r.end(), 0.0f);
                    reverb.process (l.data(), r.data(), 256);
                    for (auto v : l)
                        peak = std::max (peak, (double) std::abs (v));
                }
                return peak;
            };
            const auto w1 = windowPeak (5.0), w2 = windowPeak (5.0), w3 = windowPeak (5.0);
            const auto ok = ! stats.bad && stats.peak < 2.01 && w2 < w1 && w3 < w2;
            std::printf ("%-8s %5.1f min  peak %+6.1f dB  tail windows %.1f / %.1f / %.1f dB  %s\n",
                         algorithmNames[a], minutes, db (stats.peak), db (w1), db (w2), db (w3), ok ? "PASS" : "FAIL");
            if (! ok)
                ++failures;
        }

        std::printf ("-- extreme corners at every rate, block 1 and 4096\n");
        int corners = 0;
        for (const auto rate : { 44100.0, 48000.0, 96000.0, 192000.0 })
            for (int a = 0; a < numAlgorithms; ++a)
                for (int c = 0; c < numCharacters; ++c)
                    for (int bits = 0; bits < 16; ++bits)
                    {
                        Params p;
                        p.timeSeconds = (bits & 1) != 0 ? 20.0f : 0.2f;
                        p.damping = (bits & 2) != 0 ? 1.0f : 0.0f;
                        p.modDepth = (bits & 4) != 0 ? 1.0f : 0.0f;
                        p.lowMultiplier = (bits & 8) != 0 ? 2.5f : 0.25f;
                        p.midMultiplier = (bits & 8) != 0 ? 0.25f : 2.5f;
                        p.crossoverHz = (bits & 1) != 0 ? 2000.0f : 100.0f;
                        p.predelayMs = (bits & 2) != 0 ? 250.0f : 0.0f;
                        p.mix = 1.0f;
                        p.algorithm = a;
                        p.character = c;

                        Reverb reverb;
                        reverb.prepare (rate, p);
                        std::mt19937 rng (9u);
                        std::uniform_real_distribution<float> noise (-1.0f, 1.0f);
                        const auto block = (bits % 2) == 0 ? 1 : 4096;
                        const auto stats = run (reverb, 0.6, rate, block,
                            [&] (long long n, float& l, float& r)
                            {
                                // 含过载、NaN 和 Inf 的输入。
                                if (n == 1000)                  { l = std::numeric_limits<float>::quiet_NaN(); r = std::numeric_limits<float>::infinity(); }
                                else if (n < (long long) (0.2 * rate)) { l = 8.0f * noise (rng); r = 8.0f * noise (rng); }
                                else                            { l = r = 0.0f; }
                            },
                            [] (long long) {});

                        ++corners;
                        // 上限：干声（干湿 100 % 时为 0）加湿声软限的 +6 dB。
                        if (stats.bad || stats.peak > 2.01)
                        {
                            std::printf ("FAIL %s %s rate %.0f bits %d peak %.2f bad %d\n", algorithmNames[a], characterNames[c], rate, bits, stats.peak, (int) stats.bad);
                            ++failures;
                        }
                    }
        std::printf ("%d corners, failures so far %d\n", corners, failures);
        return failures;
    }

    //==========================================================================
    // 三阶差分相当于高通，稳定的低频音经混响后几乎没有能量，跳变会变成尖峰。
    struct ClickMeter
    {
        float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
        double peak = 0.0;
        int primed = 0;

        void add (float x)
        {
            const auto d = (double) x - 3.0 * (double) x1 + 3.0 * (double) x2 - (double) x3;
            if (primed >= 3)
                peak = std::max (peak, std::abs (d));
            else
                ++primed;
            x3 = x2;
            x2 = x1;
            x1 = x;
        }
    };

    struct ClickCase
    {
        const char* name;
        std::function<void (Params&, int)> apply;
    };

    int testClicks()
    {
        DenormalGuard guard;
        auto failures = 0;
        const double rate = 48000.0;

        const std::vector<ClickCase> cases {
            { "time sweep 0.2<->20 s", [] (Params& p, int step) { p.timeSeconds = (step & 1) != 0 ? 20.0f : 0.2f; } },
            { "algorithm cycling",     [] (Params& p, int step) { p.algorithm = step % numAlgorithms; } },
            { "algorithm rapid A/B",   [] (Params& p, int step) { p.algorithm = step % 2; } },
            { "character cycling",     [] (Params& p, int step) { p.character = step % numCharacters; } },
            { "predelay 0<->250 ms",   [] (Params& p, int step) { p.predelayMs = (step & 1) != 0 ? 250.0f : 0.0f; } },
            { "mix 0<->100 %",         [] (Params& p, int step) { p.mix = (step & 1) != 0 ? 1.0f : 0.0f; } },
            { "damping 0<->100 %",     [] (Params& p, int step) { p.damping = (step & 1) != 0 ? 1.0f : 0.0f; } },
            { "low/mid decay flip",    [] (Params& p, int step) { p.lowMultiplier = (step & 1) != 0 ? 2.5f : 0.25f; p.midMultiplier = (step & 1) != 0 ? 0.25f : 2.5f; } },
            { "mod depth 0<->100 %",   [] (Params& p, int step) { p.modDepth = (step & 1) != 0 ? 1.0f : 0.0f; } },
            { "crossover 100<->2000",  [] (Params& p, int step) { p.lowMultiplier = 2.5f; p.crossoverHz = (step & 1) != 0 ? 2000.0f : 100.0f; } },
        };

        std::printf ("-- 3rd-difference peak with a 220 Hz tone at -12 dB, dB re output RMS; PASS below -40 dB, or within 6 dB of the steady baseline (the 12 bit quantiser has its own HF floor)\n");
        std::printf ("%-24s %-8s %-7s %10s %10s\n", "case", "algo", "char", "steady dB", "change dB");

        for (const auto& testCase : cases)
            for (int a = 0; a < numAlgorithms; ++a)
                for (const auto c : { vintage24, modern, vintage12 })
                {
                    Params p;
                    p.mix = 0.5f;
                    p.timeSeconds = 2.2f;
                    p.algorithm = a;
                    p.character = c;

                    Reverb reverb;
                    reverb.prepare (rate, p);

                    const auto tone = [&] (long long n, float& l, float& r)
                    {
                        l = r = 0.25f * (float) std::sin (2.0 * 3.141592653589793 * 220.0 * (double) n / rate);
                    };

                    // 先跑稳，再测 2 秒稳态基线，再每 150 ms 改一次参数。
                    run (reverb, 2.0, rate, 256, tone, [] (long long) {});

                    std::vector<float> l (256), r (256);
                    long long pos = (long long) (2.0 * rate);
                    ClickMeter steady, change;
                    double level = 0.0;
                    long long count = 0;

                    for (long long k = 0; k < (long long) (2.0 * rate) / 256; ++k)
                    {
                        for (int i = 0; i < 256; ++i)
                            tone (pos + i, l[(size_t) i], r[(size_t) i]);
                        reverb.process (l.data(), r.data(), 256);
                        for (auto v : l)
                        {
                            steady.add (v);
                            level += (double) v * (double) v;
                            ++count;
                        }
                        pos += 256;
                    }

                    auto base = p;
                    for (int step = 1; step <= 24; ++step)
                    {
                        testCase.apply (base, step);
                        reverb.setParams (base);
                        for (long long k = 0; k < (long long) (0.15 * rate) / 256; ++k)
                        {
                            for (int i = 0; i < 256; ++i)
                                tone (pos + i, l[(size_t) i], r[(size_t) i]);
                            reverb.process (l.data(), r.data(), 256);
                            for (auto v : l)
                                change.add (v);
                            pos += 256;
                        }
                    }

                    const auto rms = std::sqrt (level / (double) std::max (count, 1LL));
                    const auto relative = db (change.peak) - db (rms);
                    const auto baseline = db (steady.peak) - db (rms);
                    const auto ok = relative < std::max (-40.0, baseline + 6.0);
                    if (! ok || a == 0)
                        std::printf ("%-24s %-8s %-7s %10.1f %10.1f  %s\n", testCase.name, algorithmNames[a], characterNames[c],
                                     db (steady.peak) - db (rms), relative, ok ? "PASS" : "FAIL");
                    if (! ok)
                        ++failures;
                }

        std::printf ("click failures: %d\n", failures);
        return failures;
    }

    //==========================================================================
    int testBlocks()
    {
        DenormalGuard guard;
        auto failures = 0;
        const double rate = 48000.0;

        for (int a = 0; a < numAlgorithms; ++a)
            for (int c = 0; c < numCharacters; ++c)
            {
                Params p;
                p.mix = 0.5f;
                p.predelayMs = 12.0f;
                p.algorithm = a;
                p.character = c;

                std::mt19937 rng (5u);
                std::uniform_real_distribution<float> noise (-0.5f, 0.5f);
                std::vector<float> inputL ((size_t) (rate * 1.5)), inputR (inputL.size());
                for (size_t i = 0; i < (size_t) (rate * 0.2); ++i)
                {
                    inputL[i] = noise (rng);
                    inputR[i] = noise (rng);
                }

                const auto render = [&] (int block, bool randomBlocks)
                {
                    Reverb reverb;
                    reverb.prepare (rate, p);
                    auto l = inputL, r = inputR;
                    std::mt19937 sizes (77u);
                    size_t pos = 0;
                    while (pos < l.size())
                    {
                        const auto wanted = randomBlocks ? (int) (sizes() % 700u) + 1 : block;
                        const auto n = std::min ((size_t) wanted, l.size() - pos);
                        reverb.process (l.data() + pos, r.data() + pos, (int) n);
                        pos += n;
                    }
                    return std::make_pair (l, r);
                };

                const auto reference = render (256, false);
                for (const auto block : { 1, 7, 32, 1024, 4096, -1 })
                {
                    const auto other = render (block, block < 0);
                    double maxDiff = 0.0;
                    for (size_t i = 0; i < reference.first.size(); ++i)
                    {
                        maxDiff = std::max (maxDiff, (double) std::abs (reference.first[i] - other.first[i]));
                        maxDiff = std::max (maxDiff, (double) std::abs (reference.second[i] - other.second[i]));
                    }
                    if (maxDiff != 0.0)
                    {
                        std::printf ("FAIL %s %s block %d max diff %.3e\n", algorithmNames[a], characterNames[c], block, maxDiff);
                        ++failures;
                    }
                }
            }

        std::printf ("block-size invariance failures: %d\n", failures);

        // 单声道路径：输出必须等于立体声左声道喂相同信号时的结果。
        {
            Params p;
            p.mix = 0.5f;
            Reverb mono, stereo;
            mono.prepare (rate, p);
            stereo.prepare (rate, p);
            std::vector<float> m (48000), l (48000), r (48000);
            std::mt19937 rng (3u);
            std::uniform_real_distribution<float> noise (-0.5f, 0.5f);
            for (size_t i = 0; i < 4800; ++i)
                m[i] = l[i] = r[i] = noise (rng);
            mono.process (m.data(), nullptr, 48000);
            stereo.process (l.data(), r.data(), 48000);
            double maxDiff = 0.0;
            for (size_t i = 0; i < m.size(); ++i)
                maxDiff = std::max (maxDiff, (double) std::abs (m[i] - l[i]));
            std::printf ("mono vs stereo-left max diff %.3e  %s\n", maxDiff, maxDiff == 0.0 ? "PASS" : "FAIL");
            if (maxDiff != 0.0)
                ++failures;
        }

        // 干声零延迟：干湿 0 % 时输出与输入逐采样完全相同。
        {
            Params p;
            p.mix = 0.0f;
            Reverb reverb;
            reverb.prepare (rate, p);
            std::vector<float> l (48000), r (48000), dry (48000);
            std::mt19937 rng (4u);
            std::uniform_real_distribution<float> noise (-1.0f, 1.0f);
            for (size_t i = 0; i < dry.size(); ++i)
                dry[i] = l[i] = r[i] = noise (rng);
            reverb.process (l.data(), r.data(), 48000);
            auto same = true;
            for (size_t i = 0; i < dry.size(); ++i)
                same = same && dry[i] == l[i] && dry[i] == r[i];
            std::printf ("mix 0 %% bit-exact dry path: %s\n", same ? "PASS" : "FAIL");
            if (! same)
                ++failures;
        }
        return failures;
    }

    //==========================================================================
    int testCpu()
    {
        DenormalGuard guard;
        std::printf ("-- single instance, stereo, tail active, one thread; percent of one core\n");
        std::printf ("%-8s %-7s %8s %8s %8s %8s\n", "algo", "char", "44.1k/64", "48k/256", "96k/256", "192k/256");
        for (int a = 0; a < numAlgorithms; ++a)
            for (int c = 0; c < numCharacters; ++c)
            {
                std::printf ("%-8s %-7s", algorithmNames[a], characterNames[c]);
                const std::array<std::pair<double, int>, 4> settings { { { 44100.0, 64 }, { 48000.0, 256 }, { 96000.0, 256 }, { 192000.0, 256 } } };
                for (const auto& setting : settings)
                {
                    const auto rate = setting.first;
                    const auto block = setting.second;
                    Params p;
                    p.mix = 0.5f;
                    p.algorithm = a;
                    p.character = c;
                    Reverb reverb;
                    reverb.prepare (rate, p);
                    std::mt19937 rng (6u);
                    std::uniform_real_distribution<float> noise (-0.5f, 0.5f);
                    std::vector<float> sourceL ((size_t) block * 200), sourceR (sourceL.size());
                    for (auto& v : sourceL) v = noise (rng);
                    for (auto& v : sourceR) v = noise (rng);
                    std::vector<float> l ((size_t) block), r ((size_t) block);

                    const auto seconds = 3.0;
                    const auto blocks = (long long) (seconds * rate) / block;
                    const auto start = std::chrono::steady_clock::now();
                    for (long long k = 0; k < blocks; ++k)
                    {
                        const auto offset = (size_t) (k % 200) * (size_t) block;
                        std::memcpy (l.data(), sourceL.data() + offset, (size_t) block * sizeof (float));
                        std::memcpy (r.data(), sourceR.data() + offset, (size_t) block * sizeof (float));
                        reverb.process (l.data(), r.data(), block);
                    }
                    const auto elapsed = std::chrono::duration<double> (std::chrono::steady_clock::now() - start).count();
                    std::printf (" %7.2f%%", 100.0 * elapsed / seconds);
                }
                std::printf ("\n");
            }
        return 0;
    }

    //==========================================================================
    int testSilence()
    {
        auto failures = 0;
        const double rate = 48000.0;
        for (int a = 0; a < numAlgorithms; ++a)
            for (int c = 0; c < numCharacters; ++c)
            {
                DenormalGuard guard;
                Params p;
                p.mix = 1.0f;
                p.timeSeconds = 0.5f;
                p.algorithm = a;
                p.character = c;
                Reverb reverb;
                reverb.prepare (rate, p);

                std::vector<float> l (256), r (256);
                std::mt19937 rng (8u);
                std::uniform_real_distribution<float> noise (-0.5f, 0.5f);
                for (auto& v : l) v = noise (rng);
                for (auto& v : r) v = noise (rng);
                reverb.process (l.data(), r.data(), 256);

                Stats stats;
                double lastPeak = 0.0;
                const auto start = std::chrono::steady_clock::now();
                const auto seconds = 30.0;
                for (long long k = 0; k < (long long) (seconds * rate) / 256; ++k)
                {
                    std::fill (l.begin(), l.end(), 0.0f);
                    std::fill (r.begin(), r.end(), 0.0f);
                    reverb.process (l.data(), r.data(), 256);
                    double peak = 0.0;
                    for (size_t i = 0; i < 256; ++i)
                    {
                        stats.add (l[i]);
                        stats.add (r[i]);
                        peak = std::max ({ peak, (double) std::abs (l[i]), (double) std::abs (r[i]) });
                    }
                    lastPeak = peak;
                }
                const auto elapsed = std::chrono::duration<double> (std::chrono::steady_clock::now() - start).count();
                const auto ok = ! stats.bad && stats.subnormal == 0 && lastPeak < 1.0e-6;
                std::printf ("%-8s %-7s subnormal %lld  last-block peak %.1f dB  idle CPU %.3f%%  %s\n", algorithmNames[a], characterNames[c],
                             stats.subnormal, db (lastPeak), 100.0 * elapsed / seconds, ok ? "PASS" : "FAIL");
                if (! ok)
                    ++failures;
            }
        return failures;
    }

    //==========================================================================
    int testConverter()
    {
        DenormalGuard guard;
        const double rate = 48000.0;
        const auto tau = 2.0 * 3.141592653589793;

        std::printf ("-- one converter stage: SINAD of a 1 kHz sine at each level, dB; magnitude response at each frequency, dB\n");
        std::printf ("%-8s", "char");
        for (const auto level : { -6, -20, -40, -60, -80 })
            std::printf (" %5ddB", level);
        for (const auto hz : { 5000, 8000, 10000, 12000, 15000, 18000, 20000 })
            std::printf (" %6dHz", hz);
        std::printf ("\n");

        for (int c = 0; c < numCharacters; ++c)
        {
            std::printf ("%-8s", characterNames[c]);

            const auto measure = [&] (double hz, double amplitude, double& gainDb, double& sinadDb)
            {
                Converter converter;
                converter.prepare ((Character) c, rate);
                std::vector<float> l (96000), r (96000);
                for (size_t i = 0; i < l.size(); ++i)
                    l[i] = r[i] = (float) (amplitude * std::sin (tau * hz * (double) i / rate));
                std::array<float, 32> a, b2;
                for (size_t i = 0; i + 32 <= l.size(); i += 32)
                {
                    std::copy_n (l.begin() + (long) i, 32, a.begin());
                    std::copy_n (r.begin() + (long) i, 32, b2.begin());
                    converter.process (a.data(), b2.data(), 32);
                    std::copy_n (a.begin(), 32, l.begin() + (long) i);
                }

                // 最小二乘拟合基波，余下的就是噪声和失真。跳过滤波器的建立段。
                double ss = 0.0, cc = 0.0, sc = 0.0, sy = 0.0, cy = 0.0, yy = 0.0;
                for (size_t i = 48000; i < 96000 - 32; ++i)
                {
                    const auto sn = std::sin (tau * hz * (double) i / rate), cs = std::cos (tau * hz * (double) i / rate);
                    const auto y = (double) l[i];
                    ss += sn * sn; cc += cs * cs; sc += sn * cs; sy += sn * y; cy += cs * y; yy += y * y;
                }
                const auto det = ss * cc - sc * sc;
                const auto ca = (sy * cc - cy * sc) / det, cb = (cy * ss - sy * sc) / det;
                const auto fitted = ca * sy + cb * cy;
                const auto residual = std::max (yy - fitted, 1.0e-30);
                gainDb = 10.0 * std::log10 (std::max (fitted, 1.0e-30) / (0.5 * amplitude * amplitude * 47968.0));
                sinadDb = 10.0 * std::log10 (std::max (fitted, 1.0e-30) / residual);
            };

            for (const auto level : { -6, -20, -40, -60, -80 })
            {
                double gain = 0.0, sinad = 0.0;
                measure (1000.0, std::pow (10.0, level / 20.0), gain, sinad);
                std::printf (" %7.1f", sinad);
            }
            for (const auto hz : { 5000, 8000, 10000, 12000, 15000, 18000, 20000 })
            {
                double gain = 0.0, sinad = 0.0;
                measure ((double) hz, 0.5, gain, sinad);
                std::printf (" %8.1f", gain);
            }
            std::printf ("\n");
        }
        return 0;
    }
}

int main (int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "stability")
        return testStability (argc > 2 ? std::atof (argv[2]) : 10.0) == 0 ? 0 : 1;
    if (mode == "clicks")
        return testClicks() == 0 ? 0 : 1;
    if (mode == "blocks")
        return testBlocks() == 0 ? 0 : 1;
    if (mode == "cpu")
        return testCpu();
    if (mode == "silence")
        return testSilence() == 0 ? 0 : 1;
    if (mode == "converter")
        return testConverter();

    std::printf ("usage: elysium_dsp_bench stability|clicks|blocks|cpu|silence|converter\n");
    return 2;
}
