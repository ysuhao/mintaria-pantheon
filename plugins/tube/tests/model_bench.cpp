// 把声音模型导出给 tests/verify_tube.py，用来测传输曲线本身和未做抗混叠的基线。
#include <TubeModel.h>

using namespace vesta;

#define BENCH_EXPORT extern "C" __declspec (dllexport)

// 曲线和原函数，用于检查单调性和原函数的正确性。
BENCH_EXPORT void curve (const double* w, double* y, double* first, double* second, int n, double even)
{
    const TubeCurve c (voicing());
    for (int i = 0; i < n; ++i)
    {
        const auto s = c.shape (w[i]);
        const auto r = c.evaluate (w[i]);
        y[i] = s + even * s * s;
        first[i] = r.first;
        second[i] = r.second;
    }
}

BENCH_EXPORT double even_at (double driveDb)
{
    return evenAt (voicing(), driveDb);
}

// 同一条链路，电子管级不做 ADAA，也没有后级补偿滤波器。
BENCH_EXPORT void run_naive (const double* in, double* out, int n, double rate, double driveDb, double recoverySeconds)
{
    static const CompensationTable table;
    const auto& v = voicing();
    const TubeCurve c (v);
    const auto even = evenAt (v, driveDb);
    const auto drive = v.inputScale * std::pow (10.0, driveDb / 20.0);
    const auto attack = 1.0 - std::exp (-1.0 / (v.biasAttackSeconds * rate));
    const auto recovery = 1.0 - std::exp (-1.0 / (recoverySeconds * rate));

    ChannelChain chain;
    chain.prepare (v, rate);
    chain.setDrive (v, driveDb);

    auto bias = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const auto u = drive * chain.pre (in[i]);
        const auto excess = u - v.biasThreshold;
        const auto target = excess > 0.0 ? v.biasMax * (1.0 - std::exp (-v.biasRatio * excess / v.biasMax)) : 0.0;
        bias += (target > bias ? attack : recovery) * (target - bias);

        const auto s = c.shape (u - bias);
        const auto rest = c.shape (-bias);
        out[i] = chain.post ((s + even * s * s) - (rest + even * rest * rest)) * table.gainFor (driveDb);
    }
}
