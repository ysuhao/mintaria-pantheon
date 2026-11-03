#pragma once

#include <algorithm>
#include <array>
#include <cmath>

// 自动推子的检测与增益计算。逐采样运行，结果不依赖缓冲区大小；不分配内存、不加锁。
namespace helios
{

// 电平刻度：RMS，满幅正弦读 0 dB（峰值 -18 dB 的正弦读 -18 dB）。
inline constexpr float rmsCalibrationDb = 3.0103f;

// 检测里可能走低频长时间常数，用 double 存状态。
struct Biquad
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0, z1 = 0.0, z2 = 0.0;

    void reset() noexcept     { z1 = z2 = 0.0; }

    double process (double x) noexcept
    {
        const auto y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

    void setLowpass (double rate, double freq, double q) noexcept
    {
        const auto w = 2.0 * 3.14159265358979323846 * freq / rate;
        const auto alpha = std::sin (w) / (2.0 * q);
        const auto c = std::cos (w);
        const auto a0 = 1.0 + alpha;
        b0 = (1.0 - c) * 0.5 / a0;
        b1 = (1.0 - c) / a0;
        b2 = b0;
        a1 = -2.0 * c / a0;
        a2 = (1.0 - alpha) / a0;
    }

    void setHighpass (double rate, double freq, double q) noexcept
    {
        const auto w = 2.0 * 3.14159265358979323846 * freq / rate;
        const auto alpha = std::sin (w) / (2.0 * q);
        const auto c = std::cos (w);
        const auto a0 = 1.0 + alpha;
        b0 = (1.0 + c) * 0.5 / a0;
        b1 = -(1.0 + c) / a0;
        b2 = b0;
        a1 = -2.0 * c / a0;
        a2 = (1.0 - alpha) / a0;
    }
};

// 周期性检测：把人声带内的信号降到约 4 kHz，每 5 ms 算一次 40 ms 窗的归一化自相关峰值。
// 浊音接近 1，呼吸、齿音、底噪明显偏低。只影响判断，音频通路不因此增加延迟。
class VoicingDetector
{
public:
    void prepare (double rate) noexcept
    {
        decimation = std::max (1, (int) std::lround (rate / 4000.0));
        const auto lowRate = rate / decimation;
        windowLength = std::min ((int) std::lround (0.040 * lowRate), maxWindow);
        minLag = std::max (2, (int) std::lround (lowRate / 800.0));
        maxLag = std::min ((int) std::lround (lowRate / 60.0), maxLagLimit);
        hop = std::max (1, (int) std::lround (0.005 * lowRate));
        lowpassA.setLowpass (rate, 1000.0, 0.54119610);
        lowpassB.setLowpass (rate, 1000.0, 1.30656296);
        reset();
    }

    void reset() noexcept
    {
        lowpassA.reset();
        lowpassB.reset();
        ring.fill (0.0f);
        writePos = 0;
        decimationCount = 0;
        hopCount = 0;
        periodicity = 0.0f;
    }

    // 返回 true 表示本采样产生了新的周期性结果。fullBandPower 是同一信号未滤波的平均功率。
    bool push (double sample, bool worthAnalysing, double fullBandPower) noexcept
    {
        const auto low = lowpassB.process (lowpassA.process (sample));
        if (++decimationCount < decimation)
            return false;
        decimationCount = 0;

        ring[(size_t) writePos] = (float) low;
        writePos = (writePos + 1) & ringMask;

        if (++hopCount < hop)
            return false;
        hopCount = 0;

        periodicity = worthAnalysing ? analyse (fullBandPower) : 0.0f;
        return true;
    }

    float getPeriodicity() const noexcept     { return periodicity; }

private:
    float analyse (double fullBandPower) const noexcept
    {
        std::array<float, ringSize> frame;
        const auto total = windowLength + maxLag;
        auto read = (writePos - total) & ringMask;
        for (int i = 0; i < total; ++i)
        {
            frame[(size_t) i] = ring[(size_t) read];
            read = (read + 1) & ringMask;
        }

        const auto* current = frame.data() + maxLag;
        float energyNow = 0.0f;
        for (int i = 0; i < windowLength; ++i)
            energyNow += current[i] * current[i];

        // 齿音经过降采样后只剩很小的窄带残余，看起来也有周期性，所以先要求人声带内有足够能量。
        if (energyNow < 1.0e-12f || (double) energyNow < minLowBandShare * fullBandPower * windowLength)
            return 0.0f;

        auto best = 0.0f;
        for (int lag = minLag; lag <= maxLag; ++lag)
        {
            const auto* past = current - lag;
            float cross = 0.0f, energyPast = 0.0f;
            for (int i = 0; i < windowLength; ++i)
            {
                cross += current[i] * past[i];
                energyPast += past[i] * past[i];
            }
            if (cross > 0.0f && energyPast > 1.0e-12f)
                best = std::max (best, cross / std::sqrt (energyNow * energyPast));
        }
        return best;
    }

    static constexpr int ringSize = 512, ringMask = ringSize - 1, maxWindow = 200, maxLagLimit = 80;
    static constexpr double minLowBandShare = 0.05;     // -13 dB

    Biquad lowpassA, lowpassB;
    std::array<float, ringSize> ring {};
    int writePos = 0, decimation = 12, decimationCount = 0, hop = 20, hopCount = 0;
    int windowLength = 160, minLag = 5, maxLag = 67;
    float periodicity = 0.0f;
};

struct RiderSettings
{
    float targetDb = -18.0f;
    float rangeUpDb = 6.0f, rangeDownDb = 6.0f;
    float speed = 0.5f;         // 0 慢推子，1 快修正
    float gateAmount = 0.5f;    // 0 只防静音，1 最严
    float follow = 1.0f;        // 伴奏跟随量
};

class RiderEngine
{
public:
    // 伴奏为 -18 dB 时目标不动。伴奏每高 1 dB，目标抬高 1 dB × 跟随量。
    static constexpr float musicReferenceDb = -18.0f, musicFloorDb = -60.0f, maxMusicShiftDb = 12.0f;

    void prepare (double newRate) noexcept
    {
        rate = newRate;
        hpLeft.setHighpass (rate, 60.0, 0.7071);
        hpRight = hpLeft;
        voicing.prepare (rate);

        fastCoef = coefficient (5.0f);
        mediumCoef = coefficient (30.0f);
        slowCoef = coefficient (60.0f);
        refDecayPerSample = std::pow (10.0, -refDecayDbPerSecond / (10.0 * rate));
        targetCoef = coefficient (30.0f);
        musicPowerCoef = coefficient (300.0f);
        musicPresenceCoef = coefficient (50.0f);
        musicFloorPower = powerFromDb (musicFloorDb);
        musicShiftCoef = coefficient (500.0f);
        hangFrames = 6;

        lastSpeed = lastGate = -1.0f;
        applySettings (settings);
        reset();
    }

    void reset() noexcept
    {
        hpLeft.reset();
        hpRight.reset();
        voicing.reset();
        fast = medium = slow = midPower = reference = 0.0;
        allA = allB = voicedA = voicedB = 0.0;
        seeded = false;
        voiced = false;
        hang = 0;
        confirmed = 0;
        gateOpen = false;
        gainA = gainB = 0.0;
        musicA = musicB = musicPresence = 0.0;
        musicShift = 0.0;
        musicDb = -120.0f;
        smoothedTarget = settings.targetDb;
    }

    void applySettings (const RiderSettings& s) noexcept
    {
        settings = s;

        if (s.speed != lastSpeed)
        {
            lastSpeed = s.speed;
            // 响应时间常数按对数分布，一头是慢推子，一头是快修正。
            const auto tau = maxTauMs * std::pow (minTauMs / maxTauMs, s.speed);
            // 检测必须上下对称，否则功率的纹波会把读数往峰值拉。快下慢上的手感放在增益级。
            detectCoef = coefficient (std::max (0.4f * tau, 4.0f));
            gainDownCoef = coefficient (std::max (0.2f * tau, 1.0f));
            gainUpCoef = coefficient (std::max (0.2f * tau, 12.0f));
        }

        if (s.gateAmount != lastGate)
        {
            lastGate = s.gateAmount;
            gateVoicingEnabled = s.gateAmount > 0.001f;
            floorPower = powerFromDb (-70.0f + 40.0f * s.gateAmount);
            relativeRatio = (double) std::pow (10.0f, -(36.0f - 24.0f * s.gateAmount) * 0.1f);
            // 噪声类信号的峰值实测不超过 0.51，真实浊音多在 0.7 以上（数据见 NOTES.md）。
            periodicityThreshold = 0.35f + 0.5f * s.gateAmount;
        }
    }

    // sidechainPower 是当前采样伴奏的平均通道功率。没接时传 0。
    float processFrame (float left, float right, double sidechainPower) noexcept
    {
        if (! std::isfinite (left))  left = 0.0f;
        if (! std::isfinite (right)) right = 0.0f;

        const auto hl = hpLeft.process (left);
        const auto hr = hpRight.process (right);
        const auto power = 0.5 * (hl * hl + hr * hr);

        fast += fastCoef * (power - fast);
        medium += mediumCoef * (power - medium);
        slow += slowCoef * (power - slow);

        const auto mid = 0.5 * (hl + hr);
        midPower += mediumCoef * (mid * mid - midPower);
        updateVoicing (mid, hl);

        reference *= refDecayPerSample;
        if (voiced && fast > reference)
            reference = fast;

        gateOpen = fast >= floorPower;
        if (gateVoicingEnabled)
            gateOpen = gateOpen && voiced && fast >= reference * relativeRatio && fast >= slow * dropRatio;

        // 全程包络只用来防止响的非人声（爆破、齿音）冲出去；推子只跟随门开时的人声包络。
        follow (allA, allB, power);
        if (gateOpen)
        {
            if (! seeded)
            {
                voicedA = voicedB = fast;
                seeded = true;
            }
            follow (voicedA, voicedB, power);
        }

        updateMusic (sidechainPower);
        smoothedTarget += targetCoef * ((double) settings.targetDb + musicShift - smoothedTarget);

        // 门关时目标就是当前增益，也就是保持不动。只有响的非人声才允许往下走。
        auto desired = gainB;
        if (gateOpen)
        {
            desired = smoothedTarget - (double) levelDb (voicedB);

            // 句首包络还没跟上时不要往上推：瞬时电平已经够了，再推就是过冲。
            if (desired > gainB && (double) levelDb (fast) + gainB > smoothedTarget + onsetMarginDb)
                desired = gainB;
        }
        else if (gateVoicingEnabled)
        {
            const auto excess = (double) levelDb (allB) + gainB - (smoothedTarget + loudNonVoiceMarginDb);
            if (excess > 0.0)
                desired = gainB - excess;
        }
        desired = std::max ((double) -settings.rangeDownDb, std::min ((double) settings.rangeUpDb, desired));

        gainA += (desired < gainA ? gainDownCoef : gainUpCoef) * (desired - gainA);
        gainB += (gainA < gainB ? gainDownCoef : gainUpCoef) * (gainA - gainB);

        return std::exp ((float) gainB * dbToNeper);
    }

    float getGainDb() const noexcept            { return (float) gainB; }
    bool isGateOpen() const noexcept            { return gateOpen; }
    float getEffectiveTargetDb() const noexcept { return (float) smoothedTarget; }
    float getMusicDb() const noexcept           { return musicDb; }

private:
    static constexpr float maxTauMs = 2000.0f, minTauMs = 5.0f;
    static constexpr float dbToNeper = 0.11512925f;
    static constexpr double refDecayDbPerSecond = 2.0;
    static constexpr double dropRatio = 0.2511886;      // 比 60 ms 包络低 6 dB 视为句尾正在落下
    static constexpr double onsetMarginDb = 1.5;
    static constexpr float loudNonVoiceMarginDb = 6.0f;

    template <typename T>
    static T powerFloor (T p) noexcept          { return std::max (p, (T) 1.0e-14); }

    static float levelDb (double power) noexcept
    {
        return 10.0f * std::log10 ((float) powerFloor (power)) + rmsCalibrationDb;
    }

    static double powerFromDb (float db) noexcept
    {
        return (double) std::pow (10.0f, (db - rmsCalibrationDb) * 0.1f);
    }

    double coefficient (float ms) const noexcept
    {
        return 1.0 - std::exp (-1.0 / (0.001 * (double) ms * rate));
    }

    void follow (double& a, double& b, double power) const noexcept
    {
        a += detectCoef * (power - a);
        b += detectCoef * (a - b);
    }

    void updateVoicing (double mid, double side) noexcept
    {
        // 左右反相时中置为零，改看单通道，避免把反相的人声判成静音。
        const auto input = midPower * 100.0 < medium ? side : mid;
        if (! voicing.push (input, fast >= floorPower * 0.25, medium))
            return;

        // 噪声刚开始时分析窗还没填满，自相关偶尔会很高。连续 4 帧（20 ms）都有周期性才算人声。
        if (voicing.getPeriodicity() >= periodicityThreshold)
        {
            if (++confirmed >= confirmFrames)
                hang = hangFrames;
        }
        else
        {
            confirmed = 0;
            if (hang > 0)
                --hang;
        }
        voiced = hang > 0;
    }

    void updateMusic (double sidechainPower) noexcept
    {
        if (! std::isfinite (sidechainPower))
            sidechainPower = 0.0;

        // 响度读数按 300 ms 平滑，不跟鼓点。是否还有伴奏另用 50 ms 判断，
        // 否则伴奏停止时读数要几秒才落到底，目标会先被拖低再回来。
        musicPresence += musicPresenceCoef * (sidechainPower - musicPresence);
        auto wanted = 0.0f;
        if (musicPresence < musicFloorPower)
        {
            musicA = musicB = 0.0;
            musicDb = -120.0f;
        }
        else
        {
            musicA += musicPowerCoef * (sidechainPower - musicA);
            musicB += musicPowerCoef * (musicA - musicB);
            musicDb = levelDb (musicB);
            wanted = settings.follow * std::max (-maxMusicShiftDb, std::min (maxMusicShiftDb, musicDb - musicReferenceDb));
        }
        musicShift += musicShiftCoef * ((double) wanted - musicShift);
    }

    double rate = 48000.0;
    RiderSettings settings;
    float lastSpeed = -1.0f, lastGate = -1.0f;

    Biquad hpLeft, hpRight;
    VoicingDetector voicing;

    double fastCoef = 0.0, mediumCoef = 0.0, slowCoef = 0.0, targetCoef = 0.0, musicPowerCoef = 0.0, musicPresenceCoef = 0.0, musicShiftCoef = 0.0;
    double musicFloorPower = 0.0;
    double detectCoef = 0.0, gainDownCoef = 0.0, gainUpCoef = 0.0;
    double refDecayPerSample = 1.0, floorPower = 0.0, relativeRatio = 0.0;
    float periodicityThreshold = 0.5f;
    bool gateVoicingEnabled = true;
    static constexpr int confirmFrames = 4;
    int hangFrames = 6, hang = 0, confirmed = 0;

    double fast = 0.0, medium = 0.0, slow = 0.0, midPower = 0.0, reference = 0.0;
    double allA = 0.0, allB = 0.0, voicedA = 0.0, voicedB = 0.0;
    bool seeded = false, voiced = false, gateOpen = false;

    // 慢速时每采样的增量很小，增益状态用 double，否则会停在目标附近走不动。
    double gainA = 0.0, gainB = 0.0;
    double smoothedTarget = -18.0;

    double musicA = 0.0, musicB = 0.0, musicPresence = 0.0, musicShift = 0.0;
    float musicDb = -120.0f;
};

} // namespace helios
