#include "LetheCore.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace lethe
{
namespace
{
    constexpr int fadeMs = 1;            // 回退的淡出，也是音频线程提前判断的距离
    constexpr double recoverMs = 10.0;   // 从干声淡回人声
    constexpr double holdOffMs = 100.0;  // 一次回退后至少保持干声这么久，避免反复切换
    constexpr double switchFadeMs = 5.0; // 切换引擎时的静音淡变
    constexpr double gainRampMs = 20.0;  // 旋钮平滑
    constexpr int zeroCrossings = 32;
    constexpr float voiceLimit = 8.0f;   // 超过 +18 dBFS 就当网络发散

    int64_t floorDiv (int64_t a, int64_t b) noexcept
    {
        const auto q = a / b;
        return (a % b != 0 && a < 0) ? q - 1 : q;
    }

    int nextPowerOfTwo (int value) noexcept
    {
        auto result = 1;
        while (result < value)
            result <<= 1;
        return result;
    }

    double besselI0 (double x) noexcept
    {
        auto sum = 1.0, term = 1.0;
        for (int k = 1; k < 32; ++k)
        {
            term *= (x * 0.5 / k) * (x * 0.5 / k);
            sum += term;
        }
        return sum;
    }

    float sanitise (float value) noexcept
    {
        return isFinite (value) ? value : 0.0f;
    }
}

void SincTable::prepare (double cutoffNormalised, int zeros)
{
    halfWidth = juce::jmin (maxHalfWidth, (int) std::ceil (zeros / cutoffNormalised));

    const auto width = (double) halfWidth;
    constexpr auto beta = 9.0;
    const auto kernel = [&] (double x)
    {
        if (std::abs (x) >= width)
            return 0.0;
        const auto ratio = x / width;
        const auto w = besselI0 (beta * std::sqrt (1.0 - ratio * ratio)) / besselI0 (beta);
        const auto arg = juce::MathConstants<double>::pi * cutoffNormalised * x;
        const auto sinc = std::abs (arg) < 1.0e-9 ? 1.0 : std::sin (arg) / arg;
        return cutoffNormalised * sinc * w;
    };

    const auto taps = halfWidth * 2;
    table.assign ((size_t) ((numPhases + 1) * taps), 0.0f);

    for (int phase = 0; phase <= numPhases; ++phase)
    {
        const auto fraction = (double) phase / numPhases;
        auto* row = table.data() + phase * taps;
        auto sum = 0.0;
        for (int t = 0; t < taps; ++t)
            sum += kernel ((double) (t - halfWidth + 1) - fraction);
        for (int t = 0; t < taps; ++t)
            row[t] = (float) (kernel ((double) (t - halfWidth + 1) - fraction) / sum);
    }
}

void SincTable::getCoefficients (double fraction, float* out) const noexcept
{
    const auto taps = halfWidth * 2;
    const auto position = fraction * numPhases;
    const auto phase = juce::jmin ((int) position, numPhases - 1);
    const auto mix = (float) (position - phase);
    const auto* a = table.data() + phase * taps;
    const auto* b = a + taps;
    for (int t = 0; t < taps; ++t)
        out[t] = a[t] + mix * (b[t] - a[t]);
}

Core::Core() : juce::Thread ("Lethe inference")
{
    wakeEvent = CreateEventW (nullptr, FALSE, FALSE, nullptr);
}

Core::~Core()
{
    release();
    CloseHandle (static_cast<HANDLE> (wakeEvent));
}

void Core::wake() noexcept
{
    SetEvent (static_cast<HANDLE> (wakeEvent));
}

int Core::computeLatency (EngineKind kind) const noexcept
{
    const auto delay = getEngineDelay (kind);
    const auto budget = getEngineBudget (kind);

    if (! resampling)
    {
        const auto cycles = (budget + maxBlock - 1) / maxBlock;
        return delay + hopSize + cycles * maxBlock + fadeSamples48;
    }

    const auto ratio = (double) hostRate / engineRate;
    const auto block48 = (int) std::ceil (maxBlock / ratio) + 1;
    const auto cycles = (budget + block48 - 1) / block48;
    const auto total48 = delay + hopSize + cycles * block48 + fadeSamples48 + upTable.getHalfWidth() + 1;
    return (int) std::ceil (total48 * ratio) + downTable.getHalfWidth() + 1;
}

void Core::prepare (double sampleRate, int maxBlockSize, int numChannels, EngineKind initialKind)
{
    release();

    hostRate = juce::jmax ((int64_t) 8000, (int64_t) std::llround (sampleRate));
    maxBlock = juce::jmax (1, maxBlockSize);
    resampling = hostRate != engineRate;

    if (channelCount != numChannels)
        for (auto& engine : engines)
            engine.reset();
    channelCount = juce::jmax (1, numChannels);

    fadeSamples48 = engineRate * fadeMs / 1000;
    fadeHost = (int) std::ceil ((double) fadeSamples48 * (double) hostRate / engineRate);

    if (resampling)
    {
        const auto cutoffHz = 0.46 * (double) juce::jmin (hostRate, (int64_t) engineRate);
        downTable.prepare (cutoffHz / (0.5 * (double) hostRate), zeroCrossings);
        upTable.prepare (cutoffHz / (0.5 * engineRate), zeroCrossings);
    }

    for (size_t i = 0; i < latency.size(); ++i)
        latency[i] = computeLatency ((EngineKind) i);

    const auto ratio = (double) hostRate / engineRate;
    const auto longest = *std::max_element (latency.begin(), latency.end());
    const auto drySize = nextPowerOfTwo (longest + maxBlock + 2 * downTable.getHalfWidth() + 64);
    const auto ringSize = nextPowerOfTwo ((int) (longest / ratio) + 4 * maxBlock + 8 * hopSize);
    dryMask = drySize - 1;
    ringMask = ringSize - 1;

    dryRing.assign ((size_t) channelCount, std::vector<float> ((size_t) drySize, 0.0f));
    inputRing.assign ((size_t) channelCount, std::vector<float> ((size_t) ringSize, 0.0f));
    voiceRing.assign ((size_t) channelCount, std::vector<float> ((size_t) ringSize, 0.0f));

    frameIn.assign ((size_t) channelCount, std::vector<float> ((size_t) hopSize, 0.0f));
    frameOut.assign ((size_t) channelCount, std::vector<float> ((size_t) hopSize, 0.0f));
    frameInPointers.clear();
    frameOutPointers.clear();
    for (auto& channel : frameIn)  frameInPointers.push_back (channel.data());
    for (auto& channel : frameOut) frameOutPointers.push_back (channel.data());

    hostWritten = 0;
    downIndex = downRemainder = 0;
    blend = 0.0f;
    switchGain = 1.0f;
    holdOff = 0;
    switching = false;
    pendingKind = initialKind;
    smoothedVoice = smoothedAmbience = -1.0f;
    gainStep = (float) (1.0 / (gainRampMs * 0.001 * (double) hostRate));

    inputWritten.store (0);
    readPosition.store (std::numeric_limits<int64_t>::min());
    voiceStart.store (0);
    voiceEnd.store (0);
    voiceGeneration.store (-1);
    generationStart.store (0);
    generation.store (0);
    fallingBack.store (false);

    activeKind.store (initialKind);
    activeLatency = latency[(size_t) initialKind];
    activeDelay = getEngineDelay (initialKind);
    resetOutputClock();

    engineState.store (EngineState::loading);
    startThread (juce::Thread::Priority::high);
}

void Core::release()
{
    signalThreadShouldExit();
    wake();
    stopThread (10000);
}

int Core::getLatencySamples (EngineKind kind) const noexcept
{
    return latency[(size_t) kind];
}

juce::String Core::getEngineError() const
{
    const juce::SpinLock::ScopedLockType lock (errorLock);
    return engineError;
}

void Core::resetOutputClock() noexcept
{
    const auto scaled = (hostWritten - activeLatency) * (int64_t) engineRate;
    upIndex = floorDiv (scaled, hostRate);
    upRemainder = scaled - upIndex * hostRate;
}

// 音频线程：把一块输入写进干声线，并产生相应的 48 kHz 输入。
void Core::pushInput (const float* const* channels, int numSamples) noexcept
{
    for (int ch = 0; ch < channelCount; ++ch)
    {
        auto* ring = dryRing[(size_t) ch].data();
        const auto* in = channels[ch];
        for (int i = 0; i < numSamples; ++i)
            ring[(hostWritten + i) & dryMask] = sanitise (in[i]);
    }

    const auto written = hostWritten + numSamples;
    auto produced = inputWritten.load (std::memory_order_relaxed);
    const auto before = produced / hopSize;

    if (! resampling)
    {
        for (int ch = 0; ch < channelCount; ++ch)
        {
            const auto* dry = dryRing[(size_t) ch].data();
            auto* ring = inputRing[(size_t) ch].data();
            for (int i = 0; i < numSamples; ++i)
                ring[(produced + i) & ringMask] = dry[(hostWritten + i) & dryMask];
        }
        produced += numSamples;
    }
    else
    {
        const auto half = downTable.getHalfWidth();
        const auto taps = half * 2;
        while (downIndex + half < written)
        {
            downTable.getCoefficients ((double) downRemainder / engineRate, tapBuffer.data());
            const auto first = downIndex - half + 1;
            for (int ch = 0; ch < channelCount; ++ch)
            {
                const auto* dry = dryRing[(size_t) ch].data();
                auto sum = 0.0f;
                for (int t = 0; t < taps; ++t)
                    sum += tapBuffer[(size_t) t] * dry[(first + t) & dryMask];
                inputRing[(size_t) ch][(size_t) (produced & ringMask)] = sum;
            }
            ++produced;

            downRemainder += hostRate;
            while (downRemainder >= engineRate)
            {
                downRemainder -= engineRate;
                ++downIndex;
            }
        }
    }

    inputWritten.store (produced, std::memory_order_release);
    if (produced / hopSize != before)
    {
        lastInputTicks.store (juce::Time::getHighResolutionTicks(), std::memory_order_relaxed);
        wake();
    }
}

int64_t Core::voiceIndexForOutput (int64_t outputIndex) const noexcept
{
    // 读数范围的最后一个 48 kHz 序号，含重采样抽头和提前判断的余量。
    const auto index = resampling ? floorDiv (outputIndex * engineRate, hostRate) + upTable.getHalfWidth()
                                  : outputIndex;
    return index + activeDelay + fadeSamples48;
}

void Core::waitUntilVoiceReady (int numSamples) noexcept
{
    const auto needed = voiceIndexForOutput (hostWritten + numSamples - 1 - activeLatency);
    const auto gen = generation.load (std::memory_order_relaxed);
    if (needed < generationStart.load (std::memory_order_relaxed))
        return;

    // 模型加载可能要几秒，只有引擎就绪之后才算超时。
    auto deadline = juce::Time::getMillisecondCounter() + 30000;
    auto loaded = false;
    while (juce::Time::getMillisecondCounter() < deadline)
    {
        if (voiceGeneration.load (std::memory_order_acquire) == gen && voiceEnd.load (std::memory_order_acquire) > needed)
            return;

        const auto state = engineState.load (std::memory_order_acquire);
        if (state == EngineState::failed)
            return;
        if (state == EngineState::ready && ! loaded)
        {
            loaded = true;
            deadline = juce::Time::getMillisecondCounter() + 2000;
        }
        juce::Thread::yield();
    }
}

void Core::beginKind (EngineKind kind) noexcept
{
    activeKind.store (kind, std::memory_order_relaxed);
    activeLatency = latency[(size_t) kind];
    activeDelay = getEngineDelay (kind);
    resetOutputClock();
    blend = 0.0f;

    const auto produced = inputWritten.load (std::memory_order_relaxed);
    const auto next = (produced + hopSize - 1) / hopSize * hopSize;
    generationStart.store (next, std::memory_order_relaxed);
    generation.fetch_add (1, std::memory_order_release);
    wake();
}

void Core::process (float* const* channels, int numSamples, float voiceGain, float ambienceGain,
                    EngineKind requestedKind, bool bypassed, bool waitForVoice) noexcept
{
    if (channelCount == 0)
        return;

    if (requestedKind != activeKind.load (std::memory_order_relaxed) || switching)
    {
        pendingKind = requestedKind;
        switching = true;
    }

    float* pointers[2] = {};
    for (int done = 0; done < numSamples; )
    {
        const auto chunk = juce::jmin (maxBlock, numSamples - done);
        for (int ch = 0; ch < channelCount; ++ch)
            pointers[ch] = channels[ch] + done;
        processChunk (pointers, chunk, voiceGain, ambienceGain, bypassed, waitForVoice);
        done += chunk;
    }
}

void Core::processChunk (float* const* channels, int numSamples, float voiceGain, float ambienceGain,
                         bool bypassed, bool waitForVoice) noexcept
{
    pushInput (channels, numSamples);

    if (waitForVoice && ! bypassed && ! switching)
        waitUntilVoiceReady (numSamples);

    const auto gen = generation.load (std::memory_order_relaxed);
    const auto usable = ! bypassed && voiceGeneration.load (std::memory_order_acquire) == gen;
    const auto end = usable ? voiceEnd.load (std::memory_order_acquire) : (int64_t) 0;
    const auto start = usable ? voiceStart.load (std::memory_order_acquire) : (int64_t) 0;

    const auto targetVoice = bypassed ? 1.0f : voiceGain;
    const auto targetAmbience = bypassed ? 0.0f : ambienceGain;
    if (smoothedVoice < 0.0f)
    {
        smoothedVoice = targetVoice;
        smoothedAmbience = targetAmbience;
    }

    const auto downStep = 1.0f / (float) fadeHost;
    const auto upStep = (float) (1000.0 / (recoverMs * (double) hostRate));
    const auto holdSamples = (int) (holdOffMs * 0.001 * (double) hostRate);
    const auto halfUp = resampling ? upTable.getHalfWidth() : 0;
    const auto taps = halfUp * 2;
    const auto first = hostWritten - activeLatency;

    for (int i = 0; i < numSamples; ++i)
    {
        const auto index48 = resampling ? upIndex : first + i;
        const auto lastNeeded = index48 + halfUp + activeDelay;
        const auto firstNeeded = index48 - juce::jmax (0, halfUp - 1) + activeDelay;
        const auto ready = usable && lastNeeded + fadeSamples48 < end && firstNeeded >= start;

        if (! ready)
        {
            if (blend > 0.0f)
            {
                if (holdOff == 0 && usable)
                    stats.underruns.fetch_add (1, std::memory_order_relaxed);
                holdOff = holdSamples;
            }
            blend = juce::jmax (0.0f, blend - downStep);
        }
        else if (holdOff > 0)
        {
            --holdOff;
            blend = juce::jmax (0.0f, blend - downStep);
        }
        else
        {
            blend = juce::jmin (1.0f, blend + upStep);
        }

        smoothedVoice += juce::jlimit (-gainStep, gainStep, targetVoice - smoothedVoice);
        smoothedAmbience += juce::jlimit (-gainStep, gainStep, targetAmbience - smoothedAmbience);

        if (blend > 0.0f && resampling)
            upTable.getCoefficients ((double) upRemainder / (double) hostRate, tapBuffer.data());

        for (int ch = 0; ch < channelCount; ++ch)
        {
            const auto dry = dryRing[(size_t) ch][(size_t) ((first + i) & dryMask)];
            auto voice = dry;

            if (blend > 0.0f)
            {
                const auto* ring = voiceRing[(size_t) ch].data();
                auto estimate = 0.0f;
                if (resampling)
                {
                    const auto base = index48 - halfUp + 1 + activeDelay;
                    for (int t = 0; t < taps; ++t)
                        estimate += tapBuffer[(size_t) t] * ring[(base + t) & ringMask];
                }
                else
                {
                    estimate = ring[(index48 + activeDelay) & ringMask];
                }
                voice = dry + blend * (estimate - dry);
            }

            channels[ch][i] = smoothedVoice * voice + smoothedAmbience * (dry - voice);
        }

        if (resampling)
        {
            upRemainder += engineRate;
            while (upRemainder >= hostRate)
            {
                upRemainder -= hostRate;
                ++upIndex;
            }
        }
    }

    hostWritten += numSamples;
    fallingBack.store (holdOff > 0 && ! bypassed, std::memory_order_relaxed);
    readPosition.store (voiceIndexForOutput (hostWritten - 1 - activeLatency), std::memory_order_relaxed);

    if (switching)
        applySwitchFade (channels, numSamples);
}

// 切换引擎时干声的读位置也会跳，所以先静音再跳，两边都是 5 ms。
void Core::applySwitchFade (float* const* channels, int numSamples) noexcept
{
    const auto step = (float) (1000.0 / (switchFadeMs * (double) hostRate));
    const auto wanted = pendingKind;
    auto swapped = wanted == activeKind.load (std::memory_order_relaxed);

    for (int i = 0; i < numSamples; ++i)
    {
        switchGain = swapped ? juce::jmin (1.0f, switchGain + step) : juce::jmax (0.0f, switchGain - step);
        for (int ch = 0; ch < channelCount; ++ch)
            channels[ch][i] *= switchGain;
        if (! swapped && switchGain <= 0.0f)
        {
            // 这一块剩下的采样静音，下一块开始用新延迟。
            for (int ch = 0; ch < channelCount; ++ch)
                juce::FloatVectorOperations::clear (channels[ch] + i, numSamples - i);
            beginKind (wanted);
            swapped = true;
            return;
        }
    }

    if (swapped && switchGain >= 1.0f)
        switching = false;
}

bool Core::ensureEngine (EngineKind kind)
{
    auto& slot = engines[(size_t) kind];
    if (slot == nullptr)
    {
        // 加载很慢，不占实时优先级。
        engineState.store (EngineState::loading, std::memory_order_release);
        SetThreadPriority (GetCurrentThread(), THREAD_PRIORITY_NORMAL);

        juce::String error;
        slot = createEngine (kind, channelCount, error);

        if (slot == nullptr)
        {
            const juce::SpinLock::ScopedLockType lock (errorLock);
            engineError = error;
            return false;
        }
    }

    SetThreadPriority (GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    return true;
}

// 工作线程：处理一帧，写进 voice ring。返回 false 表示这一帧不可用。
bool Core::processOneFrame (int64_t frameStart, VoiceEngine& engine)
{
    for (int ch = 0; ch < channelCount; ++ch)
    {
        const auto* ring = inputRing[(size_t) ch].data();
        auto* out = frameIn[(size_t) ch].data();
        for (int i = 0; i < hopSize; ++i)
            out[i] = ring[(frameStart + i) & ringMask];
    }

    const auto startTicks = juce::Time::getHighResolutionTicks();
    auto good = false;
    try
    {
        good = engine.process (frameInPointers.data(), frameOutPointers.data(), channelCount);
    }
    catch (const std::exception&)
    {
        good = false;
    }
    const auto micros = (int) (juce::Time::highResolutionTicksToSeconds (juce::Time::getHighResolutionTicks() - startTicks) * 1.0e6);

    stats.inferMicrosTotal.fetch_add (micros, std::memory_order_relaxed);
    if (micros > stats.inferMicrosMax.load (std::memory_order_relaxed))
        stats.inferMicrosMax.store (micros, std::memory_order_relaxed);

    if (good)
        for (int ch = 0; ch < channelCount && good; ++ch)
            for (int i = 0; i < hopSize; ++i)
            {
                const auto s = frameOut[(size_t) ch][(size_t) i];
                if (! isFinite (s) || std::abs (s) > voiceLimit)
                {
                    good = false;
                    break;
                }
            }

    if (! good)
    {
        stats.framesFailed.fetch_add (1, std::memory_order_relaxed);
        try { engine.reset(); } catch (const std::exception&) {}
        return false;
    }

    for (int ch = 0; ch < channelCount; ++ch)
    {
        auto* ring = voiceRing[(size_t) ch].data();
        const auto* in = frameOut[(size_t) ch].data();
        for (int i = 0; i < hopSize; ++i)
            ring[(frameStart + i) & ringMask] = in[i];
    }

    stats.framesProcessed.fetch_add (1, std::memory_order_relaxed);
    return true;
}

void Core::recordPublishTime() noexcept
{
    const auto now = juce::Time::getHighResolutionTicks();
    const auto ms = juce::Time::highResolutionTicksToSeconds (now - lastInputTicks.load (std::memory_order_relaxed)) * 1000.0;
    static constexpr double edges[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0 };
    size_t bucket = 0;
    while (bucket < 8 && ms >= edges[bucket])
        ++bucket;
    stats.publishHistogram[bucket].fetch_add (1, std::memory_order_relaxed);
}

void Core::run()
{
    juce::ScopedNoDenormals noDenormals;

    auto currentGeneration = -1;
    int64_t nextFrame = 0;
    VoiceEngine* engine = nullptr;
    auto broken = false;

    while (! threadShouldExit())
    {
        const auto gen = generation.load (std::memory_order_acquire);
        if (gen != currentGeneration)
        {
            const auto kind = activeKind.load (std::memory_order_relaxed);
            voiceGeneration.store (-1, std::memory_order_release);

            if (! ensureEngine (kind))
            {
                engineState.store (EngineState::failed, std::memory_order_release);
                engine = nullptr;
                currentGeneration = gen;
                WaitForSingleObject (static_cast<HANDLE> (wakeEvent), 100);
                continue;
            }

            engine = engines[(size_t) kind].get();
            engine->reset();
            engineState.store (EngineState::ready, std::memory_order_release);

            // 加载模型可能花了几百毫秒，从当前能赶上的帧开始。
            currentGeneration = gen;
            nextFrame = generationStart.load (std::memory_order_relaxed);
            voiceStart.store (nextFrame, std::memory_order_relaxed);
            voiceEnd.store (nextFrame, std::memory_order_relaxed);
            voiceGeneration.store (gen, std::memory_order_release);
            broken = false;
        }

        auto didWork = false;
        while (engine != nullptr
               && generation.load (std::memory_order_acquire) == currentGeneration
               && nextFrame + hopSize <= inputWritten.load (std::memory_order_acquire)
               && ! threadShouldExit())
        {
            didWork = true;

            // 已经赶不上输出的帧不算，直接丢，保证负载回落后能立刻追上。
            if (nextFrame + hopSize <= readPosition.load (std::memory_order_relaxed))
            {
                stats.framesDropped.fetch_add (1, std::memory_order_relaxed);
                nextFrame += hopSize;
                broken = true;
                continue;
            }

            if (const auto stall = testWorkerStallMs.exchange (0, std::memory_order_relaxed); stall > 0)
                Sleep ((DWORD) stall);

            if (! processOneFrame (nextFrame, *engine))
            {
                nextFrame += hopSize;
                broken = true;
                continue;
            }

            if (broken)
            {
                // 断开之后，分析窗要再过一帧才全是新数据，输出还要再过一个算法延迟才对应到这些数据。
                voiceStart.store (nextFrame + getEngineDelay (activeKind.load (std::memory_order_relaxed)) + hopSize,
                                  std::memory_order_release);
                broken = false;
            }

            nextFrame += hopSize;
            voiceEnd.store (nextFrame, std::memory_order_release);
            recordPublishTime();
        }

        if (! didWork)
            WaitForSingleObject (static_cast<HANDLE> (wakeEvent), 50);
    }
}

} // namespace lethe
