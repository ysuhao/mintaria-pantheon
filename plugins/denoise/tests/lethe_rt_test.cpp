// 实时压力测试：按真实时间节奏调用核心通路，统计推理耗时、丢帧、回退和样点跳变。
//
//   lethe_rt_test <in.wav> [--kind standard|low|light] [--rate 48000] [--block 256] [--seconds 20]
//                 [--channels 1|2] [--burn N] [--force-late-ms M] [--instances N] [--switch-sec S] [--out out.wav]
//
//   --burn N      再开 N 个忙等线程，模拟高负载。
//   --jitter-ms J 每个回调随机晚到 0 到 J 毫秒，下一个回调照原定时刻，模拟驱动抖动。
//   --instances N 同一进程再开 N - 1 个实例一起跑。
//   --switch-sec S 每 S 秒切一次引擎，验证切换没有跳变。
//   --engine-only 不走通路，帧网格和文件开头对齐地直接跑引擎，用来和官方实现逐样点比较。
//   --force-late-ms M 每 2 秒让工作线程迟到 M 毫秒，强制走回退，验证切换时没有跳变。
#include "LetheCore.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{
    std::string arg (int argc, wchar_t** argv, const wchar_t* key, const char* fallback)
    {
        for (int i = 1; i + 1 < argc; ++i)
            if (wcscmp (argv[i], key) == 0)
                return juce::String (argv[i + 1]).toStdString();
        return fallback;
    }

    struct Wav
    {
        std::vector<std::vector<float>> channels;
        int rate = 48000;
    };

    Wav readWav (const juce::File& file)
    {
        Wav wav;
        juce::FileInputStream in (file);
        if (! in.openedOk())
            return wav;

        char id[4];
        in.read (id, 4);
        in.readInt();
        in.read (id, 4);
        int format = 1, channels = 1, bits = 16;
        while (! in.isExhausted())
        {
            in.read (id, 4);
            const auto size = in.readInt();
            const auto next = in.getPosition() + size + (size & 1);
            if (memcmp (id, "fmt ", 4) == 0)
            {
                format = in.readShort();
                channels = in.readShort();
                wav.rate = in.readInt();
                in.readInt();
                in.readShort();
                bits = in.readShort();
            }
            else if (memcmp (id, "data", 4) == 0)
            {
                const auto frames = size / (channels * bits / 8);
                wav.channels.assign ((size_t) channels, std::vector<float> ((size_t) frames));
                for (int i = 0; i < frames; ++i)
                    for (int c = 0; c < channels; ++c)
                    {
                        float s = 0.0f;
                        if (format == 3 && bits == 32)
                            s = in.readFloat();
                        else if (bits == 16)
                            s = (float) in.readShort() / 32768.0f;
                        else if (bits == 24)
                        {
                            unsigned char b[3];
                            in.read (b, 3);
                            s = (float) ((int) ((unsigned) b[0] << 8 | (unsigned) b[1] << 16 | (unsigned) b[2] << 24) >> 8) / 8388608.0f;
                        }
                        wav.channels[(size_t) c][(size_t) i] = s;
                    }
                break;
            }
            in.setPosition (next);
        }
        return wav;
    }

    void writeWav (const juce::File& file, const std::vector<std::vector<float>>& channels, int rate)
    {
        file.deleteFile();
        juce::FileOutputStream out (file);
        const auto numChannels = (int) channels.size();
        const auto frames = (int) channels[0].size();
        const auto dataBytes = frames * numChannels * 4;
        out.write ("RIFF", 4);
        out.writeInt (36 + dataBytes);
        out.write ("WAVEfmt ", 8);
        out.writeInt (16);
        out.writeShort (3);
        out.writeShort ((short) numChannels);
        out.writeInt (rate);
        out.writeInt (rate * numChannels * 4);
        out.writeShort ((short) (numChannels * 4));
        out.writeShort (32);
        out.write ("data", 4);
        out.writeInt (dataBytes);
        for (int i = 0; i < frames; ++i)
            for (int c = 0; c < numChannels; ++c)
                out.writeFloat (channels[(size_t) c][(size_t) i]);
    }
}

int wmain (int argc, wchar_t** argv)
{
    if (argc < 2)
    {
        std::puts ("usage: lethe_rt_test <in.wav> [options]");
        return 2;
    }

    const auto rate = std::stoi (arg (argc, argv, L"--rate", "48000"));
    const auto block = std::stoi (arg (argc, argv, L"--block", "256"));
    const auto seconds = std::stod (arg (argc, argv, L"--seconds", "20"));
    const auto numChannels = std::stoi (arg (argc, argv, L"--channels", "2"));
    const auto burners = std::stoi (arg (argc, argv, L"--burn", "0"));
    const auto lateMs = std::stoi (arg (argc, argv, L"--force-late-ms", "0"));
    const auto instances = std::stoi (arg (argc, argv, L"--instances", "1"));
    const auto switchSeconds = std::stod (arg (argc, argv, L"--switch-sec", "0"));
    const auto jitterMs = std::stod (arg (argc, argv, L"--jitter-ms", "0"));
    const auto kindName = arg (argc, argv, L"--kind", "standard");
    const auto kind = kindName == "low" ? lethe::EngineKind::lowLatency : kindName == "light" ? lethe::EngineKind::light : lethe::EngineKind::standard;
    const auto outPath = arg (argc, argv, L"--out", "");

    const auto wav = readWav (juce::File (juce::String (argv[1])));
    if (wav.channels.empty())
    {
        std::puts ("FAIL cannot read input");
        return 1;
    }
    if (wav.rate != rate)
        std::printf ("note: file rate %d, running at %d without conversion\n", wav.rate, rate);

    const auto& source = wav.channels[0];

    if (arg (argc, argv, L"--engine-only", "0") != "0")
    {
        juce::String error;
        auto engine = lethe::createEngine (kind, 1, error);
        if (engine == nullptr)
        {
            std::printf ("FAIL %s\n", error.toRawUTF8());
            return 1;
        }

        const auto frames = (int) source.size() / lethe::hopSize;
        std::vector<std::vector<float>> result (1, std::vector<float> ((size_t) (frames * lethe::hopSize)));
        const auto start = juce::Time::getMillisecondCounterHiRes();
        for (int f = 0; f < frames; ++f)
        {
            const float* in = source.data() + f * lethe::hopSize;
            float* out = result[0].data() + f * lethe::hopSize;
            engine->process (&in, &out, 1);
        }
        std::printf ("%d frames, %.3f ms/frame\n", frames, (juce::Time::getMillisecondCounterHiRes() - start) / frames);
        writeWav (juce::File (juce::String (outPath)), result, wav.rate);
        return 0;
    }

    const auto total = (int) (seconds * rate);

    std::vector<std::vector<float>> output ((size_t) numChannels, std::vector<float> ((size_t) total, 0.0f));
    std::vector<std::vector<float>> input ((size_t) numChannels, std::vector<float> ((size_t) total, 0.0f));
    for (int c = 0; c < numChannels; ++c)
        for (int i = 0; i < total; ++i)
            input[(size_t) c][(size_t) i] = source[(size_t) (i % (int) source.size())];

    lethe::Core core;
    core.prepare (rate, block, numChannels, kind);

    std::vector<std::unique_ptr<lethe::Core>> extraCores;
    std::vector<std::vector<float>> extraBuffers ((size_t) numChannels, std::vector<float> ((size_t) block));
    for (int i = 1; i < instances; ++i)
    {
        extraCores.push_back (std::make_unique<lethe::Core>());
        extraCores.back()->prepare (rate, block, numChannels, kind);
    }
    const auto latency = core.getLatencySamples (kind);
    std::printf ("latency %d samples (%.2f ms), block %d, rate %d, channels %d\n", latency, latency * 1000.0 / rate, block, rate, numChannels);

    const auto loadStart = juce::Time::getMillisecondCounterHiRes();
    {
        std::vector<float> silence ((size_t) block, 0.0f);
        std::vector<float> silence2 ((size_t) block, 0.0f);
        float* pointers[2] = { silence.data(), silence2.data() };
        while (core.getEngineState() == lethe::Core::EngineState::loading && juce::Time::getMillisecondCounterHiRes() - loadStart < 30000.0)
        {
            core.process (pointers, block, 1.0f, 0.0f, kind, false, false);
            Sleep (1);
        }
    }
    if (core.getEngineState() != lethe::Core::EngineState::ready)
    {
        std::printf ("FAIL engine not ready: %s\n", core.getEngineError().toRawUTF8());
        return 1;
    }
    std::printf ("engine ready after %.0f ms\n", juce::Time::getMillisecondCounterHiRes() - loadStart);

    std::atomic<bool> stop { false };
    std::vector<std::thread> burnThreads;
    for (int i = 0; i < burners; ++i)
        burnThreads.emplace_back ([&stop] { volatile double x = 1.0; while (! stop.load (std::memory_order_relaxed)) x = std::sqrt (x + 1.0); });

    SetThreadPriority (GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    timeBeginPeriod (1);

    // 加载期间的计数不算：那时输入一直在走，帧会被判定为赶不上。
    const auto before = core.stats.framesProcessed.load();
    const auto droppedBefore = core.stats.framesDropped.load();
    const auto underrunsBefore = core.stats.underruns.load();
    core.stats.inferMicrosMax.store (0);
    const auto inferBefore = core.stats.inferMicrosTotal.load();
    auto currentKind = kind;
    auto nextSwitch = switchSeconds;

    auto processMax = 0.0, processTotal = 0.0;
    int blocks = 0;
    LARGE_INTEGER previousCall {};
    auto intervalMax = 0.0;
    std::vector<int64_t> underrunTimeline ((size_t) seconds + 1, 0);

    LARGE_INTEGER frequency, startTime;
    QueryPerformanceFrequency (&frequency);
    QueryPerformanceCounter (&startTime);
    const auto blockTicks = (double) frequency.QuadPart * block / rate;
    auto nextStall = 2.0;
    juce::Random jitterRandom (3);

    std::vector<float> scratch ((size_t) block * 2);
    for (int pos = 0; pos + block <= total; pos += block, ++blocks)
    {
        // 精确等到这一块的“设备时刻”。
        const auto jitterTicks = (double) frequency.QuadPart * jitterRandom.nextDouble() * jitterMs * 0.001;
        const auto deadline = startTime.QuadPart + (LONGLONG) (blockTicks * (blocks + 1) + jitterTicks);
        for (;;)
        {
            LARGE_INTEGER now;
            QueryPerformanceCounter (&now);
            const auto remaining = (double) (deadline - now.QuadPart) / (double) frequency.QuadPart;
            if (remaining <= 0.0)
                break;
            // 回调之间不能 Sleep：系统满载时 Sleep (1) 会过冲好几毫秒，抖动比真实声卡驱动大得多。
            YieldProcessor();
        }

        const auto t = (double) pos / rate;
        if (lateMs > 0 && t >= nextStall)
        {
            nextStall += 2.0;
            core.testWorkerStallMs.store (lateMs);
        }

        float* pointers[2];
        for (int c = 0; c < numChannels; ++c)
        {
            std::copy_n (input[(size_t) c].data() + pos, block, output[(size_t) c].data() + pos);
            pointers[c] = output[(size_t) c].data() + pos;
        }

        LARGE_INTEGER a, b;
        QueryPerformanceCounter (&a);
        if (previousCall.QuadPart != 0)
            intervalMax = std::max (intervalMax, (double) (a.QuadPart - previousCall.QuadPart) / (double) frequency.QuadPart);
        previousCall = a;
        core.process (pointers, block, 1.0f, 0.0f, currentKind, false, false);
        QueryPerformanceCounter (&b);

        for (auto& extra : extraCores)
        {
            float* extraPointers[2];
            for (int c = 0; c < numChannels; ++c)
            {
                std::copy_n (input[(size_t) c].data() + pos, block, extraBuffers[(size_t) c].data());
                extraPointers[c] = extraBuffers[(size_t) c].data();
            }
            extra->process (extraPointers, block, 1.0f, 0.0f, kind, false, false);
        }

        if (switchSeconds > 0.0 && t >= nextSwitch)
        {
            nextSwitch += switchSeconds;
            currentKind = (lethe::EngineKind) (((int) currentKind + 1) % lethe::numEngineKinds);
        }

        const auto took = (double) (b.QuadPart - a.QuadPart) / (double) frequency.QuadPart;
        processMax = std::max (processMax, took);
        processTotal += took;

        const auto second = (size_t) (pos / rate);
        if (second < underrunTimeline.size())
            underrunTimeline[second] = core.stats.underruns.load (std::memory_order_relaxed);
    }

    const auto wall = seconds;
    stop.store (true);
    for (auto& t : burnThreads) t.join();
    timeEndPeriod (1);

    const auto frames = core.stats.framesProcessed.load() - before;
    const auto dropped = core.stats.framesDropped.load() - droppedBefore;
    const auto failed = core.stats.framesFailed.load();
    const auto underruns = core.stats.underruns.load() - underrunsBefore;
    const auto inferTotal = core.stats.inferMicrosTotal.load() - inferBefore;
    const auto inferMax = core.stats.inferMicrosMax.load();
    core.release();
    extraCores.clear();

    // 样点跳变：输出的逐点差和输入的最大逐点差比较。末尾不足一块的部分没有处理，不算。
    const auto processed = blocks * block;
    auto inputJump = 0.0f, outputJump = 0.0f;
    auto peak = 0.0f;
    auto bad = 0;
    for (int c = 0; c < numChannels; ++c)
        for (int i = 1; i < processed; ++i)
        {
            inputJump = std::max (inputJump, std::abs (input[(size_t) c][(size_t) i] - input[(size_t) c][(size_t) i - 1]));
            outputJump = std::max (outputJump, std::abs (output[(size_t) c][(size_t) i] - output[(size_t) c][(size_t) i - 1]));
            peak = std::max (peak, std::abs (output[(size_t) c][(size_t) i]));
            if (! lethe::isFinite (output[(size_t) c][(size_t) i]))
                ++bad;
        }

    std::printf ("frames %lld, dropped %lld, failed %lld, underruns %lld\n", (long long) frames, (long long) dropped, (long long) failed, (long long) underruns);
    std::printf ("infer avg %.3f ms, max %.3f ms; worker share of one core %.1f%%\n",
                 frames > 0 ? (double) inferTotal / (double) frames / 1000.0 : 0.0, inferMax / 1000.0, (double) inferTotal / (wall * 1.0e6) * 100.0);
    std::printf ("audio callback avg %.1f us, max %.1f us (block period %.0f us), avg share of period %.2f%%\n",
                 processTotal / blocks * 1.0e6, processMax * 1.0e6, block * 1.0e6 / rate, processTotal / blocks / (block / (double) rate) * 100.0);
    std::printf ("callback interval max %.2f ms (nominal %.2f)\n", intervalMax * 1000.0, block * 1000.0 / rate);
    std::printf ("frame ready -> published:");
    {
        static const char* labels[] = { "<1", "<2", "<3", "<4", "<5", "<7", "<10", "<15", ">=15" };
        for (size_t i = 0; i < core.stats.publishHistogram.size(); ++i)
            std::printf (" %s:%lld", labels[i], (long long) core.stats.publishHistogram[i].load());
    }
    std::printf (" ms\n");
    std::printf ("peak %.3f, largest step input %.3f output %.3f, non-finite %d\n", peak, inputJump, outputJump, bad);

    if (underruns > 0)
    {
        std::printf ("underruns by second:");
        for (const auto value : underrunTimeline)
            std::printf (" %lld", (long long) value);
        std::printf ("\n");
    }

    if (! outPath.empty())
        writeWav (juce::File (juce::String (outPath)), output, rate);

    return bad == 0 ? 0 : 1;
}
