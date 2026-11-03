// Aegis 的浮点级验证工具，直接托管 VST3，不经过 WAV 量化。
//
//   aegis_bench stress <aegis.vst3> [--cases N]
//       随机抽参数和缓冲区大小，用恶意信号验证输出采样峰值绝不超过上限。
//   aegis_bench params <plugin.vst3> [--name 插件名]
//   aegis_bench render <plugin.vst3> <in.wav> <out.wav> [--name 插件名] [--set 参数名=归一化值]... [--block N]
//       输出 32 位浮点 WAV，已去掉上报延迟。--name 用来从多插件容器里选插件。
//   aegis_bench batch <plugin.vst3> <jobs.txt> [--name 插件名] [--block N] [--show]
//       一次加载跑一批 render。jobs 每行：输入<TAB>输出<TAB>参数名=值<TAB>…
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#define NOMINMAX
#include <windows.h>

#include <iostream>
#include <map>
#include <vector>

namespace
{
    // VST3 公开 ABI 里枚举类需要的部分。多插件容器的全量扫描会把每个插件都实例化一次，
    // 既慢又可能被其中某个插件弄崩，所以只枚举类名，只实例化要用的那一个，和 DAW 的做法相同。
    struct ClassInfo
    {
        char cid[16];
        int cardinality;
        char category[32];
        char name[64];
    };

    struct Factory
    {
        virtual int queryInterface (const char*, void**) = 0;
        virtual unsigned addRef() = 0;
        virtual unsigned release() = 0;
        virtual int getFactoryInfo (void*) = 0;
        virtual int countClasses() = 0;
        virtual int getClassInfo (int, ClassInfo*) = 0;
    };

    bool describeClass (const juce::String& path, const juce::String& name, juce::PluginDescription& description, juce::StringArray& seen)
    {
        const auto previous = juce::File::getCurrentWorkingDirectory();
        juce::File (path).getParentDirectory().setAsCurrentWorkingDirectory();
        auto module = LoadLibraryW (path.toWideCharPointer());
        previous.setAsCurrentWorkingDirectory();

        if (module == nullptr)
            return false;

        using InitDll = bool (*)();
        if (auto init = (InitDll) (void*) GetProcAddress (module, "InitDll"))
            init();

        using GetFactory = Factory* (*)();
        auto getFactory = (GetFactory) (void*) GetProcAddress (module, "GetPluginFactory");
        auto* factory = getFactory != nullptr ? getFactory() : nullptr;
        if (factory == nullptr)
            return false;

        for (int i = 0; i < factory->countClasses(); ++i)
        {
            ClassInfo info {};
            if (factory->getClassInfo (i, &info) != 0 || std::strcmp (info.category, "Audio Module Class") != 0)
                continue;

            const auto className = juce::String (juce::CharPointer_UTF8 (info.name)).trim();
            seen.add (className);

            if (className == name)
            {
                juce::uint32 hash = 0;
                for (const auto c : info.cid)
                    hash = hash * 31 + (juce::uint32) c;

                description.fileOrIdentifier = path;
                description.name = className;
                description.pluginFormatName = "VST3";
                description.deprecatedUid = (int) hash;
                return true;
            }
        }

        return false;
    }

    std::unique_ptr<juce::AudioPluginInstance> load (const juce::String& path, const juce::String& name, double rate, int block)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);

        juce::PluginDescription chosen;
        if (name.isNotEmpty())
        {
            juce::StringArray seen;
            if (! describeClass (path, name, chosen, seen))
            {
                std::cout << "FAIL plugin not found (" << seen.size() << " classes in file)\n" << seen.joinIntoString ("\n") << "\n";
                return nullptr;
            }
        }
        else
        {
            juce::OwnedArray<juce::PluginDescription> found;
            for (auto* format : manager.getFormats())
                if (format->fileMightContainThisPluginType (path))
                    format->findAllTypesForFile (found, path);

            if (found.isEmpty())
            {
                std::cout << "FAIL no plugin found in " << path << "\n";
                return nullptr;
            }
            chosen = *found[0];
        }

        juce::String error;
        auto instance = manager.createPluginInstance (chosen, rate, block, error);
        if (instance == nullptr)
            std::cout << "FAIL could not load: " << error << "\n";
        return instance;
    }

    juce::AudioProcessorParameter* findParam (juce::AudioPluginInstance& plugin, const juce::String& name)
    {
        for (auto* param : plugin.getParameters())
            if (param->getName (64) == name)
                return param;
        return nullptr;
    }

    juce::String argValue (const juce::StringArray& args, const juce::String& key, const juce::String& fallback)
    {
        const auto index = args.indexOf (key);
        return index >= 0 && index + 1 < args.size() ? args[index + 1] : fallback;
    }

    float ceilingDbFromNorm (float norm) { return -30.0f + 0.1f * std::round (norm * 300.0f); }

    using Generator = float (*) (int n, int ch, juce::Random& random, double rate);

    float burst (int n, int, juce::Random& random, double rate)
    {
        const auto gate = n % (int) (rate * 0.08) < (int) (rate * 0.004);
        return gate ? (random.nextBool() ? 1.0f : -1.0f) : 0.0f;
    }
    float square50 (int n, int, juce::Random&, double rate)    { return std::fmod ((double) n * 50.0 / rate, 1.0) < 0.5 ? 1.0f : -1.0f; }
    float squareHigh (int n, int, juce::Random&, double)       { return (n / 8) % 2 == 0 ? 1.0f : -1.0f; }
    float nyquist (int n, int, juce::Random&, double)          { return n % 2 == 0 ? 1.0f : -1.0f; }
    float dcStep (int n, int, juce::Random&, double rate)      { return n < rate * 0.05 ? 0.0f : 1.0f; }
    float noise (int, int, juce::Random& random, double)       { return random.nextFloat() * 2.0f - 1.0f; }
    float over12 (int, int, juce::Random& random, double)      { return (random.nextFloat() * 2.0f - 1.0f) * 4.0f; }
    float huge (int n, int, juce::Random&, double)             { return n % 97 == 0 ? 1.0e20f : 0.1f; }
    float impulses (int n, int ch, juce::Random&, double)      { return n % 256 == 0 ? (ch == 0 ? 1.0f : -1.0f) : 0.0f; }
    float rightOnly (int n, int ch, juce::Random&, double)     { return ch == 1 && n % 64 < 8 ? 1.0f : 0.0f; }

    float clippedSine (int n, int, juce::Random&, double rate)
    {
        return juce::jlimit (-1.0f, 1.0f, 8.0f * (float) std::sin (juce::MathConstants<double>::twoPi * 220.0 * n / rate));
    }

    float sweep (int n, int, juce::Random&, double rate)
    {
        const auto t = n / rate;
        return (float) std::sin (juce::MathConstants<double>::twoPi * (40.0 * t + 4000.0 * t * t));
    }

    float nonFinite (int n, int, juce::Random&, double)
    {
        if (n % 101 == 0)
            return std::numeric_limits<float>::quiet_NaN();
        if (n % 103 == 0)
            return std::numeric_limits<float>::infinity();
        return n % 2 == 0 ? 0.9f : -0.9f;
    }

    struct Signal { const char* name; Generator generate; };
    const std::vector<Signal> signals {
        { "burst", burst }, { "square50", square50 }, { "squareHigh", squareHigh }, { "nyquist", nyquist },
        { "dcStep", dcStep }, { "noise", noise }, { "over12dB", over12 }, { "huge", huge }, { "impulses", impulses },
        { "rightOnly", rightOnly }, { "clippedSine", clippedSine }, { "sweep", sweep }, { "nonFinite", nonFinite }
    };

    struct Tally
    {
        double worst = -1000.0;     // 峰值相对上限的 dB，<= 0 才算过关。
        juce::String where;
        void add (double db, const juce::String& context)
        {
            if (db > worst)
            {
                worst = db;
                where = context;
            }
        }
    };

    int nextBlock (juce::Random& random, int maxBlock)
    {
        static const int sizes[] { 1, 7, 32, 64, 128, 129, 256, 480, 512, 1024, 2048 };
        const auto choice = sizes[random.nextInt ((int) std::size (sizes))];
        return juce::jmin (choice, maxBlock);
    }

    int runStress (const juce::String& path, const juce::StringArray& args)
    {
        const auto numCases = argValue (args, "--cases", "400").getIntValue();
        const double rates[] { 44100.0, 48000.0, 96000.0, 192000.0 };
        const float lookaheadNorms[] { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
        const char* lookaheadNames[] { "0 ms", "1.5 ms", "5 ms", "10 ms" };

        constexpr int maxBlock = 2048;
        auto failures = 0;
        std::map<int, Tally> byLookahead;
        std::map<juce::String, Tally> bySignal;
        juce::Random random (20261102);

        auto plugin = load (path, {}, 48000.0, maxBlock);
        if (plugin == nullptr)
            return 1;

        auto* threshold = findParam (*plugin, "Threshold");
        auto* ceiling = findParam (*plugin, "Ceiling");
        auto* release = findParam (*plugin, "Release");
        auto* autoRelease = findParam (*plugin, "Auto Release");
        auto* look = findParam (*plugin, "Look-ahead");
        if (threshold == nullptr || ceiling == nullptr || release == nullptr || autoRelease == nullptr || look == nullptr)
        {
            std::cout << "FAIL parameters not found\n";
            return 1;
        }

        juce::AudioBuffer<float> buffer (2, maxBlock);
        juce::MidiBuffer midi;

        const auto processRun = [&] (Generator generate, int total, double rate, double limit, Tally* tally, const juce::String& context, bool checkPeaks)
        {
            auto peak = 0.0f;
            auto bad = false;
            for (int pos = 0; pos < total;)
            {
                const auto n = juce::jmin (nextBlock (random, maxBlock), total - pos);
                juce::AudioBuffer<float> block (buffer.getArrayOfWritePointers(), 2, 0, n);
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < n; ++i)
                        block.setSample (ch, i, generate == nullptr ? 0.0f : generate (pos + i, ch, random, rate));
                plugin->processBlock (block, midi);
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < n; ++i)
                    {
                        const auto y = block.getSample (ch, i);
                        bad = bad || ! std::isfinite (y);
                        peak = juce::jmax (peak, std::abs (y));
                    }
                pos += n;
            }

            if (bad)
            {
                std::cout << "FAIL non-finite output: " << context << "\n";
                ++failures;
            }

            const auto db = juce::Decibels::gainToDecibels ((double) peak, -200.0) - limit;
            if (checkPeaks && tally != nullptr)
                tally->add (db, context);
            return db;
        };

        for (int c = 0; c < numCases; ++c)
        {
            const auto rate = rates[random.nextInt (4)];
            const auto lookIndex = random.nextInt (4);
            const float ceilingChoices[] { 1.0f, 299.0f / 300.0f, 290.0f / 300.0f, 0.8f, 0.0f, random.nextFloat() };
            const float thresholdChoices[] { 1.0f, 0.8f, 0.6f, 0.0f, random.nextFloat() };
            const auto ceilingNorm = ceilingChoices[random.nextInt (6)];
            const auto thresholdNorm = thresholdChoices[random.nextInt (5)];
            const float releaseChoices[] { 0.0f, 0.1f, 0.5f, 1.0f, random.nextFloat() };
            const auto releaseNorm = releaseChoices[random.nextInt (5)];
            const auto autoNorm = random.nextBool() ? 1.0f : 0.0f;
            const auto& signal = signals[(size_t) random.nextInt ((int) signals.size())];

            threshold->setValueNotifyingHost (thresholdNorm);
            ceiling->setValueNotifyingHost (ceilingNorm);
            release->setValueNotifyingHost (releaseNorm);
            autoRelease->setValueNotifyingHost (autoNorm);
            look->setValueNotifyingHost (lookaheadNorms[lookIndex]);

            plugin->setRateAndBufferSizeDetails (rate, maxBlock);
            plugin->prepareToPlay (rate, maxBlock);

            const auto limitDb = (double) ceilingDbFromNorm (ceiling->getValue());
            const auto context = juce::String (signal.name) + " look=" + lookaheadNames[lookIndex] + " thr=" + threshold->getCurrentValueAsText()
                                 + " ceil=" + ceiling->getCurrentValueAsText() + " rel=" + release->getCurrentValueAsText()
                                 + " auto=" + juce::String (autoNorm) + " rate=" + juce::String (rate);

            // 参数经由 VST3 队列到达，先跑静音让平滑和切换走完。
            processRun (nullptr, (int) (rate * 0.1), rate, limitDb, nullptr, context, false);

            const auto total = (int) (rate * 0.25);
            const auto db = processRun (signal.generate, total, rate, limitDb, nullptr, context, true);
            byLookahead[lookIndex].add (db, context);
            bySignal[signal.name].add (db, context);
            plugin->releaseResources();
        }

        std::cout << "== static settings: peak relative to ceiling (dB), " << numCases << " random cases\n";
        for (const auto& [index, tally] : byLookahead)
            std::cout << "look " << lookaheadNames[index] << ": " << juce::String (tally.worst, 7) << "  [" << tally.where << "]\n";
        for (const auto& [name, tally] : bySignal)
            std::cout << "sig  " << name << ": " << juce::String (tally.worst, 7) << "\n";

        // 运行中随机切模式、阈值、释放：上限固定，任何时刻都不能超。
        plugin->setRateAndBufferSizeDetails (48000.0, maxBlock);
        plugin->prepareToPlay (48000.0, maxBlock);
        ceiling->setValueNotifyingHost (290.0f / 300.0f);
        processRun (nullptr, 9600, 48000.0, -1.0, nullptr, "settle", false);
        Tally live;
        for (int c = 0; c < 240; ++c)
        {
            threshold->setValueNotifyingHost (random.nextFloat());
            release->setValueNotifyingHost (random.nextFloat());
            autoRelease->setValueNotifyingHost (random.nextBool() ? 1.0f : 0.0f);
            look->setValueNotifyingHost (lookaheadNorms[random.nextInt (4)]);
            const auto& signal = signals[(size_t) random.nextInt ((int) signals.size())];
            live.add (processRun (signal.generate, 4800, 48000.0, (double) ceilingDbFromNorm (ceiling->getValue()), nullptr, signal.name, true), signal.name);
        }
        std::cout << "== live switching (mode/threshold/release changed every 100 ms): " << juce::String (live.worst, 7) << " dB  [" << live.where << "]\n";

        // 上限本身在运行中下调：10 ms 平滑结束后必须服从新值。
        ceiling->setValueNotifyingHost (1.0f);
        processRun (noise, 9600, 48000.0, 0.0, nullptr, "ceiling high", false);
        ceiling->setValueNotifyingHost (240.0f / 300.0f);
        const auto duringRamp = processRun (noise, 1024, 48000.0, -6.0, nullptr, "ramp", false);
        const auto afterRamp = processRun (noise, 9600, 48000.0, -6.0, nullptr, "after ramp", false);
        std::cout << "== ceiling lowered live 0 -> -6 dB: first 21 ms " << juce::String (duringRamp, 3) << " dB over new value, afterwards "
                  << juce::String (afterRamp, 7) << " dB\n";

        // 静音输入：不能产生 denormal。
        threshold->setValueNotifyingHost (0.5f);
        processRun (noise, 9600, 48000.0, 0.0, nullptr, "before silence", false);
        auto subnormals = 0;
        for (int pos = 0; pos < 48000 * 3; pos += 256)
        {
            juce::AudioBuffer<float> block (buffer.getArrayOfWritePointers(), 2, 0, 256);
            block.clear();
            plugin->processBlock (block, midi);
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 256; ++i)
                    if (block.getSample (ch, i) != 0.0f && std::abs (block.getSample (ch, i)) < std::numeric_limits<float>::min())
                        ++subnormals;
        }
        plugin->releaseResources();
        std::cout << "== denormal samples after 3 s of silence: " << subnormals << "\n";
        failures += subnormals > 0 ? 1 : 0;

        // 默认状态：prepareToPlay 一返回，宿主读到的延迟就必须是真值（默认 1.5 ms）。
        for (const auto rate : rates)
        {
            auto fresh = load (path, {}, rate, 256);
            fresh->prepareToPlay (rate, 256);
            const auto expected = juce::roundToInt (0.0015 * rate);
            std::cout << "== default latency right after prepare at " << rate << " Hz: " << fresh->getLatencySamples()
                      << " samples (expected " << expected << ")\n";
            failures += fresh->getLatencySamples() == expected ? 0 : 1;
            fresh->releaseResources();
        }

        // 单声道总线。
        auto monoPlugin = load (path, {}, 48000.0, maxBlock);
        auto monoLayout = monoPlugin->getBusesLayout();
        monoLayout.inputBuses.getReference (0) = juce::AudioChannelSet::mono();
        monoLayout.outputBuses.getReference (0) = juce::AudioChannelSet::mono();
        if (monoPlugin->setBusesLayout (monoLayout))
        {
            monoPlugin->prepareToPlay (48000.0, 256);
            std::cout << "mono channels in/out: " << monoPlugin->getTotalNumInputChannels() << "/" << monoPlugin->getTotalNumOutputChannels() << "\n";
            juce::AudioBuffer<float> mono (1, 256);
            auto peak = 0.0f;
            for (int pos = 0; pos < 48000; pos += 256)
            {
                for (int i = 0; i < 256; ++i)
                    mono.setSample (0, i, square50 (pos + i, 0, random, 48000.0));
                monoPlugin->processBlock (mono, midi);
                peak = juce::jmax (peak, mono.getMagnitude (0, 256));
            }
            monoPlugin->releaseResources();
            std::cout << "== mono bus, default settings, square: peak " << juce::String (juce::Decibels::gainToDecibels (peak), 5) << " dBFS (ceiling -1.0)\n";
        }
        else
        {
            std::cout << "FAIL mono layout rejected\n";
            ++failures;
        }

        const auto worst = juce::jmax (live.worst, afterRamp, [&]
        {
            auto w = -1000.0;
            for (const auto& entry : byLookahead)
                w = juce::jmax (w, entry.second.worst);
            return w;
        }());

        // 上限的 dB 值在换算成线性增益时有一个 float 舍入的量级（约 1e-6 dB），容差只放这么多。
        constexpr double toleranceDb = 2.0e-5;
        std::cout << "worst overall: " << juce::String (worst, 7) << " dB relative to ceiling, tolerance " << toleranceDb << " dB\n";
        if (worst > toleranceDb)
            ++failures;

        std::cout << (failures == 0 ? "PASS\n" : "FAILED\n");
        return failures == 0 ? 0 : 1;
    }

    int runParams (const juce::String& path, const juce::StringArray& args)
    {
        auto plugin = load (path, argValue (args, "--name", {}), 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        std::cout << "name: " << plugin->getName() << " latency: " << plugin->getLatencySamples() << "\n";
        for (auto* param : plugin->getParameters())
            std::cout << param->getParameterIndex() << "\t" << param->getName (64) << "\t"
                      << param->getValue() << "\t" << param->getCurrentValueAsText() << "\n";
        return 0;
    }

    // 把 input 渲染到 output（32 位浮点，已去掉上报延迟）。sets 是 "参数名=归一化值" 列表。
    int renderOne (juce::AudioPluginInstance& plugin, const juce::File& input, const juce::File& output,
                   const juce::StringArray& sets, int block, double settleSeconds, bool show)
    {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (input));
        if (reader == nullptr)
        {
            std::cout << "FAIL cannot read " << input.getFullPathName() << "\n";
            return 1;
        }

        const auto rate = reader->sampleRate;

        for (const auto& item : sets)
        {
            const auto name = item.upToFirstOccurrenceOf ("=", false, false);
            const auto value = item.fromFirstOccurrenceOf ("=", false, false).getFloatValue();
            if (auto* param = findParam (plugin, name))
                param->setValueNotifyingHost (value);
            else
                std::cout << "WARN no parameter named " << name << "\n";
        }

        plugin.enableAllBuses();
        plugin.prepareToPlay (rate, block);

        const auto channels = juce::jmax (2, plugin.getTotalNumInputChannels(), plugin.getTotalNumOutputChannels());
        juce::AudioBuffer<float> settle (channels, block);
        juce::MidiBuffer midi;
        for (int i = 0; i < (int) (rate * settleSeconds) / block; ++i)
        {
            settle.clear();
            plugin.processBlock (settle, midi);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto latency = plugin.getLatencySamples();

        if (show)
            for (auto* param : plugin.getParameters())
                std::cout << param->getName (64) << " = " << param->getCurrentValueAsText() << "\n";

        const auto total = (int) reader->lengthInSamples;
        juce::AudioBuffer<float> all (channels, total + latency);
        all.clear();
        reader->read (&all, 0, total, 0, true, true);

        juce::AudioBuffer<float> chunk (channels, block);
        for (int pos = 0; pos < all.getNumSamples(); pos += block)
        {
            const auto n = juce::jmin (block, all.getNumSamples() - pos);
            chunk.clear();
            for (int ch = 0; ch < channels; ++ch)
                chunk.copyFrom (ch, 0, all, ch, pos, n);
            plugin.processBlock (chunk, midi);
            for (int ch = 0; ch < channels; ++ch)
                all.copyFrom (ch, pos, chunk, ch, 0, n);
        }

        output.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (output.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}.withSampleRate (rate).withNumChannels (2).withBitsPerSample (32)
                           .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write " << output.getFullPathName() << "\n";
            return 1;
        }

        juce::AudioBuffer<float> firstTwo (all.getArrayOfWritePointers(), 2, all.getNumSamples());
        writer->writeFromAudioSampleBuffer (firstTwo, latency, total);
        std::cout << "rendered " << total << " samples, reported latency " << latency << "\n";
        return 0;
    }

    juce::StringArray collectSets (const juce::StringArray& args)
    {
        juce::StringArray sets;
        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--set")
                sets.add (args[i + 1]);
        return sets;
    }

    int runRender (const juce::String& path, const juce::File& input, const juce::File& output, const juce::StringArray& args)
    {
        const auto block = argValue (args, "--block", "256").getIntValue();

        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (input));
        if (reader == nullptr)
        {
            std::cout << "FAIL cannot read " << input.getFullPathName() << "\n";
            return 1;
        }

        auto plugin = load (path, argValue (args, "--name", {}), reader->sampleRate, block);
        if (plugin == nullptr)
            return 1;

        const auto result = renderOne (*plugin, input, output, collectSets (args), block, 0.2, args.contains ("--show"));
        plugin->releaseResources();
        return result;
    }

    // 对标插件加载很慢，一次加载跑一批。jobs 文件每行：输入<TAB>输出<TAB>参数名=值<TAB>…
    int runBatch (const juce::String& path, const juce::File& jobs, const juce::StringArray& args)
    {
        const auto block = argValue (args, "--block", "256").getIntValue();
        auto plugin = load (path, argValue (args, "--name", {}), 48000.0, block);
        if (plugin == nullptr)
            return 1;

        auto failures = 0;
        for (const auto& line : juce::StringArray::fromLines (jobs.loadFileAsString()))
        {
            auto fields = juce::StringArray::fromTokens (line, "\t", {});
            if (fields.size() < 2)
                continue;

            const juce::File input (fields[0]), output (fields[1]);
            fields.removeRange (0, 2);
            std::cout << output.getFileName() << ": ";
            // 对标插件的部分参数只在激活时读取，每条任务都重新激活。释放可能很长，跑够静音把上一条任务的状态放掉。
            plugin->releaseResources();
            failures += renderOne (*plugin, input, output, fields, block, 3.0, args.contains ("--show"));
            std::cout.flush();
        }

        plugin->releaseResources();
        return failures == 0 ? 0 : 1;
    }
}

// 用宽字符入口，中文 Windows 上含非 ASCII 字符的路径才不会被 GBK 解码弄坏。
int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (argv[i]));

    auto result = 2;
    if (args.size() >= 2 && args[0] == "stress")
        result = runStress (args[1], args);
    else if (args.size() >= 2 && args[0] == "params")
        result = runParams (args[1], args);
    else if (args.size() >= 4 && args[0] == "render")
        result = runRender (args[1], juce::File (args[2]), juce::File (args[3]), args);
    else if (args.size() >= 3 && args[0] == "batch")
        result = runBatch (args[1], juce::File (args[2]), args);
    else
        std::cout << "usage: aegis_bench stress|params|render|batch <plugin.vst3> ...\n";

    // 有的多插件容器在进程退出卸载时会崩，结果此时已经写完，直接结束进程。
    std::cout.flush();
    TerminateProcess (GetCurrentProcess(), (UINT) result);
    return result;
}
