// mint_probe 的 render 只能设静态参数，这里补一个能在流中切换参数的小宿主，用于咔嗒测试和边界测试。
//
//   sirens_switch_host <plugin.vst3> sequence <in.wav> <out.wav> [--block N] [--interval-ms N]
//       依次走遍 16 种按钮组合（Gray 码顺序再加全部两两跳变），每隔 interval 切一次，并把每次切换的采样位置写到 <out.wav>.txt
//   sirens_switch_host <plugin.vst3> toggle <in.wav> <out.wav> <参数名> [--block N] [--interval-ms N]
//       反复开关一个参数
//   sirens_switch_host <plugin.vst3> stress
//       44.1–192 kHz、1 采样缓冲、单声道和立体声、静音输入，检查 NaN/Inf/denormal 和静音输出
//   sirens_switch_host <plugin.vst3> speed
//       四个模式全开时处理 1 秒音频的耗时，换算成单实例 CPU 占比
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <iostream>
#include <vector>

namespace
{
    const std::array<const char*, 4> modeNames { "Mode I", "Mode II", "Mode III", "Mode IV" };

    std::unique_ptr<juce::AudioPluginInstance> load (const juce::String& path, double rate, int block)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);

        juce::OwnedArray<juce::PluginDescription> found;
        for (auto* format : manager.getFormats())
            if (format->fileMightContainThisPluginType (path))
                format->findAllTypesForFile (found, path);

        if (found.isEmpty())
        {
            std::cout << "FAIL no plugin found in " << path << "\n";
            return nullptr;
        }

        juce::String error;
        auto instance = manager.createPluginInstance (*found[0], rate, block, error);
        if (instance == nullptr)
            std::cout << "FAIL could not load: " << error << "\n";
        return instance;
    }

    juce::AudioProcessorParameter* findParam (juce::AudioProcessor& plugin, const juce::String& name)
    {
        for (auto* param : plugin.getParameters())
            if (param->getName (64) == name)
                return param;
        return nullptr;
    }

    void setCombination (juce::AudioProcessor& plugin, int bits)
    {
        for (size_t i = 0; i < modeNames.size(); ++i)
            if (auto* param = findParam (plugin, modeNames[i]))
                param->setValueNotifyingHost ((bits >> i) & 1 ? 1.0f : 0.0f);
    }

    juce::String argValue (const juce::StringArray& args, const juce::String& key, const juce::String& fallback)
    {
        const auto index = args.indexOf (key);
        return index >= 0 && index + 1 < args.size() ? args[index + 1] : fallback;
    }

    // 每个有序对 (a→b) 都走到一次，覆盖所有可能的跳变。
    std::vector<int> allTransitions()
    {
        std::vector<int> order;
        std::vector<std::vector<bool>> used (16, std::vector<bool> (16, false));
        auto current = 0;
        order.push_back (current);
        for (;;)
        {
            auto next = -1;
            for (int b = 0; b < 16; ++b)
                if (b != current && ! used[(size_t) current][(size_t) b])
                {
                    next = b;
                    break;
                }

            if (next < 0)
            {
                // 回到还有未用出边的状态继续走。
                for (int a = 0; a < 16 && next < 0; ++a)
                    for (int b = 0; b < 16 && next < 0; ++b)
                        if (a != b && ! used[(size_t) a][(size_t) b])
                            next = a;
                if (next < 0)
                    break;
                order.push_back (next);
                current = next;
                continue;
            }

            used[(size_t) current][(size_t) next] = true;
            order.push_back (next);
            current = next;
        }
        return order;
    }

    int runRender (const juce::String& path, const juce::String& mode, const juce::File& input, const juce::File& output, const juce::StringArray& args)
    {
        const auto block = argValue (args, "--block", "256").getIntValue();
        const auto intervalMs = argValue (args, "--interval-ms", "150").getDoubleValue();

        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (input));
        if (reader == nullptr)
        {
            std::cout << "FAIL cannot read input\n";
            return 1;
        }

        const auto rate = reader->sampleRate;
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        juce::AudioProcessorParameter* toggled = nullptr;
        if (mode == "toggle")
        {
            toggled = findParam (*plugin, args[4]);
            if (toggled == nullptr)
            {
                std::cout << "FAIL no parameter " << args[4] << "\n";
                return 1;
            }
        }

        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--set")
                if (auto* param = findParam (*plugin, args[i + 1].upToFirstOccurrenceOf ("=", false, false)))
                    param->setValueNotifyingHost (args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue());

        plugin->enableAllBuses();
        plugin->prepareToPlay (rate, block);

        const auto total = (int) reader->lengthInSamples;
        juce::AudioBuffer<float> all (2, total);
        reader->read (&all, 0, total, 0, true, true);

        const auto order = allTransitions();
        const auto interval = juce::jmax (block, (int) (intervalMs * 0.001 * rate));
        juce::StringArray marks;
        juce::MidiBuffer midi;
        juce::AudioBuffer<float> chunk (2, block);

        auto nextSwitch = interval;
        size_t step = 0;
        auto flag = false;

        for (int pos = 0; pos < total; pos += block)
        {
            if (pos >= nextSwitch)
            {
                nextSwitch += interval;
                if (mode == "sequence" && step < order.size())
                {
                    setCombination (*plugin, order[step++]);
                    marks.add (juce::String (pos));
                }
                else if (toggled != nullptr)
                {
                    flag = ! flag;
                    toggled->setValueNotifyingHost (flag ? 1.0f : 0.0f);
                    marks.add (juce::String (pos));
                }
            }

            const auto n = juce::jmin (block, total - pos);
            chunk.clear();
            for (int ch = 0; ch < 2; ++ch)
                chunk.copyFrom (ch, 0, all, ch, pos, n);
            plugin->processBlock (chunk, midi);
            for (int ch = 0; ch < 2; ++ch)
                all.copyFrom (ch, pos, chunk, ch, 0, n);
        }

        plugin->releaseResources();

        output.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (output.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}.withSampleRate (rate).withNumChannels (2).withBitsPerSample (32)
                           .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write output\n";
            return 1;
        }
        writer->writeFromAudioSampleBuffer (all, 0, total);
        writer.reset();

        juce::File (output.getFullPathName() + ".txt").replaceWithText (marks.joinIntoString ("\n"));
        std::cout << "rendered " << total << " samples, " << marks.size() << " switches, " << order.size() << " planned\n";
        return 0;
    }

    int runStress (const juce::String& path)
    {
        auto failures = 0;
        const auto denormalLimit = 1.0e-30f;

        for (const auto rate : { 44100.0, 48000.0, 88200.0, 96000.0, 192000.0 })
        {
            for (const auto block : { 1, 7, 64, 2048 })
            {
                for (const auto channels : { 1, 2 })
                {
                    auto plugin = load (path, rate, block);
                    if (plugin == nullptr)
                        return 1;

                    juce::AudioProcessor::BusesLayout layout;
                    const auto set = channels == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo();
                    layout.inputBuses.add (set);
                    layout.outputBuses.add (set);
                    if (! plugin->setBusesLayout (layout))
                    {
                        std::cout << "FAIL layout rejected, channels " << channels << "\n";
                        ++failures;
                        continue;
                    }

                    plugin->setRateAndBufferSizeDetails (rate, block);
                    plugin->prepareToPlay (rate, block);
                    setCombination (*plugin, 15);
                    // 噪声默认关，这里打开，它的包络尾巴也要过一遍。
                    findParam (*plugin, "Noise")->setValueNotifyingHost (1.0f);

                    juce::AudioBuffer<float> buffer (channels, block);
                    juce::MidiBuffer midi;
                    juce::Random random (7);
                    const auto burst = (int) (rate * 0.1);
                    const auto silenceStart = (int) (rate * 0.2);
                    // 噪声包络约 1.7 s 才完全归零，尾段要留够。
                    const auto end = (int) (rate * 3.0);

                    auto bad = false, sawDenormal = false;
                    auto peak = 0.0f, tailPeak = 0.0f;

                    for (int pos = 0; pos < end; pos += block)
                    {
                        // 换组合也要在这个条件下走一遍。
                        if (pos / (int) (rate * 0.05) != (pos + block) / (int) (rate * 0.05))
                            setCombination (*plugin, 1 + random.nextInt (15));
                        if (pos >= silenceStart)
                            setCombination (*plugin, 15);

                        for (int ch = 0; ch < channels; ++ch)
                            for (int i = 0; i < block; ++i)
                                buffer.setSample (ch, i, pos + i < burst ? (random.nextFloat() * 2.0f - 1.0f) * 0.9f
                                                         : pos + i < silenceStart ? 1.0e-37f : 0.0f);

                        plugin->processBlock (buffer, midi);

                        for (int ch = 0; ch < channels; ++ch)
                            for (int i = 0; i < block; ++i)
                            {
                                const auto v = buffer.getSample (ch, i);
                                bad = bad || ! std::isfinite (v);
                                const auto a = std::abs (v);
                                sawDenormal = sawDenormal || (a > 0.0f && a < std::numeric_limits<float>::min());
                                peak = juce::jmax (peak, a);
                                if (pos > end - (int) (rate * 0.1))
                                    tailPeak = juce::jmax (tailPeak, a);
                            }
                    }

                    plugin->releaseResources();

                    const auto tailOk = tailPeak < denormalLimit;
                    if (bad || sawDenormal || ! tailOk || peak > 4.0f)
                    {
                        std::cout << "FAIL rate " << rate << " block " << block << " ch " << channels
                                  << " bad " << (int) bad << " denormal " << (int) sawDenormal
                                  << " tail " << tailPeak << " peak " << peak << "\n";
                        ++failures;
                    }
                }
            }
        }

        std::cout << (failures == 0 ? "PASS stress\n" : "FAILED stress\n");
        return failures == 0 ? 0 : 1;
    }

    int runSpeed (const juce::String& path)
    {
        for (const auto rate : { 48000.0, 192000.0 })
        {
            const auto block = 64;
            auto plugin = load (path, rate, block);
            if (plugin == nullptr)
                return 1;

            plugin->enableAllBuses();
            plugin->prepareToPlay (rate, block);
            setCombination (*plugin, 15);
            // 噪声默认关，这里打开，读数才是最坏情况。
            findParam (*plugin, "Noise")->setValueNotifyingHost (1.0f);

            juce::AudioBuffer<float> buffer (2, block);
            juce::MidiBuffer midi;
            juce::Random random (3);
            const auto blocks = (int) (rate * 10.0) / block;
            const auto start = juce::Time::getMillisecondCounterHiRes();
            for (int n = 0; n < blocks; ++n)
            {
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < block; ++i)
                        buffer.setSample (ch, i, random.nextFloat() * 0.2f - 0.1f);
                plugin->processBlock (buffer, midi);
            }
            const auto seconds = (juce::Time::getMillisecondCounterHiRes() - start) * 0.001;
            std::cout << rate << " Hz: " << seconds / 10.0 * 100.0 << " % of one core (test signal generation included)\n";
            plugin->releaseResources();
        }
        return 0;
    }
}

int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (argv[i]));

    if (args.size() >= 2 && args[1] == "stress")
        return runStress (args[0]);
    if (args.size() >= 2 && args[1] == "speed")
        return runSpeed (args[0]);
    if (args.size() >= 4 && args[1] == "sequence")
        return runRender (args[0], args[1], juce::File (args[2]), juce::File (args[3]), args);
    if (args.size() >= 5 && args[1] == "toggle")
    {
        // args: plugin toggle in out name → 参数名放到第 4 位
        return runRender (args[0], args[1], juce::File (args[2]), juce::File (args[3]), args);
    }

    std::cout << "usage: sirens_switch_host <plugin.vst3> sequence|toggle|stress ...\n";
    return 2;
}
