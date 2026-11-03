// Helios 的验证宿主，也能加载对标插件做黑盒渲染。
//
//   mint_rider_bench render <plugin.vst3> <in.wav> <out.wav> [--sc sc.wav] [--no-sc-bus] [--mono]
//                           [--block N] [--random-blocks] [--set name=normalised]...
//                           [--at seconds:name=normalised]...   播放中途在块边界上改参数
//   mint_rider_bench write  <plugin.vst3> <in.wav> <out.wav> <log.csv> [--block N] [--offline] [--set name=normalised]...
//
// render 把输出写成 32 位浮点 WAV，不跑预热段，也不去除延迟。
// write 打开写自动化，音频放在独立线程上按实时节奏处理，消息线程正常运转，
// 记录 ride_gain 每次通知的线程、时间和数值。
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

namespace
{
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

    juce::String argValue (const juce::StringArray& args, const juce::String& key, const juce::String& fallback)
    {
        const auto index = args.indexOf (key);
        return index >= 0 && index + 1 < args.size() ? args[index + 1] : fallback;
    }

    void applySets (juce::AudioProcessor& plugin, const juce::StringArray& args)
    {
        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--set")
                continue;
            const auto name = args[i + 1].upToFirstOccurrenceOf ("=", false, false);
            const auto value = args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue();
            auto matched = false;
            for (auto* param : plugin.getParameters())
                if (param->getName (64) == name || juce::String (param->getParameterIndex()) == name)
                {
                    param->setValueNotifyingHost (value);
                    matched = true;
                }
            if (! matched)
                std::cout << "WARN no parameter named " << name << "\n";
        }
    }

    struct TimedSet
    {
        int sample;
        juce::AudioProcessorParameter* parameter;
        float value;
    };

    std::vector<TimedSet> parseTimedSets (juce::AudioProcessor& plugin, const juce::StringArray& args, double rate)
    {
        std::vector<TimedSet> sets;
        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--at")
                continue;
            const auto seconds = args[i + 1].upToFirstOccurrenceOf (":", false, false).getDoubleValue();
            const auto rest = args[i + 1].fromFirstOccurrenceOf (":", false, false);
            const auto name = rest.upToFirstOccurrenceOf ("=", false, false);
            for (auto* param : plugin.getParameters())
                if (param->getName (64) == name)
                    sets.push_back ({ (int) (seconds * rate), param, rest.fromFirstOccurrenceOf ("=", false, false).getFloatValue() });
        }
        return sets;
    }

    bool readWav (const juce::File& file, juce::AudioBuffer<float>& into, double& rate)
    {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
        if (reader == nullptr)
        {
            std::cout << "FAIL cannot read " << file.getFullPathName() << "\n";
            return false;
        }
        rate = reader->sampleRate;
        into.setSize (2, (int) reader->lengthInSamples);
        reader->read (&into, 0, (int) reader->lengthInSamples, 0, true, true);
        return true;
    }

    bool writeWav (const juce::File& file, const juce::AudioBuffer<float>& data, double rate)
    {
        file.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (file.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}
                           .withSampleRate (rate)
                           .withNumChannels (data.getNumChannels())
                           .withBitsPerSample (32)
                           .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write " << file.getFullPathName() << "\n";
            return false;
        }
        return writer->writeFromAudioSampleBuffer (data, 0, data.getNumSamples());
    }

    // 把主输入与侧链依次放进整块缓冲区，通道排布和 VST3 宿主一致：主总线在前，侧链在后。
    struct Streams
    {
        juce::AudioBuffer<float> main, sidechain;
        int mainChannels = 2, sidechainChannels = 0;
    };

    void processRange (juce::AudioProcessor& plugin, Streams& s, juce::AudioBuffer<float>& chunk,
                       juce::AudioBuffer<float>& out, int pos, int n)
    {
        juce::AudioBuffer<float> view (chunk.getArrayOfWritePointers(), chunk.getNumChannels(), n);
        view.clear();
        for (int ch = 0; ch < s.mainChannels; ++ch)
            view.copyFrom (ch, 0, s.main, ch, pos, n);
        for (int ch = 0; ch < s.sidechainChannels; ++ch)
            if (pos < s.sidechain.getNumSamples())
                view.copyFrom (s.mainChannels + ch, 0, s.sidechain, ch, pos, juce::jmin (n, s.sidechain.getNumSamples() - pos));

        juce::MidiBuffer midi;
        plugin.processBlock (view, midi);

        for (int ch = 0; ch < s.mainChannels; ++ch)
            out.copyFrom (ch, pos, view, ch, 0, n);
    }

    bool configure (juce::AudioProcessor& plugin, const juce::StringArray& args, Streams& s)
    {
        const auto mono = args.contains ("--mono");
        const auto hasSidechainBus = plugin.getBusCount (true) > 1;
        const auto noBus = args.contains ("--no-sc-bus") || ! hasSidechainBus;

        auto layout = plugin.getBusesLayout();
        const auto mainSet = mono ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo();
        layout.getChannelSet (true, 0) = mainSet;
        layout.getChannelSet (false, 0) = mainSet;
        if (hasSidechainBus)
            layout.getChannelSet (true, 1) = noBus ? juce::AudioChannelSet::disabled() : juce::AudioChannelSet::stereo();

        if (! plugin.setBusesLayout (layout))
        {
            std::cout << "FAIL layout rejected\n";
            return false;
        }

        s.mainChannels = mono ? 1 : 2;
        s.sidechainChannels = noBus ? 0 : 2;
        return true;
    }

    int runRender (const juce::String& path, const juce::File& in, const juce::File& outFile, const juce::StringArray& args)
    {
        Streams s;
        double rate = 48000.0;
        if (! readWav (in, s.main, rate))
            return 1;

        const auto scPath = argValue (args, "--sc", {});
        if (scPath.isNotEmpty())
        {
            double scRate = rate;
            if (! readWav (juce::File (scPath), s.sidechain, scRate))
                return 1;
        }

        const auto block = argValue (args, "--block", "256").getIntValue();
        const auto randomBlocks = args.contains ("--random-blocks");

        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        if (! configure (*plugin, args, s))
            return 1;
        applySets (*plugin, args);
        plugin->prepareToPlay (rate, block);
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto latency = plugin->getLatencySamples();

        const auto total = s.main.getNumSamples();
        juce::AudioBuffer<float> out (s.mainChannels, total);
        juce::AudioBuffer<float> chunk (juce::jmax (plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels(), 2), block);
        juce::Random random (7);
        const auto timedSets = parseTimedSets (*plugin, args, rate);
        auto processingSeconds = 0.0;

        for (int pos = 0; pos < total;)
        {
            const auto wanted = randomBlocks ? 1 + random.nextInt (block) : block;
            const auto n = juce::jmin (wanted, total - pos);

            for (const auto& timed : timedSets)
                if (timed.sample >= pos && timed.sample < pos + n)
                    timed.parameter->setValueNotifyingHost (timed.value);

            const auto before = juce::Time::getMillisecondCounterHiRes();
            processRange (*plugin, s, chunk, out, pos, n);
            processingSeconds += 0.001 * (juce::Time::getMillisecondCounterHiRes() - before);
            pos += n;
        }

        plugin->releaseResources();
        if (! writeWav (outFile, out, rate))
            return 1;
        std::cout << "rendered " << total << " samples, reported latency " << latency
                  << ", cpu " << 100.0 * processingSeconds / (total / rate) << " %\n";
        return 0;
    }

    struct WriteLog final : public juce::AudioProcessorListener
    {
        struct Event { int kind; int samplePos; float value; bool fromAudioThread; bool fromMessageThread; };

        void audioProcessorParameterChanged (juce::AudioProcessor*, int index, float value) override
        {
            if (index == rideIndex)
                add (0, value);
        }

        void audioProcessorChanged (juce::AudioProcessor*, const ChangeDetails&) override {}

        void audioProcessorParameterChangeGestureBegin (juce::AudioProcessor*, int index) override
        {
            if (index == rideIndex)
                add (1, 0.0f);
        }

        void audioProcessorParameterChangeGestureEnd (juce::AudioProcessor*, int index) override
        {
            if (index == rideIndex)
                add (2, 0.0f);
        }

        void add (int kind, float value)
        {
            const juce::ScopedLock lock (guard);
            events.push_back ({ kind, samplePos.load(),
                                value,
                                juce::Thread::getCurrentThreadId() == audioThreadId.load(),
                                juce::MessageManager::getInstance()->isThisTheMessageThread() });
        }

        juce::CriticalSection guard;
        std::vector<Event> events;
        std::atomic<int> samplePos { 0 };
        std::atomic<juce::Thread::ThreadID> audioThreadId { nullptr };
        int rideIndex = -1;
    };

    int runWrite (const juce::String& path, const juce::File& in, const juce::File& outFile, const juce::File& logFile, const juce::StringArray& args)
    {
        Streams s;
        double rate = 48000.0;
        if (! readWav (in, s.main, rate))
            return 1;

        const auto block = argValue (args, "--block", "256").getIntValue();
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        if (! configure (*plugin, args, s))
            return 1;
        applySets (*plugin, args);

        WriteLog log;
        for (auto* param : plugin->getParameters())
            if (param->getName (64) == "Ride Gain")
                log.rideIndex = param->getParameterIndex();
        if (log.rideIndex < 0)
        {
            std::cout << "FAIL no Ride Gain parameter\n";
            return 1;
        }
        plugin->addListener (&log);

        plugin->setNonRealtime (args.contains ("--offline"));
        plugin->prepareToPlay (rate, block);

        const auto total = s.main.getNumSamples();
        juce::AudioBuffer<float> out (s.mainChannels, total);
        juce::AudioBuffer<float> chunk (juce::jmax (plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels(), 2), block);
        std::atomic<bool> finished { false };

        auto* writeSwitch = [&]() -> juce::AudioProcessorParameter*
        {
            for (auto* param : plugin->getParameters())
                if (param->getName (64) == "Write Automation")
                    return param;
            return nullptr;
        }();

        if (writeSwitch == nullptr)
        {
            std::cout << "FAIL no Write Automation parameter\n";
            return 1;
        }

        // 前 90% 打开写入，之后关闭，用来确认手势正确收尾、关闭后不再写。
        const auto writeOffAt = (int) (total * 0.9);
        writeSwitch->setValueNotifyingHost (1.0f);

        std::thread audio ([&]
        {
            log.audioThreadId.store (juce::Thread::getCurrentThreadId());
            const auto start = juce::Time::getMillisecondCounterHiRes();
            for (int pos = 0; pos < total;)
            {
                const auto n = juce::jmin (block, total - pos);
                processRange (*plugin, s, chunk, out, pos, n);
                pos += n;
                log.samplePos.store (pos);
                const auto dueMs = start + 1000.0 * pos / rate;
                const auto nowMs = juce::Time::getMillisecondCounterHiRes();
                if (dueMs > nowMs + 1.0)
                    juce::Thread::sleep ((int) (dueMs - nowMs));
            }
            finished.store (true);
        });

        auto switched = false;
        while (! finished.load())
        {
            juce::MessageManager::getInstance()->runDispatchLoopUntil (5);
            if (! switched && log.samplePos.load() >= writeOffAt)
            {
                writeSwitch->setValueNotifyingHost (0.0f);
                switched = true;
            }
        }
        audio.join();
        juce::MessageManager::getInstance()->runDispatchLoopUntil (200);

        plugin->removeListener (&log);
        plugin->releaseResources();

        if (! writeWav (outFile, out, rate))
            return 1;

        juce::String csv ("kind,sample,value,audio_thread,message_thread\n");
        int fromAudio = 0, notFromMessage = 0, begins = 0, ends = 0, values = 0;
        for (const auto& e : log.events)
        {
            csv << e.kind << "," << e.samplePos << "," << e.value << "," << (int) e.fromAudioThread << "," << (int) e.fromMessageThread << "\n";
            fromAudio += e.fromAudioThread ? 1 : 0;
            notFromMessage += e.fromMessageThread ? 0 : 1;
            begins += e.kind == 1 ? 1 : 0;
            ends += e.kind == 2 ? 1 : 0;
            values += e.kind == 0 ? 1 : 0;
        }
        logFile.replaceWithText (csv);

        std::cout << "values " << values << " gestures " << begins << "/" << ends
                  << " from_audio_thread " << fromAudio << " not_from_message_thread " << notFromMessage << "\n";
        return 0;
    }
}

int main (int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI init;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String::fromUTF8 (argv[i]));

    if (args.size() >= 4 && args[0] == "render")
        return runRender (args[1], juce::File (args[2]), juce::File (args[3]), args);
    if (args.size() >= 5 && args[0] == "write")
        return runWrite (args[1], juce::File (args[2]), juce::File (args[3]), juce::File (args[4]), args);

    std::cout << "usage: mint_rider_bench render|write ...\n";
    return 2;
}
