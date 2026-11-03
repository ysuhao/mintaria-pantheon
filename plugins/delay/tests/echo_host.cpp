// 延迟插件的测试宿主。把插件当黑盒通过 VST3 加载，也能选中外壳型 VST3 里的单个插件做对比。
//
//   mint_delay_host list   <file.vst3> [--cache scan.xml]
//   mint_delay_host classes <file.vst3> [--name X --write-cache scan.xml]
//   mint_delay_host params <file.vst3> [--name X] [--set Name=norm]... [--bpm 120]
//   mint_delay_host render <file.vst3> in.wav out.wav [--name X] [--set Name=norm]... [--block 256] [--mono]
//                          [--bpm 120] [--bpm-at seconds bpm]... [--at seconds Name=norm]...
//   mint_delay_host shot   <file.vst3> out.png [--set Name=norm]...
//
// 所有命令都接受 --cache scan.xml，外壳型 VST3 只需要扫描一次。
// params 额外打印 latency 和 tail，经由 VST3 接口读回，与真实宿主看到的一致。
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

bool captureWindowPixels (void* windowHandle, std::vector<unsigned char>& pixels, int& width, int& height);

struct FactoryClass
{
    std::string name, category;
    char cid[16];
};

bool listFactoryClasses (const wchar_t* path, std::vector<FactoryClass>& classes);

namespace
{
    struct Change { double seconds; juce::String name; float value; };
    struct TempoChange { double seconds; double bpm; };

    // 假的宿主走带。bpm <= 0 时不报速度，用来测回退。
    class FakePlayHead final : public juce::AudioPlayHead
    {
    public:
        double bpm = 0.0;
        juce::int64 samples = 0;
        double rate = 48000.0;

        juce::Optional<PositionInfo> getPosition() const override
        {
            PositionInfo info;
            info.setTimeInSamples (samples);
            info.setTimeInSeconds ((double) samples / rate);
            info.setIsPlaying (true);
            if (bpm > 0.0)
                info.setBpm (bpm);
            return info;
        }
    };

    // 外壳型 VST3 扫一次要几分钟，给了 --cache 就把描述存下来，之后直接加载。
    juce::OwnedArray<juce::PluginDescription> scan (const juce::String& path, const juce::String& cachePath)
    {
        juce::OwnedArray<juce::PluginDescription> found;
        const juce::File cache (cachePath);

        if (cachePath.isNotEmpty() && cache.existsAsFile())
        {
            if (auto xml = juce::parseXML (cache))
                for (auto* element : xml->getChildIterator())
                {
                    auto description = std::make_unique<juce::PluginDescription>();
                    if (description->loadFromXml (*element))
                        found.add (std::move (description));
                }
            return found;
        }

        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);
        for (auto* format : manager.getFormats())
            if (format->fileMightContainThisPluginType (path))
                format->findAllTypesForFile (found, path);

        if (cachePath.isNotEmpty())
        {
            juce::XmlElement root ("PLUGINS");
            for (auto* description : found)
                root.addChildElement (description->createXml().release());
            root.writeTo (cache);
        }
        return found;
    }

    std::unique_ptr<juce::AudioPluginInstance> load (const juce::String& path, const juce::StringArray& args, double rate, int block)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);

        const auto name = args.contains ("--name") ? args[args.indexOf ("--name") + 1] : juce::String();
        auto found = scan (path, args.contains ("--cache") ? args[args.indexOf ("--cache") + 1] : juce::String());
        const juce::PluginDescription* chosen = nullptr;
        for (auto* description : found)
            if (name.isEmpty() || description->name.equalsIgnoreCase (name))
                if (chosen == nullptr)
                    chosen = description;

        if (chosen == nullptr)
        {
            std::cout << "FAIL no plugin '" << name << "' in " << path << " (" << found.size() << " types)\n";
            return nullptr;
        }

        juce::String error;
        auto instance = manager.createPluginInstance (*chosen, rate, block, error);
        if (instance == nullptr)
            std::cout << "FAIL could not load: " << error << "\n";
        return instance;
    }

    juce::String argValue (const juce::StringArray& args, const juce::String& key, const juce::String& fallback)
    {
        const auto index = args.indexOf (key);
        return index >= 0 && index + 1 < args.size() ? args[index + 1] : fallback;
    }

    void configureBuses (juce::AudioPluginInstance& plugin, const juce::StringArray& args)
    {
        plugin.enableAllBuses();
        if (args.contains ("--mono"))
        {
            juce::AudioProcessor::BusesLayout layout;
            layout.inputBuses.add (juce::AudioChannelSet::mono());
            layout.outputBuses.add (juce::AudioChannelSet::mono());
            if (! plugin.setBusesLayout (layout))
                std::cout << "FAIL mono layout refused\n";
        }
    }

    void setParameter (juce::AudioPluginInstance& plugin, const juce::String& name, float value)
    {
        for (auto* param : plugin.getParameters())
            if (param->getName (64) == name)
                param->setValueNotifyingHost (value);
    }

    void applySets (juce::AudioPluginInstance& plugin, const juce::StringArray& args)
    {
        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--set")
                setParameter (plugin,
                              args[i + 1].upToFirstOccurrenceOf ("=", false, false),
                              args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue());
    }

    // 让挂起的参数改动和 Timer 上报生效。
    void settle (juce::AudioPluginInstance& plugin, double rate, int block)
    {
        juce::AudioBuffer<float> buffer (juce::jmax (2, plugin.getTotalNumInputChannels(), plugin.getTotalNumOutputChannels()), block);
        juce::MidiBuffer midi;
        // 预跑的总采样数固定，LFO 相位才不会随块大小变化，不同块大小的输出可以直接比较。
        for (int remaining = (int) (rate * 0.05); remaining > 0; remaining -= block)
        {
            buffer.setSize (buffer.getNumChannels(), juce::jmin (block, remaining), false, false, true);
            buffer.clear();
            plugin.processBlock (buffer, midi);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (100);
    }

    int runList (const juce::String& path, const juce::StringArray& args)
    {
        for (auto* description : scan (path, args.contains ("--cache") ? args[args.indexOf ("--cache") + 1] : juce::String()))
            std::cout << description->name << "\t" << description->manufacturerName << "\t"
                      << description->fileOrIdentifier << "\n";
        return 0;
    }

    // 不实例化，只列工厂里的类。给了 --write-cache 就为指定的类写一份描述，之后用 --cache 加载。
    int runClasses (const juce::String& path, const juce::StringArray& args)
    {
        std::vector<FactoryClass> classes;
        if (! listFactoryClasses (path.toWideCharPointer(), classes))
        {
            std::cout << "FAIL cannot read factory of " << path << "\n";
            return 1;
        }

        const auto wanted = argValue (args, "--name", {});
        const auto cachePath = argValue (args, "--write-cache", {});
        juce::XmlElement root ("PLUGINS");

        for (const auto& item : classes)
        {
            if (item.category != "Audio Module Class")
                continue;

            std::cout << item.name << "\n";

            if (cachePath.isNotEmpty() && juce::String (item.name) == wanted)
            {
                // JUCE 按名字和 deprecatedUid 匹配类，deprecatedUid 是对原始类 ID 逐字节做的哈希。
                juce::uint32 hash = 0;
                for (const auto byte : item.cid)
                    hash = (hash * 31) + (juce::uint32) byte;

                juce::PluginDescription description;
                description.name = description.descriptiveName = item.name;
                description.pluginFormatName = "VST3";
                description.fileOrIdentifier = path;
                description.deprecatedUid = (int) hash;
                description.uniqueId = (int) hash;
                description.numInputChannels = description.numOutputChannels = 2;
                root.addChildElement (description.createXml().release());
            }
        }

        if (cachePath.isNotEmpty())
        {
            if (root.getNumChildElements() == 0)
            {
                std::cout << "FAIL no class named " << wanted << "\n";
                return 1;
            }
            root.writeTo (juce::File (cachePath));
        }

        // 外壳模块只调了 InitDll 就被留在进程里，正常退出时它的析构会崩溃。结果已经写完，直接结束。
        std::cout.flush();
        std::_Exit (0);
    }

    int runParams (const juce::String& path, const juce::StringArray& args)
    {
        const auto rate = argValue (args, "--rate", "48000").getDoubleValue();
        const auto block = 256;
        auto plugin = load (path, args, rate, block);
        if (plugin == nullptr)
            return 1;

        FakePlayHead head;
        head.bpm = argValue (args, "--bpm", "0").getDoubleValue();
        head.rate = rate;
        plugin->setPlayHead (&head);
        applySets (*plugin, args);
        configureBuses (*plugin, args);
        plugin->prepareToPlay (rate, block);
        settle (*plugin, rate, block);

        for (auto* param : plugin->getParameters())
            std::cout << param->getParameterIndex() << "\t" << param->getName (64) << "\t"
                      << param->getValue() << "\t" << param->getCurrentValueAsText() << "\n";
        std::cout << "latency\t" << plugin->getLatencySamples() << "\n";
        std::cout << "tail\t" << plugin->getTailLengthSeconds() << "\n";

        // --probe 名字：打印这个参数归一化值到显示文字的对应，对标插件的范围要靠它查。
        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--probe")
                for (auto* param : plugin->getParameters())
                    if (param->getName (64) == args[i + 1])
                        for (int step = 0; step <= 1000; ++step)
                            std::cout << "probe\t" << args[i + 1] << "\t" << (float) step * 0.001f << "\t"
                                      << param->getText ((float) step * 0.001f, 64) << "\n";

        plugin->releaseResources();
        return 0;
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

        const auto rate = reader->sampleRate;
        auto plugin = load (path, args, rate, block);
        if (plugin == nullptr)
            return 1;

        std::vector<Change> changes;
        std::vector<TempoChange> tempos;
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--at" && i + 2 < args.size())
                changes.push_back ({ args[i + 1].getDoubleValue(),
                                     args[i + 2].upToFirstOccurrenceOf ("=", false, false),
                                     args[i + 2].fromFirstOccurrenceOf ("=", false, false).getFloatValue() });
            else if (args[i] == "--bpm-at" && i + 2 < args.size())
                tempos.push_back ({ args[i + 1].getDoubleValue(), args[i + 2].getDoubleValue() });
        }

        FakePlayHead head;
        head.bpm = argValue (args, "--bpm", "0").getDoubleValue();
        head.rate = rate;
        plugin->setPlayHead (&head);
        applySets (*plugin, args);
        configureBuses (*plugin, args);
        plugin->prepareToPlay (rate, block);
        settle (*plugin, rate, block);
        const auto latency = plugin->getLatencySamples();

        const auto outputs = juce::jmax (2, plugin->getTotalNumOutputChannels());
        const auto channels = juce::jmax (outputs, plugin->getTotalNumInputChannels());
        const auto total = (int) reader->lengthInSamples;
        juce::AudioBuffer<float> all (channels, total + latency);
        all.clear();
        reader->read (&all, 0, total, 0, true, true);
        for (int ch = outputs; ch < channels; ++ch)
            all.clear (ch, 0, all.getNumSamples());

        juce::AudioBuffer<float> chunk (channels, block);
        juce::MidiBuffer midi;
        size_t nextChange = 0, nextTempo = 0;
        std::stable_sort (changes.begin(), changes.end(), [] (const Change& a, const Change& b) { return a.seconds < b.seconds; });
        std::stable_sort (tempos.begin(), tempos.end(), [] (const TempoChange& a, const TempoChange& b) { return a.seconds < b.seconds; });

        const auto started = std::chrono::steady_clock::now();
        for (int pos = 0; pos < all.getNumSamples(); pos += block)
        {
            const auto now = (double) pos / rate;
            for (; nextChange < changes.size() && changes[nextChange].seconds <= now; ++nextChange)
                setParameter (*plugin, changes[nextChange].name, changes[nextChange].value);
            for (; nextTempo < tempos.size() && tempos[nextTempo].seconds <= now; ++nextTempo)
                head.bpm = tempos[nextTempo].bpm;
            head.samples = pos;

            const auto n = juce::jmin (block, all.getNumSamples() - pos);
            chunk.setSize (channels, n, false, false, true);
            for (int ch = 0; ch < channels; ++ch)
                chunk.copyFrom (ch, 0, all, ch, pos, n);
            plugin->processBlock (chunk, midi);
            for (int ch = 0; ch < plugin->getTotalNumOutputChannels(); ++ch)
                all.copyFrom (ch, pos, chunk, ch, 0, n);
        }
        // 单声道总线时第二声道不是插件的输出。
        for (int ch = plugin->getTotalNumOutputChannels(); ch < outputs; ++ch)
            all.clear (ch, 0, all.getNumSamples());
        const auto elapsed = std::chrono::duration<double> (std::chrono::steady_clock::now() - started).count();

        std::cout << "realtime_ratio\t" << elapsed / ((double) all.getNumSamples() / rate) << "\n";
        std::cout << "latency\t" << latency << "\ntail\t" << plugin->getTailLengthSeconds() << "\n";
        plugin->releaseResources();

        // 32 位浮点，超过满度的样本和极低电平都能如实保存。
        output.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (output.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}.withSampleRate (rate).withNumChannels (outputs).withBitsPerSample (32)
                           .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write " << output.getFullPathName() << "\n";
            return 1;
        }

        juce::AudioBuffer<float> mainOutputs (all.getArrayOfWritePointers(), outputs, all.getNumSamples());
        writer->writeFromAudioSampleBuffer (mainOutputs, latency, total);
        return 0;
    }

    int runShot (const juce::String& path, const juce::File& output, const juce::StringArray& args)
    {
        auto plugin = load (path, args, 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        applySets (*plugin, args);
        plugin->prepareToPlay (48000.0, 256);
        settle (*plugin, 48000.0, 256);

        std::unique_ptr<juce::AudioProcessorEditor> editor (plugin->createEditorIfNeeded());
        if (editor == nullptr)
        {
            std::cout << "FAIL plugin has no editor\n";
            return 1;
        }

        juce::DocumentWindow window ("mint_delay_host", juce::Colours::black, 0);
        window.setUsingNativeTitleBar (true);
        window.setContentNonOwned (editor.get(), true);
        window.setTopLeftPosition (-6000, 100);
        window.setVisible (true);
        juce::MessageManager::getInstance()->runDispatchLoopUntil (1000);

        std::vector<unsigned char> pixels;
        int w = 0, h = 0;
        if (! captureWindowPixels (window.getWindowHandle(), pixels, w, h))
        {
            std::cout << "FAIL cannot capture window\n";
            return 1;
        }

        juce::Image image (juce::Image::ARGB, w, h, true);
        {
            const juce::Image::BitmapData data (image, juce::Image::BitmapData::writeOnly);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const auto* p = pixels.data() + ((size_t) y * (size_t) w + (size_t) x) * 4;
                    data.setPixelColour (x, y, juce::Colour (p[2], p[1], p[0], (juce::uint8) 255));
                }
        }

        window.clearContentComponent();
        editor.reset();
        plugin->releaseResources();

        output.deleteFile();
        juce::FileOutputStream stream (output);
        juce::PNGImageFormat png;
        if (! stream.openedOk() || ! png.writeImageToStream (image, stream))
        {
            std::cout << "FAIL cannot write " << output.getFullPathName() << "\n";
            return 1;
        }

        std::cout << "wrote " << image.getWidth() << "x" << image.getHeight() << "\n";
        return 0;
    }
}

// 用宽字符入口，中文 Windows 上含非 ASCII 字符的路径才不会被 GBK 解码弄坏。
int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (argv[i]));

    if (args.size() >= 2 && args[0] == "list")
        return runList (args[1], args);
    if (args.size() >= 2 && args[0] == "classes")
        return runClasses (args[1], args);
    if (args.size() >= 2 && args[0] == "params")
        return runParams (args[1], args);
    if (args.size() >= 4 && args[0] == "render")
        return runRender (args[1], juce::File (args[2]), juce::File (args[3]), args);
    if (args.size() >= 3 && args[0] == "shot")
        return runShot (args[1], juce::File (args[2]), args);

    std::cout << "usage: mint_delay_host list|params|render|shot <file.vst3> ...\n";
    return 2;
}
