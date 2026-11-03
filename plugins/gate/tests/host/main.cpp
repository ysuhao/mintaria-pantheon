// Janus 的测试宿主。mint_probe 不喂侧链、不选 Shell 子插件，这里补上，并支持渲染中途改参数。
// 既加载 Janus，也把对标插件当黑盒加载，只走标准 VST3 接口。
//
//   gate_host list   <plugin.vst3> [--cache descriptions.xml]
//   gate_host classes <plugin.vst3> [--match TEXT] [--cache descriptions.xml]
//   gate_host params <plugin.vst3> [--sub NAME] [--sweep NAME]   --sweep 打印该参数在 0..1 上的显示值，用来读映射
//   gate_host render <plugin.vst3> <in.wav> <out.wav> [--sub NAME] [--sc sc.wav] [--block N]
//                    [--set name=normalised]... [--at seconds:name=normalised]... [--no-sc-bus]
//                    [--bypass-at seconds:0|1]... [--mono] [--cache descriptions.xml]
//   gate_host shot   <plugin.vst3> <out.png> [--set name=normalised]... [--tone dB] [--sc-tone dB]
//                    --tone、--sc-tone 截图时持续喂 1 kHz 正弦，用来看电平表和侧链状态。
//
// 大 Shell 的完整扫描要实例化每个子插件，很慢，遇到一个闪退的就全没了。
// classes 只走 VST3 工厂的类列表，不实例化任何子插件。带 --cache 时把描述存下来，后面用同一个 --cache 直接加载。
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <windows.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

bool captureWindowPixels (void* windowHandle, std::vector<unsigned char>& pixels, int& width, int& height);

namespace
{
    juce::String argValue (const juce::StringArray& args, const juce::String& key, const juce::String& fallback = {})
    {
        const auto index = args.indexOf (key);
        return index >= 0 && index + 1 < args.size() ? args[index + 1] : fallback;
    }

    juce::StringArray argValues (const juce::StringArray& args, const juce::String& key)
    {
        juce::StringArray values;
        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == key)
                values.add (args[i + 1]);
        return values;
    }

    void scan (juce::AudioPluginFormatManager& manager, const juce::String& path, juce::OwnedArray<juce::PluginDescription>& found)
    {
        for (auto* format : manager.getFormats())
            if (format->fileMightContainThisPluginType (path))
                format->findAllTypesForFile (found, path);
    }

    std::unique_ptr<juce::AudioPluginInstance> load (juce::AudioPluginFormatManager& manager, const juce::String& path,
                                                     const juce::String& sub, const juce::String& cache, double rate, int block)
    {
        juce::OwnedArray<juce::PluginDescription> found;
        if (cache.isNotEmpty() && juce::File (cache).existsAsFile())
        {
            if (auto xml = juce::parseXML (juce::File (cache)))
                for (auto* item : xml->getChildIterator())
                    if (auto d = std::make_unique<juce::PluginDescription>(); d->loadFromXml (*item))
                        found.add (d.release());
        }
        else
        {
            scan (manager, path, found);
        }

        const juce::PluginDescription* chosen = nullptr;
        for (auto* d : found)
            if (sub.isEmpty() || d->name == sub)
            {
                chosen = d;
                break;
            }

        if (chosen == nullptr)
        {
            std::cout << "FAIL plugin not found: " << sub << "\n";
            return nullptr;
        }

        juce::String error;
        auto instance = manager.createPluginInstance (*chosen, rate, block, error);
        if (instance == nullptr)
            std::cout << "FAIL could not load: " << error << "\n";
        return instance;
    }

    void setParam (juce::AudioPluginInstance& plugin, const juce::String& name, float value)
    {
        for (auto* param : plugin.getParameters())
            if (param->getName (64) == name || juce::String (param->getParameterIndex()) == name)
            {
                param->setValueNotifyingHost (value);
                return;
            }
        std::cout << "WARN no parameter named " << name << "\n";
    }

    struct Automation
    {
        juce::int64 position;
        juce::String name;
        float value;
    };

    int runList (const juce::String& path, const juce::StringArray& args)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);
        juce::OwnedArray<juce::PluginDescription> found;
        scan (manager, path, found);
        for (auto* d : found)
            std::cout << d->name << "\t" << d->numInputChannels << "in\t" << d->numOutputChannels << "out\n";

        if (args.contains ("--cache"))
        {
            juce::XmlElement root ("PLUGINS");
            for (auto* d : found)
                root.addChildElement (d->createXml().release());
            root.writeTo (juce::File (argValue (args, "--cache")));
        }
        return 0;
    }

    // VST3 工厂接口的最小声明，布局和调用约定来自公开的 VST3 规范。
    struct FactoryClassInfo
    {
        char cid[16];
        int cardinality;
        char category[32];
        char name[64];
    };

    struct PluginFactory
    {
        virtual int __stdcall queryInterface (const char*, void**) = 0;
        virtual unsigned int __stdcall addRef() = 0;
        virtual unsigned int __stdcall release() = 0;
        virtual int __stdcall getFactoryInfo (void*) = 0;
        virtual int __stdcall countClasses() = 0;
        virtual int __stdcall getClassInfo (int, FactoryClassInfo*) = 0;
    };

    int runClasses (const juce::String& path, const juce::StringArray& args)
    {
        // Shell 文件是单个 DLL。
        auto* module = LoadLibraryW (path.toWideCharPointer());
        if (module == nullptr)
        {
            std::cout << "FAIL cannot load module " << (int) GetLastError() << "\n";
            return 1;
        }

        if (auto* init = reinterpret_cast<bool (__stdcall*)()> (GetProcAddress (module, "InitDll")))
            init();

        auto* getFactory = reinterpret_cast<PluginFactory* (__stdcall*)()> (GetProcAddress (module, "GetPluginFactory"));
        auto* factory = getFactory != nullptr ? getFactory() : nullptr;
        if (factory == nullptr)
        {
            std::cout << "FAIL no plugin factory\n";
            return 1;
        }

        const auto match = argValue (args, "--match");
        juce::XmlElement root ("PLUGINS");

        for (int i = 0; i < factory->countClasses(); ++i)
        {
            FactoryClassInfo info {};
            if (factory->getClassInfo (i, &info) != 0 || juce::String (info.category) != "Audio Module Class")
                continue;

            const auto name = juce::String (juce::CharPointer_UTF8 (info.name)).trim();
            if (match.isNotEmpty() && ! name.containsIgnoreCase (match))
                continue;

            std::cout << name << "\n";

            // 和 JUCE 宿主用同样的方式由类 ID 算出标识，加载时它只按名字和这个标识找类。
            juce::uint32 deprecated = 0;
            for (auto c : info.cid)
                deprecated = deprecated * 31 + (juce::uint32) c;

            juce::uint32 normalised = 0;
            for (int word = 0; word < 4; ++word)
            {
                juce::uint32 v = 0;
                for (int b = 0; b < 4; ++b)
                    v |= (juce::uint32) (juce::uint8) info.cid[word * 4 + b] << (8 * b);

                // 规范化的 TUID 在 Windows 上把前两个字段按 COM GUID 的字节序重排。
                if (word == 1)
                    v = (v >> 16) | (v << 16);
                else if (word >= 2)
                    v = juce::ByteOrder::swap (v);

                normalised = normalised * 31 + v;
            }

            juce::PluginDescription d;
            d.name = name;
            d.pluginFormatName = "VST3";
            d.fileOrIdentifier = path;
            d.deprecatedUid = (int) deprecated;
            d.uniqueId = (int) normalised;
            root.addChildElement (d.createXml().release());
        }

        if (args.contains ("--cache"))
            root.writeTo (juce::File (argValue (args, "--cache")));

        return 0;
    }

    int runParams (const juce::String& path, const juce::StringArray& args)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);
        auto plugin = load (manager, path, argValue (args, "--sub"), argValue (args, "--cache"), 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        for (auto* param : plugin->getParameters())
            std::cout << param->getParameterIndex() << "\t" << param->getName (64) << "\t"
                      << param->getValue() << "\t" << param->getCurrentValueAsText() << "\n";

        std::cout << "buses in=" << plugin->getBusCount (true) << " out=" << plugin->getBusCount (false)
                  << " latency=" << plugin->getLatencySamples() << "\n";

        for (const auto& name : argValues (args, "--sweep"))
            for (auto* param : plugin->getParameters())
                if (param->getName (64) == name)
                {
                    for (int i = 0; i <= 20; ++i)
                    {
                        param->setValueNotifyingHost ((float) i / 20.0f);
                        juce::MessageManager::getInstance()->runDispatchLoopUntil (20);
                        std::cout << name << "\t" << (float) i / 20.0f << "\t" << param->getValue() << "\t" << param->getCurrentValueAsText() << "\n";
                    }
                }
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
        const auto total = (int) reader->lengthInSamples;

        juce::AudioBuffer<float> sidechain;
        if (args.contains ("--sc"))
        {
            std::unique_ptr<juce::AudioFormatReader> scReader (formats.createReaderFor (juce::File (argValue (args, "--sc"))));
            if (scReader == nullptr)
            {
                std::cout << "FAIL cannot read sidechain\n";
                return 1;
            }
            sidechain.setSize (2, total + (int) rate);
            sidechain.clear();
            scReader->read (&sidechain, 0, juce::jmin (total, (int) scReader->lengthInSamples), 0, true, true);
        }

        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);
        auto plugin = load (manager, path, argValue (args, "--sub"), argValue (args, "--cache"), rate, block);
        if (plugin == nullptr)
            return 1;

        for (const auto& item : argValues (args, "--set"))
            setParam (*plugin, item.upToFirstOccurrenceOf ("=", false, false),
                      item.fromFirstOccurrenceOf ("=", false, false).getFloatValue());

        std::vector<Automation> automation;
        for (const auto& item : argValues (args, "--at"))
        {
            const auto rest = item.fromFirstOccurrenceOf (":", false, false);
            automation.push_back ({ (juce::int64) (item.upToFirstOccurrenceOf (":", false, false).getDoubleValue() * rate),
                                    rest.upToFirstOccurrenceOf ("=", false, false),
                                    rest.fromFirstOccurrenceOf ("=", false, false).getFloatValue() });
        }

        std::map<juce::int64, bool> bypassChanges;
        for (const auto& item : argValues (args, "--bypass-at"))
            bypassChanges[(juce::int64) (item.upToFirstOccurrenceOf (":", false, false).getDoubleValue() * rate)]
                = item.fromFirstOccurrenceOf (":", false, false).getIntValue() != 0;

        auto layout = plugin->getBusesLayout();
        if (args.contains ("--mono"))
        {
            layout.inputBuses.getReference (0) = juce::AudioChannelSet::mono();
            layout.outputBuses.getReference (0) = juce::AudioChannelSet::mono();
        }
        if (layout.inputBuses.size() > 1)
            layout.inputBuses.getReference (1) = args.contains ("--no-sc-bus") ? juce::AudioChannelSet::disabled()
                                                                               : juce::AudioChannelSet::stereo();
        if (! plugin->setBusesLayout (layout))
            std::cout << "WARN layout refused\n";

        plugin->setRateAndBufferSizeDetails (rate, block);
        plugin->prepareToPlay (rate, block);

        const auto mainIns  = plugin->getMainBusNumInputChannels();
        const auto mainOuts = plugin->getMainBusNumOutputChannels();
        const auto channels = juce::jmax (plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels());
        const auto scChannels = plugin->getTotalNumInputChannels() - mainIns;

        // 多跑一段让参数平滑和 Timer 上报的延迟稳定下来。
        juce::AudioBuffer<float> chunk (channels, block);
        juce::MidiBuffer midi;
        // 预跑的总长度不随缓冲区大小变化，不同缓冲区的渲染结果才能逐采样比较。
        for (int remaining = (int) (rate * 0.2); remaining > 0; remaining -= block)
        {
            chunk.clear();
            juce::AudioBuffer<float> alias (chunk.getArrayOfWritePointers(), channels, juce::jmin (block, remaining));
            plugin->processBlock (alias, midi);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto latency = plugin->getLatencySamples();

        juce::AudioBuffer<float> source (2, total + latency);
        source.clear();
        reader->read (&source, 0, total, 0, true, true);
        if (reader->numChannels == 1)
            source.copyFrom (1, 0, source, 0, 0, total);

        juce::AudioBuffer<float> result (juce::jmax (1, mainOuts), total + latency);
        result.clear();

        std::sort (automation.begin(), automation.end(), [] (const auto& a, const auto& b) { return a.position < b.position; });
        size_t nextAutomation = 0;
        auto bypassed = false;
        auto* bypassParam = plugin->getBypassParameter();

        for (int pos = 0; pos < result.getNumSamples(); pos += block)
        {
            while (nextAutomation < automation.size() && automation[nextAutomation].position <= pos)
            {
                setParam (*plugin, automation[nextAutomation].name, automation[nextAutomation].value);
                ++nextAutomation;
            }

            for (const auto& [position, state] : bypassChanges)
                if (position > pos - block && position <= pos)
                {
                    bypassed = state;
                    if (bypassParam != nullptr)
                        bypassParam->setValueNotifyingHost (state ? 1.0f : 0.0f);
                }

            const auto n = juce::jmin (block, result.getNumSamples() - pos);
            chunk.clear();
            for (int ch = 0; ch < mainIns; ++ch)
                chunk.copyFrom (ch, 0, source, juce::jmin (ch, 1), pos, n);
            if (sidechain.getNumSamples() >= pos + n)
                for (int ch = 0; ch < scChannels; ++ch)
                    chunk.copyFrom (mainIns + ch, 0, sidechain, juce::jmin (ch, 1), pos, n);

            // 尾块不足时缓冲区长度也要等于 n。
            juce::AudioBuffer<float> alias (chunk.getArrayOfWritePointers(), channels, n);
            if (bypassed)
                plugin->processBlockBypassed (alias, midi);
            else
                plugin->processBlock (alias, midi);

            for (int ch = 0; ch < mainOuts; ++ch)
                result.copyFrom (ch, pos, chunk, ch, 0, n);
        }

        const auto finalLatency = plugin->getLatencySamples();
        plugin->releaseResources();

        output.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (output.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}.withSampleRate (rate).withNumChannels (mainOuts).withBitsPerSample (32);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write " << output.getFullPathName() << "\n";
            return 1;
        }

        writer->writeFromAudioSampleBuffer (result, latency, total);
        writer.reset();
        std::cout << "rendered " << total << " samples, reported latency " << latency << " final " << finalLatency << "\n";
        return 0;
    }

    // 与 mint_probe shot 相同，多一步设参数，用来看闪避模式下才出现的界面。
    int runShot (const juce::String& path, const juce::File& output, const juce::StringArray& args)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);
        auto plugin = load (manager, path, argValue (args, "--sub"), argValue (args, "--cache"), 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        for (const auto& item : argValues (args, "--set"))
            setParam (*plugin, item.upToFirstOccurrenceOf ("=", false, false),
                      item.fromFirstOccurrenceOf ("=", false, false).getFloatValue());

        if (args.contains ("--sc-tone"))
        {
            auto layout = plugin->getBusesLayout();
            if (layout.inputBuses.size() > 1)
                layout.inputBuses.getReference (1) = juce::AudioChannelSet::stereo();
            plugin->setBusesLayout (layout);
        }

        plugin->prepareToPlay (48000.0, 256);

        const auto mainIns = plugin->getMainBusNumInputChannels();
        const auto channels = juce::jmax (plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels());
        const auto toneGain = juce::Decibels::decibelsToGain (argValue (args, "--tone", "-200").getFloatValue());
        const auto scGain = juce::Decibels::decibelsToGain (argValue (args, "--sc-tone", "-200").getFloatValue());
        juce::AudioBuffer<float> chunk (channels, 256);
        juce::MidiBuffer midi;
        auto phase = 0.0f;

        const auto feed = [&] (int blocks)
        {
            for (int block = 0; block < blocks; ++block)
            {
                for (int i = 0; i < 256; ++i)
                {
                    phase += juce::MathConstants<float>::twoPi * 1000.0f / 48000.0f;
                    const auto x = std::sin (phase);
                    for (int ch = 0; ch < channels; ++ch)
                        chunk.setSample (ch, i, x * (ch < mainIns ? toneGain : scGain));
                }

                plugin->processBlock (chunk, midi);
            }
        };

        // 屏幕外的窗口只有第一帧抓得到，所以先把音频跑到稳态，再开窗口。
        feed (400);
        std::cout << "output peak after tone: " << juce::Decibels::gainToDecibels (chunk.getMagnitude (0, 0, 256)) << " dB\n";

        std::unique_ptr<juce::AudioProcessorEditor> editor (plugin->createEditorIfNeeded());
        if (editor == nullptr)
        {
            std::cout << "FAIL plugin has no editor\n";
            return 1;
        }

        juce::DocumentWindow window ("gate_host", juce::Colours::black, 0);
        window.setUsingNativeTitleBar (true);
        window.setContentNonOwned (editor.get(), true);
        window.setTopLeftPosition (-6000, 100);

        // 电平表 30 Hz 取值，窗口显示前让它们先取到值。
        for (int i = 0; i < 12; ++i)
        {
            feed (4);
            juce::MessageManager::getInstance()->runDispatchLoopUntil (20);
        }

        window.setVisible (true);

        for (int i = 0; i < 60; ++i)
        {
            feed (2);
            juce::MessageManager::getInstance()->runDispatchLoopUntil (20);
        }

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

    std::cout << "usage: gate_host list|params|render|shot <plugin.vst3> ...\n";
    return 2;
}
