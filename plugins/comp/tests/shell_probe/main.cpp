// 黑盒对比工具。mint_probe 只会加载文件里的第一个子插件，这里按名字选，别的做法和它的 render 一样。
//
//   atlas_shell_probe list   <shell.vst3>
//   atlas_shell_probe params <shell.vst3> <subPluginName>
//   atlas_shell_probe texts  <shell.vst3> <subPluginName> <paramName>
//   atlas_shell_probe render <shell.vst3> <subPluginName> <in.wav> <out.wav> [--set name=normalised]...
//                            [--at seconds:name=normalised]... [--mono] [--block 256]
//
// 只通过标准 VST3 宿主接口加载和渲染，不读取任何资源文件。
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
    // VST3 模块工厂接口的二进制布局（Steinberg 公开 SDK 的 IPluginFactory 和 PClassInfo），每个 VST3 宿主都这样列举类。
    struct ClassInfo
    {
        char cid[16];
        juce::int32 cardinality;
        char category[32];
        char name[64];
    };

    struct Factory
    {
        virtual juce::int32 __stdcall queryInterface (const char*, void**) = 0;
        virtual juce::uint32 __stdcall addRef() = 0;
        virtual juce::uint32 __stdcall release() = 0;
        virtual juce::int32 __stdcall getFactoryInfo (void*) = 0;
        virtual juce::int32 __stdcall countClasses() = 0;
        virtual juce::int32 __stdcall getClassInfo (juce::int32, ClassInfo*) = 0;
    };

    // 外壳文件里的子插件很多。JUCE 的 findAllTypesForFile 会逐个实例化，既慢（每个外壳几分钟），
    // 遇到有问题的子插件还会让整个进程崩溃。这里只读工厂的类清单，用到哪个才实例化哪个。
    juce::OwnedArray<juce::PluginDescription> listClasses (const juce::String& path)
    {
        juce::OwnedArray<juce::PluginDescription> found;

        auto file = juce::File (path);
        if (file.isDirectory())
            file = file.getChildFile ("Contents/x86_64-win").getChildFile (file.getFileName());

        // 故意不释放：之后 JUCE 会加载同一个模块。
        auto* library = new juce::DynamicLibrary (file.getFullPathName());
        using InitFunction = bool (__stdcall*)();
        using GetFactoryFunction = Factory* (__stdcall*)();

        if (auto init = (InitFunction) library->getFunction ("InitDll"))
            init();

        auto getFactory = (GetFactoryFunction) library->getFunction ("GetPluginFactory");
        auto* factory = getFactory != nullptr ? getFactory() : nullptr;
        if (factory == nullptr)
            return found;

        for (juce::int32 i = 0; i < factory->countClasses(); ++i)
        {
            ClassInfo info {};
            if (factory->getClassInfo (i, &info) != 0 || std::strcmp (info.category, "Audio Module Class") != 0)
                continue;

            // 与 JUCE 里 deprecatedUid 的算法一致，它靠名字和这个哈希找到要实例化的类。
            juce::uint32 hash = 0;
            for (const auto item : info.cid)
                hash = hash * 31 + (juce::uint32) item;

            auto description = std::make_unique<juce::PluginDescription>();
            description->name = juce::String (info.name).trim();
            description->pluginFormatName = "VST3";
            description->fileOrIdentifier = path;
            description->deprecatedUid = (int) hash;
            found.add (description.release());
        }

        factory->release();
        return found;
    }

    std::unique_ptr<juce::AudioPluginInstance> load (juce::AudioPluginFormatManager& manager, const juce::String& path,
                                                     const juce::String& name, double rate, int block)
    {
        auto found = listClasses (path);
        for (auto* description : found)
        {
            if (description->name != name)
                continue;

            juce::String error;
            auto instance = manager.createPluginInstance (*description, rate, block, error);
            if (instance == nullptr)
                std::cout << "FAIL could not load: " << error << "\n";
            return instance;
        }

        std::cout << "FAIL no sub-plugin named " << name << "\n";
        return nullptr;
    }

    int runRender (juce::AudioPluginInstance& plugin, const juce::File& output, juce::AudioFormatReader& reader, int block,
                   const juce::StringArray& args)
    {
        const auto rate = reader.sampleRate;

        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--set")
                continue;
            const auto key = args[i + 1].upToFirstOccurrenceOf ("=", false, false);
            const auto value = args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue();
            auto matched = false;
            for (auto* param : plugin.getParameters())
                if (param->getName (64) == key)
                {
                    param->setValueNotifyingHost (value);
                    matched = true;
                }
            if (! matched)
                std::cout << "WARN no parameter named " << key << "\n";
        }

        plugin.enableAllBuses();
        if (args.contains ("--mono"))
        {
            juce::AudioProcessor::BusesLayout layout;
            layout.inputBuses.add (juce::AudioChannelSet::mono());
            layout.outputBuses.add (juce::AudioChannelSet::mono());
            if (! plugin.setBusesLayout (layout))
            {
                std::cout << "FAIL mono layout refused\n";
                return 1;
            }
        }
        plugin.prepareToPlay (rate, block);

        const auto channels = juce::jmax (2, plugin.getTotalNumInputChannels(), plugin.getTotalNumOutputChannels());
        const auto outputs = juce::jmax (1, plugin.getTotalNumOutputChannels());

        juce::AudioBuffer<float> settle (channels, block);
        juce::MidiBuffer midi;
        for (int i = 0; i < (int) (rate * 0.5) / block; ++i)
        {
            settle.clear();
            plugin.processBlock (settle, midi);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (500);
        const auto latency = plugin.getLatencySamples();

        const auto total = (int) reader.lengthInSamples;
        juce::AudioBuffer<float> all (channels, total + latency);
        all.clear();
        reader.read (&all, 0, total, 0, true, true);

        struct Change { int sample; juce::String name; float value; bool done; };
        std::vector<Change> changes;
        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--at")
                continue;
            const auto seconds = args[i + 1].upToFirstOccurrenceOf (":", false, false).getDoubleValue();
            const auto rest = args[i + 1].fromFirstOccurrenceOf (":", false, false);
            changes.push_back ({ (int) (seconds * rate), rest.upToFirstOccurrenceOf ("=", false, false),
                                 rest.fromFirstOccurrenceOf ("=", false, false).getFloatValue(), false });
        }

        juce::AudioBuffer<float> chunk (channels, block);
        for (int pos = 0; pos < all.getNumSamples(); pos += block)
        {
            for (auto& change : changes)
                if (! change.done && pos >= change.sample)
                {
                    for (auto* param : plugin.getParameters())
                        if (param->getName (64) == change.name)
                            param->setValueNotifyingHost (change.value);
                    change.done = true;
                }

            const auto n = juce::jmin (block, all.getNumSamples() - pos);
            chunk.clear();
            for (int ch = 0; ch < channels; ++ch)
                chunk.copyFrom (ch, 0, all, ch, pos, n);
            plugin.processBlock (chunk, midi);
            for (int ch = 0; ch < outputs; ++ch)
                all.copyFrom (ch, pos, chunk, ch, 0, n);
        }

        plugin.releaseResources();

        output.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (output.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}.withSampleRate (rate).withNumChannels (outputs).withBitsPerSample (32);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write " << output.getFullPathName() << "\n";
            return 1;
        }

        juce::AudioBuffer<float> mainOutputs (all.getArrayOfWritePointers(), outputs, all.getNumSamples());
        writer->writeFromAudioSampleBuffer (mainOutputs, latency, total);
        std::cout << "rendered " << total << " samples, reported latency " << latency << "\n";
        return 0;
    }
}

// 有的外壳在进程退出时崩溃。结果到这里已经写完，直接结束，不走析构。
static int finish (int result)
{
    std::cout.flush();
    std::_Exit (result);
}

int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (argv[i]));

    juce::AudioPluginFormatManager manager;
    juce::addDefaultFormatsToManager (manager);

    if (args.size() >= 2 && args[0] == "list")
    {
        for (auto* description : listClasses (args[1]))
            std::cout << description->name << "\n";
        return finish (0);
    }

    if (args.size() >= 3 && args[0] == "params")
    {
        auto plugin = load (manager, args[1], args[2], 48000.0, 256);
        if (plugin == nullptr)
            return 1;
        for (auto* param : plugin->getParameters())
            std::cout << param->getParameterIndex() << "\t" << param->getName (64) << "\t"
                      << param->getValue() << "\t" << param->getCurrentValueAsText() << "\n";
        return finish (0);
    }

    // 打印某个参数的归一化值和显示文字的对应表，脚本用它把物理值换算成归一化值。
    if (args.size() >= 4 && args[0] == "texts")
    {
        auto plugin = load (manager, args[1], args[2], 48000.0, 256);
        if (plugin == nullptr)
            return 1;
        for (auto* param : plugin->getParameters())
            if (param->getName (64) == args[3])
                for (int i = 0; i <= 400; ++i)
                    std::cout << (float) i / 400.0f << "\t" << param->getText ((float) i / 400.0f, 64) << "\n";
        return finish (0);
    }

    if (args.size() >= 5 && args[0] == "render")
    {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (juce::File (args[3])));
        if (reader == nullptr)
        {
            std::cout << "FAIL cannot read " << args[3] << "\n";
            return 1;
        }

        const auto blockIndex = args.indexOf ("--block");
        const auto block = blockIndex >= 0 ? args[blockIndex + 1].getIntValue() : 256;

        auto plugin = load (manager, args[1], args[2], reader->sampleRate, block);
        if (plugin == nullptr)
            return 1;
        return finish (runRender (*plugin, juce::File (args[4]), *reader, block, args));
    }

    std::cout << "usage: atlas_shell_probe list|params|render <shell.vst3> ...\n";
    return 2;
}
