// 通用 VST3 检测工具，既能查自研插件，也能对对标插件做黑盒渲染，用于 A/B。
//
//   mint_probe check  <plugin.vst3> [--expect-latency N]
//   mint_probe render <plugin.vst3> <in.wav> <out.wav> [--set paramId=normalised]... [--rate 48000] [--block 256]
//   mint_probe params <plugin.vst3>
//   mint_probe shot   <plugin.vst3> <out.png>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <iostream>
#include <vector>

bool captureWindowPixels (void* windowHandle, std::vector<unsigned char>& pixels, int& width, int& height);

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

    bool hasBadSamples (const juce::AudioBuffer<float>& buffer)
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                if (! std::isfinite (buffer.getSample (ch, i)))
                    return true;
        return false;
    }

    int runCheck (const juce::String& path, const juce::StringArray& args)
    {
        auto failures = 0;
        const auto expected = argValue (args, "--expect-latency", "-1").getIntValue();

        for (const auto rate : { 44100.0, 48000.0, 96000.0 })
        {
            for (const auto block : { 32, 64, 256, 1024 })
            {
                auto plugin = load (path, rate, block);
                if (plugin == nullptr)
                    return 1;

                plugin->setRateAndBufferSizeDetails (rate, block);
                plugin->enableAllBuses();
                plugin->prepareToPlay (rate, block);

                juce::AudioBuffer<float> buffer (juce::jmax (2, plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels()), block);
                juce::MidiBuffer midi;
                juce::Random random (1);
                auto bad = false;

                for (int n = 0; n < (int) (rate * 2.0) / block; ++n)
                {
                    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                        for (int i = 0; i < block; ++i)
                            buffer.setSample (ch, i, n < 4 ? (random.nextFloat() * 2.0f - 1.0f)
                                                           : 0.5f * std::sin (0.05f * (float) (n * block + i)));

                    plugin->processBlock (buffer, midi);
                    bad = bad || hasBadSamples (buffer);
                }

                plugin->releaseResources();

                if (bad)
                {
                    std::cout << "FAIL non-finite output at " << rate << " Hz, block " << block << "\n";
                    ++failures;
                }
            }
        }

        auto plugin = load (path, 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        plugin->prepareToPlay (48000.0, 256);
        const auto latency = plugin->getLatencySamples();
        std::cout << "name: " << plugin->getName() << "\nlatency: " << latency << " samples\nparams: "
                  << plugin->getParameters().size() << "\n";

        if (expected >= 0 && latency != expected)
        {
            std::cout << "FAIL expected latency " << expected << "\n";
            ++failures;
        }

        juce::MemoryBlock first, second;
        plugin->getStateInformation (first);
        plugin->setStateInformation (first.getData(), (int) first.getSize());
        plugin->getStateInformation (second);
        if (first != second)
        {
            std::cout << "FAIL state does not round-trip\n";
            ++failures;
        }

        plugin->releaseResources();
        std::cout << (failures == 0 ? "PASS\n" : "FAILED\n");
        return failures == 0 ? 0 : 1;
    }

    int runParams (const juce::String& path)
    {
        auto plugin = load (path, 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        for (auto* param : plugin->getParameters())
            std::cout << param->getParameterIndex() << "\t" << param->getName (64) << "\t"
                      << param->getValue() << "\t" << param->getCurrentValueAsText() << "\n";
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
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--set")
                continue;
            const auto index = args[i + 1].upToFirstOccurrenceOf ("=", false, false);
            const auto value = args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue();
            for (auto* param : plugin->getParameters())
                if (param->getName (64) == index || juce::String (param->getParameterIndex()) == index)
                    param->setValueNotifyingHost (value);
        }

        plugin->enableAllBuses();
        plugin->prepareToPlay (rate, block);

        // 多跑一段让参数平滑和 Timer 上报的延迟稳定下来。
        juce::AudioBuffer<float> settle (juce::jmax (2, plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels()), block);
        juce::MidiBuffer midi;
        for (int i = 0; i < (int) (rate * 0.2) / block; ++i)
        {
            settle.clear();
            plugin->processBlock (settle, midi);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto latency = plugin->getLatencySamples();

        // 带侧链的插件，缓冲区要包含侧链通道。render 不喂侧链，侧链保持静音。
        const auto outputs = juce::jmax (2, plugin->getTotalNumOutputChannels());
        const auto channels = juce::jmax (outputs, plugin->getTotalNumInputChannels());
        const auto total = (int) reader->lengthInSamples;
        juce::AudioBuffer<float> all (channels, total + latency);
        all.clear();
        reader->read (&all, 0, total, 0, true, true);
        for (int ch = outputs; ch < channels; ++ch)
            all.clear (ch, 0, all.getNumSamples());

        juce::AudioBuffer<float> chunk (channels, block);
        for (int pos = 0; pos < all.getNumSamples(); pos += block)
        {
            const auto n = juce::jmin (block, all.getNumSamples() - pos);
            chunk.clear();
            for (int ch = 0; ch < channels; ++ch)
                chunk.copyFrom (ch, 0, all, ch, pos, n);
            plugin->processBlock (chunk, midi);
            for (int ch = 0; ch < outputs; ++ch)
                all.copyFrom (ch, pos, chunk, ch, 0, n);
        }

        plugin->releaseResources();

        output.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::OutputStream> stream (output.createOutputStream());
        auto options = juce::AudioFormatWriterOptions {}.withSampleRate (rate).withNumChannels (outputs).withBitsPerSample (24);
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, options));
        if (writer == nullptr)
        {
            std::cout << "FAIL cannot write " << output.getFullPathName() << "\n";
            return 1;
        }

        // 去掉插件报告的延迟，输出与输入对齐，便于直接做差。
        juce::AudioBuffer<float> mainOutputs (all.getArrayOfWritePointers(), outputs, all.getNumSamples());
        writer->writeFromAudioSampleBuffer (mainOutputs, latency, total);
        std::cout << "rendered " << total << " samples, reported latency " << latency << "\n";
        return 0;
    }

    int runShot (const juce::String& path, const juce::File& output)
    {
        auto plugin = load (path, 48000.0, 256);
        if (plugin == nullptr)
            return 1;

        plugin->prepareToPlay (48000.0, 256);
        std::unique_ptr<juce::AudioProcessorEditor> editor (plugin->createEditorIfNeeded());
        if (editor == nullptr)
        {
            std::cout << "FAIL plugin has no editor\n";
            return 1;
        }

        // 宿主加载的 VST3 界面是原生子窗口，离屏快照抓不到，必须放进真实窗口再抓。
        // 窗口放在屏幕外，不打扰正在用电脑的人。
        juce::DocumentWindow window ("mint_probe", juce::Colours::black, 0);
        window.setUsingNativeTitleBar (true);
        window.setContentNonOwned (editor.get(), true);
        window.setTopLeftPosition (-6000, 100);
        window.setVisible (true);
        juce::MessageManager::getInstance()->runDispatchLoopUntil (1200);

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

    if (args.size() >= 2 && args[0] == "check")
        return runCheck (args[1], args);
    if (args.size() >= 2 && args[0] == "params")
        return runParams (args[1]);
    if (args.size() >= 4 && args[0] == "render")
        return runRender (args[1], juce::File (args[2]), juce::File (args[3]), args);
    if (args.size() >= 3 && args[0] == "shot")
        return runShot (args[1], juce::File (args[2]));

    std::cout << "usage: mint_probe check|params|render|shot <plugin.vst3> ...\n";
    return 2;
}
