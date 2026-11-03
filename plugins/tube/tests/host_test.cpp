// mint_probe 覆盖不到的项目：单声道、1 采样缓冲区、运行中切换档位、参数自动化、非法输入、静音、CPU。
//
//   mint_tube_host_test <plugin.vst3>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <chrono>
#include <cmath>
#include <iostream>

namespace
{
    int failures = 0;

    void report (const juce::String& name, bool ok, const juce::String& detail = {})
    {
        std::cout << (ok ? "PASS " : "FAIL ") << name << (detail.isEmpty() ? "" : "  " + detail) << "\n";
        if (! ok)
            ++failures;
    }

    std::unique_ptr<juce::AudioPluginInstance> load (const juce::String& path, double rate, int block)
    {
        juce::AudioPluginFormatManager manager;
        juce::addDefaultFormatsToManager (manager);
        juce::OwnedArray<juce::PluginDescription> found;
        for (auto* format : manager.getFormats())
            if (format->fileMightContainThisPluginType (path))
                format->findAllTypesForFile (found, path);
        if (found.isEmpty())
            return nullptr;
        juce::String error;
        return manager.createPluginInstance (*found[0], rate, block, error);
    }

    juce::AudioProcessorParameter* param (juce::AudioPluginInstance& plugin, const juce::String& name)
    {
        for (auto* p : plugin.getParameters())
            if (p->getName (64) == name)
                return p;
        return nullptr;
    }

    struct Scan
    {
        bool bad = false;
        float peak = 0.0f, maxStep = 0.0f, previous = 0.0f;
        bool started = false;

        void add (const juce::AudioBuffer<float>& buffer, int numChannels)
        {
            for (int ch = 0; ch < numChannels; ++ch)
                for (int i = 0; i < buffer.getNumSamples(); ++i)
                {
                    const auto v = buffer.getSample (ch, i);
                    if (! std::isfinite (v) || (v != 0.0f && std::fpclassify (v) == FP_SUBNORMAL))
                        bad = true;
                    peak = juce::jmax (peak, std::abs (v));
                    if (ch == 0)
                    {
                        if (started)
                            maxStep = juce::jmax (maxStep, std::abs (v - previous));
                        previous = v;
                        started = true;
                    }
                }
        }
    };

    // 流式处理：fill 写入输入，scan 收集输出。
    template <typename Fill>
    Scan stream (juce::AudioPluginInstance& plugin, int numChannels, int block, int total, Fill&& fill)
    {
        juce::AudioBuffer<float> buffer (numChannels, block);
        juce::MidiBuffer midi;
        Scan scan;
        for (int pos = 0; pos < total; pos += block)
        {
            for (int ch = 0; ch < numChannels; ++ch)
                for (int i = 0; i < block; ++i)
                    buffer.setSample (ch, i, fill (pos + i, ch));
            plugin.processBlock (buffer, midi);
            scan.add (buffer, numChannels);
        }
        return scan;
    }

    float sine (int n, double rate, double hz, float amp)
    {
        return amp * (float) std::sin (juce::MathConstants<double>::twoPi * hz * n / rate);
    }

    void testLayouts (const juce::String& path)
    {
        for (const auto rate : { 44100.0, 48000.0, 96000.0, 192000.0 })
            for (const auto block : { 1, 7, 512 })
                for (const auto numChannels : { 1, 2 })
                    for (const auto hq : { 0.0f, 1.0f })
                    {
                        auto plugin = load (path, rate, block);
                        const auto set = numChannels == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo();
                        juce::AudioProcessor::BusesLayout layout;
                        layout.inputBuses.add (set);
                        layout.outputBuses.add (set);
                        const auto accepted = plugin->setBusesLayout (layout);
                        param (*plugin, "Drive")->setValue (1.0f);
                        param (*plugin, "Quality")->setValue (hq);
                        plugin->setRateAndBufferSizeDetails (rate, block);
                        plugin->prepareToPlay (rate, block);

                        const auto loud = stream (*plugin, numChannels, block, (int) (rate * 0.1),
                                                  [&] (int n, int) { return sine (n, rate, 440.0, 0.7f); });
                        const auto quiet = stream (*plugin, numChannels, block, (int) (rate * 3.0), [] (int, int) { return 0.0f; });
                        plugin->releaseResources();

                        report (juce::String (rate, 0) + " Hz block " + juce::String (block) + " ch " + juce::String (numChannels) + (hq > 0.5f ? " HQ" : " RT"),
                                accepted && ! loud.bad && ! quiet.bad && loud.peak > 0.01f,
                                "peak " + juce::String (loud.peak, 3));
                    }
    }

    // 参数稳定时，输出不应取决于宿主缓冲区大小。
    void testBlockIndependence (const juce::String& path)
    {
        const auto rate = 48000.0;
        const auto total = 48000;
        juce::AudioBuffer<float> reference;

        for (const auto block : { 512, 1, 33 })
        {
            auto plugin = load (path, rate, block);
            param (*plugin, "Drive")->setValue (0.6f);
            plugin->setRateAndBufferSizeDetails (rate, block);
            plugin->prepareToPlay (rate, block);

            juce::AudioBuffer<float> buffer (2, block), all (2, total);
            juce::MidiBuffer midi;
            for (int pos = 0; pos + block <= total; pos += block)
            {
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < block; ++i)
                        buffer.setSample (ch, i, sine (pos + i, rate, 997.0, 0.3f));
                plugin->processBlock (buffer, midi);
                for (int ch = 0; ch < 2; ++ch)
                    all.copyFrom (ch, pos, buffer, ch, 0, block);
            }
            plugin->releaseResources();

            if (block == 512)
            {
                reference = all;
                continue;
            }

            auto worst = 0.0f;
            for (int i = 0; i < 40000; ++i)
                worst = juce::jmax (worst, std::abs (all.getSample (0, i) - reference.getSample (0, i)));
            report ("block independence " + juce::String (block), worst < 1.0e-5f, "max diff " + juce::String (worst, 8));
        }
    }

    // 猛推驱动不应产生明显大于稳态的单样本跳变。
    void testAutomation (const juce::String& path)
    {
        const auto rate = 48000.0;
        const auto block = 64;
        auto plugin = load (path, rate, block);
        auto* drive = param (*plugin, "Drive");
        plugin->setRateAndBufferSizeDetails (rate, block);
        plugin->prepareToPlay (rate, block);

        auto input = [&] (int n, int) { return sine (n, rate, 220.0, 0.1f); };

        drive->setValue (0.0f);
        stream (*plugin, 2, block, 24000, input);
        const auto steady = stream (*plugin, 2, block, 24000, input);

        juce::AudioBuffer<float> buffer (2, block);
        juce::MidiBuffer midi;
        Scan moving;
        juce::Random random (7);
        for (int pos = 0; pos < 48000 * 4; pos += block)
        {
            if ((pos / block) % 200 == 0)
                drive->setValue (random.nextFloat());
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < block; ++i)
                    buffer.setSample (ch, i, input (pos + i, ch));
            plugin->processBlock (buffer, midi);
            moving.add (buffer, 2);
        }
        plugin->releaseResources();

        report ("drive automation has no clicks", ! moving.bad && moving.maxStep < 6.0f * steady.maxStep,
                "steady step " + juce::String (steady.maxStep, 4) + ", moving step " + juce::String (moving.maxStep, 4));
    }

    void testQualitySwitch (const juce::String& path)
    {
        const auto rate = 48000.0;
        const auto block = 128;
        auto plugin = load (path, rate, block);
        auto* quality = param (*plugin, "Quality");
        plugin->setRateAndBufferSizeDetails (rate, block);
        plugin->prepareToPlay (rate, block);

        // 输入相位和扫描器全程连续，否则测到的是测试自己造成的跳变。
        juce::AudioBuffer<float> buffer (2, block);
        juce::MidiBuffer midi;
        auto position = 0;
        auto runSection = [&] (Scan& scan, int total)
        {
            for (int done = 0; done < total; done += block)
            {
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < block; ++i)
                        buffer.setSample (ch, i, sine (position + i, rate, 220.0, 0.1f));
                position += block;
                plugin->processBlock (buffer, midi);
                scan.add (buffer, 2);
            }
        };

        Scan steady, switching;
        runSection (steady, 24000);
        switching.previous = steady.previous;
        switching.started = true;

        const auto rt0 = plugin->getLatencySamples();
        quality->setValue (1.0f);
        runSection (switching, 24000);
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto hqLatency = plugin->getLatencySamples();
        quality->setValue (0.0f);
        runSection (switching, 24000);
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto backLatency = plugin->getLatencySamples();
        plugin->releaseResources();

        report ("quality switch latency", rt0 == 0 && hqLatency > 0 && backLatency == 0,
                "rt " + juce::String (rt0) + ", hq " + juce::String (hqLatency) + ", back " + juce::String (backLatency));
        report ("quality switch has no clicks", ! switching.bad && switching.maxStep < 4.0f * steady.maxStep,
                "steady step " + juce::String (steady.maxStep, 4) + ", switching step " + juce::String (switching.maxStep, 4));
    }

    void testBadInput (const juce::String& path)
    {
        const auto rate = 48000.0;
        const auto block = 256;
        for (const auto hq : { 0.0f, 1.0f })
        {
            auto plugin = load (path, rate, block);
            param (*plugin, "Quality")->setValue (hq);
            param (*plugin, "Drive")->setValue (1.0f);
            plugin->setRateAndBufferSizeDetails (rate, block);
            plugin->prepareToPlay (rate, block);

            const auto nan = std::numeric_limits<float>::quiet_NaN();
            const auto inf = std::numeric_limits<float>::infinity();
            const auto hit = stream (*plugin, 2, block, 4800, [&] (int n, int)
            {
                return n % 97 == 0 ? nan : (n % 101 == 0 ? inf : (n % 103 == 0 ? 1.0e30f : sine (n, rate, 440.0, 0.5f)));
            });
            stream (*plugin, 2, block, 48000 * 3, [] (int, int) { return 0.0f; });
            const auto after = stream (*plugin, 2, block, 4800, [&] (int n, int) { return sine (n, rate, 440.0, 0.1f); });
            plugin->releaseResources();

            report (juce::String ("NaN/Inf input recovers") + (hq > 0.5f ? " HQ" : " RT"), ! hit.bad && ! after.bad && after.peak > 0.01f,
                    "peak after " + juce::String (after.peak, 3));
        }
    }

    void testCpu (const juce::String& path)
    {
        const auto rate = 48000.0;
        const auto block = 256;
        for (const auto hq : { 0.0f, 1.0f })
        {
            auto plugin = load (path, rate, block);
            param (*plugin, "Quality")->setValue (hq);
            param (*plugin, "Drive")->setValue (0.8f);
            plugin->setRateAndBufferSizeDetails (rate, block);
            plugin->prepareToPlay (rate, block);

            juce::AudioBuffer<float> buffer (2, block);
            juce::MidiBuffer midi;
            const auto blocks = (int) (rate * 10.0) / block;
            std::chrono::duration<double> spent {};
            for (int b = 0; b < blocks; ++b)
            {
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < block; ++i)
                        buffer.setSample (ch, i, sine (b * block + i, rate, 220.0, 0.3f));
                const auto start = std::chrono::steady_clock::now();
                plugin->processBlock (buffer, midi);
                spent += std::chrono::steady_clock::now() - start;
            }
            plugin->releaseResources();

            std::cout << "INFO cpu " << (hq > 0.5f ? "HQ" : "RT") << ": " << juce::String (100.0 * spent.count() / 10.0, 2) << "% of one core, stereo 48 kHz\n";
        }
    }
}

int wmain (int argc, wchar_t* argv[])
{
    if (argc < 2)
    {
        std::cout << "usage: mint_tube_host_test <plugin.vst3>\n";
        return 2;
    }

    juce::ScopedJuceInitialiser_GUI gui;
    const juce::String path (argv[1]);

    testLayouts (path);
    testBlockIndependence (path);
    testAutomation (path);
    testQualitySwitch (path);
    testBadInput (path);
    testCpu (path);

    std::cout << (failures == 0 ? "ALL PASS\n" : "FAILED " + juce::String (failures).toStdString() + "\n");
    return failures == 0 ? 0 : 1;
}
