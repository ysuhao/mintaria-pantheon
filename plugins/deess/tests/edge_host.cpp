// 用真实的插件实例跑 mint_probe 没覆盖的边界：单声道、1 采样缓冲区、192 kHz、静音后的 denormal、实时倍率。
//
//   medusa_edge_host "<Mintaria Medusa.vst3>"
#include <juce_audio_utils/juce_audio_utils.h>

#include <cfloat>
#include <iostream>
#include <memory>
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
            return nullptr;
        juce::String error;
        return manager.createPluginInstance (*found[0], rate, block, error);
    }

    // 元音类谐波加 4–10 kHz 附近的噪声爆发（白噪声一阶差分近似），再接静音。
    float testSample (int i, double rate, juce::Random& random, float& last)
    {
        const auto t = (double) i / rate;
        if (t > 3.0)
            return 0.0f;
        const auto phase = juce::MathConstants<double>::twoPi * 180.0 * t;
        auto x = 0.08 * std::sin (phase) + 0.03 * std::sin (2.0 * phase) + 0.01 * std::sin (5.0 * phase);
        if (std::fmod (t, 0.5) > 0.3)
        {
            const auto w = random.nextFloat() * 2.0f - 1.0f;
            x += 0.2 * (w - last);
            last = w;
        }
        return (float) x;
    }

    struct Result { bool ok = true; float peak = 0.0f; int subnormal = 0; double speed = 0.0; };

    Result runOne (const juce::String& path, double rate, int block, bool mono, std::vector<float>* capture)
    {
        Result result;
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return { false };

        if (mono)
        {
            juce::AudioProcessor::BusesLayout layout;
            layout.inputBuses.add (juce::AudioChannelSet::mono());
            layout.outputBuses.add (juce::AudioChannelSet::mono());
            if (! plugin->setBusesLayout (layout))
            {
                std::cout << "  cannot set mono layout\n";
                return { false };
            }
        }

        plugin->setRateAndBufferSizeDetails (rate, block);
        plugin->prepareToPlay (rate, block);

        const auto channels = mono ? 1 : 2;
        juce::AudioBuffer<float> buffer (channels, block);
        juce::MidiBuffer midi;
        juce::Random random (1);
        auto last = 0.0f;
        const auto total = (int) (rate * 8.0);
        const auto startTime = juce::Time::getMillisecondCounterHiRes();

        for (int pos = 0; pos + block <= total; pos += block)
        {
            for (int i = 0; i < block; ++i)
            {
                const auto x = testSample (pos + i, rate, random, last);
                for (int ch = 0; ch < channels; ++ch)
                    buffer.setSample (ch, i, x);
            }

            plugin->processBlock (buffer, midi);

            for (int ch = 0; ch < channels; ++ch)
                for (int i = 0; i < block; ++i)
                {
                    const auto y = buffer.getSample (ch, i);
                    if (! std::isfinite (y))
                        result.ok = false;
                    result.peak = juce::jmax (result.peak, std::abs (y));
                    if (y != 0.0f && std::abs (y) < FLT_MIN)
                        ++result.subnormal;
                }

            if (capture != nullptr)
                for (int i = 0; i < block; ++i)
                    capture->push_back (buffer.getSample (0, i));
        }

        result.speed = 8000.0 / (juce::Time::getMillisecondCounterHiRes() - startTime);
        plugin->releaseResources();
        return result;
    }
}

int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    if (argc < 2)
    {
        std::cout << "usage: medusa_edge_host <plugin.vst3>\n";
        return 2;
    }

    const juce::String path (argv[1]);
    auto failures = 0;

    for (const auto rate : { 44100.0, 48000.0, 96000.0, 192000.0 })
        for (const auto block : { 1, 7, 256, 4096 })
            for (const auto mono : { true, false })
            {
                const auto r = runOne (path, rate, block, mono, nullptr);
                const auto pass = r.ok && r.subnormal == 0;
                std::cout << (pass ? "PASS " : "FAIL ") << rate << " Hz, block " << block << ", " << (mono ? "mono" : "stereo")
                          << ": peak " << r.peak << ", subnormal " << r.subnormal << ", speed x" << (int) r.speed << "\n";
                failures += pass ? 0 : 1;
            }

    // 自动化：每块都动强度和频率，并反复切换监听，输出的采样间跳变不应比静止时大出很多。
    {
        const auto measure = [&path] (bool automate)
        {
            const auto rate = 48000.0;
            const auto block = 64;
            auto plugin = load (path, rate, block);
            if (plugin == nullptr)
                return 1.0e9f;
            plugin->setRateAndBufferSizeDetails (rate, block);
            plugin->prepareToPlay (rate, block);

            juce::AudioProcessorParameter* strength = nullptr;
            juce::AudioProcessorParameter* frequency = nullptr;
            juce::AudioProcessorParameter* monitor = nullptr;
            for (auto* param : plugin->getParameters())
            {
                if (param->getName (64) == "Strength")   strength = param;
                if (param->getName (64) == "Frequency")  frequency = param;
                if (param->getName (64) == "Monitor")    monitor = param;
            }
            if (strength == nullptr || frequency == nullptr || monitor == nullptr)
                return 1.0e9f;

            juce::AudioBuffer<float> buffer (2, block);
            juce::MidiBuffer midi;
            juce::Random random (1);
            auto last = 0.0f, previous = 0.0f, jump = 0.0f;
            for (int pos = 0; pos + block <= (int) (rate * 3.0); pos += block)
            {
                if (automate)
                {
                    const auto t = (float) pos / (float) rate;
                    strength->setValue (0.5f + 0.5f * std::sin (9.0f * t));
                    frequency->setValue (0.5f + 0.5f * std::sin (5.0f * t));
                    monitor->setValue ((pos / block) % 400 < 200 ? 0.0f : 1.0f);
                }
                for (int i = 0; i < block; ++i)
                {
                    const auto x = testSample (pos + i, rate, random, last);
                    buffer.setSample (0, i, x);
                    buffer.setSample (1, i, x);
                }
                plugin->processBlock (buffer, midi);
                for (int i = 0; i < block; ++i)
                {
                    const auto y = buffer.getSample (0, i);
                    jump = juce::jmax (jump, std::abs (y - previous));
                    previous = y;
                }
            }
            plugin->releaseResources();
            return jump;
        };

        const auto still = measure (false);
        const auto moving = measure (true);
        const auto pass = moving <= still * 1.5f;
        std::cout << (pass ? "PASS " : "FAIL ") << "automation max sample jump " << moving << " vs static " << still << "\n";
        failures += pass ? 0 : 1;
    }

    // 运行中改 Look-ahead：不能有满幅台阶，改完要恢复出声。
    {
        const auto rate = 48000.0;
        const auto block = 128;
        if (auto plugin = load (path, rate, block))
        {
            plugin->setRateAndBufferSizeDetails (rate, block);
            plugin->prepareToPlay (rate, block);
            juce::AudioProcessorParameter* la = nullptr;
            for (auto* param : plugin->getParameters())
                if (param->getName (64) == "Look-ahead")
                    la = param;

            juce::AudioBuffer<float> buffer (2, block);
            juce::MidiBuffer midi;
            auto previous = 0.0f, jump = 0.0f, tailPeak = 0.0f;
            const auto total = (int) (rate * 3.0);
            const float steps[] = { 0.0f, 0.5f, 1.0f, 0.25f, 0.0f };
            for (int pos = 0; pos + block <= total; pos += block)
            {
                if (la != nullptr && pos % (int) (rate * 0.5) < block)
                    la->setValue (steps[juce::jmin (4, pos / (int) (rate * 0.5))]);
                for (int i = 0; i < block; ++i)
                {
                    const auto x = (float) (0.3 * std::sin (juce::MathConstants<double>::twoPi * 200.0 * (double) (pos + i) / rate));
                    buffer.setSample (0, i, x);
                    buffer.setSample (1, i, x);
                }
                plugin->processBlock (buffer, midi);
                for (int i = 0; i < block; ++i)
                {
                    const auto y = buffer.getSample (0, i);
                    jump = juce::jmax (jump, std::abs (y - previous));
                    previous = y;
                    if (pos > total - (int) (rate * 0.2))
                        tailPeak = juce::jmax (tailPeak, std::abs (y));
                }
            }
            plugin->releaseResources();
            // 200 Hz、0.3 幅度的正弦，自身的采样间跳变约 0.008，淡入淡出 5 ms 再加约 0.0125。
            const auto pass = jump < 0.03f && tailPeak > 0.29f;
            std::cout << (pass ? "PASS " : "FAIL ") << "look-ahead switch max sample jump " << jump << ", tail peak " << tailPeak << "\n";
            failures += pass ? 0 : 1;
        }
    }

    // 单声道结果必须等于立体声（双单声道）的任一声道：联动用的是取大，不是求和。
    std::vector<float> monoOut, stereoOut;
    runOne (path, 48000.0, 256, true, &monoOut);
    runOne (path, 48000.0, 256, false, &stereoOut);
    auto maxDiff = 0.0f;
    for (size_t i = 0; i < juce::jmin (monoOut.size(), stereoOut.size()); ++i)
        maxDiff = juce::jmax (maxDiff, std::abs (monoOut[i] - stereoOut[i]));
    const auto same = maxDiff == 0.0f;
    std::cout << (same ? "PASS " : "FAIL ") << "mono == stereo channel, max diff " << maxDiff << "\n";
    failures += same ? 0 : 1;

    std::cout << (failures == 0 ? "ALL PASS\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
