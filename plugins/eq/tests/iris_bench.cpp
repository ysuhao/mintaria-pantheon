// Iris 的验证工具，把 VST3 当黑盒加载。mint_probe 做不到的项目放在这里：
//
//   mint_eq_bench design
//       直接调用设计函数：遍历参数、检查数值健全，统计单次设计耗时。不需要 VST3。
//   mint_eq_bench render <plugin.vst3> <in.wav> <out.wav> --block N [--set "名称=归一化值"]... [--sweep "名称=从:到:周期秒"]...
//       和 mint_probe render 类似，但参数可以在渲染过程中按三角波扫动（模拟快速拖动），并把上报的延迟补偿掉。
//       加 --mono 时以单声道总线运行，只有左声道有效。
//   mint_eq_bench cpu <plugin.vst3> --rate 48000 --block 128 [--set ...]...
//       实测 processBlock 平均耗时，给出占实时的百分比。
//   mint_eq_bench query <plugin.vst3> --set "名称=归一化值"...
//       打印参数被设置后的文字值，用来反查黑盒插件的归一化映射。
//   mint_eq_bench shot <plugin.vst3> <out.png> [--audio in.wav] [--set ...]... [--mouse "动作"]...
//       打开编辑器，可选喂音频（频谱才有内容），可选模拟鼠标，再截图。
//       动作：dbl:x:y  click:x:y  rclick:x:y  drag:x1:y1:x2:y2  wheel:x:y:delta  move:x:y  wait:毫秒
//             dump（打印所有已用频段参数和手势计数）  cpu:毫秒（这段时间内本进程占用的 CPU，单位 ms/s）
//   mint_eq_bench standalone <Standalone.exe> <out.png> [--offset x:y] [--mouse ...]...
//       启动 Standalone，只向确认属于这个进程的窗口句柄发鼠标消息，截图后只结束这个进程。
#include "../IrisFilterDesign.h"
#include "../IrisDsp.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if JUCE_WINDOWS
 #define NOMINMAX
 #include <windows.h>
#endif

bool captureWindowPixels (void* windowHandle, std::vector<unsigned char>& pixels, int& width, int& height);

namespace
{
    struct Sweep
    {
        juce::AudioProcessorParameter* parameter;
        float from, to, periodSeconds;
    };

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

    juce::AudioProcessorParameter* findParameter (juce::AudioProcessor& plugin, const juce::String& name)
    {
        for (auto* param : plugin.getParameters())
            if (param->getName (64) == name)
                return param;
        std::cout << "FAIL no parameter named " << name << "\n";
        return nullptr;
    }

    void applySets (juce::AudioProcessor& plugin, const juce::StringArray& args)
    {
        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--set")
                if (auto* p = findParameter (plugin, args[i + 1].upToFirstOccurrenceOf ("=", false, false)))
                    p->setValueNotifyingHost (args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue());
    }

    std::unique_ptr<juce::AudioFormatReader> openWav (const juce::File& file)
    {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        return std::unique_ptr<juce::AudioFormatReader> (formats.createReaderFor (file));
    }

    int runRender (const juce::String& path, const juce::File& input, const juce::File& output, const juce::StringArray& args)
    {
        const auto block = argValue (args, "--block", "256").getIntValue();
        auto reader = openWav (input);
        if (reader == nullptr)
        {
            std::cout << "FAIL cannot read input\n";
            return 1;
        }

        const auto rate = reader->sampleRate;
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        applySets (*plugin, args);

        std::vector<Sweep> sweeps;
        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--sweep")
                continue;
            juce::StringArray fields;
            fields.addTokens (args[i + 1].fromFirstOccurrenceOf ("=", false, false), ":", "");
            if (auto* p = findParameter (*plugin, args[i + 1].upToFirstOccurrenceOf ("=", false, false)))
                sweeps.push_back ({ p, fields[0].getFloatValue(), fields[1].getFloatValue(), fields[2].getFloatValue() });
        }

        plugin->enableAllBuses();
        const auto mono = args.contains ("--mono");
        if (mono)
        {
            juce::AudioProcessor::BusesLayout layout;
            layout.inputBuses.add (juce::AudioChannelSet::mono());
            layout.outputBuses.add (juce::AudioChannelSet::mono());
            if (! plugin->setBusesLayout (layout))
            {
                std::cout << "FAIL mono layout rejected\n";
                return 1;
            }
        }
        plugin->prepareToPlay (rate, block);

        // 预热的总采样数不随缓冲区变化，这样不同缓冲区之间的控制周期相位相同，输出可以直接比较。
        // 黑盒插件可能带侧链总线，缓冲区要包含这些声道，保持静音。
        const auto channels = mono ? 1 : juce::jmax (2, plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels());
        juce::AudioBuffer<float> settle (channels, block);
        juce::MidiBuffer midi;
        for (int remaining = (int) std::ceil (rate * 0.1); remaining > 0; remaining -= block)
        {
            juce::AudioBuffer<float> view (settle.getArrayOfWritePointers(), channels, 0, juce::jmin (block, remaining));
            view.clear();
            plugin->processBlock (view, midi);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        const auto latency = plugin->getLatencySamples();

        const auto total = (int) reader->lengthInSamples;
        juce::AudioBuffer<float> all (channels, total + latency);
        all.clear();
        reader->read (&all, 0, total, 0, true, ! mono);
        for (int ch = 2; ch < channels; ++ch)
            all.clear (ch, 0, all.getNumSamples());

        juce::AudioBuffer<float> chunk (2, block);
        for (int pos = 0; pos < all.getNumSamples(); pos += block)
        {
            const auto n = juce::jmin (block, all.getNumSamples() - pos);
            const auto seconds = (double) pos / rate;
            for (const auto& s : sweeps)
            {
                const auto phase = std::fmod (seconds / s.periodSeconds, 1.0);
                const auto tri = phase < 0.5 ? 2.0 * phase : 2.0 - 2.0 * phase;
                s.parameter->setValue (s.from + (float) tri * (s.to - s.from));
            }

            juce::AudioBuffer<float> view (all.getArrayOfWritePointers(), channels, pos, n);
            plugin->processBlock (view, midi);
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
        juce::AudioBuffer<float> mainOutputs (all.getArrayOfWritePointers(), juce::jmin (2, channels), all.getNumSamples());
        if (mono)
        {
            juce::AudioBuffer<float> doubled (2, all.getNumSamples());
            doubled.copyFrom (0, 0, all, 0, 0, all.getNumSamples());
            doubled.copyFrom (1, 0, all, 0, 0, all.getNumSamples());
            writer->writeFromAudioSampleBuffer (doubled, latency, total);
        }
        else
        {
            writer->writeFromAudioSampleBuffer (mainOutputs, latency, total);
        }
        std::cout << "rendered " << total << " samples, reported latency " << latency << "\n";
        return 0;
    }

    int runCpu (const juce::String& path, const juce::StringArray& args)
    {
        const auto rate = argValue (args, "--rate", "48000").getDoubleValue();
        const auto block = argValue (args, "--block", "128").getIntValue();
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        applySets (*plugin, args);
        plugin->enableAllBuses();
        plugin->prepareToPlay (rate, block);

        juce::AudioBuffer<float> buffer (2, block), source (2, block);
        juce::Random random (7);
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < block; ++i)
                source.setSample (ch, i, 0.25f * (random.nextFloat() * 2.0f - 1.0f));

        juce::MidiBuffer midi;
        const auto seconds = 0.5;
        const auto blocks = (int) (seconds * rate / block);

        for (int i = 0; i < blocks; ++i)
        {
            buffer.makeCopyOf (source, true);
            plugin->processBlock (buffer, midi);
        }

        // 机器是共用的，别的进程会把单次结果拉高，所以测 12 轮取最小值和中位数。
        std::vector<double> rounds;
        for (int round = 0; round < 12; ++round)
        {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < blocks; ++i)
            {
                buffer.makeCopyOf (source, true);
                plugin->processBlock (buffer, midi);
            }
            rounds.push_back (100.0 * std::chrono::duration<double> (std::chrono::steady_clock::now() - start).count() / seconds);
        }
        plugin->releaseResources();
        std::sort (rounds.begin(), rounds.end());

        std::cout << "cpu min " << juce::String (rounds.front(), 3) << " % median " << juce::String (rounds[rounds.size() / 2], 3) << " % at " << rate << " Hz, block " << block << "\n";
        return 0;
    }

    int runQuery (const juce::String& path, const juce::StringArray& args)
    {
        auto plugin = load (path, 48000.0, 256);
        if (plugin == nullptr)
            return 1;
        applySets (*plugin, args);
        for (int i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--set")
                if (auto* p = findParameter (*plugin, args[i + 1].upToFirstOccurrenceOf ("=", false, false)))
                    std::cout << p->getName (64) << "\t" << p->getValue() << "\t" << p->getCurrentValueAsText() << "\n";
        return 0;
    }

    int runDesign()
    {
        using namespace iris;
        const std::array<SectionShape, 9> shapes { SectionShape::bell, SectionShape::lowShelf, SectionShape::highShelf, SectionShape::tilt,
                                                   SectionShape::highPass2, SectionShape::lowPass2, SectionShape::highPass1, SectionShape::lowPass1, SectionShape::notch };

        auto designs = 0, invalid = 0;
        double worst = 0.0;
        std::array<double, 9> worstByShape {};
        std::array<juce::String, 9> worstText;
        const auto start = std::chrono::steady_clock::now();

        for (const auto rate : { 44100.0, 48000.0, 88200.0, 96000.0, 192000.0 })
            for (auto f = 10.0; f <= 30000.0; f *= 1.17)
                for (const auto gain : { -30.0, -12.0, -3.0, -0.1, 0.0, 0.1, 3.0, 12.0, 30.0 })
                    for (const auto q : { 0.025, 0.1, 0.5, 0.7071, 1.0, 3.0, 10.0, 40.0 })
                        for (size_t s = 0; s < shapes.size(); ++s)
                        {
                            const auto d = designSection (shapes[s], f, gain, q, rate);
                            ++designs;
                            const auto finite = std::isfinite (d.g) && std::isfinite (d.k) && std::isfinite (d.mHp) && std::isfinite (d.mBp) && std::isfinite (d.mLp);
                            if (! finite || d.g <= 0.0 || d.k <= 0.0)
                            {
                                ++invalid;
                                continue;
                            }

                            // 中等 Q 和增益下记录相对模拟目标的最大误差。
                            if (q < 0.5 || q > 10.0 || std::abs (gain) > 12.0 || f > 0.45 * rate)
                                continue;
                            const auto proto = analogPrototype (shapes[s], std::abs (gain), q);
                            for (auto x = 20.0; x < 0.49 * rate; x *= 1.05)
                            {
                                auto target = proto.magnitudeSquared (x / f);
                                if (isReciprocalShape (shapes[s]) && gain < 0.0)
                                    target = 1.0 / target;
                                if (target < 1.0e-6)
                                    continue;
                                const auto e = std::abs (10.0 * std::log10 (d.magnitudeSquared (x, rate) / target));
                                worst = std::max (worst, e);
                                if (e > worstByShape[s])
                                {
                                    worstByShape[s] = e;
                                    worstText[s] = "f0=" + juce::String (f, 1) + " gain=" + juce::String (gain) + " q=" + juce::String (q) + " rate=" + juce::String (rate) + " at " + juce::String (x, 0) + " Hz target " + juce::String (10.0 * std::log10 (target), 1) + " dB got " + juce::String (10.0 * std::log10 (d.magnitudeSquared (x, rate)), 1) + " dB";
                                }
                            }
                        }

        const auto elapsed = std::chrono::duration<double> (std::chrono::steady_clock::now() - start).count();
        std::cout << "designs " << designs << " invalid " << invalid << " worst error (dB, |gain|<=12, 0.5<=Q<=10, f<=0.45 fs) " << worst << "\n";
        const std::array<const char*, 9> names { "bell", "lowShelf", "highShelf", "tilt", "highPass2", "lowPass2", "highPass1", "lowPass1", "notch" };
        for (size_t s = 0; s < names.size(); ++s)
            std::cout << "  " << names[s] << " worst " << worstByShape[s] << " dB  (" << worstText[s] << ")\n";

        // 单次设计耗时，分别测跳过拟合和完整拟合。
        const auto timeIt = [] (double f, SectionShape shape)
        {
            constexpr int runs = 200000;
            volatile double sink = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < runs; ++i)
                sink = sink + designSection (shape, f + 0.01 * (i % 7), 6.0, 1.0, 48000.0).g;
            return std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count() / runs * 1.0e6;
        };
        std::cout << "design time: bell 200 Hz (uncorrected) " << timeIt (200.0, SectionShape::bell) << " us, bell 8 kHz (fitted) " << timeIt (8000.0, SectionShape::bell)
                  << " us, high pass 8 kHz (fitted) " << timeIt (8000.0, SectionShape::highPass2) << " us\n";
        juce::ignoreUnused (elapsed);
        return invalid == 0 ? 0 : 1;
    }

   #if JUCE_WINDOWS
    HWND findDeepestChild (HWND parent)
    {
        struct Search { HWND best; } search { nullptr };
        EnumChildWindows (parent, [] (HWND child, LPARAM data) -> BOOL
        {
            auto* s = reinterpret_cast<Search*> (data);
            RECT rect;
            GetClientRect (child, &rect);
            if (rect.right - rect.left >= 800)
                s->best = child;
            return TRUE;
        }, (LPARAM) &search);
        return search.best != nullptr ? search.best : parent;
    }

    void pump (int ms)
    {
        juce::MessageManager::getInstance()->runDispatchLoopUntil (ms);
    }

    // 测试窗口压在其他窗口下面，真实光标也不在它上面，系统会让它丢掉鼠标捕获，JUCE 随即用真实光标位置结束拖动。
    // 这里给窗口加一层过滤：挡掉捕获变化、鼠标离开和真实的鼠标移动，只放行本工具发出的消息（wParam 带 simulatedFlag）。
    constexpr WPARAM simulatedFlag = 0x8000;
    WNDPROC originalProc = nullptr;

    double processCpuMilliseconds()
    {
        FILETIME created, exited, kernel, user;
        GetProcessTimes (GetCurrentProcess(), &created, &exited, &kernel, &user);
        const auto toMs = [] (const FILETIME& t) { return (double) ((juce::uint64) t.dwHighDateTime << 32 | t.dwLowDateTime) * 1.0e-4; };
        return toMs (kernel) + toMs (user);
    }

    LRESULT CALLBACK filteredProc (HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        if (message == WM_CAPTURECHANGED || message == WM_MOUSELEAVE)
            return 0;
        if (message == WM_MOUSEMOVE)
        {
            if ((wParam & simulatedFlag) == 0)
                return 0;
            wParam &= ~simulatedFlag;
        }
        return CallWindowProc (originalProc, hwnd, message, wParam, lParam);
    }

    void postMove (HWND target, WPARAM buttons, LPARAM position)
    {
        PostMessage (target, WM_MOUSEMOVE, buttons | simulatedFlag, position);
    }

    // 向编辑器窗口发真实的 Windows 鼠标消息，走完整的 JUCE 事件路径。
    void sendMouse (HWND target, const juce::String& action)
    {
        juce::StringArray f;
        f.addTokens (action, ":", "");
        const auto lp = [] (int x, int y) { return MAKELPARAM (x, y); };
        const auto x = f[1].getIntValue(), y = f[2].getIntValue();

        if (f[0] == "dbl")
        {
            for (int i = 0; i < 2; ++i)
            {
                postMove (target, 0, lp (x, y));
                PostMessage (target, WM_LBUTTONDOWN, MK_LBUTTON, lp (x, y));
                PostMessage (target, WM_LBUTTONUP, 0, lp (x, y));
                pump (40);
            }
        }
        else if (f[0] == "click")
        {
            postMove (target, 0, lp (x, y));
            PostMessage (target, WM_LBUTTONDOWN, MK_LBUTTON, lp (x, y));
            PostMessage (target, WM_LBUTTONUP, 0, lp (x, y));
        }
        else if (f[0] == "rclick")
        {
            postMove (target, 0, lp (x, y));
            PostMessage (target, WM_RBUTTONDOWN, MK_RBUTTON, lp (x, y));
            PostMessage (target, WM_RBUTTONUP, 0, lp (x, y));
        }
        else if (f[0] == "drag")
        {
            const auto x2 = f[3].getIntValue(), y2 = f[4].getIntValue();
            postMove (target, 0, lp (x, y));
            PostMessage (target, WM_LBUTTONDOWN, MK_LBUTTON, lp (x, y));
            pump (30);
            for (int i = 1; i <= 8; ++i)
            {
                postMove (target, MK_LBUTTON, lp (x + (x2 - x) * i / 8, y + (y2 - y) * i / 8));
                pump (20);
            }
            PostMessage (target, WM_LBUTTONUP, 0, lp (x2, y2));
        }
        else if (f[0] == "move")
        {
            postMove (target, 0, lp (x, y));
        }
        else if (f[0] == "wait")
        {
            pump (x);
        }
        else if (f[0] == "cpu")
        {
            const auto before = processCpuMilliseconds();
            pump (x);
            std::cout << "cpu " << (processCpuMilliseconds() - before) * 1000.0 / (double) x << " ms/s\n";
            return;
        }
        else if (f[0] == "wheel")
        {
            POINT p { x, y };
            ClientToScreen (target, &p);
            postMove (target, 0, lp (x, y));
            pump (20);
            PostMessage (target, WM_MOUSEWHEEL, MAKEWPARAM (0, f[3].getIntValue()), MAKELPARAM (p.x, p.y));
        }
        pump (120);
    }
   #endif

    // 统计宿主收到的手势开始和结束，检查是否成对。
    struct GestureCounter final : juce::AudioProcessorParameter::Listener
    {
        void parameterValueChanged (int, float) override {}
        void parameterGestureChanged (int index, bool starting) override
        {
            (starting ? begins : ends) += 1;
            open[index] += starting ? 1 : -1;
            if (starting)
                perParameter[index] += 1;
        }

        int unpaired() const
        {
            auto total = 0;
            for (const auto& entry : open)
                total += std::abs (entry.second);
            return total;
        }

        int begins = 0, ends = 0;
        std::map<int, int> open, perParameter;
    };

    GestureCounter* gestureCounter = nullptr;

    void dumpUsedBands (juce::AudioProcessor& plugin)
    {
        const auto params = plugin.getParameters();
        for (int b = 1; b <= iris::numBands; ++b)
        {
            const auto prefix = "B" + juce::String (b).paddedLeft ('0', 2) + " ";
            auto* used = findParameter (plugin, prefix + "Used");
            if (used == nullptr || used->getValue() < 0.5f)
                continue;
            std::cout << "band " << b << ":";
            for (auto* p : params)
                if (p->getName (64).startsWith (prefix))
                    std::cout << " " << p->getName (64).fromFirstOccurrenceOf (" ", false, false) << "=" << p->getCurrentValueAsText();
            std::cout << "\n";
        }
        if (gestureCounter != nullptr)
        {
            std::cout << "gestures begin=" << gestureCounter->begins << " end=" << gestureCounter->ends << " unpaired=" << gestureCounter->unpaired();
            for (const auto& entry : gestureCounter->perParameter)
                std::cout << " " << params[entry.first]->getName (64).replace (" ", "_") << ":" << entry.second;
            std::cout << "\n";
        }
        std::cout << "dump end\n";
    }

    int runShot (const juce::String& path, const juce::File& output, const juce::StringArray& args)
    {
        constexpr double rate = 48000.0;
        constexpr int block = 256;
        auto plugin = load (path, rate, block);
        if (plugin == nullptr)
            return 1;

        applySets (*plugin, args);
        plugin->enableAllBuses();
        plugin->prepareToPlay (rate, block);

        GestureCounter counter;
        gestureCounter = &counter;
        for (auto* p : plugin->getParameters())
            p->addListener (&counter);

        std::unique_ptr<juce::AudioProcessorEditor> editor (plugin->createEditorIfNeeded());
        if (editor == nullptr)
        {
            std::cout << "FAIL plugin has no editor\n";
            return 1;
        }

        juce::DocumentWindow window ("mint_eq_bench", juce::Colours::black, 0);
        window.setUsingNativeTitleBar (true);
        window.setContentNonOwned (editor.get(), true);
        // 屏幕外或被盖住的窗口，系统不合成画面，截到的是首帧；JUCE 的命中判断也会看 WindowFromPoint，被盖住时点击会丢。
        // 所以放在屏幕内并置顶，不抢焦点。只在这个测试里出现几秒。
        window.setTopLeftPosition (40, 40);
        window.setVisible (true);
       #if JUCE_WINDOWS
        SetWindowPos ((HWND) window.getWindowHandle(), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
       #endif

        // 音频在后台线程喂，模拟宿主。
        juce::AudioBuffer<float> audio;
        const auto audioPath = argValue (args, "--audio", {});
        if (audioPath.isNotEmpty())
            if (auto reader = openWav (juce::File (audioPath)))
            {
                audio.setSize (2, (int) reader->lengthInSamples);
                reader->read (&audio, 0, audio.getNumSamples(), 0, true, true);
            }

        struct Feeder final : juce::Thread
        {
            Feeder (juce::AudioProcessor& p, const juce::AudioBuffer<float>& a) : juce::Thread ("feeder"), plugin (p), source (a) {}
            void run() override
            {
                juce::AudioBuffer<float> chunk (2, block);
                juce::MidiBuffer midi;
                auto pos = 0;
                while (! threadShouldExit())
                {
                    if (source.getNumSamples() >= block)
                    {
                        for (int ch = 0; ch < 2; ++ch)
                            chunk.copyFrom (ch, 0, source, ch, pos, block);
                        pos += block;
                        if (pos + block > source.getNumSamples())
                            pos = 0;
                    }
                    else
                    {
                        chunk.clear();
                    }
                    plugin.processBlock (chunk, midi);
                    wait ((int) (1000.0 * block / rate));
                }
            }
            juce::AudioProcessor& plugin;
            const juce::AudioBuffer<float>& source;
        } feeder (*plugin, audio);
        feeder.startThread();

        juce::MessageManager::getInstance()->runDispatchLoopUntil (2000);

       #if JUCE_WINDOWS
        const auto target = findDeepestChild ((HWND) window.getWindowHandle());
        originalProc = (WNDPROC) SetWindowLongPtr (target, GWLP_WNDPROC, (LONG_PTR) &filteredProc);
        for (int i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--mouse")
                continue;
            if (args[i + 1] == "dump")
                dumpUsedBands (*plugin);
            else
                sendMouse (target, args[i + 1]);
        }
       #endif

       #if JUCE_WINDOWS
        SetWindowLongPtr (target, GWLP_WNDPROC, (LONG_PTR) originalProc);

       #endif

        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);

        // 被盖住的窗口不一定收到重绘，这里强制几次，让 JUCE 把最新状态画出来。
       #if JUCE_WINDOWS
        for (int i = 0; i < 4; ++i)
        {
            RedrawWindow (target, nullptr, nullptr, RDW_INVALIDATE | RDW_ALLCHILDREN | RDW_UPDATENOW);
            juce::MessageManager::getInstance()->runDispatchLoopUntil (150);
        }
       #endif

        std::vector<unsigned char> pixels;
        int w = 0, h = 0;
        const auto captured = captureWindowPixels (window.getWindowHandle(), pixels, w, h);
        window.setVisible (false);
        feeder.stopThread (2000);
        if (! captured)
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
        juce::MessageManager::getInstance()->runDispatchLoopUntil (100);
        std::cout << "gestures after close begin=" << counter.begins << " end=" << counter.ends << " unpaired=" << counter.unpaired() << "\n";
        for (auto* p : plugin->getParameters())
            p->removeListener (&counter);
        gestureCounter = nullptr;
        plugin->releaseResources();

        output.deleteFile();
        juce::FileOutputStream stream (output);
        juce::PNGImageFormat png;
        if (! stream.openedOk() || ! png.writeImageToStream (image, stream))
        {
            std::cout << "FAIL cannot write png\n";
            return 1;
        }
        std::cout << "wrote " << w << "x" << h << "\n";
        return 0;
    }

   #if JUCE_WINDOWS
    struct WindowSearch
    {
        DWORD processId;
        HWND found;
    };

    // 只接受属于 processId 的可见顶层窗口。
    HWND findWindowOfProcess (DWORD processId)
    {
        WindowSearch search { processId, nullptr };
        EnumWindows ([] (HWND hwnd, LPARAM data) -> BOOL
        {
            auto* s = reinterpret_cast<WindowSearch*> (data);
            DWORD owner = 0;
            GetWindowThreadProcessId (hwnd, &owner);
            RECT rect;
            GetClientRect (hwnd, &rect);
            if (owner == s->processId && IsWindowVisible (hwnd) && rect.right - rect.left >= 800)
            {
                s->found = hwnd;
                return FALSE;
            }
            return TRUE;
        }, (LPARAM) &search);
        return search.found;
    }

    int runStandalone (const juce::String& exe, const juce::File& output, const juce::StringArray& args)
    {
        STARTUPINFOW startup {};
        startup.cb = sizeof (startup);
        PROCESS_INFORMATION info {};
        const auto commandLine = "\"" + exe + "\"";
        std::wstring mutableLine (commandLine.toWideCharPointer());
        if (! CreateProcessW (nullptr, mutableLine.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &info))
        {
            std::cout << "FAIL cannot start " << exe << "\n";
            return 1;
        }

        auto result = 1;
        HWND window = nullptr;
        for (int i = 0; i < 600 && window == nullptr; ++i)
        {
            pump (100);
            window = findWindowOfProcess (info.dwProcessId);
        }

        if (window == nullptr)
        {
            DWORD exitCode = 0;
            GetExitCodeProcess (info.hProcess, &exitCode);
            std::cout << "FAIL no window from process " << (int) info.dwProcessId << ", exit code " << (juce::int64) exitCode << "\n";
            EnumWindows ([] (HWND hwnd, LPARAM data) -> BOOL
            {
                DWORD owner = 0;
                GetWindowThreadProcessId (hwnd, &owner);
                if (owner == (DWORD) data)
                {
                    RECT rect;
                    GetClientRect (hwnd, &rect);
                    std::cout << "  window " << (rect.right - rect.left) << "x" << (rect.bottom - rect.top) << " visible " << (IsWindowVisible (hwnd) != 0) << "\n";
                }
                return TRUE;
            }, (LPARAM) info.dwProcessId);
        }
        else
        {
            SetWindowPos (window, HWND_TOPMOST, 40, 40, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
            pump (1000);

            juce::StringArray offsetFields;
            offsetFields.addTokens (argValue (args, "--offset", "0:0"), ":", "");
            const auto dx = offsetFields[0].getIntValue(), dy = offsetFields[1].getIntValue();

            for (int i = 0; i + 1 < args.size(); ++i)
            {
                if (args[i] != "--mouse")
                    continue;

                // 动作里的坐标是编辑器坐标，这里加上 Standalone 窗口内的偏移。
                juce::StringArray fields;
                fields.addTokens (args[i + 1], ":", "");
                if (fields[0] != "wait" && fields[0] != "cpu")
                {
                    fields.set (1, juce::String (fields[1].getIntValue() + dx));
                    fields.set (2, juce::String (fields[2].getIntValue() + dy));
                    if (fields[0] == "drag")
                    {
                        fields.set (3, juce::String (fields[3].getIntValue() + dx));
                        fields.set (4, juce::String (fields[4].getIntValue() + dy));
                    }
                }
                sendMouse (window, fields.joinIntoString (":"));
            }

            pump (600);
            std::vector<unsigned char> pixels;
            int w = 0, h = 0;
            if (captureWindowPixels (window, pixels, w, h))
            {
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
                output.deleteFile();
                juce::FileOutputStream stream (output);
                juce::PNGImageFormat png;
                if (stream.openedOk() && png.writeImageToStream (image, stream))
                {
                    std::cout << "wrote " << w << "x" << h << "\n";
                    result = 0;
                }
            }
        }

        // 只结束自己启动的这个进程。
        TerminateProcess (info.hProcess, 0);
        WaitForSingleObject (info.hProcess, 3000);
        CloseHandle (info.hThread);
        CloseHandle (info.hProcess);
        return result;
    }
   #endif
}

int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (argv[i]));

    if (args.size() >= 1 && args[0] == "design")
        return runDesign();
    if (args.size() >= 4 && args[0] == "render")
        return runRender (args[1], juce::File (args[2]), juce::File (args[3]), args);
    if (args.size() >= 2 && args[0] == "cpu")
        return runCpu (args[1], args);
    if (args.size() >= 2 && args[0] == "query")
        return runQuery (args[1], args);
    if (args.size() >= 3 && args[0] == "shot")
        return runShot (args[1], juce::File (args[2]), args);
   #if JUCE_WINDOWS
    if (args.size() >= 3 && args[0] == "standalone")
        return runStandalone (args[1], juce::File (args[2]), args);
   #endif

    std::cout << "usage: mint_eq_bench design|render|cpu|query|shot ...\n";
    return 2;
}
