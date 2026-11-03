// DeepFilterNet3 的流式推理。DSP 部分按 libDF 的 tract.rs 和 lib.rs 移植，保持同样的特征和滤波；
// 不同之处是多声道时只跑一次网络：特征取各声道的平均谱，同一组增益和滤波系数作用到每个声道，
// 这样立体声的 CPU 和单声道相同，声像也不会被两套独立的掩码拉偏。
#include "LetheEngine.h"
#include "LetheFft.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace lethe
{
namespace
{
    using Complex = std::complex<float>;

    constexpr int fftSize = 960;
    constexpr int numFreqs = fftSize / 2 + 1;
    constexpr int numErb = 32;
    constexpr int numDf = 96;
    constexpr int dfOrder = 5;
    constexpr int minErbFreqs = 2;
    constexpr float normAlpha = 0.99f; // libDF 的 calc_norm_alpha (48000, 480, 1.0)
    constexpr float minDbThresh = -15.0f, maxDbErbThresh = 35.0f, maxDbDfThresh = 35.0f;

    // 含这段代码的模块，也就是 VST3 或测试程序本身。不用 currentExecutableFile，
    // 宿主里它可能指向宿主进程。
    juce::File getModuleDirectory()
    {
        HMODULE module = nullptr;
        GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR> (&getModuleDirectory), &module);
        wchar_t path[32768] = {};
        GetModuleFileNameW (module, path, 32768);
        return juce::File (juce::String (path)).getParentDirectory();
    }

    // VST3 包是 Contents/x86_64-win/*.vst3 和 Contents/Resources/lethe；
    // Standalone 和测试程序是 exe 旁边的 Resources/lethe。
    juce::File findModelRoot()
    {
        const auto binary = getModuleDirectory();
        const auto inBundle = binary.getSiblingFile ("Resources").getChildFile ("lethe");
        return inBundle.isDirectory() ? inBundle : binary.getChildFile ("Resources").getChildFile ("lethe");
    }

    // 必须用完整路径加载：Windows 11 的 System32 里也有一个 onnxruntime.dll，版本对不上。
    // LOAD_WITH_ALTERED_SEARCH_PATH 让它的依赖（VC 运行库）优先从同一目录找。
    // 进程退出前不卸载，其他实例还可能在用。
    const OrtApi* loadRuntime (juce::String& error)
    {
        static const OrtApi* api = nullptr;
        static juce::String loadError;
        static std::once_flag once;

        std::call_once (once, []
        {
            const auto dll = getModuleDirectory().getChildFile ("onnxruntime.dll");
            auto* module = LoadLibraryExW (dll.getFullPathName().toWideCharPointer(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (module == nullptr)
            {
                loadError = "cannot load " + dll.getFullPathName() + ", error " + juce::String ((int) GetLastError());
                return;
            }

            using GetApiBase = const OrtApiBase* (ORT_API_CALL*) ();
            auto getBase = reinterpret_cast<GetApiBase> (reinterpret_cast<void*> (GetProcAddress (module, "OrtGetApiBase")));
            api = getBase != nullptr ? getBase()->GetApi (ORT_API_VERSION) : nullptr;
            if (api == nullptr)
                loadError = "onnxruntime.dll does not provide API version " + juce::String (ORT_API_VERSION);
        });

        error = loadError;
        return api;
    }

    Ort::Env& getEnvironment()
    {
        // 故意不析构：运行库 DLL 也不卸载，静态析构顺序靠不住。
        static auto* env = []
        {
            auto* e = new Ort::Env (ORT_LOGGING_LEVEL_ERROR, "lethe");
            e->DisableTelemetryEvents();
            return e;
        }();
        return *env;
    }

    // 一个有状态子网。输入缓冲在构造时一次分配好，形状从模型里读。
    class Net
    {
    public:
        Net (const juce::File& file, const Ort::SessionOptions& options)
        {
            juce::MemoryBlock bytes;
            if (! file.loadFileAsData (bytes))
                throw std::runtime_error ("cannot read " + file.getFullPathName().toStdString());

            session = std::make_unique<Ort::Session> (getEnvironment(), bytes.getData(), bytes.getSize(), options);

            Ort::AllocatorWithDefaultOptions allocator;
            const auto memory = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);

            for (size_t i = 0; i < session->GetInputCount(); ++i)
            {
                inputNames.emplace_back (session->GetInputNameAllocated (i, allocator).get());
                shapes.push_back (session->GetInputTypeInfo (i).GetTensorTypeAndShapeInfo().GetShape());
                size_t count = 1;
                for (const auto d : shapes.back())
                    count *= (size_t) d;
                buffers.emplace_back (count, 0.0f);
            }

            for (size_t i = 0; i < inputNames.size(); ++i)
                inputValues.push_back (Ort::Value::CreateTensor<float> (memory, buffers[i].data(), buffers[i].size(),
                                                                        shapes[i].data(), shapes[i].size()));

            for (size_t i = 0; i < session->GetOutputCount(); ++i)
                outputNames.emplace_back (session->GetOutputNameAllocated (i, allocator).get());

            for (const auto& name : inputNames)  inputPointers.push_back (name.c_str());
            for (const auto& name : outputNames) outputPointers.push_back (name.c_str());

            for (size_t i = 0; i < inputNames.size(); ++i)
                if (inputNames[i].rfind ("h_", 0) == 0 || inputNames[i].rfind ("p_", 0) == 0)
                    states.push_back ({ i, indexOf (outputNames, inputNames[i] + "_out"), inputNames[i][0] == 'p' });
        }

        float* input (const std::string& name)
        {
            return buffers[indexOf (inputNames, name)].data();
        }

        void reset()
        {
            for (const auto& s : states)
                std::fill (buffers[s.input].begin(), buffers[s.input].end(), 0.0f);
        }

        void run()
        {
            outputs = session->Run (runOptions, inputPointers.data(), inputValues.data(), inputValues.size(),
                                    outputPointers.data(), outputPointers.size());

            for (const auto& s : states)
            {
                const auto* out = outputs[s.output].GetTensorData<float>();
                auto& buffer = buffers[s.input];

                if (! s.isHistory)
                {
                    std::copy_n (out, buffer.size(), buffer.begin());
                    continue;
                }

                // 历史帧的形状是 [1, C, T, F]：每个 C 内把时间轴前移一帧，新帧放到末尾。
                const auto channels = (size_t) shapes[s.input][1];
                const auto frames = (size_t) shapes[s.input][2];
                const auto bins = (size_t) shapes[s.input][3];
                for (size_t c = 0; c < channels; ++c)
                {
                    auto* base = buffer.data() + c * frames * bins;
                    std::copy (base + bins, base + frames * bins, base);
                    std::copy_n (out + c * bins, bins, base + (frames - 1) * bins);
                }
            }
        }

        const float* output (const std::string& name) const
        {
            return outputs[indexOf (outputNames, name)].GetTensorData<float>();
        }

        // 尺寸由本网络的输入决定。按名字查，单帧只有几次调用，不值得缓存。
        void copyFrom (const Net& other, const std::string& name)
        {
            auto& buffer = buffers[indexOf (inputNames, name)];
            std::copy_n (other.output (name), buffer.size(), buffer.begin());
        }

    private:
        struct State { size_t input, output; bool isHistory; };

        static size_t indexOf (const std::vector<std::string>& names, const std::string& name)
        {
            const auto it = std::find (names.begin(), names.end(), name);
            if (it == names.end())
                throw std::runtime_error ("model has no tensor " + name);
            return (size_t) std::distance (names.begin(), it);
        }

        std::unique_ptr<Ort::Session> session;
        Ort::RunOptions runOptions { nullptr };
        std::vector<std::string> inputNames, outputNames;
        std::vector<const char*> inputPointers, outputPointers;
        std::vector<std::vector<float>> buffers;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<Ort::Value> inputValues, outputs;
        std::vector<State> states;
    };

    float freqToErb (float hz) noexcept { return 9.265f * std::log1p (hz / (24.7f * 9.265f)); }
    float erbToFreq (float erb) noexcept { return 24.7f * 9.265f * (std::exp (erb / 9.265f) - 1.0f); }

    std::vector<int> makeErbBands()
    {
        const auto freqWidth = (float) engineRate / (float) fftSize;
        const auto erbLow = freqToErb (0.0f);
        const auto erbHigh = freqToErb ((float) engineRate * 0.5f);
        const auto step = (erbHigh - erbLow) / (float) numErb;

        std::vector<int> bands ((size_t) numErb);
        auto previous = 0, over = 0;
        for (int i = 1; i <= numErb; ++i)
        {
            const auto bin = (int) std::round (erbToFreq (erbLow + (float) i * step) / freqWidth);
            auto count = bin - previous - over;
            if (count < minErbFreqs)
            {
                over = minErbFreqs - count;
                count = minErbFreqs;
            }
            else
            {
                over = 0;
            }
            bands[(size_t) i - 1] = count;
            previous = bin;
        }

        bands.back() += 1;
        auto total = 0;
        for (const auto b : bands)
            total += b;
        if (total > numFreqs)
            bands.back() -= total - numFreqs;
        return bands;
    }

    class DfEngine final : public VoiceEngine
    {
    public:
        DfEngine (const juce::File& modelDir, int lookaheadFrames, int numChannels)
            : lookahead (lookaheadFrames), channels (numChannels),
              erbBands (makeErbBands()), fft (fftSize), window ((size_t) fftSize), timeBuffer ((size_t) fftSize),
              freqBuffer ((size_t) fftSize)
        {
            for (int i = 0; i < fftSize; ++i)
            {
                const auto s = std::sin (0.5 * juce::MathConstants<double>::pi * (i + 0.5) / (fftSize / 2));
                window[(size_t) i] = (float) std::sin (0.5 * juce::MathConstants<double>::pi * s * s);
            }

            Ort::SessionOptions options;
            options.SetIntraOpNumThreads (1);
            options.SetInterOpNumThreads (1);
            options.SetExecutionMode (ORT_SEQUENTIAL);
            options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.AddConfigEntry ("session.set_denormal_as_zero", "1");
            options.AddConfigEntry ("session.intra_op.allow_spinning", "0");
            options.DisableProfiling();

            enc = std::make_unique<Net> (modelDir.getChildFile ("enc.onnx"), options);
            erbDec = std::make_unique<Net> (modelDir.getChildFile ("erb_dec.onnx"), options);
            dfDec = std::make_unique<Net> (modelDir.getChildFile ("df_dec.onnx"), options);

            spec.assign ((size_t) channels, std::vector<Complex> ((size_t) numFreqs));
            mixSpec.resize ((size_t) numFreqs);
            bufferY.assign ((size_t) channels, std::vector<std::vector<Complex>> ((size_t) (dfOrder + lookahead), std::vector<Complex> ((size_t) numFreqs)));
            bufferX.assign ((size_t) channels, std::vector<std::vector<Complex>> ((size_t) dfOrder, std::vector<Complex> ((size_t) numFreqs)));
            analysisMem.assign ((size_t) channels, std::vector<float> ((size_t) hopSize));
            synthesisMem.assign ((size_t) channels, std::vector<float> ((size_t) hopSize));
            outSpec.resize ((size_t) numFreqs);
            bandGain.resize ((size_t) numFreqs);

            reset();
            warmUp();
            reset();
        }

        void reset() override
        {
            for (auto* net : { enc.get(), erbDec.get(), dfDec.get() })
                net->reset();

            for (auto& channel : bufferY) for (auto& frame : channel) std::fill (frame.begin(), frame.end(), Complex());
            for (auto& channel : bufferX) for (auto& frame : channel) std::fill (frame.begin(), frame.end(), Complex());
            for (auto& mem : analysisMem)  std::fill (mem.begin(), mem.end(), 0.0f);
            for (auto& mem : synthesisMem) std::fill (mem.begin(), mem.end(), 0.0f);

            for (int i = 0; i < numErb; ++i)
                meanNorm[(size_t) i] = -60.0f + (float) i * (-90.0f + 60.0f) / (float) (numErb - 1);
            for (int i = 0; i < numDf; ++i)
                unitNorm[(size_t) i] = 0.001f + (float) i * (0.0001f - 0.001f) / (float) (numDf - 1);

            skipCounter = 0;
        }

        bool process (const float* const* input, float* const* output, int numChannels) override
        {
            jassert (numChannels == channels);

            auto energy = 0.0f;
            for (int ch = 0; ch < numChannels; ++ch)
                for (int i = 0; i < hopSize; ++i)
                    energy += input[ch][i] * input[ch][i];

            skipCounter = energy / (float) (hopSize * numChannels) < 1.0e-7f ? skipCounter + 1 : 0;
            if (skipCounter > 5)
            {
                for (int ch = 0; ch < numChannels; ++ch)
                    std::fill_n (output[ch], hopSize, 0.0f);
                return true;
            }

            for (int ch = 0; ch < numChannels; ++ch)
            {
                analysis (ch, input[ch]);
                std::rotate (bufferY[(size_t) ch].begin(), bufferY[(size_t) ch].begin() + 1, bufferY[(size_t) ch].end());
                std::rotate (bufferX[(size_t) ch].begin(), bufferX[(size_t) ch].begin() + 1, bufferX[(size_t) ch].end());
                bufferY[(size_t) ch].back() = spec[(size_t) ch];
                bufferX[(size_t) ch].back() = spec[(size_t) ch];
            }

            computeFeatures();

            enc->run();
            const auto lsnr = *enc->output ("lsnr");
            if (! isFinite (lsnr))
            {
                recoverFromBadFrame (input, output);
                return false;
            }

            const auto applyGains = lsnr >= minDbThresh && lsnr <= maxDbErbThresh;
            const auto applyZeros = lsnr < minDbThresh;
            const auto applyDf = lsnr >= minDbThresh && lsnr <= maxDbDfThresh;

            const float* mask = nullptr;
            if (applyGains)
            {
                for (const auto* key : { "emb", "e3", "e2", "e1", "e0" })
                    erbDec->copyFrom (*enc, key);
                erbDec->run();
                mask = erbDec->output ("m");
            }

            const float* coefs = nullptr;
            if (applyDf)
            {
                dfDec->copyFrom (*enc, "emb");
                dfDec->copyFrom (*enc, "c0");
                dfDec->run();
                coefs = dfDec->output ("coefs");
            }

            if (mask != nullptr)
            {
                auto bin = 0;
                for (int band = 0; band < numErb; ++band)
                    for (int j = 0; j < erbBands[(size_t) band]; ++j)
                        bandGain[(size_t) bin++] = mask[band];
                if (! allFinite (bandGain.data(), numFreqs))
                {
                    recoverFromBadFrame (input, output);
                    return false;
                }
            }

            if (coefs != nullptr && ! allFinite (coefs, numDf * dfOrder * 2))
            {
                recoverFromBadFrame (input, output);
                return false;
            }

            for (int ch = 0; ch < numChannels; ++ch)
            {
                auto& target = bufferY[(size_t) ch][(size_t) dfOrder - 1];

                if (mask != nullptr)
                    for (int f = 0; f < numFreqs; ++f)
                        target[(size_t) f] *= bandGain[(size_t) f];
                else if (applyZeros)
                    std::fill (target.begin(), target.end(), Complex());

                outSpec = target;

                if (coefs != nullptr)
                {
                    std::fill_n (outSpec.begin(), numDf, Complex());
                    for (int n = 0; n < dfOrder; ++n)
                    {
                        const auto& past = bufferX[(size_t) ch][(size_t) n];
                        for (int f = 0; f < numDf; ++f)
                        {
                            const auto* c = coefs + (f * dfOrder + n) * 2;
                            outSpec[(size_t) f] += past[(size_t) f] * Complex (c[0], c[1]);
                        }
                    }
                }

                synthesis (ch, output[ch]);
            }

            return true;
        }

    private:
        static bool allFinite (const float* data, int count) noexcept
        {
            for (int i = 0; i < count; ++i)
                if (! isFinite (data[i]))
                    return false;
            return true;
        }

        void recoverFromBadFrame (const float* const* input, float* const* output)
        {
            reset();
            for (int ch = 0; ch < channels; ++ch)
                std::copy_n (input[ch], hopSize, output[ch]);
        }

        void analysis (int ch, const float* in) noexcept
        {
            auto& mem = analysisMem[(size_t) ch];
            for (int i = 0; i < hopSize; ++i)
            {
                timeBuffer[(size_t) i] = mem[(size_t) i] * window[(size_t) i];
                timeBuffer[(size_t) (i + hopSize)] = in[i] * window[(size_t) (i + hopSize)];
            }
            std::copy_n (in, hopSize, mem.begin());

            fft.forward (timeBuffer.data(), freqBuffer.data());

            constexpr auto norm = 1.0f / (float) fftSize; // wnorm = 2 * hop / fft^2，这里 hop = fft / 2
            auto& out = spec[(size_t) ch];
            for (int f = 0; f < numFreqs; ++f)
                out[(size_t) f] = freqBuffer[(size_t) f] * norm;
        }

        void synthesis (int ch, float* out) noexcept
        {
            for (int f = 0; f < numFreqs; ++f)
                freqBuffer[(size_t) f] = outSpec[(size_t) f];
            freqBuffer[0] = Complex (outSpec[0].real(), 0.0f);
            freqBuffer[(size_t) numFreqs - 1] = Complex (outSpec[(size_t) numFreqs - 1].real(), 0.0f);
            for (int f = 1; f < numFreqs - 1; ++f)
                freqBuffer[(size_t) (fftSize - f)] = std::conj (outSpec[(size_t) f]);

            fft.inverse (freqBuffer.data(), timeBuffer.data());

            auto& mem = synthesisMem[(size_t) ch];
            for (int i = 0; i < hopSize; ++i)
            {
                out[i] = timeBuffer[(size_t) i].real() * window[(size_t) i] + mem[(size_t) i];
                mem[(size_t) i] = timeBuffer[(size_t) (i + hopSize)].real() * window[(size_t) (i + hopSize)];
            }
        }

        void computeFeatures() noexcept
        {
            const auto* mix = spec[0].data();
            if (channels > 1)
            {
                const auto scale = 1.0f / (float) channels;
                for (int f = 0; f < numFreqs; ++f)
                {
                    auto sum = Complex();
                    for (int ch = 0; ch < channels; ++ch)
                        sum += spec[(size_t) ch][(size_t) f];
                    mixSpec[(size_t) f] = sum * scale;
                }
                mix = mixSpec.data();
            }

            auto* feature = enc->input ("feat_erb");
            auto bin = 0;
            for (int band = 0; band < numErb; ++band)
            {
                const auto size = erbBands[(size_t) band];
                const auto k = 1.0f / (float) size;
                auto power = 0.0f;
                for (int j = 0; j < size; ++j, ++bin)
                    power += std::norm (mix[bin]) * k;

                const auto db = std::log10 (power + 1.0e-10f) * 10.0f;
                auto& state = meanNorm[(size_t) band];
                state = db * (1.0f - normAlpha) + state * normAlpha;
                feature[band] = (db - state) / 40.0f;
            }

            auto* complexFeature = enc->input ("feat_spec");
            for (int f = 0; f < numDf; ++f)
            {
                auto& state = unitNorm[(size_t) f];
                state = std::abs (mix[f]) * (1.0f - normAlpha) + state * normAlpha;
                const auto scale = 1.0f / std::sqrt (state);
                complexFeature[f] = mix[f].real() * scale;
                complexFeature[numDf + f] = mix[f].imag() * scale;
            }
        }

        // 第一次推理会碰到还没进缓存的页面，耗时是常态的十倍以上，放在就绪之前跑掉。
        void warmUp()
        {
            std::vector<float> in ((size_t) hopSize), out ((size_t) hopSize);
            juce::Random random (1);
            for (auto& s : in)
                s = 0.01f * (random.nextFloat() * 2.0f - 1.0f);

            std::vector<const float*> inputs ((size_t) channels, in.data());
            std::vector<std::vector<float>> outBuffers ((size_t) channels, out);
            std::vector<float*> outputs;
            for (auto& b : outBuffers)
                outputs.push_back (b.data());

            for (int i = 0; i < 8; ++i)
                process (inputs.data(), outputs.data(), channels);
        }

        const int lookahead, channels;
        const std::vector<int> erbBands;

        std::unique_ptr<Net> enc, erbDec, dfDec;
        MixedRadixFft fft;
        std::vector<float> window;
        std::vector<Complex> timeBuffer, freqBuffer, mixSpec, outSpec;
        std::vector<float> bandGain;
        std::vector<std::vector<Complex>> spec;
        std::vector<std::vector<std::vector<Complex>>> bufferY, bufferX;
        std::vector<std::vector<float>> analysisMem, synthesisMem;
        std::array<float, numErb> meanNorm {};
        std::array<float, numDf> unitNorm {};
        int skipCounter = 0;
    };

    struct EngineSpec
    {
        const char* directory; // nullptr 表示不走 ONNX
        int lookahead;         // 帧
        int delay;             // 算法延迟，48 kHz 采样，全部实测
        int budget;            // 48 kHz 采样，单帧推理峰值加唤醒和回调抖动的余量，见 NOTES.md
    };

    constexpr EngineSpec engineSpecs[numEngineKinds] =
    {
        { "df3",    2, 1440, 240 }, // DeepFilterNet3
        { "df3_ll", 0, 480,  336 }, // DeepFilterNet3 LL
        { nullptr,  0, 960,  144 }, // RNNoise
    };

    const EngineSpec& getSpec (EngineKind kind) noexcept
    {
        return engineSpecs[(size_t) kind];
    }
}

int getEngineDelay (EngineKind kind) noexcept
{
    return getSpec (kind).delay;
}

int getEngineBudget (EngineKind kind) noexcept
{
    return getSpec (kind).budget;
}

std::unique_ptr<VoiceEngine> createEngine (EngineKind kind, int numChannels, juce::String& error)
{
    const auto& spec = getSpec (kind);
    if (spec.directory == nullptr)
        return createRnnoiseEngine (numChannels);

    const auto* api = loadRuntime (error);
    if (api == nullptr)
        return nullptr;

    Ort::InitApi (api);

    try
    {
        return std::make_unique<DfEngine> (findModelRoot().getChildFile (spec.directory), spec.lookahead, numChannels);
    }
    catch (const std::exception& e)
    {
        error = e.what();
    }
    return nullptr;
}

} // namespace lethe
