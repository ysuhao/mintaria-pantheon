// “轻量”档：RNNoise 的 little 模型。CPU 约为 DeepFilterNet3 的五分之一，语句间隙压得更干净，
// 人声的 SI-SDR 低一些。库本身不分配内存，但一帧要几百微秒，放在工作线程上才不会拖累小缓冲区。
#include "LetheEngine.h"

#include <rnnoise.h>

#include <algorithm>
#include <array>
#include <vector>

namespace lethe
{
namespace
{
    constexpr float int16Scale = 32768.0f; // RNNoise 按 16 位整数的幅度工作

    class RnnoiseEngine final : public VoiceEngine
    {
    public:
        explicit RnnoiseEngine (int numChannels)
        {
            for (int ch = 0; ch < numChannels; ++ch)
                states.push_back (rnnoise_create (nullptr));

            // 第一帧会碰到还没进缓存的权重页面，放在就绪之前跑掉。
            std::array<float, hopSize> silence {}, out {};
            for (auto* state : states)
                rnnoise_process_frame (state, out.data(), silence.data());
            reset();
        }

        ~RnnoiseEngine() override
        {
            for (auto* state : states)
                rnnoise_destroy (state);
        }

        void reset() override
        {
            for (auto* state : states)
                rnnoise_init (state, nullptr);
        }

        bool process (const float* const* input, float* const* output, int numChannels) override
        {
            // 左右相同（立体声轨上的单声道话筒）时只算一次。
            const auto dualMono = numChannels == 2 && std::equal (input[0], input[0] + hopSize, input[1]);

            for (int ch = 0; ch < (dualMono ? 1 : numChannels); ++ch)
            {
                for (size_t i = 0; i < scaled.size(); ++i)
                    scaled[i] = input[ch][i] * int16Scale;
                rnnoise_process_frame (states[(size_t) ch], output[ch], scaled.data());
                for (int i = 0; i < hopSize; ++i)
                    output[ch][i] /= int16Scale;
            }

            if (dualMono)
                std::copy_n (output[0], hopSize, output[1]);
            return true;
        }

    private:
        std::vector<DenoiseState*> states;
        std::array<float, hopSize> scaled {};
    };
}

std::unique_ptr<VoiceEngine> createRnnoiseEngine (int numChannels)
{
    return std::make_unique<RnnoiseEngine> (numChannels);
}

} // namespace lethe
