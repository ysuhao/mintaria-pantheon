// 打印 VST3 某个参数在一组归一化值上的显示文字和宿主读到的尾长。mint_probe 只能读默认值，所以单独做这个小工具。
//   elysium_param_text <plugin.vst3> "<参数名>" [步数]
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <iostream>

int wmain (int argc, wchar_t* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;
    if (argc < 3)
        return 2;

    const juce::String path (argv[1]), wanted (argv[2]);
    const auto steps = argc > 3 ? juce::String (argv[3]).getIntValue() : 20;

    juce::AudioPluginFormatManager manager;
    juce::addDefaultFormatsToManager (manager);

    juce::OwnedArray<juce::PluginDescription> found;
    for (auto* format : manager.getFormats())
        if (format->fileMightContainThisPluginType (path))
            format->findAllTypesForFile (found, path);

    if (found.isEmpty())
        return 1;

    juce::String error;
    auto plugin = manager.createPluginInstance (*found[0], 48000.0, 256, error);
    if (plugin == nullptr)
        return 1;

    plugin->prepareToPlay (48000.0, 256);

    for (auto* param : plugin->getParameters())
    {
        if (param->getName (64) != wanted)
            continue;

        for (int i = 0; i <= steps; ++i)
        {
            const auto value = (float) i / (float) steps;
            param->setValueNotifyingHost (value);
            std::cout << value << "\t" << param->getCurrentValueAsText() << "\ttail " << plugin->getTailLengthSeconds() << " s\n";
        }
    }

    plugin->releaseResources();
    return 0;
}
