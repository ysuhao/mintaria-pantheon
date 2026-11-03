// 截高级页，并验证分页状态和旋钮的数值输入。mint_probe shot 只能截默认的主控页，而 JUCE 判断鼠标下的控件时
// 读的是系统真实光标，给窗口发消息点不到分页。这里把插件源码直接编进来，调用分页控件的真实 mouseDown，
// 再做离屏快照，不碰真实鼠标键盘。
//
//   mint_delay_shot <main.png> <advanced.png> <main_again.png>
#include "../EchoPlugin.cpp"

#include <iostream>

namespace
{
    bool savePng (juce::Component& component, const juce::File& file)
    {
        file.deleteFile();
        juce::FileOutputStream stream (file);
        juce::PNGImageFormat png;
        return stream.openedOk()
            && png.writeImageToStream (component.createComponentSnapshot (component.getLocalBounds()), stream);
    }

    void clickAt (juce::Component& component, juce::Point<float> position)
    {
        const auto now = juce::Time::getCurrentTime();
        component.mouseDown ({ juce::Desktop::getInstance().getMainMouseSource(), position, juce::ModifierKeys::leftButtonModifier,
                               juce::MouseInputSource::defaultPressure, 0.0f, 0.0f, 0.0f, 0.0f,
                               &component, &component, now, position, now, 1, false });
    }

    PageSwitch* findPageSwitch (juce::Component& editor)
    {
        for (auto* child : editor.getChildren())
            if (auto* found = dynamic_cast<PageSwitch*> (child))
                return found;
        return nullptr;
    }
}

namespace
{
    int visibleDials (juce::Component& editor)
    {
        auto count = 0;
        for (auto* child : editor.getChildren())
            if (dynamic_cast<mint::ui::Dial*> (child) != nullptr && child->isVisible())
                ++count;
        return count;
    }

    bool report (const char* name, bool ok, const juce::String& detail = {})
    {
        std::cout << (ok ? "PASS " : "FAIL ") << name;
        if (detail.isNotEmpty())
            std::cout << ": " << detail;
        std::cout << "\n";
        return ok;
    }

    // 双击输入框里显示的文字原样提交，数值要回到显示的值。整个范围都扫一遍。
    bool checkValueEntry (juce::Component& editor, EchoProcessor& processor)
    {
        auto ok = true;
        juce::String detail;

        for (auto* child : editor.getChildren())
        {
            auto* dial = dynamic_cast<mint::ui::Dial*> (child);
            if (dial == nullptr)
                continue;

            auto& slider = dial->slider;
            const auto saved = slider.getValue();
            for (int step = 0; step <= 200; ++step)
            {
                slider.setValue (slider.proportionOfLengthToValue ((double) step / 200.0), juce::dontSendNotification);
                const auto shown = slider.getTextFromValue (slider.getValue());
                const auto parsed = slider.getValueFromText (shown);
                if (slider.getTextFromValue (parsed) != shown)
                {
                    ok = false;
                    detail << shown << " -> " << slider.getTextFromValue (parsed) << "; ";
                    break;
                }
            }
            slider.setValue (saved, juce::dontSendNotification);
        }
        ok = report ("value entry round trip on every dial", ok, detail) && ok;

        const auto parameterValue = [&processor] (const char* id, const juce::String& text)
        {
            auto* parameter = processor.apvts.getParameter (id);
            return parameter->convertFrom0to1 (parameter->getValueForText (text));
        };
        const auto near = [] (float a, float b) { return std::abs (a - b) < 0.5f; };
        ok = report ("7.0 kHz reads as 7000 Hz", near (parameterValue ("hi_cut", "7.0 kHz"), 7000.0f)) && ok;
        ok = report ("1.234 s reads as 1234 ms", near (parameterValue ("time_ms", "1.234 s"), 1234.0f)) && ok;
        ok = report ("300 ms reads as 300 ms", near (parameterValue ("time_ms", "300 ms"), 300.0f)) && ok;
        ok = report ("plain 450 reads as 450 ms", near (parameterValue ("time_ms", "450"), 450.0f)) && ok;
        return ok;
    }
}

int wmain (int argc, wchar_t* argv[])
{
    if (argc < 4)
    {
        std::cout << "usage: mint_delay_shot <main.png> <advanced.png> <main_again.png>\n";
        return 2;
    }

    juce::ScopedJuceInitialiser_GUI gui;
    EchoProcessor processor;
    processor.prepareToPlay (48000.0, 256);
    std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());

    auto* pages = findPageSwitch (*editor);
    if (pages == nullptr)
    {
        std::cout << "FAIL page switch not found\n";
        return 1;
    }

    const auto left = pages->getLocalBounds().toFloat().getCentre().withX ((float) pages->getWidth() * 0.25f);
    const auto right = pages->getLocalBounds().toFloat().getCentre().withX ((float) pages->getWidth() * 0.75f);

    auto ok = report ("starts on main view", ! processor.isAdvancedViewOpen() && visibleDials (*editor) == 3);
    ok = savePng (*editor, juce::File (juce::String (argv[1]))) && ok;
    ok = checkValueEntry (*editor, processor) && ok;

    juce::MemoryBlock mainState;
    processor.getStateInformation (mainState);

    clickAt (*pages, right);
    ok = report ("click opens advanced view", processor.isAdvancedViewOpen() && visibleDials (*editor) == 7) && ok;
    ok = savePng (*editor, juce::File (juce::String (argv[2]))) && ok;

    // 关掉窗口再开，要停在高级视图。
    editor.reset();
    editor.reset (processor.createEditor());
    pages = findPageSwitch (*editor);
    ok = report ("reopened window stays on advanced view", pages != nullptr && visibleDials (*editor) == 7) && ok;

    // 存进工程再恢复到新实例，也要停在高级视图。
    juce::MemoryBlock advancedState;
    processor.getStateInformation (advancedState);
    {
        EchoProcessor restored;
        restored.setStateInformation (advancedState.getData(), (int) advancedState.getSize());
        std::unique_ptr<juce::AudioProcessorEditor> restoredEditor (restored.createEditor());
        ok = report ("saved state restores advanced view", visibleDials (*restoredEditor) == 7) && ok;
    }

    // 窗口开着时宿主恢复工程，界面要跟上。
    processor.setStateInformation (mainState.getData(), (int) mainState.getSize());
    juce::MessageManager::getInstance()->runDispatchLoopUntil (100);
    ok = report ("state restore with window open switches view", visibleDials (*editor) == 3) && ok;

    processor.setStateInformation (advancedState.getData(), (int) advancedState.getSize());
    juce::MessageManager::getInstance()->runDispatchLoopUntil (100);
    ok = report ("state restore with window open switches back", visibleDials (*editor) == 7) && ok;

    pages = findPageSwitch (*editor);
    clickAt (*pages, left);
    ok = report ("click returns to main view", ! processor.isAdvancedViewOpen() && visibleDials (*editor) == 3) && ok;
    ok = savePng (*editor, juce::File (juce::String (argv[3]))) && ok;

    std::cout << (ok ? "PASS all\n" : "FAILED\n");
    return ok ? 0 : 1;
}
