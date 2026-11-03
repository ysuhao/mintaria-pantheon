#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <mint_core/MintCore.h>

#include <functional>
#include <map>
#include <memory>
#include <vector>

// 全系列共用的界面。插件只能通过这里取颜色和控件，不得写死色值或自绘旋钮，
// 这样换设计方向时只改这一处。
namespace mint::ui
{

struct Theme
{
    juce::Colour ground, plate, ink, inkMuted, trim, trimSoft, accent, accentEnd,
                 tickOff, tickOn, rail, bezelLight, bezelMid, bezelDark,
                 faceLight, faceMid, faceDark, pointer, meterOff, meterLow, meterHigh, meterClip,
                 waveDark, waveLight;
};

const Theme& theme();

juce::Font headingFont (float height);
juce::Font bodyFont (float height, bool bold = false);

enum class Language { en, ko, zh };

// 全系列共用一个界面语言。每款插件是独立的 DLL，所以设置存在用户目录的文件里，
// 打开的界面定时读取，在一款插件里切换后其他插件随之切换。
// 环境变量 MINT_LANG=en|ko|zh 可强制指定语言且不写文件，用于离线截图。
Language currentLanguage();
void setLanguage (Language);

// 界面文案的三语版本。源文件按 UTF-8 编译，字面量可直接写。
// 单参数形式表示三种语言相同，只用于数值、单位等不需要翻译的内容。
struct Text
{
    Text() = default;
    Text (const char* english, const char* korean, const char* chinese);
    Text (const char* all);
    Text (const juce::String& all);

    juce::String en, ko, zh;
};

// 取当前语言的文字。绘制时调用，切换语言后重绘即可更新。
juce::String tr (const Text&);

struct ProductInfo
{
    juce::String name;          // 英文名，如 "JANUS"，任何语言下都不翻译
    Text subtitle;              // 功能名，如 { "Gate", "게이트", "门限" }
    juce::String numeral;       // 系列序号，如 "I"
};

class LookAndFeel : public juce::LookAndFeel_V4
{
public:
    LookAndFeel();

    void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height, float sliderPos,
                           float startAngle, float endAngle, juce::Slider&) override;
};

// 把用户键入的数值换算成参数值。显示文字可能和参数单位不同（如参数是 Hz、显示 "7.0 kHz"），
// 所以用当前的显示文字和参数值反推比例。不带单位时按当前显示的单位理解，换算不了时用 fallback。
double parseTypedValue (const juce::String& typed, const juce::String& shown, double current, double fallback);

// 带数值和标签的旋钮。big 用于每个插件唯一的主控。
class Dial : public juce::Component
{
public:
    Dial (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, Text label, bool big = false);

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDoubleClick (const juce::MouseEvent&) override;

    juce::Slider slider;

private:
    juce::Rectangle<int> getValueArea() const;
    void showValueEditor();
    void closeValueEditor (bool apply);

    Text labelText;
    bool isBig;
    std::unique_ptr<juce::TextEditor> valueEditor;
    juce::String valueEditorStart;
    juce::AudioProcessorValueTreeState::SliderAttachment attachment;
};

// 12 段电平表，电平范围与释放特性沿用 Mintaria 的 LevelMeter。
class Meter : public juce::Component,
              private juce::Timer
{
public:
    Meter (MeterSource& source, Text label, bool isReduction = false);
    ~Meter() override;

    void paint (juce::Graphics&) override;

private:
    void timerCallback() override;

    int litSegments() const;

    MeterSource& meterSource;
    Text labelText;
    bool reduction;
    float shown = -120.0f;
    int shownSegments = 0;
};

// 绑定 AudioParameterChoice 的分段选择器，用于模式切换。
// labels 是界面上显示的三语文字，按参数选项的顺序给出；不给时显示参数自身的选项名。
class Segmented : public juce::Component,
                  private juce::Timer
{
public:
    Segmented (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, std::vector<Text> labels = {});

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;

private:
    void timerCallback() override;
    int cellAt (int x) const;
    void setHovered (int cell);

    std::vector<Text> choices;
    int selected = 0, hovered = -1, pressed = -1;
    float shownPosition = -1.0f;
    juce::ParameterAttachment attachment;
};

// 右上角的语言切换。由 EditorBase 持有，插件不用自己放。
class LanguageSwitch : public juce::Component
{
public:
    LanguageSwitch();

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;

    std::function<void()> onChange;

private:
    int hovered = -1;
};

// 底纹、外框这些不随语言和插件变化的部分。同一个 DLL 的所有界面共用，最后一个界面关闭时释放。
struct BackgroundLayers
{
    std::map<int, juce::Image> byScale;
};

// 整机外框：青瓷波纹底、象牙面板、M 标、英文名、功能名、系列序号、语言切换、延迟读数和 Look-ahead 读数。
// 控件放进 getContentBounds() 和 getFooterSlot() 内。
// 派生类必须在构造函数最后调用 setSize (width, height)，否则 resized() 不会触发，子控件尺寸为 0。
class EditorBase : public juce::AudioProcessorEditor,
                   private juce::Timer
{
public:
    static constexpr int width = 820, height = 520;

    EditorBase (mint::Processor&, ProductInfo);
    ~EditorBase() override;

    void paint (juce::Graphics&) override;

protected:
    juce::Rectangle<int> getContentBounds() const;
    juce::Rectangle<int> getFooterSlot() const;

    // 插件报告 Look-ahead 状态的文字，例如 "关闭 · 0 ms"。没有此功能的插件不覆盖。
    virtual juce::String getLookaheadText() const   { return {}; }

    // 语言变化后调用。只用 tr() 绘制的控件会自动重绘，不用处理；
    // juce::Label、TextButton 这类存了文字的控件，在这里重新设置文字。
    virtual void languageChanged() {}

private:
    void timerCallback() override;
    void renderBackground (float scale);
    void applyLanguage();

    LookAndFeel lookAndFeel;
    mint::Processor& processor;
    ProductInfo info;
    juce::SharedResourcePointer<BackgroundLayers> layers;
    juce::TooltipWindow tooltipWindow { this, 500 };
    LanguageSwitch languageSwitch;
    Language shownLanguage;
    int shownLatency = -1, languagePollTicks = 0;
    juce::String shownLookahead;

    // 外框和文字合成一次后缓存。电平表每秒重绘 30 次，每次都重画太费 CPU。
    juce::Image background;
    float backgroundScale = 0.0f;
};

} // namespace mint::ui
