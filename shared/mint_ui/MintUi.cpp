#include "MintUi.h"

namespace mint::ui
{

namespace
{
    juce::Colour c (juce::uint32 rgb)     { return juce::Colour (0xff000000u | rgb); }

    constexpr float arcStart = juce::MathConstants<float>::pi * 1.25f - juce::MathConstants<float>::twoPi;
    constexpr float arcEnd   = juce::MathConstants<float>::pi * 0.75f;

    juce::Point<float> polar (juce::Point<float> centre, float radius, float angle)
    {
        return centre.getPointOnCircumference (radius, angle);
    }

    juce::ColourGradient diagonal (juce::Rectangle<float> r, juce::Colour a, juce::Colour b, juce::Colour d)
    {
        juce::ColourGradient g (a, r.getX() + r.getWidth() * 0.15f, r.getY() + r.getHeight() * 0.05f,
                                d, r.getX() + r.getWidth() * 0.8f, r.getBottom() - r.getHeight() * 0.05f, false);
        g.addColour (0.42, b);
        return g;
    }

    void drawMark (juce::Graphics& g, juce::Rectangle<float> area, const Theme& t)
    {
        // 与 mintaria-app.svg 相同的几何：M、底线和中柱、菱形宝石。
        const auto s = area.getWidth() / 64.0f;
        const auto transform = juce::AffineTransform::scale (1.15789f)
                                   .translated (10.0f, 6.02632f)
                                   .scaled (s)
                                   .translated (area.getX(), area.getY());

        juce::Path m;
        m.startNewSubPath (5, 37); m.lineTo (5, 9); m.lineTo (19, 27); m.lineTo (33, 9); m.lineTo (33, 37);
        g.setColour (c (0x202640));
        g.strokePath (m, juce::PathStrokeType (3.2f * 1.15789f * s, juce::PathStrokeType::mitered, juce::PathStrokeType::square), transform);

        // 直接放在青瓷底上，钢蓝看不清，底线和宝石改用象牙色。
        g.setColour (t.plate);

        juce::Path base;
        base.startNewSubPath (1, 41); base.lineTo (37, 41);
        base.startNewSubPath (19, 33); base.lineTo (19, 41);
        g.strokePath (base, juce::PathStrokeType (1.6f * 1.15789f * s, juce::PathStrokeType::mitered, juce::PathStrokeType::square), transform);

        juce::Path jewel;
        jewel.startNewSubPath (19, 1); jewel.lineTo (23, 5); jewel.lineTo (19, 9); jewel.lineTo (15, 5); jewel.closeSubPath();
        g.fillPath (jewel, transform);
    }
}

//==============================================================================
namespace
{
    juce::String languageCode (Language language)
    {
        return language == Language::ko ? "ko" : (language == Language::zh ? "zh" : "en");
    }

    Language languageFromCode (const juce::String& code, Language fallback)
    {
        const auto lower = code.trim().toLowerCase();
        if (lower.startsWith ("ko")) return Language::ko;
        if (lower.startsWith ("zh")) return Language::zh;
        if (lower.startsWith ("en")) return Language::en;
        return fallback;
    }

    const juce::File& languageFile()
    {
        static const auto file = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                                     .getChildFile ("Mintaria").getChildFile ("Pantheon").getChildFile ("language.txt");
        return file;
    }

    bool languageIsForced()
    {
        static const auto forced = juce::SystemStats::getEnvironmentVariable ("MINT_LANG", {}).isNotEmpty();
        return forced;
    }

    juce::Time& lastSeenLanguageFile()
    {
        static juce::Time time;
        return time;
    }

    Language& languageStorage()
    {
        static Language language = []
        {
            // 用户没选过时跟随系统语言。
            const auto system = languageFromCode (juce::SystemStats::getDisplayLanguage(), Language::en);
            if (languageIsForced())
                return languageFromCode (juce::SystemStats::getEnvironmentVariable ("MINT_LANG", {}), system);

            const auto& file = languageFile();
            if (! file.existsAsFile())
                return system;

            lastSeenLanguageFile() = file.getLastModificationTime();
            return languageFromCode (file.loadFileAsString(), system);
        }();
        return language;
    }

    // 另一款插件可能改了设置。只查修改时间，文件变了才读内容。
    void reloadLanguageIfFileChanged()
    {
        if (languageIsForced())
            return;

        // 同一个 DLL 开多少实例，每秒也只查一次磁盘。
        static juce::uint32 lastCheck = 0;
        const auto now = juce::Time::getMillisecondCounter();
        if (lastCheck != 0 && now - lastCheck < 900)
            return;
        lastCheck = now;

        const auto& file = languageFile();
        const auto modified = file.getLastModificationTime();
        if (modified != lastSeenLanguageFile() && file.existsAsFile())
        {
            lastSeenLanguageFile() = modified;
            languageStorage() = languageFromCode (file.loadFileAsString(), languageStorage());
        }
    }

    juce::String bodyTypefaceName (Language language)
    {
        return language == Language::ko ? "Malgun Gothic" : (language == Language::zh ? "Microsoft YaHei UI" : "Segoe UI");
    }

    juce::Font bodyFontFor (Language language, float height, bool bold)
    {
        return juce::Font (juce::FontOptions (bodyTypefaceName (language), height, bold ? juce::Font::bold : juce::Font::plain));
    }
}

Language currentLanguage()
{
    return languageStorage();
}

void setLanguage (Language language)
{
    languageStorage() = language;
    if (languageIsForced())
        return;

    const auto& file = languageFile();
    file.getParentDirectory().createDirectory();
    if (file.replaceWithText (languageCode (language)))
        lastSeenLanguageFile() = file.getLastModificationTime();
}

Text::Text (const char* english, const char* korean, const char* chinese)
    : en (juce::String::fromUTF8 (english)), ko (juce::String::fromUTF8 (korean)), zh (juce::String::fromUTF8 (chinese))
{
}

Text::Text (const char* all)
    : Text (all, all, all)
{
}

Text::Text (const juce::String& all)
    : en (all), ko (all), zh (all)
{
}

juce::String tr (const Text& text)
{
    switch (currentLanguage())
    {
        case Language::ko:  return text.ko;
        case Language::zh:  return text.zh;
        case Language::en:  break;
    }
    return text.en;
}

//==============================================================================
const Theme& theme()
{
    // 青瓷神谕方向。象牙、墨蓝、黄铜、钢蓝、丁香取自 Mintaria 设计系统，青瓷取自启动图。
    static const Theme celadon {
        c (0x5f9d9c), c (0xfffaf0), c (0x202640), c (0x676976), c (0xb3a07b), c (0xded9cb),
        c (0x527fac), c (0x8a7cc8),
        c (0xbcc6d1), c (0x202640), c (0xb3a07b),
        c (0xe6d6ab), c (0xb3a07b), c (0x7f6f50),
        c (0xa8d3cc), c (0x84bab5), c (0x5f9d9c),
        c (0x202640), c (0xe8e3d6), c (0x5f9d9c), c (0x202640), c (0xeb7b74),
        c (0x4f8d8d), c (0x7fb8b4)
    };
    return celadon;
}

juce::Font headingFont (float height)
{
    return juce::Font (juce::FontOptions ("Cambria", height, juce::Font::plain));
}

juce::Font bodyFont (float height, bool bold)
{
    return bodyFontFor (currentLanguage(), height, bold);
}

//==============================================================================
LookAndFeel::LookAndFeel()
{
    setDefaultSansSerifTypefaceName (bodyTypefaceName (currentLanguage()));
}

void LookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height, float sliderPos,
                                    float startAngle, float endAngle, juce::Slider& slider)
{
    const auto& t = theme();
    const auto area = juce::Rectangle<int> (x, y, width, height).toFloat().reduced (2.0f);
    const auto radius = juce::jmin (area.getWidth(), area.getHeight()) * 0.5f;
    const auto centre = area.getCentre();
    const auto valueAngle = startAngle + sliderPos * (endAngle - startAngle);
    const auto big = radius > 80.0f;
    const auto numTicks = big ? 60 : 40;

    for (int i = 0; i <= numTicks; ++i)
    {
        const auto proportion = (float) i / (float) numTicks;
        const auto angle = startAngle + proportion * (endAngle - startAngle);
        const auto major = i % 10 == 0;
        const auto lit = proportion <= sliderPos;
        g.setColour (lit ? t.tickOn : t.tickOff);
        g.drawLine ({ polar (centre, radius * (major ? 0.80f : 0.84f), angle), polar (centre, radius * 0.90f, angle) }, lit ? 1.3f : 1.0f);
    }

    juce::Path rail;
    rail.addCentredArc (centre.x, centre.y, radius * 0.94f, radius * 0.94f, 0.0f, startAngle, valueAngle, true);
    const auto active = slider.isMouseOverOrDragging();
    g.setColour (active ? t.bezelDark : t.rail);
    g.strokePath (rail, juce::PathStrokeType (active ? 1.8f : 1.2f));

    const auto bezel = juce::Rectangle<float> (radius * 1.46f, radius * 1.46f).withCentre (centre);
    g.setGradientFill (diagonal (bezel, t.bezelLight, t.bezelMid, t.bezelDark));
    g.fillEllipse (bezel);
    g.setColour (t.bezelDark);
    g.drawEllipse (bezel, 0.7f);

    const auto face = juce::Rectangle<float> (radius * 1.26f, radius * 1.26f).withCentre (centre);
    g.setGradientFill (diagonal (face, t.faceLight, t.faceMid, t.faceDark));
    g.fillEllipse (face);
    g.setColour (t.plate);
    g.drawEllipse (face, 1.2f);

    if (big)
    {
        juce::Path crests;
        const auto r = radius * 0.15f;
        for (int row = 0; row < 2; ++row)
        {
            const auto y = centre.y + radius * (0.15f + 0.15f * (float) row);
            const auto count = 3 - row;
            for (int i = 0; i < count; ++i)
            {
                const auto cx = centre.x + r * 2.0f * ((float) i - (float) (count - 1) * 0.5f);
                crests.addCentredArc (cx, y, r, r, 0.0f, -juce::MathConstants<float>::halfPi, juce::MathConstants<float>::halfPi, true);
            }
        }
        g.setColour (t.plate.withAlpha (0.65f));
        g.strokePath (crests, juce::PathStrokeType (0.8f));
    }

    g.setColour (t.pointer);
    g.drawLine ({ polar (centre, radius * (big ? 0.47f : 0.28f), valueAngle), polar (centre, radius * 0.60f, valueAngle) }, big ? 2.4f : 2.0f);

    if (big)
    {
        juce::Path jewel;
        const auto d = radius * 0.13f;
        jewel.startNewSubPath (centre.x, centre.y - d);
        jewel.lineTo (centre.x + d, centre.y);
        jewel.lineTo (centre.x, centre.y + d);
        jewel.lineTo (centre.x - d, centre.y);
        jewel.closeSubPath();
        g.setColour (t.plate);
        g.fillPath (jewel);
        g.setColour (t.rail);
        g.strokePath (jewel, juce::PathStrokeType (1.0f));

        juce::Path core;
        const auto e = d * 0.57f;
        core.startNewSubPath (centre.x, centre.y - e);
        core.lineTo (centre.x + e, centre.y);
        core.lineTo (centre.x, centre.y + e);
        core.lineTo (centre.x - e, centre.y);
        core.closeSubPath();
        g.setColour (t.ink);
        g.fillPath (core);
    }
}

//==============================================================================
Dial::Dial (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, Text label, bool big)
    : slider (juce::Slider::RotaryVerticalDrag, juce::Slider::NoTextBox),
      labelText (std::move (label)),
      isBig (big),
      attachment (state, paramId, slider)
{
    slider.setRotaryParameters (arcStart + juce::MathConstants<float>::twoPi, arcEnd + juce::MathConstants<float>::twoPi, true);
    slider.setPopupDisplayEnabled (false, false, nullptr);
    slider.setRepaintsOnMouseActivity (true);
    slider.setMouseCursor (juce::MouseCursor::UpDownResizeCursor);
    slider.setMouseDragSensitivity (big ? 280 : 200);
    // 按住 Shift 精调。
    slider.setVelocityModeParameters (0.12, 1, 0.0, true, juce::ModifierKeys::shiftModifier);
    if (auto* parameter = state.getParameter (paramId))
        slider.setDoubleClickReturnValue (true, parameter->convertFrom0to1 (parameter->getDefaultValue()));
    slider.onValueChange = [this] { repaint (getValueArea()); };
    addAndMakeVisible (slider);
    setRepaintsOnMouseActivity (false);
}

double parseTypedValue (const juce::String& typed, const juce::String& shown, double current, double fallback)
{
    const auto unitScale = [] (juce::String text)
    {
        text = text.retainCharacters ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").toLowerCase();
        if (text == "khz")  return 1000.0;
        if (text == "hz")   return 1.0;
        if (text == "ms")   return 0.001;
        if (text == "s")    return 1.0;
        return 0.0;
    };
    const auto hasNumber = [] (const juce::String& text) { return text.containsAnyOf ("0123456789"); };

    if (! hasNumber (typed) || ! hasNumber (shown))
        return fallback;

    const auto typedNumber = typed.retainCharacters ("0123456789.-+").getDoubleValue();
    const auto shownNumber = shown.upToFirstOccurrenceOf (":", false, false).retainCharacters ("0123456789.-+").getDoubleValue();
    if (std::abs (shownNumber) < 1.0e-9)
        return fallback;

    const auto shownUnit = unitScale (shown);
    const auto typedUnit = unitScale (typed);
    const auto conversion = shownUnit > 0.0 && typedUnit > 0.0 ? typedUnit / shownUnit : 1.0;
    return typedNumber * conversion * (current / shownNumber);
}

juce::Rectangle<int> Dial::getValueArea() const
{
    return getLocalBounds().removeFromBottom (isBig ? 46 : 40).removeFromTop (isBig ? 26 : 22);
}

void Dial::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (isEnabled() && slider.isEnabled() && getValueArea().contains (e.getPosition()))
        showValueEditor();
}

void Dial::showValueEditor()
{
    const auto& t = theme();
    valueEditor = std::make_unique<juce::TextEditor>();
    valueEditor->setBounds (getValueArea().withSizeKeepingCentre (juce::jmin (getWidth(), 110), isBig ? 26 : 22));
    valueEditor->setJustification (juce::Justification::centred);
    valueEditor->setFont (headingFont (isBig ? 20.0f : 16.0f));
    valueEditor->setColour (juce::TextEditor::backgroundColourId, t.plate);
    valueEditor->setColour (juce::TextEditor::textColourId, t.ink);
    valueEditor->setColour (juce::TextEditor::outlineColourId, t.trim);
    valueEditor->setColour (juce::TextEditor::focusedOutlineColourId, t.bezelDark);
    valueEditor->setColour (juce::TextEditor::highlightColourId, t.trimSoft);
    valueEditor->setSelectAllWhenFocused (true);
    valueEditorStart = slider.getTextFromValue (slider.getValue());
    valueEditor->setText (valueEditorStart, false);
    valueEditor->onReturnKey = [this] { closeValueEditor (true); };
    valueEditor->onEscapeKey = [this] { closeValueEditor (false); };
    valueEditor->onFocusLost = [this] { closeValueEditor (true); };
    addAndMakeVisible (*valueEditor);
    valueEditor->grabKeyboardFocus();
}

void Dial::closeValueEditor (bool apply)
{
    if (valueEditor == nullptr)
        return;

    // 回调还在编辑框里运行，放到下一轮消息再销毁。
    const auto text = valueEditor->getText();
    valueEditor->setVisible (false);
    // 没改文字就不写回，否则显示精度的舍入会悄悄改掉参数。
    if (apply && text.trim() != valueEditorStart.trim())
        slider.setValue (parseTypedValue (text, valueEditorStart, slider.getValue(), slider.getValueFromText (text)), juce::sendNotificationSync);

    juce::Component::SafePointer<Dial> safe (this);
    juce::MessageManager::callAsync ([safe]
    {
        if (safe != nullptr)
            safe->valueEditor.reset();
    });
}

void Dial::resized()
{
    slider.setBounds (getLocalBounds().withTrimmedBottom (isBig ? 46 : 40));
}

void Dial::paint (juce::Graphics& g)
{
    const auto& t = theme();
    auto text = getLocalBounds().removeFromBottom (isBig ? 46 : 40);

    g.setColour (t.ink);
    g.setFont (headingFont (isBig ? 22.0f : 18.0f));
    g.drawText (slider.getTextFromValue (slider.getValue()), text.removeFromTop (isBig ? 26 : 22), juce::Justification::centred);

    g.setColour (t.inkMuted);
    g.setFont (bodyFont (12.5f));
    g.drawFittedText (tr (labelText), text, juce::Justification::centred, 1, 0.75f);
}

//==============================================================================
Meter::Meter (MeterSource& source, Text label, bool isReduction)
    : meterSource (source), labelText (std::move (label)), reduction (isReduction)
{
    startTimerHz (30);
}

Meter::~Meter()
{
    stopTimer();
}

void Meter::timerCallback()
{
    const auto incoming = meterSource.take();
    const auto releaseKeep = 0.72f;
    const auto fromGain = dbToGain (shown) * releaseKeep;
    shown = juce::jmax (incoming, gainToDb (fromGain));

    // 没有段变化就不重绘，十几个实例开着时差别很大。
    const auto lit = litSegments();
    if (lit != shownSegments)
    {
        shownSegments = lit;
        repaint (getLocalBounds().withTrimmedBottom (20));
    }
}

int Meter::litSegments() const
{
    if (shown <= -119.0f)
        return 0;
    const auto level = reduction ? juce::jmap (juce::jlimit (-24.0f, 0.0f, shown), -24.0f, 0.0f, 0.0f, 1.0f)
                                 : juce::jmap (juce::jlimit (-60.0f, -6.0f, shown), -60.0f, -6.0f, 0.0f, 1.0f);
    return juce::roundToInt (level * 12.0f);
}

void Meter::paint (juce::Graphics& g)
{
    const auto& t = theme();
    auto bounds = getLocalBounds().toFloat();
    auto labelArea = bounds.removeFromBottom (20.0f);

    constexpr int segments = 12;
    const auto pitch = bounds.getHeight() / segments;
    const auto segHeight = juce::jmin (7.0f, pitch - 3.0f);
    const auto segWidth = juce::jmin (20.0f, bounds.getWidth());
    const auto x = bounds.getCentreX() - segWidth * 0.5f;

    // 增益衰减表自上而下点亮，电平表自下而上点亮。
    for (int i = 0; i < segments; ++i)
    {
        const auto row = reduction ? i : segments - 1 - i;
        const auto seg = juce::Rectangle<float> (x, bounds.getY() + ((float) row + 0.5f) * pitch - segHeight * 0.5f, segWidth, segHeight);
        if (i < shownSegments)
        {
            g.setColour (reduction ? t.trim : (i < 8 ? t.meterLow : (i < 10 ? t.meterHigh : t.meterClip)));
            g.fillRoundedRectangle (seg, 1.5f);
        }
        else
        {
            g.setColour (t.meterOff);
            g.fillRoundedRectangle (seg, 1.5f);
            g.setColour (t.trim.withAlpha (0.55f));
            g.drawRoundedRectangle (seg.reduced (0.25f), 1.5f, 0.75f);
        }
    }

    g.setColour (t.inkMuted);
    g.setFont (bodyFont (12.0f));
    g.drawFittedText (tr (labelText), labelArea.toNearestInt(), juce::Justification::centredBottom, 1, 0.7f);
}

//==============================================================================
Segmented::Segmented (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, std::vector<Text> labels)
    : choices (std::move (labels)),
      attachment (*state.getParameter (paramId), [this] (float value)
      {
          selected = juce::roundToInt (value);
          // 第一次直接就位，之后滑过去。
          if (shownPosition < 0.0f)
              shownPosition = (float) selected;
          else
              startTimerHz (60);
          repaint();
      }, nullptr)
{
    if (auto* choice = dynamic_cast<juce::AudioParameterChoice*> (state.getParameter (paramId)))
    {
        jassert (choices.empty() || (int) choices.size() == choice->choices.size());
        if (choices.empty())
            for (const auto& name : choice->choices)
                choices.emplace_back (name);
    }
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    attachment.sendInitialUpdate();
}

void Segmented::timerCallback()
{
    const auto target = (float) selected;
    shownPosition += (target - shownPosition) * 0.35f;
    if (std::abs (target - shownPosition) < 0.01f)
    {
        shownPosition = target;
        stopTimer();
    }
    repaint();
}

int Segmented::cellAt (int x) const
{
    const auto count = (int) choices.size();
    return juce::jlimit (0, juce::jmax (0, count - 1), x * count / juce::jmax (1, getWidth()));
}

void Segmented::setHovered (int cell)
{
    if (cell != hovered)
    {
        hovered = cell;
        repaint();
    }
}

void Segmented::mouseMove (const juce::MouseEvent& e)     { setHovered (cellAt (e.x)); }
void Segmented::mouseExit (const juce::MouseEvent&)       { setHovered (-1); }

void Segmented::paint (juce::Graphics& g)
{
    const auto& t = theme();
    const auto count = (int) choices.size();
    const auto w = (float) getWidth() / (float) juce::jmax (1, count);
    const auto h = (float) getHeight();

    if (hovered >= 0 && hovered != selected)
    {
        g.setColour (t.trimSoft.withAlpha (pressed == hovered ? 0.9f : 0.45f));
        g.fillRect (juce::Rectangle<float> ((float) hovered * w, 0.0f, w, h));
    }

    g.setColour (t.ink);
    g.fillRect (juce::Rectangle<float> (juce::jmax (0.0f, shownPosition) * w, 0.0f, w, h));

    g.setColour (t.trim);
    g.drawRect (getLocalBounds().toFloat(), 1.0f);
    for (int i = 1; i < count; ++i)
        g.fillRect (juce::Rectangle<float> ((float) i * w - 0.5f, 0.0f, 1.0f, h));

    g.setFont (bodyFont (13.0f));
    for (int i = 0; i < count; ++i)
    {
        // 文字颜色跟着滑块覆盖的比例变，切换时不会闪。
        const auto covered = juce::jlimit (0.0f, 1.0f, 1.0f - std::abs ((float) i - shownPosition));
        g.setColour (t.ink.interpolatedWith (t.plate, covered));
        g.drawFittedText (tr (choices[(size_t) i]), juce::Rectangle<float> ((float) i * w, 0.0f, w, h).reduced (3.0f, 0.0f).toNearestInt(),
                          juce::Justification::centred, 1, 0.75f);
    }
}

void Segmented::mouseDown (const juce::MouseEvent& e)
{
    if (choices.empty() || ! e.mods.isLeftButtonDown())
        return;
    pressed = cellAt (e.x);
    repaint();
    // 点已选中的项不改参数，免得给宿主多记一步撤销。
    if (pressed != selected)
        attachment.setValueAsCompleteGesture ((float) pressed);
}

void Segmented::mouseUp (const juce::MouseEvent&)
{
    pressed = -1;
    repaint();
}

//==============================================================================
LanguageSwitch::LanguageSwitch()
{
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
}

void LanguageSwitch::paint (juce::Graphics& g)
{
    // 每种语言的按钮用自己的字体，这样任何界面语言下都能显示正确。
    struct Cell { Language language; const char* label; };
    static constexpr Cell cells[] { { Language::en, "EN" }, { Language::ko, "한" }, { Language::zh, "中" } };

    const auto& t = theme();
    const auto w = (float) getWidth() / 3.0f;

    for (int i = 0; i < 3; ++i)
    {
        const auto cell = juce::Rectangle<float> ((float) i * w, 0.0f, w, (float) getHeight());
        const auto active = cells[i].language == currentLanguage();
        if (active || i == hovered)
        {
            g.setColour (active ? t.ink : t.trimSoft.withAlpha (0.6f));
            g.fillRect (cell);
        }
        g.setColour (t.trim);
        g.drawRect (cell, 1.0f);
        g.setColour (active ? t.plate : t.ink);
        g.setFont (bodyFontFor (cells[i].language, 12.0f, false));
        g.drawText (juce::String::fromUTF8 (cells[i].label), cell, juce::Justification::centred);
    }
}

void LanguageSwitch::mouseMove (const juce::MouseEvent& e)
{
    const auto cell = juce::jlimit (0, 2, e.x * 3 / juce::jmax (1, getWidth()));
    if (cell != hovered)
    {
        hovered = cell;
        repaint();
    }
}

void LanguageSwitch::mouseExit (const juce::MouseEvent&)
{
    hovered = -1;
    repaint();
}

void LanguageSwitch::mouseDown (const juce::MouseEvent& e)
{
    static constexpr Language order[] { Language::en, Language::ko, Language::zh };
    const auto language = order[juce::jlimit (0, 2, e.x * 3 / juce::jmax (1, getWidth()))];
    if (! e.mods.isLeftButtonDown() || language == currentLanguage())
        return;
    setLanguage (language);
    if (onChange)
        onChange();
}

//==============================================================================
EditorBase::EditorBase (mint::Processor& p, ProductInfo productInfo)
    : juce::AudioProcessorEditor (p), processor (p), info (std::move (productInfo))
{
    reloadLanguageIfFileChanged();
    shownLanguage = currentLanguage();
    lookAndFeel.setDefaultSansSerifTypefaceName (bodyTypefaceName (shownLanguage));
    setLookAndFeel (&lookAndFeel);
    setOpaque (true);

    languageSwitch.onChange = [this] { applyLanguage(); };
    languageSwitch.setBounds (width - 172, 36, 130, 22);
    addAndMakeVisible (languageSwitch);

    startTimerHz (4);
}

EditorBase::~EditorBase()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

juce::Rectangle<int> EditorBase::getContentBounds() const
{
    return { 62, 116, width - 124, 290 };
}

juce::Rectangle<int> EditorBase::getFooterSlot() const
{
    return { 200, 428, width - 400, 44 };
}

void EditorBase::applyLanguage()
{
    shownLanguage = currentLanguage();
    lookAndFeel.setDefaultSansSerifTypefaceName (bodyTypefaceName (shownLanguage));
    background = {};
    languageChanged();
    sendLookAndFeelChange();
    repaint();
}

void EditorBase::timerCallback()
{
    if (processor.getLatencySamples() != shownLatency)
    {
        shownLatency = processor.getLatencySamples();
        repaint (62, 444, 136, 22);
    }

    const auto lookahead = getLookaheadText();
    if (lookahead != shownLookahead)
    {
        shownLookahead = lookahead;
        repaint (width - 192, 426, 130, 40);
    }

    // 约每秒查一次设置文件，跟上其他插件里的切换。
    if (++languagePollTicks >= 4)
    {
        languagePollTicks = 0;
        reloadLanguageIfFileChanged();
    }

    if (currentLanguage() != shownLanguage)
        applyLanguage();
}

namespace
{
    // 青海波是周期图案，画一块再平铺，比逐条描边快两个数量级。
    juce::Image makeWaveTile (float scale, const Theme& t)
    {
        constexpr float period = 60.0f, rowStep = 30.0f;
        const auto size = juce::jmax (1, juce::roundToInt (period * scale));
        const auto halfPi = juce::MathConstants<float>::halfPi;

        juce::Path dark, light;
        for (float y = -rowStep; y <= period + rowStep * 2.0f; y += rowStep)
        {
            for (float x = -period; x <= period * 2.0f; x += period)
            {
                for (const auto r : { 30.0f, 22.0f, 14.0f, 6.0f })
                {
                    dark.startNewSubPath (x + period * 0.5f - r, y);
                    dark.addCentredArc (x + period * 0.5f, y, r, r, 0.0f, -halfPi, halfPi);
                }
                for (const auto r : { 30.0f, 22.0f, 14.0f })
                {
                    light.startNewSubPath (x - r, y - rowStep * 0.5f);
                    light.addCentredArc (x, y - rowStep * 0.5f, r, r, 0.0f, -halfPi, halfPi);
                }
            }
        }

        juce::Image tile (juce::Image::ARGB, size, size, true, juce::SoftwareImageType());
        juce::Graphics g (tile);
        g.addTransform (juce::AffineTransform::scale ((float) size / period));
        g.setColour (t.waveLight);
        g.strokePath (light, juce::PathStrokeType (0.7f));
        g.setColour (t.waveDark);
        g.strokePath (dark, juce::PathStrokeType (1.0f));
        return tile;
    }

    juce::Image makeStaticLayer (float scale, const Theme& t)
    {
        const auto w = juce::roundToInt ((float) EditorBase::width * scale);
        const auto h = juce::roundToInt ((float) EditorBase::height * scale);
        juce::Image layer (juce::Image::ARGB, w, h, true, juce::SoftwareImageType());
        juce::Graphics g (layer);
        g.fillAll (t.ground);
        g.setTiledImageFill (makeWaveTile (scale, t), 0, 0, 1.0f);
        g.fillAll();
        g.addTransform (juce::AffineTransform::scale (scale));

    juce::ColourGradient brass (c (0xd8c697), 0.0f, 0.0f, c (0x8f7f5e), 0.0f, (float) EditorBase::height, false);
    brass.addColour (0.5, t.trim);

    g.setGradientFill (brass);
    g.drawRect (juce::Rectangle<float> ((float) EditorBase::width, (float) EditorBase::height).reduced (12.0f), 1.4f);

    // 象牙主面板。
    const juce::Rectangle<float> plate (36.0f, 100.0f, (float) EditorBase::width - 72.0f, 396.0f);
    g.setColour (t.plate);
    g.fillRoundedRectangle (plate, 8.0f);
    g.setGradientFill (brass);
    g.drawRoundedRectangle (plate, 8.0f, 1.6f);
    g.setColour (t.trim);
    g.drawRoundedRectangle (plate.reduced (6.0f), 5.0f, 0.5f);

    // 铭牌。
    juce::Path cartouche;
    cartouche.startNewSubPath (250.0f, 28.0f);
    cartouche.lineTo (570.0f, 28.0f);
    cartouche.lineTo (586.0f, 62.0f);
    cartouche.lineTo (570.0f, 96.0f);
    cartouche.lineTo (250.0f, 96.0f);
    cartouche.lineTo (234.0f, 62.0f);
    cartouche.closeSubPath();
    g.setColour (t.plate);
    g.fillPath (cartouche);
    g.setGradientFill (brass);
    g.strokePath (cartouche, juce::PathStrokeType (1.6f));

    // 铭牌与面板之间的菱形扣。
    const auto diamond = [] (float cx, float cy, float d)
    {
        juce::Path p;
        p.startNewSubPath (cx, cy - d);
        p.lineTo (cx + d, cy);
        p.lineTo (cx, cy + d);
        p.lineTo (cx - d, cy);
        p.closeSubPath();
        return p;
    };
    g.setColour (t.plate);
    g.fillPath (diamond (410.0f, 100.0f, 12.0f));
    g.setGradientFill (brass);
    g.strokePath (diamond (410.0f, 100.0f, 12.0f), juce::PathStrokeType (1.4f));
    g.setColour (t.ground);
    g.fillPath (diamond (410.0f, 100.0f, 6.0f));

    drawMark (g, { 30.0f, 28.0f, 60.0f, 60.0f }, t);

    // 系列牌，上方留给语言切换，高度与铭牌对齐。
    const juce::Rectangle<float> series ((float) EditorBase::width - 180.0f, 28.0f, 146.0f, 68.0f);
    g.setColour (t.plate);
    g.fillRoundedRectangle (series, 8.0f);
    g.setColour (t.trim);
    g.drawRoundedRectangle (series, 8.0f, 1.5f);

        return layer;
    }
}

void EditorBase::renderBackground (float scale)
{
    const auto& t = theme();
    const auto key = juce::roundToInt (scale * 100.0f);
    auto cached = layers->byScale.find (key);
    if (cached == layers->byScale.end())
        cached = layers->byScale.emplace (key, makeStaticLayer (scale, t)).first;

    background = juce::Image (juce::Image::ARGB, cached->second.getWidth(), cached->second.getHeight(), true);
    backgroundScale = scale;

    juce::Graphics g (background);
    g.drawImageAt (cached->second, 0, 0);
    g.addTransform (juce::AffineTransform::scale (scale));

    const juce::Rectangle<float> series ((float) width - 180.0f, 28.0f, 146.0f, 68.0f);

    g.setColour (t.ink);
    g.setFont (headingFont (11.0f).withExtraKerningFactor (0.2f));
    g.drawText ("MINTARIA", (int) series.getX(), 60, (int) series.getWidth(), 15, juce::Justification::centred);
    g.setColour (t.inkMuted);
    g.setFont (headingFont (10.0f).withExtraKerningFactor (0.14f));
    g.drawText (juce::String::fromUTF8 ("PANTHEON N° ") + info.numeral, (int) series.getX(), 75, (int) series.getWidth(), 15, juce::Justification::centred);

    g.setColour (t.ink);
    g.setFont (headingFont (32.0f).withExtraKerningFactor (0.3f));
    g.drawText (info.name, 234, 30, 352, 38, juce::Justification::centred);

    g.setColour (t.inkMuted);
    g.setFont (bodyFont (14.0f));
    g.drawText (tr (info.subtitle), 234, 66, 352, 20, juce::Justification::centred);

    g.setColour (t.trim);
    g.drawHorizontalLine (416, 62.0f, (float) width - 62.0f);

    g.setColour (t.inkMuted);
    g.setFont (bodyFont (12.0f));
    g.drawText (tr ({ "Plugin latency", "플러그인 레이턴시", "插件延迟" }), 62, 426, 130, 16, juce::Justification::centredLeft);
}

void EditorBase::paint (juce::Graphics& g)
{
    const auto& t = theme();
    const auto scale = g.getInternalContext().getPhysicalPixelScaleFactor();
    if (background.isNull() || ! juce::approximatelyEqual (scale, backgroundScale))
        renderBackground (scale);

    g.drawImage (background, getLocalBounds().toFloat());

    // 主读数用毫秒，采样数给需要对齐的人看。
    const auto latency = processor.getLatencySamples();
    const auto rate = processor.getSampleRate();
    const auto milliseconds = rate > 0.0 ? 1000.0 * (double) latency / rate : 0.0;
    g.setColour (t.ink);
    g.setFont (headingFont (18.0f));
    const auto reading = latency == 0 ? juce::String ("0 ms") : juce::String (milliseconds, 1) + " ms";
    g.drawText (reading, 62, 444, 76, 22, juce::Justification::centredLeft);
    if (latency > 0)
    {
        g.setColour (t.inkMuted);
        g.setFont (bodyFont (12.0f));
        g.drawText (juce::String (latency) + " " + tr ({ "samples", "샘플", "采样" }), 138, 444, 60, 22, juce::Justification::centredLeft);
    }

    const auto lookahead = getLookaheadText();
    if (lookahead.isNotEmpty())
    {
        g.setColour (t.inkMuted);
        g.setFont (bodyFont (12.0f));
        g.drawText (tr ({ "Look-ahead", "룩어헤드", "前瞻" }), width - 192, 426, 130, 16, juce::Justification::centredRight);
        g.setColour (t.ink);
        g.setFont (bodyFont (14.0f));
        g.drawText (lookahead, width - 192, 444, 130, 22, juce::Justification::centredRight);
    }
}

} // namespace mint::ui
