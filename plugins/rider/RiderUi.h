#pragma once

#include <mint_ui/MintUi.h>

#include <array>
#include <atomic>
#include <functional>
#include <vector>

// 共享层没有的控件：正负增益表、增益曲线、不绑定参数的分段切换。颜色一律取 theme()。
namespace helios
{

// 音频线程写、界面和写自动化定时器读。
struct RiderTelemetry
{
    std::atomic<float> gainDb { 0.0f };
    std::atomic<bool> gateOpen { false };
    std::atomic<float> effectiveTargetDb { -18.0f };
    std::atomic<float> musicDb { -120.0f };
    std::atomic<bool> sidechainConnected { false };
};

// 表的量程跟随上下范围取 ±6、±12、±18、±24，这样 ±6 dB 的默认范围也看得清楚。
inline float meterSpanDb (float up, float down) noexcept
{
    const auto widest = juce::jmax (up, down);
    return widest <= 6.0f ? 6.0f : widest <= 12.0f ? 12.0f : widest <= 18.0f ? 18.0f : 24.0f;
}

inline juce::String signedDb (float db)
{
    const auto rounded = std::abs (db) < 0.05f ? 0.0f : db;
    return (rounded > 0.0f ? "+" : "") + juce::String (rounded, 1);
}

// 竖向增益表：中线 0 dB，向上是上推，向下是下压，黄铜刻线标出当前允许的范围。
class GainMeter final : public juce::Component,
                        private juce::Timer
{
public:
    GainMeter (RiderTelemetry& source, juce::AudioProcessorValueTreeState& state)
        : telemetry (source),
          rangeUp (state.getRawParameterValue ("range_up_db")),
          rangeDown (state.getRawParameterValue ("range_down_db"))
    {
        startTimerHz (30);
    }

    ~GainMeter() override   { stopTimer(); }

    void paint (juce::Graphics& g) override
    {
        const auto& t = mint::ui::theme();
        auto bounds = getLocalBounds();

        g.setColour (t.ink);
        g.setFont (mint::ui::headingFont (18.0f));
        g.drawText (signedDb (shownDb), bounds.removeFromTop (22), juce::Justification::centred);

        // 标签和两侧电平表的标签同一高度、同一字号。
        const auto caption = bounds.removeFromBottom (20);
        g.setColour (t.inkMuted);
        g.setFont (mint::ui::bodyFont (12.0f));
        g.drawText (mint::ui::tr ({ "Gain", "게인", "增益" }), caption, juce::Justification::centredBottom);

        paintStatus (g, bounds.removeFromBottom (18).withSizeKeepingCentre (54, 18).toFloat());

        // 最下面的刻度文字比柱子低 7 px，这里再留 7 px 空隙。
        bounds.removeFromTop (6);
        bounds.removeFromBottom (14);

        const auto up = rangeUp->load();
        const auto down = rangeDown->load();
        const auto span = meterSpanDb (up, down);
        const auto bar = bounds.removeFromLeft (26).withTrimmedLeft (4).toFloat();
        const auto scale = bounds.toFloat().withTrimmedLeft (4.0f);
        const auto yFor = [&] (float db) { return juce::jmap (db, span, -span, bar.getY(), bar.getBottom()); };

        g.setColour (t.meterOff);
        g.fillRoundedRectangle (bar, 3.0f);

        const auto zeroY = yFor (0.0f);
        const auto valueY = yFor (juce::jlimit (-span, span, shownDb));
        auto fill = juce::Rectangle<float> (bar.getX(), juce::jmin (zeroY, valueY), bar.getWidth(), std::abs (valueY - zeroY));
        g.setColour (open ? (shownDb >= 0.0f ? t.meterLow : t.accent) : t.inkMuted);
        g.fillRect (fill);

        g.setColour (t.ink);
        g.drawRoundedRectangle (bar.reduced (0.5f), 3.0f, 1.0f);
        g.fillRect (bar.getX() - 2.0f, zeroY - 0.75f, bar.getWidth() + 4.0f, 1.5f);

        g.setColour (t.trim);
        g.fillRect (bar.getX() - 2.0f, yFor (up) - 1.0f, bar.getWidth() + 4.0f, 2.0f);
        g.fillRect (bar.getX() - 2.0f, yFor (-down) - 1.0f, bar.getWidth() + 4.0f, 2.0f);

        g.setColour (t.inkMuted);
        g.setFont (mint::ui::bodyFont (10.0f));
        for (const auto mark : { span, span * 0.5f, 0.0f, -span * 0.5f, -span })
            g.drawText (signedDb (mark).upToFirstOccurrenceOf (".", false, false),
                        scale.withY (yFor (mark) - 7.0f).withHeight (14.0f), juce::Justification::centredLeft);
    }

private:
    // 人声 / 保持是状态，不是标签：做成带指示灯的小牌，开门时填满。
    void paintStatus (juce::Graphics& g, juce::Rectangle<float> area) const
    {
        const auto& t = mint::ui::theme();
        g.setColour (open ? t.ink : t.plate);
        g.fillRoundedRectangle (area, 9.0f);
        g.setColour (open ? t.ink : t.trim);
        g.drawRoundedRectangle (area.reduced (0.5f), 8.5f, 1.0f);

        auto text = area;
        const auto lamp = text.removeFromLeft (17.0f).withTrimmedLeft (7.0f).withSizeKeepingCentre (6.0f, 6.0f);
        if (open)
        {
            g.setColour (t.faceLight);
            g.fillEllipse (lamp);
        }
        else
        {
            g.setColour (t.trim);
            g.drawEllipse (lamp.reduced (0.5f), 1.0f);
        }

        g.setColour (open ? t.plate : t.inkMuted);
        g.setFont (mint::ui::bodyFont (12.0f));
        g.drawFittedText (open ? mint::ui::tr ({ "Vocal", "보컬", "人声" }) : mint::ui::tr ({ "Hold", "홀드", "保持" }),
                          text.withTrimmedRight (5.0f).toNearestInt(), juce::Justification::centred, 1, 0.8f);
    }

    void timerCallback() override
    {
        const auto newGain = telemetry.gainDb.load (std::memory_order_relaxed);
        const auto newOpen = telemetry.gateOpen.load (std::memory_order_relaxed);
        if (std::abs (newGain - shownDb) < 0.01f && newOpen == open)
            return;
        shownDb = newGain;
        open = newOpen;
        repaint();
    }

    RiderTelemetry& telemetry;
    std::atomic<float>* rangeUp;
    std::atomic<float>* rangeDown;
    float shownDb = 0.0f;
    bool open = false;
};

// 最近 8 秒的增益曲线，下方读数给出伴奏跟随后的有效目标。
class GainTrace final : public juce::Component,
                        private juce::Timer
{
public:
    GainTrace (RiderTelemetry& source, juce::AudioProcessorValueTreeState& state)
        : telemetry (source),
          rangeUp (state.getRawParameterValue ("range_up_db")),
          rangeDown (state.getRawParameterValue ("range_down_db"))
    {
        history.fill (0.0f);
        startTimerHz (framesPerSecond);
    }

    ~GainTrace() override   { stopTimer(); }

    void paint (juce::Graphics& g) override
    {
        const auto& t = mint::ui::theme();
        auto bounds = getLocalBounds().toFloat();
        auto readout = bounds.removeFromBottom (20.0f);

        const auto up = rangeUp->load();
        const auto down = rangeDown->load();
        const auto span = meterSpanDb (up, down);
        const auto yFor = [&] (float db) { return juce::jmap (juce::jlimit (-span, span, db), span, -span, bounds.getY() + 2.0f, bounds.getBottom() - 2.0f); };

        const auto rowLine = [&] (float db) { g.fillRect (bounds.getX(), std::round (yFor (db)), bounds.getWidth(), 1.0f); };

        // 底色就是面板的象牙色，只把当前允许的范围淡淡托出来。
        g.setColour (t.trimSoft.withAlpha (0.28f));
        g.fillRect (bounds.getX(), yFor (up), bounds.getWidth(), yFor (-down) - yFor (up));

        // 横线在量程的一半处，竖线每 2 秒一条。
        g.setColour (t.trimSoft);
        rowLine (span * 0.5f);
        rowLine (-span * 0.5f);
        for (int i = 1; i < 4; ++i)
            g.fillRect (std::round (bounds.getX() + bounds.getWidth() * (float) i * 0.25f), bounds.getY(), 1.0f, bounds.getHeight());

        g.setColour (t.trim);
        rowLine (0.0f);
        g.drawRect (bounds, 1.0f);

        juce::Path line;
        for (int i = 0; i < numPoints; ++i)
        {
            const auto value = history[(size_t) ((writeIndex + i) % numPoints)];
            const auto x = bounds.getX() + bounds.getWidth() * (float) i / (float) (numPoints - 1);
            if (i == 0)
                line.startNewSubPath (x, yFor (value));
            else
                line.lineTo (x, yFor (value));
        }
        g.setColour (t.waveDark);
        g.strokePath (line, juce::PathStrokeType (1.3f));

        g.setColour (t.inkMuted);
        g.setFont (mint::ui::bodyFont (12.0f));
        const auto targetText = mint::ui::tr ({ "Effective Target ", "실제 타겟 ", "有效目标 " }) + juce::String (telemetry.effectiveTargetDb.load (std::memory_order_relaxed), 1) + " dB";
        const auto music = telemetry.musicDb.load (std::memory_order_relaxed);
        const auto musicText = ! telemetry.sidechainConnected.load (std::memory_order_relaxed) ? mint::ui::tr ({ "Music Not Connected", "반주 미연결", "伴奏未连接" })
                             : music <= -60.0f ? mint::ui::tr ({ "Music No Signal", "반주 신호 없음", "伴奏无信号" })
                                               : mint::ui::tr ({ "Music ", "반주 ", "伴奏 " }) + juce::String (music, 1) + " dB";
        g.drawText (targetText, readout, juce::Justification::centredLeft);
        g.drawText (musicText, readout, juce::Justification::centredRight);
    }

private:
    void timerCallback() override
    {
        history[(size_t) writeIndex] = telemetry.gainDb.load (std::memory_order_relaxed);
        writeIndex = (writeIndex + 1) % numPoints;
        if (isVisible())
            repaint();
    }

    static constexpr int framesPerSecond = 30, numPoints = framesPerSecond * 8;

    RiderTelemetry& telemetry;
    std::atomic<float>* rangeUp;
    std::atomic<float>* rangeDown;
    std::array<float, numPoints> history;
    int writeIndex = 0;
};

// 外观和手感同 mint::ui::Segmented（悬停、滑动选中），但只切换界面状态，不产生参数。
class ViewTabs final : public juce::Component,
                       private juce::Timer
{
public:
    ViewTabs (std::vector<mint::ui::Text> names, std::function<void (int)> onChange)
        : choices (std::move (names)), changed (std::move (onChange))
    {
        setMouseCursor (juce::MouseCursor::PointingHandCursor);
    }

    // 恢复保存的状态用：直接就位，不通知。
    void setSelected (int index)
    {
        selected = index;
        shownPosition = (float) index;
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        const auto& t = mint::ui::theme();
        const auto count = (int) choices.size();
        const auto w = (float) getWidth() / (float) juce::jmax (1, count);
        const auto h = (float) getHeight();

        if (hovered >= 0 && hovered != selected)
        {
            g.setColour (t.trimSoft.withAlpha (0.45f));
            g.fillRect (juce::Rectangle<float> ((float) hovered * w, 0.0f, w, h));
        }

        g.setColour (t.ink);
        g.fillRect (juce::Rectangle<float> (shownPosition * w, 0.0f, w, h));

        g.setColour (t.trim);
        g.drawRect (getLocalBounds().toFloat(), 1.0f);
        for (int i = 1; i < count; ++i)
            g.fillRect (juce::Rectangle<float> ((float) i * w - 0.5f, 0.0f, 1.0f, h));

        g.setFont (mint::ui::bodyFont (13.0f));
        for (int i = 0; i < count; ++i)
        {
            const auto covered = juce::jlimit (0.0f, 1.0f, 1.0f - std::abs ((float) i - shownPosition));
            g.setColour (t.ink.interpolatedWith (t.plate, covered));
            g.drawFittedText (mint::ui::tr (choices[(size_t) i]), juce::Rectangle<float> ((float) i * w, 0.0f, w, h).reduced (3.0f, 0.0f).toNearestInt(),
                              juce::Justification::centred, 1, 0.75f);
        }
    }

    void mouseMove (const juce::MouseEvent& e) override     { setHovered (cellAt (e.x)); }
    void mouseExit (const juce::MouseEvent&) override       { setHovered (-1); }

    void mouseDown (const juce::MouseEvent& e) override
    {
        const auto index = cellAt (e.x);
        if (index == selected)
            return;
        selected = index;
        startTimerHz (60);
        repaint();
        changed (selected);
    }

private:
    void timerCallback() override
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

    int cellAt (int x) const
    {
        const auto count = (int) choices.size();
        return juce::jlimit (0, juce::jmax (0, count - 1), x * count / juce::jmax (1, getWidth()));
    }

    void setHovered (int cell)
    {
        if (cell != hovered)
        {
            hovered = cell;
            repaint();
        }
    }

    std::vector<mint::ui::Text> choices;
    std::function<void (int)> changed;
    int selected = 0, hovered = -1;
    float shownPosition = 0.0f;
};

} // namespace helios
