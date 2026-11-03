// 界面：交互式曲线编辑器占满内容区，选中频段后浮出参数面板。共享层没有曲线控件，
// 所以写在本目录内，颜色全部取 theme()。
#include "IrisPlugin.h"

#include <mint_ui/MintUi.h>

#include <juce_dsp/juce_dsp.h>

#include <utility>

namespace iris
{

namespace
{
    using mint::ui::theme;
    using mint::ui::tr;
    using mint::ui::Text;

    const Text bandWord { "Band", "밴드", "频段" };

    constexpr int numPoints = 349;
    const std::array<float, 3> displayRanges { 6.0f, 12.0f, 30.0f };

    float frequencyToNorm (double f)     { return (float) (std::log (f / minFrequency) / std::log (maxFrequency / minFrequency)); }
    double normToFrequency (float n)     { return minFrequency * std::exp ((double) n * std::log (maxFrequency / minFrequency)); }

    juce::String shortFrequency (double hz)
    {
        if (hz >= 10000.0)  return juce::String (hz * 0.001, 1) + " kHz";
        if (hz >= 1000.0)   return juce::String (hz * 0.001, 2) + " kHz";
        return (hz < 100.0 ? juce::String (hz, 1) : juce::String (juce::roundToInt (hz))) + " Hz";
    }

    // 从 FIFO 取采样，做加窗 FFT。只在界面线程运行，不影响音频延迟。
    class SpectrumDisplay
    {
    public:
        void configure (double rate, const std::array<double, numPoints>& frequencies)
        {
            const auto order = rate > 150000.0 ? 14 : (rate > 75000.0 ? 13 : 12);
            if (fft == nullptr || fft->getSize() != (1 << order) || ! juce::approximatelyEqual (rate, sampleRate))
            {
                sampleRate = rate;
                fft = std::make_unique<juce::dsp::FFT> (order);
                const auto size = (size_t) fft->getSize();
                ring.assign (size, 0.0f);
                workspace.assign (size * 2, 0.0f);
                window.resize (size);
                for (size_t i = 0; i < size; ++i)
                    window[i] = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) size);
                levels.fill (-120.0f);
                writePos = 0;

                for (int p = 0; p < numPoints; ++p)
                {
                    const auto lower = p > 0 ? std::sqrt (frequencies[(size_t) p] * frequencies[(size_t) p - 1]) : frequencies[0];
                    const auto upper = p + 1 < numPoints ? std::sqrt (frequencies[(size_t) p] * frequencies[(size_t) p + 1]) : frequencies[(size_t) p];
                    bins[(size_t) p] = { (int) std::floor (lower * (double) size / rate), (int) std::ceil (upper * (double) size / rate) };
                    tilt[(size_t) p] = (float) (3.0 * std::log2 (frequencies[(size_t) p] / 1000.0));
                }
            }
        }

        // 返回显示内容是否变化。没有新采样、或者整个窗口都是静音时不做 FFT，
        // 只让谱线落下去，落到底就不用再重绘。宿主待机时常常一直送静音。
        bool update (SpectrumFifo& fifo)
        {
            if (fft == nullptr)
                return false;

            const auto size = fft->getSize();
            std::array<float, 8192> incoming;
            for (;;)
            {
                const auto got = fifo.pull (incoming.data(), (int) incoming.size());
                if (got == 0)
                    break;
                for (int i = 0; i < got; ++i)
                {
                    const auto sample = incoming[(size_t) i];
                    silentSamples = std::abs (sample) > 1.0e-6f ? 0 : silentSamples + 1;
                    ring[(size_t) writePos] = sample;
                    writePos = (writePos + 1) % size;
                }
                fresh = true;
            }

            const auto received = std::exchange (fresh, false);
            if (! received || silentSamples >= size)
            {
                // 约半秒没有音频就清掉旧数据，否则恢复播放时第一帧会混进很久以前的声音。
                if (! received && ++idleFrames == 15)
                {
                    std::fill (ring.begin(), ring.end(), 0.0f);
                    silentSamples = size;
                }

                auto moving = false;
                for (auto& shown : levels)
                {
                    if (shown > displayFloorDb)
                    {
                        shown -= 1.2f;
                        moving = true;
                    }
                }
                return moving;
            }

            idleFrames = 0;

            for (int i = 0; i < size; ++i)
                workspace[(size_t) i] = ring[(size_t) ((writePos + i) % size)] * window[(size_t) i];
            std::fill (workspace.begin() + size, workspace.end(), 0.0f);
            fft->performFrequencyOnlyForwardTransform (workspace.data(), true);

            const auto scale = 4.0f / (float) size; // 汉宁窗相干增益 0.5，单边谱再乘 2
            for (int p = 0; p < numPoints; ++p)
            {
                const auto [lo, hi] = bins[(size_t) p];
                auto peak = 0.0f;
                for (int bin = juce::jmax (1, lo); bin <= juce::jmin (size / 2, juce::jmax (lo, hi)); ++bin)
                    peak = juce::jmax (peak, workspace[(size_t) bin]);

                const auto db = mint::gainToDb (peak * scale) + tilt[(size_t) p];
                auto& shown = levels[(size_t) p];
                shown = juce::jmax (db, shown - 1.2f);
            }
            return true;
        }

        const std::array<float, numPoints>& getLevels() const noexcept     { return levels; }

    private:
        static constexpr float displayFloorDb = -100.0f; // 低于绘图区底部（加上显示斜率也看不到）

        std::unique_ptr<juce::dsp::FFT> fft;
        double sampleRate = 0.0;
        std::vector<float> ring, workspace, window;
        int writePos = 0, idleFrames = 0, silentSamples = 0;
        bool fresh = false;
        std::array<float, numPoints> levels {}, tilt {};
        std::array<std::pair<int, int>, numPoints> bins {};
    };

    // 面板里的小按钮，样式和 Segmented 的格子一致。
    class ActionCell final : public juce::Component
    {
    public:
        ActionCell (Text textToShow, std::function<void()> callback)
            : text (std::move (textToShow)), onClick (std::move (callback))
        {
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
        }

        void mouseEnter (const juce::MouseEvent&) override     { repaint(); }
        void mouseExit (const juce::MouseEvent&) override      { repaint(); }

        void paint (juce::Graphics& g) override
        {
            const auto& t = theme();
            if (isMouseOver())
            {
                g.setColour (t.trimSoft.withAlpha (0.45f));
                g.fillRect (getLocalBounds());
            }
            g.setColour (t.ink);
            g.drawRect (getLocalBounds().toFloat(), 1.0f);
            g.setFont (mint::ui::bodyFont (13.0f));
            g.drawText (tr (text), getLocalBounds(), juce::Justification::centred);
        }

        void mouseUp (const juce::MouseEvent& e) override
        {
            if (getLocalBounds().contains (e.getPosition()) && onClick != nullptr)
                onClick();
        }

    private:
        Text text;
        std::function<void()> onClick;
    };

    // 页脚的数值框：上下拖动或滚轮改值，双击恢复默认。页脚放不下旋钮。
    class ValueField final : public juce::Component,
                             public juce::TooltipClient,
                             private juce::Timer
    {
    public:
        ValueField (juce::RangedAudioParameter& p, Text labelText)
            : parameter (p), label (std::move (labelText)),
              attachment (p, [this] (float) { repaint(); }, nullptr)
        {
            setMouseCursor (juce::MouseCursor::UpDownResizeCursor);
            attachment.sendInitialUpdate();
        }

        ~ValueField() override
        {
            endGesture();
        }

        juce::String getTooltip() override
        {
            return tr ({ "Drag or scroll to adjust. Double-click to reset.", "드래그 또는 스크롤로 조절, 더블클릭으로 초기화", "拖动或滚轮调整，双击恢复默认" });
        }

        void mouseEnter (const juce::MouseEvent&) override     { repaint(); }
        void mouseExit (const juce::MouseEvent&) override      { repaint(); }

        void paint (juce::Graphics& g) override
        {
            const auto& t = theme();
            auto area = getLocalBounds();
            g.setColour (t.inkMuted);
            g.setFont (mint::ui::bodyFont (12.0f));
            g.drawText (tr (label), area.removeFromTop (17), juce::Justification::centred);

            // 外框和上下箭头说明这里可以调，悬停时加深。
            const auto box = area.withSizeKeepingCentre (juce::jmin (88, area.getWidth() - 4), 25).toFloat().reduced (0.5f);
            const auto active = isMouseOver (true);
            if (active)
            {
                g.setColour (t.trimSoft.withAlpha (0.45f));
                g.fillRoundedRectangle (box, 3.0f);
            }
            g.setColour (active ? t.ink : t.trim);
            g.drawRoundedRectangle (box, 3.0f, 1.0f);

            g.setColour (t.ink);
            g.setFont (mint::ui::bodyFont (14.0f));
            g.drawText (parameter.getCurrentValueAsText(), box.withTrimmedRight (10.0f), juce::Justification::centred);

            const auto cx = box.getRight() - 8.0f;
            const auto cy = box.getCentreY();
            juce::Path arrows;
            arrows.addTriangle (cx - 3.0f, cy - 2.0f, cx + 3.0f, cy - 2.0f, cx, cy - 6.0f);
            arrows.addTriangle (cx - 3.0f, cy + 2.0f, cx + 3.0f, cy + 2.0f, cx, cy + 6.0f);
            g.setColour (active ? t.ink : t.inkMuted);
            g.fillPath (arrows);
        }

        void mouseDown (const juce::MouseEvent&) override
        {
            endGesture();
            dragStart = parameter.getValue();
            beginGesture();
        }

        void mouseDrag (const juce::MouseEvent& e) override
        {
            const auto sensitivity = e.mods.isShiftDown() ? 0.0005f : 0.004f;
            setNormalised (dragStart - (float) e.getDistanceFromDragStartY() * sensitivity);
        }

        void mouseUp (const juce::MouseEvent&) override        { endGesture(); }

        void mouseDoubleClick (const juce::MouseEvent&) override
        {
            endGesture();
            attachment.setValueAsCompleteGesture (parameter.convertFrom0to1 (parameter.getDefaultValue()));
        }

        void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& wheel) override
        {
            // 连续滚动合并成一个手势，宿主的撤销和自动化写入只算一次。
            beginGesture();
            setNormalised (parameter.getValue() + wheel.deltaY * 0.02f);
            startTimer (wheelGestureMs);
        }

    private:
        void timerCallback() override                          { endGesture(); }

        void beginGesture()
        {
            if (! gestureOpen)
            {
                gestureOpen = true;
                attachment.beginGesture();
            }
        }

        void endGesture()
        {
            stopTimer();
            if (gestureOpen)
            {
                gestureOpen = false;
                attachment.endGesture();
            }
        }

        void setNormalised (float value)
        {
            attachment.setValueAsPartOfGesture (parameter.convertFrom0to1 (juce::jlimit (0.0f, 1.0f, value)));
        }

        static constexpr int wheelGestureMs = 300;

        juce::RangedAudioParameter& parameter;
        Text label;
        juce::ParameterAttachment attachment;
        float dragStart = 0.0f;
        bool gestureOpen = false;
    };

    // 选中频段的参数面板。切换频段时重建控件，绑定到对应频段的参数。
    class BandPanel final : public juce::Component
    {
    public:
        BandPanel (IrisProcessor& p, std::function<void (int)> stepBand, std::function<void()> removeBand)
            : processor (p),
              previous ("<", [stepBand] { stepBand (-1); }),
              next (">", [stepBand] { stepBand (1); }),
              removal (Text { "Delete", "삭제", "删除" }, std::move (removeBand))
        {
            addAndMakeVisible (previous);
            addAndMakeVisible (next);
            addAndMakeVisible (removal);
        }

        void setBand (int newBand)
        {
            if (band == newBand)
                return;
            band = newBand;
            dials = {};
            onOff.reset();
            type.reset();
            stereo.reset();
            slope.reset();
            dynamicMode.reset();
            if (band < 0)
                return;

            auto& state = processor.apvts;
            const auto id = bandPrefix (band);
            // 显示文字和参数选项分开：选项名是自动化接口，不动。
            onOff = std::make_unique<mint::ui::Segmented> (state, id + "on",
                std::vector<Text> { { "Bypass", "바이패스", "旁通" }, { "On", "켬", "开启" } });
            type = std::make_unique<mint::ui::Segmented> (state, id + "type",
                std::vector<Text> { { "Bell", "벨", "钟形" }, { "L Shelf", "L 쉘프", "低架" }, { "H Shelf", "H 쉘프", "高架" },
                                    { "L Cut", "로우컷", "低切" }, { "H Cut", "하이컷", "高切" }, { "Notch", "노치", "陷波" }, { "Tilt", "틸트", "倾斜" } });
            stereo = std::make_unique<mint::ui::Segmented> (state, id + "stereo",
                std::vector<Text> { { "Stereo", "스테레오", "立体声" }, { "Left", "좌", "左" }, { "Right", "우", "右" }, { "Mid", "미드", "中" }, { "Side", "사이드", "侧" } });
            slope = std::make_unique<mint::ui::Segmented> (state, id + "slope");
            dynamicMode = std::make_unique<mint::ui::Segmented> (state, id + "dyn",
                std::vector<Text> { { "Static", "스태틱", "静态" }, { "Dynamic", "다이내믹", "动态" } });
            for (auto* item : { onOff.get(), type.get(), stereo.get(), slope.get(), dynamicMode.get() })
                addAndMakeVisible (item);

            const std::array<std::pair<const char*, Text>, 5> definitions { {
                { "freq", { "Frequency", "주파수", "频率" } }, { "gain", { "Gain", "게인", "增益" } }, { "q", "Q" },
                { "thresh", { "Threshold", "스레숄드", "阈值" } }, { "range", { "Range", "레인지", "范围" } } } };
            for (size_t i = 0; i < dials.size(); ++i)
            {
                dials[i] = std::make_unique<mint::ui::Dial> (state, id + definitions[i].first, definitions[i].second);
                addAndMakeVisible (*dials[i]);
            }

            lastSettings = {};
            hasLayout = false;
            refresh();
            repaint();
        }

        // 类型或动态开关改变时显示不同的控件。
        void refresh()
        {
            if (band < 0)
                return;

            const auto s = processor.readBand (band);
            if (hasLayout && s.type == lastSettings.type && s.dynamic == lastSettings.dynamic && s.slopeIndex == lastSettings.slopeIndex)
                return;

            lastSettings = s;
            hasLayout = true;
            resized();
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = theme();
            const auto area = getLocalBounds().toFloat().reduced (0.5f);
            g.setColour (t.plate.withAlpha (0.96f));
            g.fillRoundedRectangle (area, 5.0f);
            g.setColour (t.trim);
            g.drawRoundedRectangle (area, 5.0f, 1.0f);

            if (band >= 0)
            {
                g.setColour (t.ink);
                g.setFont (mint::ui::bodyFont (13.0f));
                g.drawText (tr (bandWord) + " " + juce::String (band + 1).paddedLeft ('0', 2), 10, 7, 54, 22, juce::Justification::centredLeft);
            }
        }

        void resized() override
        {
            if (band < 0)
                return;

            auto area = getLocalBounds().reduced (8, 7);
            auto left = area.removeFromLeft (316);
            area.removeFromLeft (12);

            auto header = left.removeFromTop (22);
            header.removeFromLeft (56);
            previous.setBounds (header.removeFromLeft (22));
            next.setBounds (header.removeFromLeft (22).translated (-1, 0));
            header.removeFromLeft (10);
            removal.setBounds (header.removeFromRight (48));
            header.removeFromRight (10);
            onOff->setBounds (header);

            left.removeFromTop (3);
            type->setBounds (left.removeFromTop (22));
            left.removeFromTop (3);
            stereo->setBounds (left.removeFromTop (22));
            left.removeFromTop (3);
            const auto bottomRow = left.removeFromTop (22);

            const auto s = lastSettings;
            const auto isCut = typeIsCut (s.type);
            const auto hasGain = typeHasGain (s.type);
            const auto showDynamic = hasGain && s.dynamic;

            slope->setVisible (isCut);
            slope->setBounds (bottomRow);
            dynamicMode->setVisible (hasGain);
            dynamicMode->setBounds (bottomRow.withWidth (150));

            std::array<bool, 5> visible { true, hasGain, ! (isCut && s.slopeIndex == 0), showDynamic, showDynamic };
            for (size_t i = 0; i < dials.size(); ++i)
            {
                dials[i]->setVisible (visible[i]);
                if (visible[i])
                    dials[i]->setBounds (area.removeFromLeft (70));
            }
        }

    private:
        IrisProcessor& processor;
        int band = -1;
        ActionCell previous, next, removal;
        std::unique_ptr<mint::ui::Segmented> onOff, type, stereo, slope, dynamicMode;
        std::array<std::unique_ptr<mint::ui::Dial>, 5> dials;
        BandSettings lastSettings;
        bool hasLayout = false;
    };

    class CurveEditor final : public juce::Component,
                              private juce::Timer
    {
    public:
        explicit CurveEditor (IrisProcessor& p)
            : processor (p),
              panel (p, [this] (int direction) { stepSelection (direction); }, [this] { removeBand (selected); })
        {
            for (int i = 0; i < numPoints; ++i)
                frequencies[(size_t) i] = normToFrequency ((float) i / (float) (numPoints - 1));

            auto& tree = processor.apvts.state;
            rangeIndex = juce::jlimit (0, 2, (int) tree.getProperty ("displayRange", 1));
            spectrumOn = (bool) tree.getProperty ("spectrumOn", true);

            addChildComponent (panel);
            setWantsKeyboardFocus (true);
            processor.spectrumWanted.store (spectrumOn, std::memory_order_relaxed);
            refreshCurves (true);
            startTimerHz (30);
        }

        ~CurveEditor() override
        {
            stopTimer();
            endDragGesture();
            endWheelGesture();
            processor.spectrumWanted.store (false, std::memory_order_relaxed);
        }

        void resized() override
        {
            // 频率刻度也按像素重算一次就够了。
            placePanel();
        }

        void paint (juce::Graphics& g) override
        {
            const auto& t = theme();
            const auto bounds = getLocalBounds().toFloat();

            g.setColour (t.plate);
            g.fillRect (bounds);

            drawGrid (g);
            if (spectrumOn)
                drawSpectrum (g);
            drawBandCurves (g);
            drawSum (g);
            drawNodes (g);
            drawReadout (g);

            g.setColour (t.trim);
            g.drawRect (bounds, 1.0f);
        }

        void mouseDown (const juce::MouseEvent& e) override
        {
            grabKeyboardFocus();
            endWheelGesture();
            const auto hit = nodeAt (e.position);

            if (e.mods.isPopupMenu())
            {
                if (hit >= 0)
                    removeBand (hit);
                return;
            }

            const auto topCells = headerCells();
            if (topCells.range.contains (e.getPosition()))
            {
                rangeIndex = (rangeIndex + 1) % (int) displayRanges.size();
                processor.apvts.state.setProperty ("displayRange", rangeIndex, nullptr);
                repaint();
                return;
            }
            if (topCells.spectrum.contains (e.getPosition()))
            {
                spectrumOn = ! spectrumOn;
                processor.apvts.state.setProperty ("spectrumOn", spectrumOn, nullptr);
                processor.spectrumWanted.store (spectrumOn, std::memory_order_relaxed);
                repaint();
                return;
            }

            setSelected (hit);
            if (hit < 0)
                return;

            // 手势按按下时的状态记录，松开和析构时照记录结束，中途改类型或删频段都能配对。
            dragging = true;
            const auto s = processor.readBand (hit);
            dragStartNorm = frequencyToNorm (s.frequency);
            dragStartGain = (float) s.gainDb;
            dragFrequency = &parameter (hit, "freq");
            dragFrequency->beginChangeGesture();
            if (typeHasGain (s.type))
            {
                dragGain = &parameter (hit, "gain");
                dragGain->beginChangeGesture();
            }
            panel.setVisible (false);
            setMouseCursor (juce::MouseCursor::DraggingHandCursor);
        }

        void mouseDrag (const juce::MouseEvent& e) override
        {
            if (! dragging || dragFrequency == nullptr)
                return;

            const auto fine = e.mods.isShiftDown() ? 0.15f : 1.0f;
            const auto norm = juce::jlimit (0.0f, 1.0f, dragStartNorm + fine * (float) e.getDistanceFromDragStartX() / (float) getWidth());
            dragFrequency->setValueNotifyingHost (dragFrequency->convertTo0to1 ((float) normToFrequency (norm)));

            if (dragGain != nullptr)
            {
                const auto gain = dragStartGain - fine * (float) e.getDistanceFromDragStartY() / pixelsPerDb();
                dragGain->setValueNotifyingHost (dragGain->convertTo0to1 (juce::jlimit (-30.0f, 30.0f, gain)));
            }

            // 拖动时不等定时器，曲线和节点立刻跟手。
            refreshCurves (false);
            repaint();
        }

        void mouseUp (const juce::MouseEvent& e) override
        {
            if (! dragging)
                return;

            endDragGesture();
            placePanel();
            updateHover (e.position);
        }

        void mouseDoubleClick (const juce::MouseEvent& e) override
        {
            const auto cells = headerCells();
            if (! e.mods.isPopupMenu() && e.position.y < plotHeight() && nodeAt (e.position) < 0
                && ! cells.range.contains (e.getPosition()) && ! cells.spectrum.contains (e.getPosition()))
                addBand (e.position);
        }

        void mouseMove (const juce::MouseEvent& e) override    { updateHover (e.position); }

        void mouseExit (const juce::MouseEvent&) override
        {
            if (! dragging && hovered >= 0)
            {
                hovered = -1;
                repaint();
            }
        }

        void mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override
        {
            auto target = nodeAt (e.position);
            if (target < 0)
                target = selected;
            if (target < 0)
                return;

            const auto s = processor.readBand (target);
            if (e.mods.isCommandDown() && typeIsCut (s.type))
            {
                const auto step = wheel.deltaY > 0.0f ? 1 : -1;
                const auto index = juce::jlimit (0, (int) slopeChoices.size() - 1, s.slopeIndex + step);
                adjustByWheel (parameter (target, "slope"), (float) index / (float) ((int) slopeChoices.size() - 1));
            }
            else
            {
                const auto rate = e.mods.isShiftDown() ? 0.4f : 2.5f;
                auto& q = parameter (target, "q");
                adjustByWheel (q, q.convertTo0to1 ((float) juce::jlimit (0.025, 40.0, s.q * std::exp ((double) (wheel.deltaY * rate)))));
            }

            refreshCurves (false);
            repaint();
        }

        bool keyPressed (const juce::KeyPress& key) override
        {
            if ((key == juce::KeyPress::deleteKey || key == juce::KeyPress::backspaceKey) && selected >= 0)
            {
                removeBand (selected);
                return true;
            }
            return false;
        }

    private:
        struct HeaderCells { juce::Rectangle<int> range, spectrum; };

        HeaderCells headerCells() const
        {
            auto strip = getLocalBounds().reduced (6).removeFromTop (18);
            HeaderCells cells;
            cells.range = strip.removeFromRight (56);
            strip.removeFromRight (4);
            cells.spectrum = strip.removeFromRight (64);
            return cells;
        }

        float plotHeight() const noexcept                     { return (float) getHeight() - 16.0f; }
        float xForFrequency (double f) const noexcept         { return frequencyToNorm (f) * (float) getWidth(); }

        // 增益和像素的换算只有这一处，节点、拖动和双击加点都用它。
        float pixelsPerDb() const noexcept                    { return (plotHeight() * 0.5f - 4.0f) / displayRanges[(size_t) rangeIndex]; }
        float yForDb (float db) const noexcept                { return plotHeight() * 0.5f - db * pixelsPerDb(); }
        float dbForY (float y) const noexcept                 { return (plotHeight() * 0.5f - y) / pixelsPerDb(); }

        // 一次性的改动也要走完整手势，宿主才能正确写自动化和撤销。
        static void setWithGesture (juce::RangedAudioParameter& p, float normalised)
        {
            p.beginChangeGesture();
            p.setValueNotifyingHost (normalised);
            p.endChangeGesture();
        }

        // 连续滚轮合并成一个手势，停下约 300 ms 后结束。
        void adjustByWheel (juce::RangedAudioParameter& p, float normalised)
        {
            if (wheelParameter != &p)
            {
                endWheelGesture();
                wheelParameter = &p;
                p.beginChangeGesture();
            }
            wheelIdleTicks = 9;
            p.setValueNotifyingHost (normalised);
        }

        void endWheelGesture()
        {
            if (wheelParameter != nullptr)
            {
                wheelParameter->endChangeGesture();
                wheelParameter = nullptr;
            }
        }

        void endDragGesture()
        {
            if (dragFrequency != nullptr)
                dragFrequency->endChangeGesture();
            if (dragGain != nullptr)
                dragGain->endChangeGesture();
            dragFrequency = dragGain = nullptr;
            dragging = false;
        }

        void updateHover (juce::Point<float> position)
        {
            const auto node = nodeAt (position);
            const auto cells = headerCells();
            const auto point = position.roundToInt();
            setMouseCursor (node >= 0 || cells.range.contains (point) || cells.spectrum.contains (point)
                                ? juce::MouseCursor::PointingHandCursor : juce::MouseCursor::NormalCursor);
            if (node != hovered)
            {
                hovered = node;
                repaint();
            }
        }

        juce::RangedAudioParameter& parameter (int band, const char* suffix)
        {
            return *processor.apvts.getParameter (bandPrefix (band) + suffix);
        }

        void setPlain (int band, const char* suffix, float value)
        {
            auto& p = parameter (band, suffix);
            setWithGesture (p, p.convertTo0to1 (value));
        }

        void setChoice (int band, const char* suffix, int index, int count)
        {
            setWithGesture (parameter (band, suffix), (float) index / (float) (count - 1));
        }

        juce::Point<float> nodePosition (const BandSettings& s) const
        {
            auto y = yForDb (0.0f);
            if (typeHasGain (s.type))
                y = yForDb ((float) s.gainDb);
            else if (typeIsCut (s.type))
                y = yForDb (-3.0f);
            return { juce::jlimit (9.0f, (float) getWidth() - 9.0f, xForFrequency (s.frequency)), juce::jlimit (9.0f, plotHeight() - 9.0f, y) };
        }

        int nodeAt (juce::Point<float> position) const
        {
            auto best = -1;
            auto bestDistance = 13.0f;
            for (int b = 0; b < numBands; ++b)
            {
                if (! processor.isBandUsed (b))
                    continue;
                const auto distance = nodePosition (processor.readBand (b)).getDistanceFrom (position);
                if (distance < bestDistance || (b == selected && distance <= bestDistance))
                {
                    best = b;
                    bestDistance = distance;
                }
            }
            return best;
        }

        void setSelected (int band)
        {
            if (selected == band)
                return;
            selected = band;
            panel.setBand (band);
            placePanel();
            repaint();
        }

        void stepSelection (int direction)
        {
            if (selected < 0)
                return;
            for (int i = 1; i <= numBands; ++i)
            {
                const auto candidate = (selected + direction * i + numBands * 2) % numBands;
                if (processor.isBandUsed (candidate))
                {
                    setSelected (candidate);
                    return;
                }
            }
        }

        void placePanel()
        {
            if (selected < 0 || dragging || getWidth() == 0)
            {
                panel.setVisible (false);
                return;
            }

            const auto s = processor.readBand (selected);
            const auto y = nodePosition (s).y;
            constexpr int panelHeight = 106;
            const auto top = y > plotHeight() * 0.5f;
            // 放上面时留出右上角的频谱和量程按钮。
            const auto topY = headerCells().range.getBottom() + 4;
            panel.setBounds (8, top ? topY : getHeight() - panelHeight - 4, getWidth() - 16, panelHeight);
            panel.setVisible (true);
            panel.toFront (false);
        }

        void addBand (juce::Point<float> position)
        {
            auto slot = -1;
            for (int b = 0; b < numBands && slot < 0; ++b)
                if (! processor.isBandUsed (b))
                    slot = b;
            if (slot < 0)
                return;

            endWheelGesture();
            const auto frequency = normToFrequency (juce::jlimit (0.0f, 1.0f, position.x / (float) getWidth()));
            const auto gain = juce::jlimit (-30.0f, 30.0f, dbForY (position.y));

            // 最左和最右默认是切除，其他位置默认是钟形。
            auto type = BandType::bell;
            if (frequency < 30.0)
                type = BandType::lowCut;
            else if (frequency > 16000.0)
                type = BandType::highCut;

            for (const auto* suffix : { "on", "slope", "stereo", "dyn", "thresh", "range" })
                setWithGesture (parameter (slot, suffix), parameter (slot, suffix).getDefaultValue());

            setChoice (slot, "type", (int) type, 7);
            setPlain (slot, "freq", (float) frequency);
            setPlain (slot, "gain", typeHasGain (type) ? gain : 0.0f);
            setPlain (slot, "q", 1.0f);
            setWithGesture (parameter (slot, "used"), 1.0f);
            refreshCurves (false);
            setSelected (slot);
            repaint();
        }

        void removeBand (int band)
        {
            if (band < 0)
                return;
            endWheelGesture();
            endDragGesture();
            setWithGesture (parameter (band, "used"), 0.0f);
            if (band == selected)
                setSelected (-1);
            if (band == hovered)
                hovered = -1;
            refreshCurves (false);
            repaint();
        }

        void timerCallback() override
        {
            // 只有内容变了才重绘，静止且频谱关闭时这里不产生任何绘制。
            auto changed = false;
            if (spectrumOn)
            {
                const auto rate = processor.getRate();
                inputDisplay.configure (rate, frequencies);
                outputDisplay.configure (rate, frequencies);
                changed = inputDisplay.update (processor.inputSpectrum);
                changed = outputDisplay.update (processor.outputSpectrum) || changed;
            }
            changed = refreshCurves (false) || changed;
            panel.refresh();

            if (wheelParameter != nullptr && --wheelIdleTicks <= 0)
                endWheelGesture();

            if (! dragging && selected >= 0 && ! processor.isBandUsed (selected))
                setSelected (-1);

            if (changed)
                repaint();
        }

        // 返回曲线或节点的显示内容是否变化。
        bool refreshCurves (bool force)
        {
            const auto rate = processor.getRate();
            const auto rateChanged = ! juce::approximatelyEqual (rate, curveRate);
            curveRate = rate;
            auto anyChange = false;

            for (int b = 0; b < numBands; ++b)
            {
                auto s = processor.readBand (b);
                const auto used = processor.isBandUsed (b);
                const auto offset = s.dynamic && typeHasGain (s.type) && s.active ? (double) processor.getDynamicOffset (b) : 0.0;
                const auto gain = s.gainDb + offset;
                auto& cached = curves[(size_t) b];

                // 动态范围线只影响绘制，不影响曲线，但变了也要重绘。
                if (cached.dynamic != s.dynamic || ! juce::approximatelyEqual (cached.range, s.rangeDb))
                {
                    cached.dynamic = s.dynamic;
                    cached.range = s.rangeDb;
                    anyChange = true;
                }

                if (! force && ! rateChanged && cached.used == used && cached.active == s.active && cached.type == s.type && cached.slope == s.slopeIndex
                    && juce::approximatelyEqual (cached.frequency, s.frequency) && juce::approximatelyEqual (cached.q, s.q)
                    && std::abs (cached.gain - gain) < 0.005)
                    continue;

                cached.used = used;
                cached.active = s.active;
                cached.type = s.type;
                cached.slope = s.slopeIndex;
                cached.frequency = s.frequency;
                cached.q = s.q;
                cached.gain = gain;
                anyChange = true;

                if (! used)
                    continue;

                std::array<SectionDesign, maxSections> sections;
                const auto count = designBand (s, gain, rate, sections);
                for (int p = 0; p < numPoints; ++p)
                    cached.db[(size_t) p] = (float) bandMagnitudeDb (sections, count, frequencies[(size_t) p], rate);
            }

            if (anyChange || force)
            {
                sum.fill (0.0f);
                for (const auto& curve : curves)
                    if (curve.used && curve.active)
                        for (int p = 0; p < numPoints; ++p)
                            sum[(size_t) p] += curve.db[(size_t) p];
            }
            return anyChange || force || rateChanged;
        }

        // 曲线画满全宽。奈奎斯特以上的点取奈奎斯特处的值，频谱在那里自然落到底。
        juce::Path pathFor (const std::array<float, numPoints>& values, float yMin, float yMax) const
        {
            juce::Path path;
            for (int p = 0; p < numPoints; ++p)
            {
                const auto x = (float) p / (float) (numPoints - 1) * (float) getWidth();
                const auto y = juce::jlimit (yMin, yMax, yForDb (values[(size_t) p]));
                if (p == 0)
                    path.startNewSubPath (x, y);
                else
                    path.lineTo (x, y);
            }
            return path;
        }

        void drawGrid (juce::Graphics& g) const
        {
            const auto& t = theme();
            const auto height = plotHeight();
            g.setFont (mint::ui::bodyFont (10.0f));

            for (const auto f : { 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0 })
            {
                const auto x = xForFrequency (f);
                g.setColour (t.trimSoft);
                g.drawVerticalLine ((int) x, 0.0f, height);
                g.setColour (t.inkMuted);
                const auto text = f >= 1000.0 ? juce::String ((int) (f / 1000.0)) + "k" : juce::String ((int) f);
                g.drawText (text, (int) x - 20, (int) height, 40, 16, juce::Justification::centred);
            }

            const auto range = displayRanges[(size_t) rangeIndex];
            const auto step = range > 20.0f ? 10.0f : (range > 10.0f ? 6.0f : 3.0f);
            for (auto db = -range; db <= range + 0.01f; db += step)
            {
                const auto y = yForDb (db);
                g.setColour (std::abs (db) < 0.01f ? t.trim : t.trimSoft);
                g.drawHorizontalLine ((int) y, 0.0f, (float) getWidth());

                // 数字放在线的上方，不被线穿过。最上面一条放到线的下方。
                const auto labelY = (int) y - 14 >= 1 ? (int) y - 14 : (int) y + 2;
                g.setColour (t.inkMuted);
                g.drawText ((db > 0.0f ? "+" : "") + juce::String ((int) db), 4, labelY, 26, 12, juce::Justification::centredLeft);
            }

            const auto cells = headerCells();
            g.setFont (mint::ui::bodyFont (11.0f));
            g.setColour (t.inkMuted);
            g.drawRect (cells.range, 1);
            g.drawText (juce::String::fromUTF8 ("\xc2\xb1") + juce::String ((int) range) + " dB", cells.range, juce::Justification::centred);
            g.setColour (spectrumOn ? t.ink : t.inkMuted);
            g.drawRect (cells.spectrum, 1);
            g.drawText (tr (Text { "Spectrum", "스펙트럼", "频谱" }), cells.spectrum, juce::Justification::centred);

            if (! anyBandUsed())
            {
                g.setColour (t.inkMuted);
                g.setFont (mint::ui::bodyFont (13.0f));
                // 放在 0 dB 线上方的空白处，不压线。
                const auto hint = juce::Rectangle<int> (0, (int) yForDb (0.0f) - 36, getWidth(), 20);
                g.drawText (tr (Text { "Double-click to add a band", "더블클릭으로 밴드 추가", "双击添加频段" }), hint, juce::Justification::centred);
            }
        }

        bool anyBandUsed() const
        {
            for (int b = 0; b < numBands; ++b)
                if (processor.isBandUsed (b))
                    return true;
            return false;
        }

        // 频谱用自己的纵轴：-84 到 +6 dBFS 铺满绘图区，带 3 dB/oct 的显示斜率。
        juce::Path spectrumPath (const std::array<float, numPoints>& levels) const
        {
            juce::Path path;
            const auto height = plotHeight();
            for (int p = 0; p < numPoints; ++p)
            {
                const auto x = (float) p / (float) (numPoints - 1) * (float) getWidth();
                const auto y = juce::jlimit (0.0f, height, juce::jmap (levels[(size_t) p], 6.0f, -84.0f, 0.0f, height));
                if (p == 0)
                    path.startNewSubPath (x, y);
                else
                    path.lineTo (x, y);
            }
            return path;
        }

        void drawSpectrum (juce::Graphics& g) const
        {
            const auto& t = theme();
            const auto height = plotHeight();

            auto output = spectrumPath (outputDisplay.getLevels());
            auto fill = output;
            fill.lineTo ((float) getWidth(), height);
            fill.lineTo (0.0f, height);
            fill.closeSubPath();
            g.setColour (t.ground.withAlpha (0.28f));
            g.fillPath (fill);
            g.setColour (t.ground.withAlpha (0.65f));
            g.strokePath (output, juce::PathStrokeType (1.0f));

            g.setColour (t.inkMuted.withAlpha (0.35f));
            g.strokePath (spectrumPath (inputDisplay.getLevels()), juce::PathStrokeType (0.8f));
        }

        void drawBandCurves (juce::Graphics& g) const
        {
            const auto& t = theme();
            const auto height = plotHeight();

            for (int b = 0; b < numBands; ++b)
            {
                const auto& curve = curves[(size_t) b];
                if (! curve.used)
                    continue;

                auto path = pathFor (curve.db, -2.0f, height);
                if (b == selected)
                {
                    auto fill = path;
                    fill.lineTo ((float) getWidth(), yForDb (0.0f));
                    fill.lineTo (0.0f, yForDb (0.0f));
                    fill.closeSubPath();
                    g.setColour (t.accent.withAlpha (0.14f));
                    g.fillPath (fill);
                    g.setColour (t.accent);
                    g.strokePath (path, juce::PathStrokeType (1.4f));
                }
                else if (b == hovered)
                {
                    g.setColour (t.accent.withAlpha (0.7f));
                    g.strokePath (path, juce::PathStrokeType (1.2f));
                }
                else
                {
                    g.setColour (t.trim.withAlpha (curve.active ? 0.8f : 0.3f));
                    g.strokePath (path, juce::PathStrokeType (1.0f));
                }
            }
        }

        void drawSum (juce::Graphics& g) const
        {
            g.setColour (theme().ink);
            g.strokePath (pathFor (sum, -2.0f, plotHeight()), juce::PathStrokeType (2.2f));
        }

        void drawNodes (juce::Graphics& g) const
        {
            const auto& t = theme();
            g.setFont (mint::ui::bodyFont (10.0f, true));

            for (int b = 0; b < numBands; ++b)
            {
                if (! processor.isBandUsed (b))
                    continue;

                const auto s = processor.readBand (b);
                const auto position = nodePosition (s);

                if (s.dynamic && typeHasGain (s.type))
                {
                    // 动态范围：细线是可移动的区间，菱形是当前位置。
                    const auto reach = yForDb ((float) (s.gainDb + s.rangeDb));
                    g.setColour (t.accentEnd);
                    g.drawLine (position.x, position.y, position.x, juce::jlimit (0.0f, plotHeight(), reach), 3.0f);

                    const auto now = juce::jlimit (0.0f, plotHeight(), yForDb ((float) (s.gainDb + processor.getDynamicOffset (b))));
                    juce::Path diamond;
                    diamond.startNewSubPath (position.x, now - 5.0f);
                    diamond.lineTo (position.x + 5.0f, now);
                    diamond.lineTo (position.x, now + 5.0f);
                    diamond.lineTo (position.x - 5.0f, now);
                    diamond.closeSubPath();
                    g.fillPath (diamond);
                }

                const auto isSelected = b == selected;
                const auto isHovered = b == hovered;
                const auto dot = juce::Rectangle<float> (isHovered ? 19.0f : 16.0f, isHovered ? 19.0f : 16.0f).withCentre (position);
                if (isHovered)
                {
                    g.setColour (t.accent.withAlpha (0.25f));
                    g.fillEllipse (dot.expanded (4.0f));
                }
                g.setColour (isSelected ? t.accent : (s.active ? t.plate : t.tickOff));
                g.fillEllipse (dot);
                g.setColour (t.ink);
                g.drawEllipse (dot, 1.2f);
                g.setColour (isSelected ? t.plate : t.ink);
                g.drawText (juce::String (b + 1), dot, juce::Justification::centred);
            }
        }

        void drawReadout (juce::Graphics& g) const
        {
            if (selected < 0 || panel.isVisible())
                return;

            const auto& t = theme();
            const auto s = processor.readBand (selected);
            auto text = tr (bandWord) + " " + juce::String (selected + 1) + "  " + shortFrequency (s.frequency);
            if (typeHasGain (s.type))
                text << "  " << (s.gainDb > 0.0 ? "+" : "") << juce::String (s.gainDb, 1) << " dB";
            text << "  Q " << juce::String (s.q, 2);
            g.setColour (t.ink);
            g.setFont (mint::ui::bodyFont (12.0f));
            g.drawText (text, 34, 6, 300, 18, juce::Justification::centredLeft);
        }

        struct CachedCurve
        {
            bool used = false, active = false;
            BandType type = BandType::bell;
            int slope = 0;
            double frequency = 0.0, q = 0.0, gain = 0.0, range = 0.0;
            bool dynamic = false;
            std::array<float, numPoints> db {};
        };

        IrisProcessor& processor;
        BandPanel panel;
        std::array<double, numPoints> frequencies {};
        std::array<CachedCurve, numBands> curves;
        std::array<float, numPoints> sum {};
        SpectrumDisplay inputDisplay, outputDisplay;
        double curveRate = 48000.0;
        int selected = -1, hovered = -1, rangeIndex = 1, wheelIdleTicks = 0;
        bool spectrumOn = true, dragging = false;
        float dragStartNorm = 0.0f, dragStartGain = 0.0f;
        juce::RangedAudioParameter* dragFrequency = nullptr;
        juce::RangedAudioParameter* dragGain = nullptr;
        juce::RangedAudioParameter* wheelParameter = nullptr;
    };

    class IrisEditor final : public mint::ui::EditorBase
    {
    public:
        explicit IrisEditor (IrisProcessor& p)
            : EditorBase (p, { "IRIS", { "EQ", "EQ", "均衡" }, "IV" }),
              curve (p),
              output (*p.apvts.getParameter ("output"), Text { "Output", "출력", "输出" }),
              lookaheadField (*p.apvts.getParameter ("lookahead_ms"), Text { "Look-ahead", "룩어헤드", "前瞻" }),
              attack (*p.apvts.getParameter ("attack_ms"), Text { "Attack", "어택", "起音" }),
              release (*p.apvts.getParameter ("release_ms"), Text { "Release", "릴리즈", "释放" })
        {
            addAndMakeVisible (curve);
            for (auto* field : { &output, &lookaheadField, &attack, &release })
                addAndMakeVisible (field);
            setSize (width, height);
        }

        void resized() override
        {
            curve.setBounds (getContentBounds());

            auto footer = getFooterSlot();
            const auto cell = footer.getWidth() / 4;
            output.setBounds (footer.removeFromLeft (cell));
            lookaheadField.setBounds (footer.removeFromLeft (cell));
            attack.setBounds (footer.removeFromLeft (cell));
            release.setBounds (footer);
        }

    private:
        CurveEditor curve;
        ValueField output, lookaheadField, attack, release;
    };
}

juce::AudioProcessorEditor* IrisProcessor::createEditor()
{
    return new IrisEditor (*this);
}

} // namespace iris
