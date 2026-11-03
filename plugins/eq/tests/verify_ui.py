"""界面验证：往真实的 VST3 编辑器发 Windows 鼠标消息，解析参数变化，并生成截图。

截图放在 WORK 目录，需要人工看：文字不重叠、不被裁切。
坐标是编辑器窗口坐标。曲线区左上角 (62, 116)，大小 696x290；绘图高度 274，±12 dB 时 +12 dB 在 y=120。
"""
import os
import re
import sys

import numpy as np
from scipy import signal

import iris_common as ic

failures = 0


def check(name, ok, detail):
    global failures
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        failures += 1


def x_for(freq):
    return round(62 + ic.norm_log(freq, 10.0, 30000.0) * 696)


def y_for(db):
    return round(116 + 137 - db / 12.0 * 133)


def shot(name, sets=(), mouse=(), audio=None):
    """返回每次 dump 时的频段列表。"""
    out = os.path.join(ic.WORK, f"{name}.png")
    cmd = [ic.BENCH, "shot", ic.PLUGIN, out] + ic.set_args(sets)
    if audio:
        cmd += ["--audio", audio]
    for action in mouse:
        cmd += ["--mouse", action]
    text = ic.run(cmd)
    dumps, current = [], {}
    global last_output
    last_output = text
    for line in text.splitlines():
        match = re.match(r"band (\d+): (.*)", line)
        if match:
            fields = dict(re.findall(r"(\w+)=(.*?)(?= \w+=|$)", match.group(2)))
            current[int(match.group(1))] = fields
        elif line.startswith("dump end"):
            dumps.append(current)
            current = {}
    print(f"截图 {out}")
    return dumps


def gestures_after_close():
    match = re.search(r"gestures after close begin=(\d+) end=(\d+) unpaired=(\d+)", last_output)
    return tuple(int(v) for v in match.groups())


def gesture_counts_at(index):
    """第 index 次 dump 时每个参数的手势次数。"""
    lines = [l for l in last_output.splitlines() if l.startswith("gestures begin=")]
    return dict((k, int(v)) for k, v in re.findall(r"(\S+):(\d+)", lines[index]))


def hz(text):
    value = float(text.split()[0])
    return value * 1000.0 if "kHz" in text else value


def db(text):
    return float(text.split()[0])


def voice_like_noise():
    rng = np.random.default_rng(1)
    b, a = signal.butter(1, 600 / 24000)
    x = signal.lfilter(b, a, rng.standard_normal(48000 * 6))
    return ic.write_wav("ui_noise.wav", 0.3 * x / np.max(np.abs(x)), 48000)


def main():
    noise = voice_like_noise()

    node = (x_for(1000.0), y_for(6.0))
    dumps = shot("ui_interaction", audio=noise, mouse=[
        f"dbl:{node[0]}:{node[1]}", "dump",
        f"drag:{node[0]}:{node[1]}:{node[0] + 50}:{node[1] + 30}", "dump",
        f"wheel:{node[0] + 50}:{node[1] + 30}:120", "dump",
        f"dbl:{x_for(200.0)}:{y_for(3.0)}", "dump",
        f"rclick:{node[0] + 50}:{node[1] + 30}", "dump",
    ])
    first = dumps[0].get(1, {})
    check("双击加点", 1 in dumps[0] and abs(hz(first["Freq"]) - 1000) < 30 and abs(db(first["Gain"]) - 6) < 0.3 and first["Type"] == "钟形", str(first.get("Freq")) + " " + str(first.get("Gain")))
    second = dumps[1].get(1, {})
    expected_freq = 1000 * 3000 ** (50 / 696)
    check("拖动改频率和增益", abs(hz(second["Freq"]) - expected_freq) < 40 and abs(db(second["Gain"]) - (6 - 30 / 133 * 12)) < 0.3, f"{second['Freq']} {second['Gain']}，期望约 {expected_freq:.0f} Hz")
    third = dumps[2].get(1, {})
    check("滚轮改 Q", float(third["Q"]) > float(second["Q"]) * 1.2, f"{second['Q']} -> {third['Q']}")
    check("再加一个频段", 2 in dumps[3], str(dumps[3].get(2, {}).get("Freq")))
    check("右键删除", 1 not in dumps[4] and 2 in dumps[4], f"剩余频段 {sorted(dumps[4])}")

    dumps = shot("ui_edges", mouse=[f"dbl:{x_for(15.0)}:{y_for(0.0)}", "dump", f"dbl:{x_for(20000.0)}:{y_for(0.0)}", "dump"])
    check("最左双击是低切，最右双击是高切", dumps[1][1]["Type"] == "低切" and dumps[1][2]["Type"] == "高切", f"{dumps[1][1]['Type']} {dumps[1][2]['Type']}")

    # 面板上的删除按钮。频段在图的上半部分时面板在下面。删除按钮的位置见 BandPanel::resized。
    dumps = shot("ui_panel_delete", mouse=[f"dbl:{node[0]}:{node[1]}", "dump", "click:366:310", "dump"])
    check("面板删除按钮", 1 in dumps[0] and 1 not in dumps[1], f"删除后剩余 {sorted(dumps[1])}")

    # 带频谱和多种频段的截图，选中动态频段显示全部旋钮。
    sets = ic.band_sets(1, kind="lowCut", freq=80.0, slope=24, q=1.0)
    sets += ic.band_sets(2, kind="bell", freq=320.0, gain=-4.0, q=2.0, dynamic=True, thresh=-40.0, rng=-6.0)
    sets += ic.band_sets(3, kind="bell", freq=3000.0, gain=5.0, q=1.2, stereo="mid")
    sets += ic.band_sets(4, kind="highShelf", freq=9000.0, gain=3.0, q=1.0)
    sets += ic.band_sets(5, kind="notch", freq=6000.0, q=8.0)
    dumps = shot("ui_dynamic_selected", sets=sets, audio=noise, mouse=[f"click:{x_for(320.0)}:{y_for(-4.0)}", "dump"])
    check("多频段状态读回", sorted(dumps[0]) == [1, 2, 3, 4, 5], str(sorted(dumps[0])))

    shot("ui_cut_selected", sets=sets, audio=noise, mouse=[f"click:{x_for(80.0)}:{y_for(-3.0)}"])
    shot("ui_notch_selected", sets=sets, audio=noise, mouse=[f"click:{x_for(6000.0)}:{y_for(0.0)}"])

    # 显示范围切换。
    shot("ui_range30", sets=sets, audio=noise, mouse=["click:726:129"])

    # 负增益节点：面板放上面，不能盖住右上角的频谱和量程按钮。点量程按钮要生效。
    dumps = shot("ui_panel_top", sets=ic.band_sets(1, kind="bell", freq=1000.0, gain=-6.0), mouse=[f"click:{x_for(1000.0)}:{y_for(-6.0)}", "click:726:129"])

    # 节点跟手：向下拖 100 像素，增益变化必须等于绘图比例（±12 dB 对应上下各 133 像素）。
    dumps = shot("ui_drag_scale", sets=ic.band_sets(1, kind="bell", freq=1000.0, gain=6.0), mouse=[f"drag:{node[0]}:{node[1]}:{node[0]}:{node[1] + 100}", "dump"])
    moved = db(dumps[0][1]["Gain"]) - 6.0
    check("拖动增益和绘图比例一致", abs(moved - (-100 / 133 * 12)) < 0.1, f"向下 100 像素变化 {moved:+.2f} dB，期望 {-100 / 133 * 12:+.2f} dB")

    # 底部频率刻度区双击不加点。
    dumps = shot("ui_scale_area", mouse=[f"dbl:{x_for(1000.0)}:{116 + 282}", "dump", f"dbl:{x_for(1000.0)}:{y_for(3.0)}", "dump"])
    check("频率刻度区双击不加点", len(dumps[0]) == 0 and len(dumps[1]) == 1, f"刻度区后 {len(dumps[0])} 个频段，绘图区后 {len(dumps[1])} 个")

    # 手势：加点、拖动、连续滚轮、删除都要成对，连续滚轮合并成一个手势。
    dumps = shot("ui_gestures", mouse=[
        f"dbl:{node[0]}:{node[1]}", "dump",
        f"wheel:{node[0]}:{node[1]}:60", f"wheel:{node[0]}:{node[1]}:60", f"wheel:{node[0]}:{node[1]}:60", "wait:600", "dump",
        f"drag:{node[0]}:{node[1]}:{node[0] + 20}:{node[1] + 20}", "dump",
        f"rclick:{node[0] + 20}:{node[1] + 20}", "dump",
    ])
    after_add = gesture_counts_at(0)
    after_wheel = gesture_counts_at(1)
    after_drag = gesture_counts_at(2)
    begins, ends, unpaired = gestures_after_close()
    check("加点走手势", after_add.get("B01_Used", 0) == 1 and after_add.get("B01_Freq", 0) == 1, str(after_add))
    wheel_gestures = after_wheel.get("B01_Q", 0) - after_add.get("B01_Q", 0)
    check("连续滚轮合并成一个手势", wheel_gestures == 1, f"3 次滚轮产生 {wheel_gestures} 个手势")
    check("拖动的频率和增益手势", after_drag.get("B01_Freq", 0) == after_wheel.get("B01_Freq", 0) + 1 and after_drag.get("B01_Gain", 0) == after_wheel.get("B01_Gain", 0) + 1, "各加 1")
    check("手势全部成对", begins == ends and unpaired == 0 and begins > 0, f"开始 {begins}，结束 {ends}，未配对 {unpaired}")

    # 拖动途中关窗：析构时也要把手势结束。这里用没发抬起的方式模拟不了，所以只验证滚轮未结束时关窗。
    shot("ui_close_during_wheel", mouse=[f"dbl:{node[0]}:{node[1]}", f"wheel:{node[0]}:{node[1]}:60"])
    begins, ends, unpaired = gestures_after_close()
    check("滚轮手势未结束时关窗也成对", begins == ends and unpaired == 0, f"开始 {begins}，结束 {ends}，未配对 {unpaired}")

    # 悬停反馈的截图。
    shot("ui_hover", sets=sets, audio=noise, mouse=[f"move:{x_for(3000.0)}:{y_for(5.0)}"])

    # 进程 CPU 只作参考，共用机器上噪声太大，不做判据。重绘次数的实测见 NOTES.md。
    def cpu_of(name, **kwargs):
        shot(name, sets=ic.band_sets(1, kind="bell", freq=1000.0, gain=6.0), **kwargs)
        return float(re.search(r"cpu ([\d.]+) ms/s", last_output).group(1))

    idle = cpu_of("ui_idle", mouse=["click:660:129", "wait:600", "cpu:2000"])
    active = cpu_of("ui_active", audio=noise, mouse=["wait:600", "cpu:2000"])
    print(f"进程 CPU（仅供参考）：频谱关且静止 {idle:.0f} ms/s，频谱开且有音频 {active:.0f} ms/s")

    print(f"\n{failures} 项失败")
    return failures


if __name__ == "__main__":
    sys.exit(main())
