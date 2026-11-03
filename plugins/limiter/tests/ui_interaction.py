#!/usr/bin/env python
"""用 Standalone 验证释放旋钮的禁用和恢复。需要 numpy，只能在 Windows 上跑。

    python plugins/limiter/tests/ui_interaction.py [--build D:/mint-build/limiter]

按 docs/PROJECT.md 的规定不碰真实鼠标和键盘：只向窗口句柄发消息，并且每次发送前都确认句柄属于
本脚本启动的进程，确认不了就退出。脚本结束时结束这个进程，并恢复 Standalone 的设置文件。

- 窗口要留在屏幕上（屏幕外时只显示首帧，截图不再变化），所以运行时会出现约 15 秒。脚本不主动激活它。
- 拖动过程中 JUCE 还会读真实光标，投递的拖动不能证明方向和幅度，这里只用它判断“值变没变”。
- 投递的双击不一定被识别（旋钮收到了鼠标，但数值不复位），所以双击复位可能输出“未判定”。
- 投递的点击偶尔会丢，所以每次切换自动释放后都从画面确认状态，确认不了就判失败，不让后面的判断建立在错误的前提上。
- 双击数值区的输入框这里测不了：判断命中时用的位置受真实光标影响，正面对照也观察不到。
- Standalone 启动时会打开默认音频设备，输入保持静音。MINT_LANG 让语言切换不写文件。
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetWindowDC.argtypes = [wintypes.HWND]
user32.GetWindowDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                            ctypes.c_void_p, wintypes.UINT]

WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0200, 0x0201, 0x0202
MK_LBUTTON = 0x0001
PW_RENDERFULLCONTENT = 0x2

# 编辑器坐标，来自 AegisEditor::resized()：释放旋钮、数值、自动释放开关。
DIAL_CENTRE = (608, 211)
VALUE_CENTRE = (608, 297)
VALUE_AREA = (533, 286, 683, 308)
DIAL_AREA = (533, 136, 683, 286)
AUTO_OFF = (584, 368)
AUTO_ON = (632, 368)
AUTO_ON_CELL = (612, 358, 622, 362)


def find_window(pid):
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


class Standalone:
    def __init__(self, exe, reference):
        env = dict(os.environ, MINT_LANG="en")
        self.process = subprocess.Popen([str(exe)], env=env, cwd=str(exe.parent))
        self.hwnd = None
        try:
            # 打开音频设备可能要等很久，窗口之后才出现。
            for _ in range(600):
                windows = find_window(self.process.pid)
                if windows:
                    self.hwnd = windows[0]
                    break
                time.sleep(0.1)
            if self.hwnd is None:
                raise RuntimeError("找不到被测进程的窗口")
            time.sleep(1.5)
            # 截图包含标题栏，消息用客户区坐标，两者差一个客户区原点。
            self.offset = self.locate(reference)
            outer = wintypes.RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(outer))
            origin = wintypes.POINT(0, 0)
            user32.ClientToScreen(self.hwnd, ctypes.byref(origin))
            self.client_offset = (self.offset[0] - (origin.x - outer.left), self.offset[1] - (origin.y - outer.top))
        except Exception:
            self.close()
            raise

    def close(self):
        self.process.kill()
        self.process.wait()

    def check_owner(self):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(self.hwnd, ctypes.byref(owner))
        if owner.value != self.process.pid or self.process.poll() is not None:
            raise RuntimeError("句柄不再属于被测进程，停止发送消息")

    def capture(self):
        self.check_owner()
        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        window_dc = user32.GetWindowDC(self.hwnd)
        memory_dc = gdi32.CreateCompatibleDC(window_dc)
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, w, h)
        gdi32.SelectObject(memory_dc, bitmap)
        user32.PrintWindow(self.hwnd, memory_dc, PW_RENDERFULLCONTENT)

        class Header(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("w", wintypes.LONG), ("h", wintypes.LONG), ("planes", wintypes.WORD),
                        ("bits", wintypes.WORD), ("compression", wintypes.DWORD), ("image", wintypes.DWORD),
                        ("xp", wintypes.LONG), ("yp", wintypes.LONG), ("used", wintypes.DWORD), ("important", wintypes.DWORD)]

        header = Header(ctypes.sizeof(Header), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        gdi32.GetDIBits(memory_dc, bitmap, 0, h, pixels.ctypes.data, ctypes.byref(header), 0)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(self.hwnd, window_dc)
        return pixels[:, :, :3].astype(np.int16)

    def locate(self, reference):
        """Standalone 在编辑器上方可能有提示条：用首帧截图的铭牌区找出编辑器的位置。"""
        image = self.capture()
        patch = reference[28:96, 240:580]
        best, best_error = (0, 0), None
        for dy in range(0, min(140, image.shape[0] - 96)):
            for dx in range(0, min(40, image.shape[1] - 580)):
                error = np.mean(np.abs(image[28 + dy:96 + dy, 240 + dx:580 + dx] - patch))
                if best_error is None or error < best_error:
                    best, best_error = (dx, dy), error
        if best_error > 12:
            raise RuntimeError(f"定位编辑器失败，误差 {best_error:.1f}")
        return best

    def post(self, message, wparam, point):
        self.check_owner()
        x, y = point[0] + self.client_offset[0], point[1] + self.client_offset[1]
        user32.PostMessageW(self.hwnd, message, wparam, (y << 16) | (x & 0xFFFF))
        time.sleep(0.03)

    def click(self, point):
        self.post(WM_MOUSEMOVE, 0, point)
        self.post(WM_LBUTTONDOWN, MK_LBUTTON, point)
        self.post(WM_LBUTTONUP, 0, point)
        time.sleep(0.6)

    def drag_up(self, point, pixels):
        self.post(WM_MOUSEMOVE, 0, point)
        self.post(WM_LBUTTONDOWN, MK_LBUTTON, point)
        for step in range(1, 9):
            self.post(WM_MOUSEMOVE, MK_LBUTTON, (point[0], point[1] - pixels * step // 8))
        self.post(WM_LBUTTONUP, 0, (point[0], point[1] - pixels))
        time.sleep(0.5)

    def double_click(self, point):
        self.post(WM_MOUSEMOVE, 0, point)
        self.post(WM_LBUTTONDOWN, MK_LBUTTON, point)
        self.post(WM_LBUTTONUP, 0, point)
        self.post(WM_LBUTTONDOWN, MK_LBUTTON, point)
        self.post(WM_LBUTTONUP, 0, point)
        time.sleep(0.6)

    def save(self, image, path):
        dx, dy = self.offset
        crop = image[dy:dy + 520, dx:dx + 820][:, :, [2, 1, 0]].astype(np.uint8)
        Image.fromarray(crop).save(path)

    def region(self, image, area):
        dx, dy = self.offset
        return image[area[1] + dy:area[3] + dy, area[0] + dx:area[2] + dx]


def read_png(path):
    # capture 是 BGR，这里也转成 BGR。
    return np.asarray(Image.open(path).convert("RGB"))[:, :, [2, 1, 0]].astype(np.int16)


def changed(a, b):
    return float(np.mean(np.abs(a - b))) > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="D:/mint-build/limiter")
    args = parser.parse_args()
    build = Path(args.build)
    exe = build / "plugins/limiter/mint_limiter_artefacts/Release/Standalone/Mintaria Aegis.exe"
    probe = build / "tools/mint_probe/mint_probe_artefacts/Release/mint_probe.exe"
    plugin = build / "plugins/limiter/mint_limiter_artefacts/Release/VST3/Mintaria Aegis.vst3"
    shot = build / "aegis_ui_reference.png"
    subprocess.run([str(probe), "shot", str(plugin), str(shot)], env=dict(os.environ, MINT_LANG="en"), capture_output=True)

    settings = Path(os.environ["APPDATA"]) / "Mintaria Aegis.settings"
    backup = build / "aegis_settings.backup"
    had_settings = settings.exists()
    if had_settings:
        shutil.copy2(settings, backup)

    failures = 0
    def check(ok, text):
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + text, flush=True)
        failures += 0 if ok else 1

    app = Standalone(exe, read_png(shot))
    try:
        print(f"窗口句柄属于进程 {app.process.pid}，编辑器在客户区的偏移 {app.client_offset}")
        region = app.region
        steps = build / "aegis_ui"
        steps.mkdir(exist_ok=True)
        count = 0

        def snap(name):
            nonlocal count
            image = app.capture()
            count += 1
            app.save(image, steps / f"{count:02d}_{name}.png")
            return image

        def set_auto(on, name):
            target = AUTO_ON if on else AUTO_OFF
            for _ in range(6):
                app.click(target)
                image = snap(name)
                # 选中的分段是深色底。
                if (np.mean(region(image, AUTO_ON_CELL)) < 128) == on:
                    return image
            raise RuntimeError(f"切换自动释放失败：{name}")

        on_state = set_auto(True, "auto_on")
        check(True, "切到自动释放")

        app.drag_up(DIAL_CENTRE, 60)
        after = snap("auto_on_drag")
        check(not changed(region(on_state, VALUE_AREA), region(after, VALUE_AREA)), "自动释放开：拖动释放旋钮，数值不变")
        check(not changed(region(on_state, DIAL_AREA), region(after, DIAL_AREA)), "自动释放开：拖动释放旋钮，旋钮不动")

        off_state = set_auto(False, "manual")
        check(changed(region(on_state, DIAL_AREA), region(off_state, DIAL_AREA)), "切到手动：旋钮恢复正常亮度")

        app.drag_up(DIAL_CENTRE, 60)
        dragged = snap("manual_drag")
        check(changed(region(off_state, VALUE_AREA), region(dragged, VALUE_AREA)), "手动：拖动释放旋钮，数值改变（正面对照）")

        on_dragged = set_auto(True, "auto_on_again")
        check(changed(region(dragged, DIAL_AREA), region(on_dragged, DIAL_AREA)), "切回自动：旋钮再次变暗")

        # 投递的双击不是每次都被识别，所以禁用和手动都试同样的次数，两边才可比。
        moved_when_disabled = False
        for attempt in range(3):
            app.double_click(DIAL_CENTRE)
            after = snap(f"auto_on_dblclick_dial_{attempt}")
            moved_when_disabled = moved_when_disabled or changed(region(on_dragged, VALUE_AREA), region(after, VALUE_AREA))
            time.sleep(0.6)

        before_reset = set_auto(False, "manual_again")
        moved = False
        for attempt in range(3):
            app.double_click(DIAL_CENTRE)
            after = snap(f"manual_dblclick_dial_{attempt}")
            moved = moved or changed(region(before_reset, VALUE_AREA), region(after, VALUE_AREA))
            time.sleep(0.6)

        # 没有正面对照时，“禁用时双击没反应”不能说明任何事，不算通过也不算失败。
        if moved:
            check(not moved_when_disabled, "自动释放开：双击旋钮 3 次都不复位（手动时能复位）")
        else:
            print("未判定 双击复位：手动时投递的双击也没有被识别，正面对照不成立" + ("；但禁用时数值变了，失败" if moved_when_disabled else ""))
            failures += 1 if moved_when_disabled else 0
    finally:
        app.close()
        if had_settings:
            shutil.copy2(backup, settings)
        elif settings.exists():
            settings.unlink()

    print(f"每一步的截图在 {build / 'aegis_ui'}")
    print("失败项：", failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
