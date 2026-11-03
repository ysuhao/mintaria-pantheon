#!/usr/bin/env python3
"""用 Standalone 测界面交互：视图切换、写自动化开关的显示、高级视图状态的保存、语言切换后的重绘。

    python plugins/rider/tests/ui_standalone.py

不碰真实鼠标和键盘：点击都是发给窗口句柄的消息，每次发之前确认句柄属于被测进程。
窗口不需要在前台，画面用 PrintWindow 取，不截屏幕。
MINT_LANG 设了之后语言切换不写文件，不影响用户的语言设置。
Standalone 的设置目录测试前备份、测试后恢复。
悬停效果测不了：窗口消息只能移动 JUCE 看到的位置，真实光标不在窗口上，悬停状态留不住。
"""
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes

import numpy as np
from PIL import Image

import riderlib as rl

EXE = f"{rl.BUILD}/plugins/rider/mint_rider_artefacts/Release/Standalone/Mintaria Helios.exe"
OUT = f"{rl.TMP}/ui"
SETTINGS = os.path.join(os.environ["APPDATA"], "Mintaria Helios")
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_CLOSE = 0x0200, 0x0201, 0x0202, 0x0010
MK_LBUTTON = 1
PW_RENDERFULLCONTENT = 2
EDITOR_W, EDITOR_H = 820, 520

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]

failures = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)

def windows_of(pid):
    found = []
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        if owner.value == pid and user32.IsWindowVisible(hwnd) and rect.right >= EDITOR_W:
            found.append(hwnd)
        return True
    user32.EnumWindows(visit, 0)
    return found

class App:
    def __init__(self, lang="en"):
        self.process = subprocess.Popen([EXE], env={**os.environ, "MINT_LANG": lang})
        self.hwnd = None
        # 音频设备初始化可能要几十秒，窗口在那之后才出现。
        for _ in range(600):
            time.sleep(0.2)
            found = windows_of(self.process.pid)
            if found:
                self.hwnd = found[0]
                break
        if self.hwnd is None:
            self.process.kill()
            raise RuntimeError("Standalone 窗口没有出现")
        time.sleep(1.5)
        rect = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(rect))
        self.client = (rect.right, rect.bottom)
        self.scale = user32.GetDpiForWindow(self.hwnd) / 96.0
        # 编辑器在客户区里贴底，四周有一圈窗口边框，上面是标题栏和静音提示。
        border = (rect.right - EDITOR_W * self.scale) / 2
        self.origin = (border, rect.bottom - EDITOR_H * self.scale - border)

    def post(self, message, wparam, x, y):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(self.hwnd, ctypes.byref(owner))
        if owner.value != self.process.pid:
            raise RuntimeError("句柄不属于被测进程，停止")
        px, py = int(self.origin[0] + x * self.scale), int(self.origin[1] + y * self.scale)
        if not user32.PostMessageW(self.hwnd, message, wparam, (py << 16) | (px & 0xffff)):
            raise ctypes.WinError(ctypes.get_last_error())

    def click(self, x, y, wait=0.5):
        self.post(WM_MOUSEMOVE, 0, x, y)
        time.sleep(0.05)
        self.post(WM_LBUTTONDOWN, MK_LBUTTON, x, y)
        time.sleep(0.05)
        self.post(WM_LBUTTONUP, 0, x, y)
        time.sleep(wait)

    def image(self, name=None):
        """整个窗口画到内存位图，再裁出编辑器。返回 RGB 数组，坐标是物理像素。"""
        outer = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(outer))
        w, h = outer.right - outer.left, outer.bottom - outer.top
        screen = user32.GetDC(self.hwnd)
        dc = gdi32.CreateCompatibleDC(screen)
        bitmap = gdi32.CreateCompatibleBitmap(screen, w, h)
        gdi32.SelectObject(dc, bitmap)
        user32.PrintWindow(self.hwnd, dc, PW_RENDERFULLCONTENT)
        header = (ctypes.c_uint32 * 10)(40, w, ctypes.c_uint32(-h).value, (32 << 16) | 1, 0, 0, 0, 0, 0, 0)
        data = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(dc, bitmap, 0, h, data, header, 0)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(dc)
        user32.ReleaseDC(self.hwnd, screen)
        pixels = np.frombuffer(data.raw, dtype=np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0]]

        point = wintypes.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(point))
        left = point.x - outer.left + int(self.origin[0])
        top = point.y - outer.top + int(self.origin[1])
        crop = pixels[top:top + int(EDITOR_H * self.scale), left:left + int(EDITOR_W * self.scale)]
        if name:
            Image.fromarray(crop).save(f"{OUT}/{name}.png")
        return crop

    def region(self, image, x, y, w, h):
        s = self.scale
        return image[int(y * s):int((y + h) * s), int(x * s):int((x + w) * s)].astype(np.int32)

    def close(self):
        user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise RuntimeError("窗口关闭后进程没有退出")

def differs(a, b, threshold=8.0):
    return np.abs(a - b).max() > threshold

# 编辑器坐标（RiderEditor::resized）。
TAB_MAIN, TAB_ADV = (360, 456), (460, 456)
SWITCH_OFF, SWITCH_WRITE = (225, 456), (275, 456)
LANG = dict(en=(669, 47), ko=(713, 47), zh=(756, 47))
TRACE = (330, 270, 300, 100)      # 主视图是曲线，高级视图是三个旋钮
SWITCH = (200, 426, 100, 44)      # 写自动化的标题和开关
READOUT = (330, 392, 300, 16)

def run():
    os.makedirs(OUT, exist_ok=True)
    print("== 视图切换和写自动化开关")
    app = App()
    print(f"  窗口 {app.client}，缩放 {app.scale}，编辑器原点 {app.origin}")
    try:
        main = app.image("01_main")
        empty_switch = app.region(main, *SWITCH)
        check("首次打开是主视图，写自动化不显示", empty_switch.std(axis=(0, 1)).max() < 3.0, f"开关区标准差 {empty_switch.std(axis=(0, 1)).max():.2f}")

        app.click(*TAB_ADV)
        adv = app.image("03_advanced")
        check("点高级：中间换成旋钮", differs(app.region(adv, *TRACE), app.region(main, *TRACE)))
        check("点高级：写自动化出现", differs(app.region(adv, *SWITCH), empty_switch))

        app.click(*SWITCH_WRITE)
        writing = app.image("04_writing")
        app.click(*TAB_MAIN)
        main_writing = app.image("05_main_writing")
        check("回主视图：曲线回来", differs(app.region(main_writing, *TRACE), app.region(adv, *TRACE)))
        check("正在写自动化时主视图仍显示开关", differs(app.region(main_writing, *SWITCH), empty_switch))
        check("开关显示写入状态", differs(app.region(writing, 200, 442, 100, 28), app.region(adv, 200, 442, 100, 28)))

        app.click(*SWITCH_OFF)
        main_off = app.image("06_main_off")
        check("关掉后主视图收起开关", app.region(main_off, *SWITCH).std(axis=(0, 1)).max() < 3.0)

        print("== 语言切换后的重绘（MINT_LANG 下不写语言文件）")
        for lang in ("ko", "zh", "en"):
            before = app.image()
            app.click(*LANG[lang])
            after = app.image(f"07_lang_{lang}")
            check(f"切到 {lang}：读数和页签文字重绘", differs(app.region(before, *READOUT), app.region(after, *READOUT)) and differs(app.region(before, 310, 442, 200, 28), app.region(after, 310, 442, 200, 28)))

        print("== 高级视图状态的保存")
        app.click(*TAB_ADV)
        for lang in ("ko", "zh", "en"):
            app.click(*LANG[lang])
            app.image(f"08_advanced_{lang}")
        saved = app.image("08_before_close")
    finally:
        app.close()

    app = App()
    try:
        reopened = app.image("09_reopened")
        check("关闭再打开仍是高级视图", not differs(app.region(reopened, *TRACE), app.region(saved, *TRACE)) and differs(app.region(reopened, *TRACE), app.region(main, *TRACE)))
        check("高级页签仍是选中状态", not differs(app.region(reopened, 310, 442, 200, 28), app.region(saved, 310, 442, 200, 28)))
        app.click(*TAB_MAIN)
        back = app.image("09_back_to_main")
        check("再点增益曲线回到主视图", differs(app.region(back, *TRACE), app.region(saved, *TRACE)))
    finally:
        app.close()

    app = App()
    try:
        again = app.image("10_reopened_main")
        check("切回主视图后再打开是主视图", differs(app.region(again, *TRACE), app.region(saved, *TRACE)) and app.region(again, *SWITCH).std(axis=(0, 1)).max() < 3.0)
    finally:
        app.close()

def main():
    backup = SETTINGS + " ui_test_backup"
    had_settings = os.path.exists(SETTINGS)
    if had_settings:
        shutil.move(SETTINGS, backup)
    try:
        run()
    finally:
        shutil.rmtree(SETTINGS, ignore_errors=True)
        if had_settings:
            shutil.move(backup, SETTINGS)
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("ALL PASS")

if __name__ == "__main__":
    main()
