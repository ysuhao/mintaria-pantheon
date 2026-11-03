"""测 Standalone 里的点击。只向确认属于被测进程的窗口句柄发消息，不碰真实鼠标键盘。
  python standalone_ui.py <exe> <输出目录> probe   启动、截一张、关闭
  python standalone_ui.py <exe> <输出目录> run     展开高级区、选 12-bit、切语言、重开、收起，每步截图

JUCE 按 WindowFromPoint 判断命中，所以测试期间窗口要显示在屏幕上（置顶、不激活）。
语言用 MINT_LANG 强制，不写语言文件。Standalone 的设置文件在结束时恢复原样。
"""
import atexit, ctypes, os, subprocess, sys, time, zlib, struct
from ctypes import wintypes as W

u = ctypes.WinDLL("user32", use_last_error=True)
g = ctypes.WinDLL("gdi32")
u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
u.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
u.GetDC.restype = W.HDC; u.GetDC.argtypes = [W.HWND]
u.ReleaseDC.argtypes = [W.HWND, W.HDC]
u.PrintWindow.argtypes = [W.HWND, W.HDC, W.UINT]
u.SetWindowPos.argtypes = [W.HWND, W.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.UINT]
u.IsWindowVisible.argtypes = [W.HWND]
u.GetWindowThreadProcessId.argtypes = [W.HWND, ctypes.POINTER(W.DWORD)]
u.GetClientRect.argtypes = [W.HWND, ctypes.POINTER(W.RECT)]
g.CreateCompatibleDC.restype = W.HDC; g.CreateCompatibleDC.argtypes = [W.HDC]
g.CreateCompatibleBitmap.restype = W.HBITMAP; g.CreateCompatibleBitmap.argtypes = [W.HDC, ctypes.c_int, ctypes.c_int]
g.SelectObject.argtypes = [W.HDC, W.HGDIOBJ]; g.SelectObject.restype = W.HGDIOBJ
g.DeleteObject.argtypes = [W.HGDIOBJ]; g.DeleteDC.argtypes = [W.HDC]
g.GetDIBits.argtypes = [W.HDC, W.HBITMAP, W.UINT, W.UINT, ctypes.c_void_p, ctypes.c_void_p, W.UINT]

u.GetWindowRect.argtypes = [W.HWND, ctypes.POINTER(W.RECT)]
u.ClientToScreen.argtypes = [W.HWND, ctypes.POINTER(W.POINT)]
u.WindowFromPoint.argtypes = [W.POINT]; u.WindowFromPoint.restype = W.HWND
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_CLOSE = 0x200, 0x201, 0x202, 0x10
PROCS = []


def owner(hwnd):
    pid = W.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def find_window(pid, timeout=150.0):
    end = time.time() + timeout
    while time.time() < end:
        found = []
        proto = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
        def cb(h, _):
            rect = W.RECT()
            u.GetClientRect(h, ctypes.byref(rect))
            if owner(h) == pid and u.IsWindowVisible(h) and rect.right >= 400:
                found.append(h)
            return True
        u.EnumWindows(proto(cb), 0)
        if found:
            return found[0]
        time.sleep(0.2)
    raise SystemExit("no window for the process")


def client(hwnd):
    r = W.RECT()
    u.GetClientRect(hwnd, ctypes.byref(r))
    return r.right, r.bottom


def capture(hwnd, path):
    assert owner(hwnd) == PID
    w, h = client(hwnd)
    sdc = u.GetDC(None)
    dc = g.CreateCompatibleDC(sdc)
    bmp = g.CreateCompatibleBitmap(sdc, w, h)
    old = g.SelectObject(dc, bmp)
    u.PrintWindow(hwnd, dc, 1 | 2)   # PW_CLIENTONLY | PW_RENDERFULLCONTENT
    class BI(ctypes.Structure):
        _fields_ = [("s", W.DWORD), ("w", W.LONG), ("h", W.LONG), ("p", W.WORD), ("b", W.WORD), ("c", W.DWORD),
                    ("si", W.DWORD), ("x", W.LONG), ("y", W.LONG), ("cu", W.DWORD), ("ci", W.DWORD)]
    bi = BI(ctypes.sizeof(BI), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    g.GetDIBits(dc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    g.SelectObject(dc, old); g.DeleteObject(bmp); g.DeleteDC(dc); u.ReleaseDC(None, sdc)
    rows = bytearray()
    raw = bytes(buf)
    for y in range(h):
        row = bytearray(raw[y * w * 4:(y + 1) * w * 4])
        row[0::4], row[2::4] = row[2::4], row[0::4]
        row[3::4] = b"\xff" * w
        rows += b"\0" + row
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))
    print("shot", os.path.basename(path), w, h, flush=True)


def post(hwnd, msg, wparam, x, y):
    assert owner(hwnd) == PID, "window is not owned by the process under test"
    assert u.PostMessageW(hwnd, msg, wparam, (y << 16) | (x & 0xffff))


def click(hwnd, x, y):
    w, h = client(hwnd)
    scale = round(w / 820.0, 1)
    left = (w - 820 * scale) / 2.0
    top = h - 520 * scale - left      # 编辑器贴着客户区底部，上面是标题栏和提示条
    px, py = int(x * scale + left), int(y * scale + top)
    pt = W.POINT(px, py)
    u.ClientToScreen(hwnd, ctypes.byref(pt))
    # 别的进程的窗口可能正好盖在这里。等它走开，等不到就放弃，消息不会发到别处。
    end = time.time() + 60.0
    while u.WindowFromPoint(pt) != hwnd:
        assert time.time() < end, "another window is over the target point"
        time.sleep(0.5)
    print("click", x, y, "->", px, py, "screen", pt.x, pt.y, flush=True)

    post(hwnd, WM_MOUSEMOVE, 0, px, py)
    time.sleep(0.1)
    post(hwnd, WM_LBUTTONDOWN, 1, px, py)
    time.sleep(0.05)
    post(hwnd, WM_LBUTTONUP, 0, px, py)
    time.sleep(0.6)


def launch():
    global PID
    env = dict(os.environ, MINT_LANG=LANG)
    env.pop("ELYSIUM_SHOW_ADVANCED", None)
    proc = subprocess.Popen([EXE], env=env)
    PROCS.append(proc)
    PID = proc.pid
    t0 = time.time()
    try:
        hwnd = find_window(PID)
    except BaseException:
        proc.terminate()
        raise
    print("window after", round(time.time() - t0, 1), "s", flush=True)
    time.sleep(2.0)
    assert owner(hwnd) == PID
    # JUCE 按 WindowFromPoint 判断命中，窗口必须在屏幕上且没被遮挡。置顶但不激活，不抢键盘焦点。
    frame = W.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(frame))
    x = max(0, u.GetSystemMetrics(0) - (frame.right - frame.left) - 20)
    y = max(0, u.GetSystemMetrics(1) - (frame.bottom - frame.top) - 60)
    u.SetWindowPos(hwnd, W.HWND(-1), x, y, 0, 0, 0x0001 | 0x0010)
    time.sleep(0.5)
    return proc, hwnd


def close(proc, hwnd):
    post(hwnd, WM_CLOSE, 0, 0, 0)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(5)
        print("had to terminate", flush=True)


EXE, OUT, MODE = sys.argv[1:4]
LANG = "en"
PID = 0
os.makedirs(OUT, exist_ok=True)

SETTINGS = os.path.join(os.environ["APPDATA"], "Mintaria Elysium", "Mintaria Elysium.settings")
SAVED = open(SETTINGS, "rb").read() if os.path.exists(SETTINGS) else None


def restore_settings():
    if SAVED is not None:
        with open(SETTINGS, "wb") as f:
            f.write(SAVED)
    elif os.path.exists(SETTINGS):
        os.remove(SETTINGS)
        os.rmdir(os.path.dirname(SETTINGS))


atexit.register(restore_settings)
atexit.register(lambda: [p.terminate() for p in PROCS if p.poll() is None])
if MODE == "probe":
    proc, hwnd = launch()
    capture(hwnd, os.path.join(OUT, "probe.png"))
    close(proc, hwnd)
else:
    proc, hwnd = launch()
    capture(hwnd, os.path.join(OUT, "1_start.png"))
    click(hwnd, 589, 456)      # 高级开关：展开
    capture(hwnd, os.path.join(OUT, "2_adv_open.png"))
    click(hwnd, 428, 456)      # 特性第一格 12-bit，确认底栏命中区域
    capture(hwnd, os.path.join(OUT, "3_char_12bit.png"))
    click(hwnd, 713, 47)       # 语言切换到韩文。MINT_LANG 强制时不写语言文件
    capture(hwnd, os.path.join(OUT, "4_lang_ko.png"))
    click(hwnd, 756, 47)       # 中文
    capture(hwnd, os.path.join(OUT, "5_lang_zh.png"))
    close(proc, hwnd)
    time.sleep(1.0)
    proc, hwnd = launch()      # 重开，看展开状态有没有保存
    capture(hwnd, os.path.join(OUT, "6_reopen.png"))
    click(hwnd, 589, 456)      # 收起
    capture(hwnd, os.path.join(OUT, "7_adv_closed.png"))
    close(proc, hwnd)
