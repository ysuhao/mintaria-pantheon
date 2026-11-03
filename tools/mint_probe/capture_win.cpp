// 单独成文件，避免 windows.h 和 JUCE 的名字冲突。
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <vector>

// 抓取窗口客户区，包括宿主插件的原生子窗口。输出自上而下的 BGRA 像素。
bool captureWindowPixels (void* windowHandle, std::vector<unsigned char>& pixels, int& width, int& height)
{
    auto hwnd = (HWND) windowHandle;

    RECT client {};
    if (! GetClientRect (hwnd, &client))
        return false;

    width = client.right - client.left;
    height = client.bottom - client.top;
    if (width <= 0 || height <= 0)
        return false;

    auto screen = GetDC (nullptr);
    auto memory = CreateCompatibleDC (screen);
    auto bitmap = CreateCompatibleBitmap (screen, width, height);
    auto previous = SelectObject (memory, bitmap);

    constexpr UINT clientOnlyFullContent = 0x1 | 0x2; // PW_CLIENTONLY | PW_RENDERFULLCONTENT
    const auto ok = PrintWindow (hwnd, memory, clientOnlyFullContent) != 0;

    BITMAPINFO info {};
    info.bmiHeader.biSize = sizeof (BITMAPINFOHEADER);
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;

    pixels.resize ((size_t) width * (size_t) height * 4);
    const auto lines = GetDIBits (memory, bitmap, 0, (UINT) height, pixels.data(), &info, DIB_RGB_COLORS);

    SelectObject (memory, previous);
    DeleteObject (bitmap);
    DeleteDC (memory);
    ReleaseDC (nullptr, screen);

    return ok && lines == height;
}
