# Clicks the Sirens mode buttons, switches and language buttons in the Standalone target, saving a screenshot after each step for visual checks.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File plugins/chorus/tests/standalone_clicks.ps1 -Exe <Mintaria Sirens.exe> -OutDir <dir>
#
# ASCII only: Windows PowerShell 5.1 reads BOM-less files with the system code page.
# Real mouse and keyboard input is forbidden (docs/PROJECT.md). Everything is a message posted to one window handle,
# and each post first confirms that the handle still belongs to the process this script started.
# The Mix dial is not tested: during a drag JUCE also reads the real cursor, so posted messages alone give wrong values.
# Do not post keys: Return reaches the banner's Settings button, whose dialog takes the foreground from whoever is using the machine.
# The window must stay on screen: off screen it keeps showing its first frame, and captures stop changing.
# It is shown topmost for about 5 s without being activated by this script. Launching it may still take the foreground; the run reports that.
# MINT_LANG makes language switching memory-only, so the shared language.txt is not touched.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$OutDir
)

Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class SirensDrive
{
    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
    [DllImport("user32.dll")] static extern bool PostMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
    [DllImport("user32.dll")] static extern bool GetClientRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] static extern bool ClientToScreen(IntPtr h, ref POINT p);
    [DllImport("user32.dll")] static extern bool GetCursorPos(out POINT p);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);

    [StructLayout(LayoutKind.Sequential)] struct RECT { public int L, T, R, B; }
    [StructLayout(LayoutKind.Sequential)] struct POINT { public int X, Y; }

    const uint MOVE = 0x200, DOWN = 0x201, UP = 0x202;

    static IntPtr target;
    static uint targetPid;

    static void Confirm()
    {
        uint pid;
        if (GetWindowThreadProcessId(target, out pid) == 0 || pid != targetPid)
            throw new InvalidOperationException("window no longer belongs to the process under test");
    }

    static void Post(uint msg, IntPtr w, int x, int y)
    {
        Confirm();
        PostMessage(target, msg, w, (IntPtr)((y << 16) | (x & 0xffff)));
    }

    // Read-only, so a run can report whether anything outside the test window was disturbed.
    public static string Cursor() { POINT p; GetCursorPos(out p); return p.X + "," + p.Y; }
    public static IntPtr Foreground() { return GetForegroundWindow(); }

    public static void Prepare(IntPtr h, uint pid)
    {
        target = h;
        targetPid = pid;
        Confirm();
        SetProcessDPIAware();
        SetWindowPos(h, (IntPtr)(-1), 40, 40, 0, 0, 0x1 | 0x10); // topmost, SWP_NOSIZE | SWP_NOACTIVATE
    }

    public static void Click(int x, int y)
    {
        Post(MOVE, IntPtr.Zero, x, y);
        System.Threading.Thread.Sleep(60);
        Post(DOWN, (IntPtr)1, x, y);
        System.Threading.Thread.Sleep(60);
        Post(UP, IntPtr.Zero, x, y);
    }

    public static void Capture(string path)
    {
        Confirm();
        RECT r; GetClientRect(target, out r);
        var p = new POINT();
        ClientToScreen(target, ref p);
        using (var bmp = new Bitmap(r.R - r.L, r.B - r.T, PixelFormat.Format32bppArgb))
        using (var g = Graphics.FromImage(bmp))
        {
            g.CopyFromScreen(p.X, p.Y, 0, 0, new Size(r.R - r.L, r.B - r.T));
            bmp.Save(path, ImageFormat.Png);
        }
    }
}
'@

New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:MINT_LANG = "zh"
$cursorBefore = [SirensDrive]::Cursor()
$foregroundBefore = [SirensDrive]::Foreground()
$p = Start-Process $Exe -PassThru
# Audio device scanning can hold the window back for well over 15 s.
$deadline = (Get-Date).AddSeconds(90)
while ($p.MainWindowHandle -eq 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100; $p.Refresh() }
$h = $p.MainWindowHandle
if ($h -eq 0) { Stop-Process -Id $p.Id; Write-Output "FAIL no window"; exit 1 }

try
{
    [SirensDrive]::Prepare($h, [uint32]$p.Id)
    Start-Sleep -Seconds 1

    # Client coordinates. The Standalone title bar and input-muted banner push the editor down by 57 px.
    function Step($name) { Start-Sleep -Milliseconds 450; [SirensDrive]::Capture((Join-Path $OutDir "$name.png")) }
    function Click($x, $y) { [SirensDrive]::Click($x, $y) }

    Step "1_initial_I"
    Click 337 226; Step "2_I_II"
    Click 192 226; Click 482 226; Click 627 226; Step "3_II_III_IV"
    Click 317 507; Step "4_compat_on"
    Click 505 507; Step "5_noise_on"
    Click 671 105; Step "6_en"
    Click 714 105; Step "7_ko"
    Click 757 105; Step "8_zh"

    Write-Output "cursor before $cursorBefore, after $([SirensDrive]::Cursor())"
    Write-Output "foreground before $foregroundBefore, after $([SirensDrive]::Foreground()), test window $h"
}
finally
{
    Stop-Process -Id $p.Id
}

Write-Output "wrote screenshots to $OutDir"
