# Drives the Atlas page switch, auto release switch and Release dial in the Standalone target, saving a capture after each step.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File plugins/comp/tests/standalone_clicks.ps1 -Exe <Mintaria Atlas.exe> -OutDir <dir>
#
# ASCII only: Windows PowerShell 5.1 reads BOM-less files with the system code page.
# Real mouse and keyboard input is forbidden (docs/PROJECT.md). Everything is a message posted to one window handle,
# and each post first confirms that the handle still belongs to the process this script started.
# Dial drags are not tested: during a drag JUCE also reads the real cursor, so posted messages alone give wrong values.
# Captures read the screen, because PrintWindow returned stale frames from a window that was not in the foreground. So the
# window is shown topmost without being activated while the run lasts. Launching it may still take the foreground; the run reports that.
# Captures can lag the last action by seconds on a busy machine, so a step that should change the picture polls until it
# does and then settles. A step that should change nothing waits a fixed time, and step 06 is its control: the same
# double-click on the enabled dial has to open the editor.
# Prints PASS or FAIL per step and exits 1 on any FAIL. The pictures still have to be looked at.
# MINT_LANG makes language switching memory-only, so the shared language.txt is not touched.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$BannerOffset = 57
)

Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class AtlasDrive
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

    const uint MOVE = 0x200, DOWN = 0x201, UP = 0x202, KEYDOWN = 0x100, KEYUP = 0x101, CHAR = 0x102;

    static IntPtr target;
    static uint targetPid;
    static double scale = 1.0;
    static int offset;

    static void Confirm()
    {
        uint pid;
        if (GetWindowThreadProcessId(target, out pid) == 0 || pid != targetPid)
            throw new InvalidOperationException("window no longer belongs to the process under test");
    }

    static void Post(uint msg, IntPtr w, IntPtr l)
    {
        Confirm();
        PostMessage(target, msg, w, l);
    }

    static IntPtr Pos(int x, int y)
    {
        var px = (int) Math.Round(x * scale);
        var py = (int) Math.Round((y + offset) * scale);
        return (IntPtr)((py << 16) | (px & 0xffff));
    }

    // Read-only, so a run can report whether anything outside the test window was disturbed.
    public static string Cursor() { POINT p; GetCursorPos(out p); return p.X + "," + p.Y; }
    public static IntPtr Foreground() { return GetForegroundWindow(); }

    public static string Prepare(IntPtr h, uint pid, int bannerOffset)
    {
        target = h;
        targetPid = pid;
        Confirm();
        SetProcessDPIAware();
        SetWindowPos(h, (IntPtr)(-1), 40, 40, 0, 0, 0x1 | 0x10); // topmost, SWP_NOSIZE | SWP_NOACTIVATE
        RECT r; GetClientRect(target, out r);
        scale = Math.Round((r.R - r.L) / 820.0 * 4.0) / 4.0;
        offset = bannerOffset;
        return (r.R - r.L) + "x" + (r.B - r.T) + " scale " + scale;
    }

    public static void Click(int x, int y)
    {
        Post(MOVE, IntPtr.Zero, Pos(x, y));
        System.Threading.Thread.Sleep(80);
        Post(DOWN, (IntPtr)1, Pos(x, y));
        System.Threading.Thread.Sleep(80);
        Post(UP, IntPtr.Zero, Pos(x, y));
    }

    public static void DoubleClick(int x, int y)
    {
        Post(MOVE, IntPtr.Zero, Pos(x, y));
        System.Threading.Thread.Sleep(80);
        Post(DOWN, (IntPtr)1, Pos(x, y));
        Post(UP, IntPtr.Zero, Pos(x, y));
        System.Threading.Thread.Sleep(40);
        Post(DOWN, (IntPtr)1, Pos(x, y));
        Post(UP, IntPtr.Zero, Pos(x, y));
    }

    public static void TypeText(string text)
    {
        foreach (var c in text)
        {
            Post(CHAR, (IntPtr)c, IntPtr.Zero);
            System.Threading.Thread.Sleep(40);
        }
    }

    public static void Return()
    {
        Post(KEYDOWN, (IntPtr)0x0D, (IntPtr)0x001C0001);
        Post(KEYUP, (IntPtr)0x0D, unchecked((IntPtr)(int)0xC01C0001));
    }

    public static void Capture(string path)
    {
        Confirm();
        RECT r; GetClientRect(target, out r);
        using (var bmp = new Bitmap(r.R - r.L, r.B - r.T, PixelFormat.Format32bppArgb))
        using (var g = Graphics.FromImage(bmp))
        {
            var p = new POINT();
            ClientToScreen(target, ref p);
            g.CopyFromScreen(p.X, p.Y, 0, 0, new Size(r.R - r.L, r.B - r.T));
            // Only the editor: the title bar buttons change colour when the window gets activated.
            var top = (int) Math.Round(offset * scale);
            using (var editor = bmp.Clone(new Rectangle(0, top, bmp.Width, bmp.Height - top), bmp.PixelFormat))
                editor.Save(path, ImageFormat.Png);
        }
    }
}
'@

New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:MINT_LANG = "zh"
$cursorBefore = [AtlasDrive]::Cursor()
$foregroundBefore = [AtlasDrive]::Foreground()
$started = Get-Date
$p = Start-Process $Exe -PassThru
$deadline = (Get-Date).AddSeconds(120)
while ($p.MainWindowHandle -eq 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100; $p.Refresh() }
$h = $p.MainWindowHandle
if ($h -eq 0) { Write-Output "FAIL no window"; Stop-Process -Id $p.Id; exit 1 }
Write-Output ("window after {0:n1} s" -f ((Get-Date) - $started).TotalSeconds)

try
{
    Write-Output "client $([AtlasDrive]::Prepare($h, [uint32]$p.Id, $BannerOffset))"

    $script:failed = 0
    $script:last = ""

    function Shot($name)
    {
        $path = Join-Path $OutDir "$name.png"
        [AtlasDrive]::Capture($path)
        return (Get-FileHash $path -Algorithm MD5).Hash
    }

    function Settled($name, $hash)
    {
        for ($i = 0; $i -lt 10; ++$i)
        {
            Start-Sleep -Milliseconds 1500
            $next = Shot $name
            if ($next -eq $hash) { break }
            $hash = $next
        }
        return $hash
    }

    function Changes($name, $action)
    {
        & $action
        $began = Get-Date
        $hash = Shot $name
        while ($hash -eq $script:last -and ((Get-Date) - $began).TotalSeconds -lt 30)
        {
            Start-Sleep -Milliseconds 500
            $hash = Shot $name
        }
        if ($hash -eq $script:last) { $script:failed++; Write-Output "FAIL $name did not change"; return }
        Write-Output ("PASS {0} changed after {1:n1} s" -f $name, ((Get-Date) - $began).TotalSeconds)
        $script:last = Settled $name $hash
    }

    function Stays($name, $action)
    {
        & $action
        Start-Sleep -Seconds 8
        $hash = Shot $name
        if ($hash -eq $script:last) { Write-Output "PASS $name stayed the same" }
        else { $script:failed++; Write-Output "FAIL $name changed"; $script:last = Settled $name $hash }
    }

    # Editor coordinates, taken from AtlasEditor::resized().
    function ToMain { [AtlasDrive]::Click(522, 450) }
    function ToAdvanced { [AtlasDrive]::Click(587, 450) }
    function ToManual { [AtlasDrive]::Click(458, 203) }
    function ToAuto { [AtlasDrive]::Click(578, 203) }
    function ReleaseValue { [AtlasDrive]::DoubleClick(455, 377) }
    function ReleaseKnob { [AtlasDrive]::DoubleClick(455, 313) }

    Start-Sleep -Seconds 3
    $script:last = Settled "01_main_auto" (Shot "01_main_auto")

    Stays   "02_auto_value_doubleclick_nothing" { ReleaseValue }
    Changes "03_advanced_zh" { ToAdvanced }
    Changes "04_manual_selected" { ToManual }
    Changes "05_main_manual_bright" { ToMain }
    Changes "06_manual_value_editor_open" { ReleaseValue }
    Changes "07_manual_release_500" { [AtlasDrive]::TypeText("500"); [AtlasDrive]::Return() }
    Changes "08_advanced" { ToAdvanced }
    Changes "09_auto_selected" { ToAuto }
    Changes "10_main_auto_dim_500" { ToMain }
    Stays   "11_auto_value_doubleclick_nothing" { ReleaseValue }
    Stays   "12_auto_knob_doubleclick_keeps_500" { ReleaseKnob }
    Changes "13_advanced" { ToAdvanced }
    Changes "14_manual_selected" { ToManual }
    Changes "15_main_manual_500" { ToMain }
    Changes "16_manual_knob_doubleclick_resets" { ReleaseKnob }
    Changes "17_advanced" { ToAdvanced }
    Changes "18_lookahead_editor_open" { [AtlasDrive]::DoubleClick(260, 307) }
    Changes "19_lookahead_5_no_readout" { [AtlasDrive]::TypeText("5"); [AtlasDrive]::Return() }
    Changes "20_main_lookahead_readout" { ToMain }
    Changes "21_advanced" { ToAdvanced }
    Changes "22_advanced_en" { [AtlasDrive]::Click(669, 47) }
    Changes "23_advanced_ko" { [AtlasDrive]::Click(713, 47) }

    Write-Output "cursor before $cursorBefore, after $([AtlasDrive]::Cursor())"
    Write-Output "foreground before $foregroundBefore, after $([AtlasDrive]::Foreground()), test window $h"
}
finally
{
    Stop-Process -Id $p.Id
}

Write-Output "wrote captures to $OutDir"
if ($script:failed -gt 0) { Write-Output "FAILED $($script:failed) steps"; exit 1 }
Write-Output "ALL STEPS PASSED"
