@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
robocopy "%~dp0Mintaria Pantheon" "C:\Program Files\Common Files\VST3\Mintaria Pantheon" /E /NFL /NDL /NJH /NJS /NP
if %errorlevel% geq 8 (
    echo Install failed, error %errorlevel%
) else (
    echo Installed to C:\Program Files\Common Files\VST3\Mintaria Pantheon
)
pause
