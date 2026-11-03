@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
rmdir /s /q "C:\Program Files\Common Files\VST3\Mintaria Pantheon"
echo Removed C:\Program Files\Common Files\VST3\Mintaria Pantheon
pause
