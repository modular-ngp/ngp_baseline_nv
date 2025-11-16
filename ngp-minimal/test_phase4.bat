@echo off
echo ======================================
echo Phase 4 Test Script
echo ======================================

cd C:\Users\xayah\Desktop\ngp-baseline-nv\cmake-build-debug\ngp-minimal

echo.
echo Checking if executable exists...
if exist ngp-minimal-app.exe (
    echo [OK] ngp-minimal-app.exe found
    dir ngp-minimal-app.exe
) else (
    echo [ERROR] ngp-minimal-app.exe not found
    exit /b 1
)

echo.
echo Testing --help...
ngp-minimal-app.exe --help

echo.
echo ======================================
echo Test completed
echo ======================================

