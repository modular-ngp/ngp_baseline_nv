@echo off
REM ngp-minimal Test Script
REM Tests data loading functionality

echo ========================================
echo ngp-minimal Data Loading Test
echo ========================================
echo.

set BUILD_DIR=..\cmake-build-debug\ngp-minimal
set DATA_DIR=..\data\nerf-synthetic

echo Testing: Lego scene
echo.
%BUILD_DIR%\ngp-minimal-app.exe --scene %DATA_DIR%\lego

echo.
echo ========================================
echo Test complete!
echo ========================================

