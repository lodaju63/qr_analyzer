@echo off
chcp 65001 > nul
echo.
echo ========================================
echo   단일 EXE 파일 빌드
echo ========================================
echo.

echo [1/3] Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist\QR_Analyzer.exe" del /f /q "dist\QR_Analyzer.exe"
echo Cleanup complete!
echo.

echo [2/3] Building single executable...
call venv\Scripts\activate.bat
pyinstaller --clean qr_analyzer_onefile.spec
echo.

echo [3/3] Checking build result...
if exist "dist\QR_Analyzer.exe" (
    echo.
    echo ========================================
    echo   Build SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable: dist\QR_Analyzer.exe
    echo Password: 2017112166
    echo.
    echo ⚠️ 주의: 첫 실행 시 압축 해제로 인해 시작이 느릴 수 있습니다!
    echo.
) else (
    echo.
    echo ========================================
    echo   Build FAILED!
    echo ========================================
    echo.
    echo Check the error messages above.
)

pause
