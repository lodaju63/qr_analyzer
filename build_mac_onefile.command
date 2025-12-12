#!/bin/bash
# Mac용 단일 .app 파일 빌드 스크립트

echo ""
echo "========================================"
echo "  QR Analyzer - Mac Single App Build"
echo "========================================"
echo ""

# 현재 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 활성화
if [ -d "venv" ]; then
    echo "[1/4] Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "And: pip install -r requirements_pyqt.txt"
    exit 1
fi

# 이전 빌드 정리
echo "[2/4] Cleaning previous builds..."
rm -rf build dist/QR_Analyzer.app
echo "Cleanup complete!"
echo ""

# 빌드
echo "[3/4] Building single .app file..."
pyinstaller --clean qr_analyzer_onefile_mac.spec
echo ""

# 결과 확인
echo "[4/4] Checking build result..."
if [ -d "dist/QR_Analyzer.app" ]; then
    echo ""
    echo "========================================"
    echo "  Build SUCCESSFUL!"
    echo "========================================"
    echo ""
    echo "Location: dist/QR_Analyzer.app"
    echo "Password: 2017112166"
    echo ""
    SIZE=$(du -sh dist/QR_Analyzer.app | cut -f1)
    echo "Size: $SIZE"
    echo ""
    echo "✅ Single .app bundle created!"
    echo "✅ Double-click to run!"
    echo ""
    echo "⚠️  첫 실행 시:"
    echo "   - 압축 해제로 5-10초 소요"
    echo "   - 'App from unidentified developer' 경고 시:"
    echo "     Control+Click → Open"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  Build FAILED!"
    echo "========================================"
    echo ""
    echo "Check the error messages above."
    exit 1
fi

echo "Press Enter to exit..."
read
