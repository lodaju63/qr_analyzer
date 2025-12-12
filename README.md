# QR Analyzer - PyQt6 Desktop Application

## 🎯 프로젝트 개요

고성능 QR코드 영상 분석 데스크톱 애플리케이션

- **YOLO** 기반 QR 탐지
- **Dynamsoft** 기반 QR 해독
- **PyQt6** GUI
- **실시간 데이터 시각화**

---

## 🚀 빠른 시작

### **실행 파일 다운로드 (권장)**

GitHub Actions에서 빌드된 실행 파일 다운로드:

1. [Releases](https://github.com/lodaju63/qr_analyzer/releases) 페이지 방문
2. 최신 버전 다운로드:
   - Windows: `QR_Analyzer.exe`
   - Mac: `QR_Analyzer.dmg` 또는 `QR_Analyzer.app`
3. 실행 후 로그인:
   - 비밀번호: `2017112166`

---

## 💻 개발 환경 설정

### **1. Python 환경**

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### **2. 의존성 설치**

```bash
# PyTorch CPU 버전 (필수!)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지
pip install -r requirements_pyqt.txt
```

### **3. 실행**

```bash
python main.py
```

---

## 📦 빌드 방법

### **Windows 빌드**

```bash
# 빌드 도구 설치
pip install -r build_requirements.txt

# 빌드 실행
build_onefile.bat
```

결과: `dist\QR_Analyzer.exe` (약 595 MB)

### **Mac 빌드**

```bash
# 빌드 도구 설치
pip install -r build_requirements.txt

# 빌드 실행
chmod +x build_mac_onefile.command
./build_mac_onefile.command
```

결과: `dist/QR_Analyzer.app` (약 600 MB)

---

## 🌐 GitHub Actions 자동 빌드

코드를 push하면 자동으로 Windows + Mac 빌드:

1. 코드 수정
2. `git push`
3. [Actions](https://github.com/lodaju63/qr_analyzer/actions) 페이지에서 진행 상황 확인
4. 15-20분 후 Artifacts 다운로드

---

## 📋 주요 기능

### ✅ **핵심 기능**
- YOLO 모델 업로드 (.pt 파일)
- 영상 분석 (.mp4)
- 실시간 QR 해독
- 데이터 로그 (필터링: 전체/성공/실패)

### ✅ **고급 기능**
- 전처리 옵션 (CLAHE, Blur, Threshold 등)
- 공간 분포 히트맵
- 실시간 분석 그래프
- 프레임 시크바
- 프레임 간격 설정
- 속도 조절

---

## 📂 파일 구조

```
qr_sh/
├── main.py                          # 메인 애플리케이션
├── runtime_hook_ultralytics.py      # PyInstaller 런타임 훅
│
├── .github/workflows/
│   ├── build-mac.yml                # Mac 자동 빌드
│   └── build-all.yml                # Windows + Mac 자동 빌드
│
├── qr_analyzer_onefile.spec         # Windows 빌드 설정
├── qr_analyzer_onefile_mac.spec     # Mac 빌드 설정
│
├── requirements_pyqt.txt            # Python 패키지
├── build_requirements.txt           # 빌드 도구
│
├── build_onefile.bat                # Windows 빌드 스크립트
├── build_mac_onefile.command        # Mac 빌드 스크립트
│
└── 가이드 문서/
    ├── QUICK_START.md               # 빠른 시작 가이드
    ├── GITHUB_ACTIONS_GUIDE.md      # GitHub Actions 가이드
    ├── BUILD_GUIDE_MAC.md           # Mac 빌드 가이드
    └── README_FINAL.md              # 최종 문서
```

---

## 🔑 로그인

```
비밀번호: 2017112166
```

---

## 📊 시스템 요구사항

### **최소**
- OS: Windows 10+ / macOS 10.14+
- RAM: 4GB
- 저장공간: 2GB

### **권장**
- RAM: 8GB+
- GPU: NVIDIA (CUDA 지원)
- 저장공간: 5GB+

---

## 🐛 문제 해결

### **"ultralytics 로드 실패"**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Mac 보안 경고**
```
Control + Click → "열기"
또는
xattr -cr QR_Analyzer.app
```

### **Windows Defender 차단**
```
"자세한 정보" → "실행" 클릭
```

---

## 📝 라이선스

This project uses:
- Ultralytics YOLO (AGPL-3.0)
- Dynamsoft Barcode Reader
- PyQt6
- OpenCV

---

## 📧 문의

Issues: [GitHub Issues](https://github.com/lodaju63/qr_analyzer/issues)

---

**🚀 QR Analyzer - High Performance QR Code Video Analysis Desktop App**
