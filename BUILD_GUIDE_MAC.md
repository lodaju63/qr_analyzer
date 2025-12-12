# QR Analyzer - Mac 빌드 가이드

## 🍎 Mac에서 단일 .app 파일 빌드하기

---

## 📋 **사전 요구사항**

### 1️⃣ **Python 3.10+**
```bash
python3 --version
```

### 2️⃣ **Homebrew (권장)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## 🔧 **설정 방법**

### 1️⃣ **프로젝트 다운로드**
```bash
cd ~/Downloads
# ZIP 파일 압축 해제 또는 git clone
```

### 2️⃣ **가상환경 생성**
```bash
cd qr_sh
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ **의존성 설치**
```bash
# PyTorch CPU 버전 먼저 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
pip install -r requirements_pyqt.txt

# 빌드 도구 설치
pip install -r build_requirements.txt
```

---

## 🚀 **빌드 실행**

### 방법 1: 더블클릭 (간편) ⭐
```bash
# 실행 권한 부여 (최초 1회)
chmod +x build_mac_onefile.command

# Finder에서 더블클릭!
```

### 방법 2: 터미널에서 실행
```bash
./build_mac_onefile.command
```

---

## 📦 **빌드 결과**

```
dist/
└── QR_Analyzer.app  ← 단일 .app 번들!
```

**크기:** 약 600MB

---

## ✅ **실행 방법**

### 1️⃣ **더블클릭**
```
Finder에서 QR_Analyzer.app 더블클릭!
```

### 2️⃣ **보안 경고 시**
macOS가 "unidentified developer" 경고를 표시하면:

```
1. Control + Click (또는 우클릭)
2. "열기" 선택
3. "열기" 확인
```

또는:

```
시스템 환경설정 → 보안 및 개인 정보 보호
→ "확인 없이 열기" 클릭
```

### 3️⃣ **로그인**
```
비밀번호: 2017112166
```

---

## 🎯 **배포 방법**

### 단일 .app 파일 배포
```bash
# .app 파일을 DMG로 패키징 (선택사항)
hdiutil create -volname "QR Analyzer" -srcfolder dist/QR_Analyzer.app -ov -format UDZO QR_Analyzer.dmg
```

### 사용자에게 전달
```
1. QR_Analyzer.app 또는 QR_Analyzer.dmg 전송
2. 사용자는 Applications 폴더로 드래그
3. 더블클릭 실행!
```

---

## ⚠️ **주의사항**

### 1️⃣ **첫 실행 시간**
- 압축 해제로 5-10초 소요
- 임시 폴더에 자동 압축 해제

### 2️⃣ **Gatekeeper 경고**
- 개발자 서명이 없어 경고 표시
- Control+Click → 열기로 해결

### 3️⃣ **Code Signing (선택사항)**
프로덕션 배포 시 서명 추천:
```bash
# Apple Developer 계정 필요
codesign --force --deep --sign "Developer ID Application: Your Name" dist/QR_Analyzer.app
```

---

## 🐛 **문제 해결**

### "command not found: pyinstaller"
```bash
source venv/bin/activate
pip install pyinstaller
```

### "No module named 'PyQt6'"
```bash
pip install -r requirements_pyqt.txt
```

### ".app이 손상되었습니다"
```bash
# Quarantine 속성 제거
xattr -cr dist/QR_Analyzer.app
```

### "ultralytics 로드 실패"
- `main.py`에 `NullWriter` 클래스가 포함되어 있는지 확인
- `runtime_hook_ultralytics.py` 파일이 있는지 확인

---

## 📊 **빌드 구조**

```
QR_Analyzer.app/
├── Contents/
│   ├── MacOS/
│   │   └── QR_Analyzer  ← 실행 파일
│   ├── Resources/
│   └── Info.plist
```

---

## 🎉 **완료!**

Mac용 단일 .app 파일이 완성되었습니다!

```
dist/QR_Analyzer.app
- 크기: 600MB
- 비밀번호: 2017112166
- 단일 파일 배포
- 더블클릭 실행
```

---

## 📧 **문의**

문제가 있으면 이슈를 남겨주세요!
