# QR Analyzer - 최종 배포 가이드

## 🎯 **프로젝트 개요**

고성능 QR코드 영상 분석 데스크톱 애플리케이션
- YOLO 기반 QR 탐지
- Dynamsoft 기반 QR 해독
- PyQt6 GUI
- 실시간 데이터 시각화

---

## 📦 **배포 버전**

### ✅ **Windows - 단일 exe 파일**
```
dist\QR_Analyzer.exe (595 MB)
- 콘솔창 없음
- exe 하나만 배포
- 더블클릭 실행
```

### ✅ **Mac - 단일 .app 파일**
```
dist/QR_Analyzer.app (600 MB)
- .app 번들
- 더블클릭 실행
- Applications 폴더로 드래그
```

---

## 🚀 **빌드 방법**

### **Windows**
```cmd
build_onefile.bat
```

### **Mac**
```bash
chmod +x build_mac_onefile.command
./build_mac_onefile.command
```

---

## 🔑 **로그인 정보**

```
비밀번호: 2017112166
```

---

## 📋 **주요 기능**

### ✅ **핵심**
- YOLO 모델 업로드
- 영상 분석 (.mp4)
- QR 실시간 해독
- 데이터 로그 (필터링)

### ✅ **고급**
- 전처리 옵션 (CLAHE, Blur, Threshold 등)
- 히트맵 (QR 위치 분포)
- 실시간 그래프
- 프레임 시크바
- 프레임 간격 설정

---

## 🎯 **사용 방법**

### 1️⃣ **실행**
- Windows: `QR_Analyzer.exe` 더블클릭
- Mac: `QR_Analyzer.app` 더블클릭

### 2️⃣ **로그인**
```
비밀번호 입력: 2017112166
```

### 3️⃣ **분석**
```
1. 모델 업로드 (.pt 파일)
2. 영상 업로드 (.mp4 파일)
3. 시작 버튼 클릭
4. 실시간 분석!
```

---

## 📂 **파일 구조**

```
qr_sh/
├── main.py                          # 메인 애플리케이션
├── runtime_hook_ultralytics.py      # PyInstaller 런타임 훅
│
├── qr_analyzer_onefile.spec         # Windows 빌드 설정
├── qr_analyzer_onefile_mac.spec     # Mac 빌드 설정
│
├── build_onefile.bat                # Windows 빌드
├── build_mac_onefile.command        # Mac 빌드
│
├── requirements_pyqt.txt            # Python 패키지
├── build_requirements.txt           # 빌드 도구
│
├── BUILD_GUIDE_MAC.md               # Mac 빌드 가이드
├── README_RELEASE.md                # 배포 가이드
└── README_FINAL.md                  # 이 파일
```

---

## 🔧 **개발자용**

### **개발 환경 실행**
```bash
# 가상환경 활성화
source venv/bin/activate  # Mac/Linux
.\venv\Scripts\activate   # Windows

# 직접 실행
python main.py
```

### **디버그 빌드**
```bash
# .spec 파일에서 console=True로 변경
# Windows
pyinstaller --clean qr_analyzer_onefile.spec

# Mac
pyinstaller --clean qr_analyzer_onefile_mac.spec
```

---

## ⚠️ **중요 사항**

### **NullWriter (필수!)**
`main.py` 시작 부분에 반드시 포함:
```python
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass
    @property
    def encoding(self):
        return "utf-8"

if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()
```

이것이 없으면 `console=False` 모드에서 YOLO 로드 실패!

### **Runtime Hook**
`runtime_hook_ultralytics.py`가 경로 설정을 담당:
- `TORCH_HOME`
- `YOLO_CONFIG_DIR`
- `ULTRALYTICS_CONFIG_DIR`

---

## 🐛 **문제 해결**

### **"ultralytics 로드 실패"**
```python
# NullWriter 클래스가 main.py 맨 위에 있는지 확인!
```

### **Windows Defender 차단**
```
"자세한 정보" → "실행" 클릭
또는 예외 목록에 추가
```

### **Mac Gatekeeper 경고**
```
Control + Click → "열기"
또는
xattr -cr QR_Analyzer.app
```

---

## 📊 **시스템 요구사항**

### **최소**
- OS: Windows 10+ / macOS 10.14+
- RAM: 4GB
- 저장공간: 2GB

### **권장**
- RAM: 8GB+
- GPU: NVIDIA (CUDA)
- 저장공간: 5GB+

---

## 🎉 **완료!**

**단일 파일 배포 버전 완성!**

### **Windows**
```
dist\QR_Analyzer.exe
→ 595MB 단일 파일
→ 더블클릭 실행!
```

### **Mac**
```
dist/QR_Analyzer.app
→ 600MB 단일 앱
→ 더블클릭 실행!
```

---

## 📧 **문의**

문제가 있거나 기능 요청이 있으면 이슈를 남겨주세요!

---

**🚀 QR Analyzer v1.0 - Ready to Ship! 🎊**
