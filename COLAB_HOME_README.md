# 📱 QR 탐지 시스템 - 코랩 실행 가이드

코랩에서 `Home_colab.py`를 실행하여 멀티페이지 QR 탐지 시스템을 사용하는 방법입니다.

## 📋 필요한 파일

### 1. 메인 파일
- `Home_colab.py` - 메인 홈페이지 (진입점)

### 2. pages 폴더의 파일들 (필수)
다음 파일들이 `pages/` 폴더에 있어야 합니다:
- `pages/1__비디오_QR_탐지.py` - 비디오 QR 탐지 페이지
- `pages/2__이미지_QR_탐지.py` - 이미지 QR 탐지 페이지
- `pages/3__프레임_추출.py` - 프레임 추출 페이지

### 3. 기타 필요한 파일들
- `model1.pt` - YOLO 모델 파일 (각 페이지에서 사용)
- `yolo_dynamsoft.py` - 비디오 페이지에서 import (필요 시)

### 4. 선택적 파일들
- 데이터 파일들 (이미지, 비디오 등) - 각 페이지에서 업로드 가능

## 🚀 코랩에서 실행 방법

### 1단계: 필요한 패키지 설치

```python
# 필수 패키지 설치
!pip install -q streamlit ultralytics opencv-python numpy pillow
!pip install -q dynamsoft-barcode-reader-bundle

# 한글 폰트 설치 (선택사항)
!apt-get -qq install -y fonts-nanum
```

### 2단계: 파일 구조 설정

#### 방법 A: GitHub에서 직접 다운로드 (권장)

```python
import os

# pages 폴더 생성
os.makedirs('pages', exist_ok=True)

# 메인 파일 다운로드
!wget -q -O Home_colab.py https://raw.githubusercontent.com/[사용자명]/[저장소명]/[브랜치명]/Home_colab.py

# pages 폴더의 파일들 다운로드
!wget -q -O pages/1__비디오_QR_탐지.py https://raw.githubusercontent.com/[사용자명]/[저장소명]/[브랜치명]/pages/1__비디오_QR_탐지.py
!wget -q -O pages/2__이미지_QR_탐지.py https://raw.githubusercontent.com/[사용자명]/[저장소명]/[브랜치명]/pages/2__이미지_QR_탐지.py
!wget -q -O pages/3__프레임_추출.py https://raw.githubusercontent.com/[사용자명]/[저장소명]/[브랜치명]/pages/3__프레임_추출.py

# yolo_dynamsoft.py (비디오 페이지에서 필요)
!wget -q -O yolo_dynamsoft.py https://raw.githubusercontent.com/[사용자명]/[저장소명]/[브랜치명]/yolo_dynamsoft.py
```

#### 방법 B: Google Drive 사용

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# Drive에서 파일 복사
!cp -r /content/drive/MyDrive/qr_sh/* /content/
!cp -r /content/drive/MyDrive/qr_sh/pages /content/

# 또는 특정 파일만 복사
!cp /content/drive/MyDrive/qr_sh/Home_colab.py /content/
!cp /content/drive/MyDrive/qr_sh/pages/* /content/pages/
```

#### 방법 C: 수동 업로드

```python
from google.colab import files

# 메인 파일 업로드
print("📁 Home_colab.py 업로드:")
files.upload()

# pages 폴더의 파일들 업로드
print("📁 pages 폴더의 파일들 업로드:")
files.upload()  # 여러 파일 선택 가능
```

### 3단계: 모델 파일 준비

```python
# 모델 파일이 Drive에 있는 경우
!cp /content/drive/MyDrive/model1.pt /content/

# 또는 직접 업로드
from google.colab import files
print("📁 model1.pt 업로드:")
files.upload()
```

### 4단계: Streamlit 실행

```python
# Streamlit 실행 (백그라운드)
import subprocess
import threading

def run_streamlit():
    subprocess.run([
        'streamlit', 'run', 'Home_colab.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true'
    ])

# 백그라운드에서 실행
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

print("⏳ Streamlit 서버 시작 중... (5초 대기)")
import time
time.sleep(5)

# 터널링 URL 생성
from google.colab import output
output.serve_kernel_port_as_window(8501)
```

또는 간단하게:

```bash
!streamlit run Home_colab.py --server.port 8501 --server.address 0.0.0.0
```

그리고 코랩에서 제공하는 터널링 링크를 사용하세요.

## 📁 완전한 파일 구조

코랩에서 다음과 같은 구조가 필요합니다:

```
/content/
├── Home_colab.py              # 메인 파일
├── model1.pt                  # YOLO 모델 (필수)
├── yolo_dynamsoft.py          # 비디오 페이지에서 사용 (필수)
├── pages/
│   ├── 1__비디오_QR_탐지.py  # 비디오 페이지
│   ├── 2__이미지_QR_탐지.py  # 이미지 페이지
│   └── 3__프레임_추출.py      # 프레임 추출 페이지
└── data/                      # 데이터 파일들 (선택적)
    ├── *.mp4                  # 비디오 파일
    ├── *.jpg                  # 이미지 파일
    └── ...
```

## ⚠️ 주의사항

### 1. 런타임 재시작 시 파일 사라짐
- 코랩 런타임을 재시작하면 `/content`의 **모든 파일이 사라집니다**
- **해결책**: Google Drive에 파일을 저장하고 마운트해서 사용

### 2. Google Drive 사용 권장

```python
# 1. Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. 파일을 Drive에 저장 (최초 1회)
# - 웹에서 직접 업로드하거나
# - 코랩에서 업로드 후 복사

# 3. 런타임 재시작 후에도 이렇게 사용
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/qr_sh/* /content/
```

### 3. 필수 파일 확인

```python
import os

# 필수 파일 확인
required_files = [
    'Home_colab.py',
    'model1.pt',
    'yolo_dynamsoft.py',
    'pages/1__비디오_QR_탐지.py',
    'pages/2__이미지_QR_탐지.py',
    'pages/3__프레임_추출.py'
]

print("📋 필수 파일 확인:")
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - 없음!")
```

## 🔧 트러블슈팅

### 문제 1: pages 폴더를 찾을 수 없음

```python
# pages 폴더 생성
import os
os.makedirs('pages', exist_ok=True)
print("✅ pages 폴더 생성됨")
```

### 문제 2: 페이지 파일을 찾을 수 없음

```python
# pages 폴더 내용 확인
import os
if os.path.exists('pages'):
    print("pages 폴더 내용:")
    print(os.listdir('pages'))
else:
    print("❌ pages 폴더가 없습니다!")
```

### 문제 3: 모델 파일을 찾을 수 없음

```python
# 모델 파일 경로 확인
import os
possible_paths = [
    '/content/model1.pt',
    '/content/drive/MyDrive/model1.pt',
    './model1.pt'
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ 모델 파일 발견: {path}")
        break
else:
    print("❌ 모델 파일을 찾을 수 없습니다!")
```

## 📝 실행 예시 (전체 코드)

```python
# ===== 1. 패키지 설치 =====
!pip install -q streamlit ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle

# ===== 2. Google Drive 마운트 =====
from google.colab import drive
drive.mount('/content/drive')

# ===== 3. 파일 복사 (Drive에 있는 경우) =====
!cp -r /content/drive/MyDrive/qr_sh/* /content/
!mkdir -p /content/pages
!cp /content/drive/MyDrive/qr_sh/pages/* /content/pages/

# ===== 4. 파일 확인 =====
import os
print("📁 현재 디렉토리:", os.listdir('.'))
if os.path.exists('pages'):
    print("📁 pages 폴더:", os.listdir('pages'))

# ===== 5. Streamlit 실행 =====
!streamlit run Home_colab.py --server.port 8501 --server.address 0.0.0.0
```

## 💡 팁

1. **한 번에 실행**: 위의 코드를 하나의 셀에 모아두면 런타임 재시작 후에도 쉽게 실행 가능
2. **Drive 백업**: 중요한 설정과 파일은 Drive에 저장
3. **필수 파일 체크**: 실행 전에 필수 파일이 있는지 확인
4. **에러 확인**: 페이지가 로드되지 않으면 터미널 출력 확인

