# 📋 코랩에서 Home_colab.py 실행 시 필요한 파일 목록

## ✅ 필수 파일들

### 1. 메인 파일
- ✅ `Home_colab.py` - 메인 홈페이지 (진입점)

### 2. pages 폴더의 파일들 (필수)
- ✅ `pages/1__비디오_QR_탐지.py` - 비디오 QR 탐지 페이지
- ✅ `pages/2__이미지_QR_탐지.py` - 이미지 QR 탐지 페이지  
- ✅ `pages/3__프레임_추출.py` - 프레임 추출 페이지

### 3. 모델 및 라이브러리 파일들
- ✅ `model1.pt` - YOLO 모델 파일 (모든 페이지에서 사용)
- ✅ `yolo_dynamsoft.py` - 비디오 페이지에서 import (필수)

### 4. 선택적 파일들 (기능에 따라)
- 📁 데이터 파일들:
  - 이미지 파일: `*.jpg`, `*.png`, `*.bmp`
  - 비디오 파일: `*.mp4`, `*.avi`, `*.mov` 등

## 📦 설치해야 할 패키지

```python
!pip install streamlit ultralytics opencv-python numpy pillow
!pip install dynamsoft-barcode-reader-bundle
```

## 📁 코랩에서의 권장 파일 구조

```
/content/
├── Home_colab.py              ✅ 메인 파일
├── model1.pt                  ✅ YOLO 모델 (필수)
├── yolo_dynamsoft.py          ✅ 비디오 페이지에서 사용 (필수)
├── pages/                     ✅ pages 폴더
│   ├── 1__비디오_QR_탐지.py  ✅ 비디오 페이지
│   ├── 2__이미지_QR_탐지.py  ✅ 이미지 페이지
│   └── 3__프레임_추출.py      ✅ 프레임 추출 페이지
└── data/                      📁 선택적 데이터 폴더
    ├── *.mp4
    ├── *.jpg
    └── ...
```

## 🚀 빠른 설정 (한 번에 실행)

```python
# 1. 패키지 설치
!pip install -q streamlit ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle

# 2. Google Drive 마운트 (파일이 Drive에 있는 경우)
from google.colab import drive
drive.mount('/content/drive')

# 3. 파일 구조 확인
import os
print("현재 디렉토리:", os.listdir('.'))

# 4. 필수 파일 확인
required = [
    'Home_colab.py',
    'model1.pt',
    'yolo_dynamsoft.py',
    'pages/1__비디오_QR_탐지.py',
    'pages/2__이미지_QR_탐지.py',
    'pages/3__프레임_추출.py'
]

print("\n필수 파일 확인:")
for f in required:
    exists = os.path.exists(f)
    print(f"{'✅' if exists else '❌'} {f}")

# 5. Streamlit 실행
!streamlit run Home_colab.py --server.port 8501 --server.address 0.0.0.0
```

## ⚠️ 중요 참고사항

1. **pages 폴더 필수**: `pages/` 폴더와 그 안의 3개 파일이 반드시 있어야 합니다.
2. **yolo_dynamsoft.py 필수**: 비디오 페이지가 이 파일을 import합니다.
3. **model1.pt 필수**: 각 페이지에서 YOLO 모델로 사용됩니다.
4. **경로 설정**: 모든 파일은 `/content/` 디렉토리 또는 하위에 있어야 합니다.

## 📝 파일별 역할

| 파일 | 용도 | 필수 여부 |
|------|------|----------|
| `Home_colab.py` | 메인 홈페이지, 페이지 네비게이션 | ✅ 필수 |
| `pages/1__비디오_QR_탐지.py` | 비디오 QR 탐지 기능 | ✅ 필수 |
| `pages/2__이미지_QR_탐지.py` | 이미지 QR 탐지 기능 | ✅ 필수 |
| `pages/3__프레임_추출.py` | 비디오 프레임 추출 기능 | ✅ 필수 |
| `model1.pt` | YOLO 모델 파일 | ✅ 필수 |
| `yolo_dynamsoft.py` | 비디오 처리 함수들 | ✅ 필수 |
| 데이터 파일들 | 처리할 이미지/비디오 | 📁 선택적 |

