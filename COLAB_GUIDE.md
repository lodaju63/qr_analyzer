# 구글 코랩에서 yolo_dynamsoft.py 실행 가이드

## 📋 준비 사항

### 1. 필요한 패키지 설치

코랩 노트북의 첫 번째 셀에 다음을 실행하세요:

```python
# 필수 패키지 설치
!pip install ultralytics opencv-python numpy pillow
!pip install dynamsoft-barcode-reader-bundle

# 한글 폰트 설치 (선택사항)
!apt-get -qq install -y fonts-nanum
```

### 2. 파일 업로드

**⚠️ 중요**: 코랩에서 **런타임을 재시작하면** (GPU 설정 변경 포함) `/content` 디렉토리의 **모든 파일이 사라집니다!**

#### 방법 1: Google Drive 사용 (권장) ⭐

**런타임 재시작 후에도 파일이 유지됩니다:**

```python
# 1. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. Drive에 파일 업로드 (최초 1회만)
# - Google Drive 웹에서 직접 업로드하거나
# - 코랩에서 업로드:
from google.colab import files
uploaded = files.upload()  # model1.pt, video.mp4 등 업로드
# 업로드 후 Drive로 복사
!cp model1.pt /content/drive/MyDrive/
!cp sample_video3-1.mp4 /content/drive/MyDrive/

# 3. 런타임 재시작 후에도 이렇게 사용:
from google.colab import drive
drive.mount('/content/drive')

# Drive에서 파일 복사
!cp /content/drive/MyDrive/model1.pt /content/
!cp /content/drive/MyDrive/sample_video3-1.mp4 /content/
```

#### 방법 2: 코랩 파일 업로드 (임시 사용)

**런타임 재시작 시 파일이 사라지므로 매번 다시 업로드해야 합니다:**

```python
from google.colab import files

# 모델 파일 업로드
print("📁 model1.pt 업로드:")
uploaded = files.upload()  # model1.pt 선택

# 비디오 파일 업로드
print("📹 비디오 파일 업로드:")
uploaded = files.upload()  # video.mp4 선택
```

#### 방법 3: GitHub에서 직접 다운로드

**코드 파일은 GitHub에서 가져오는 것이 편리합니다:**

```python
# 코드 파일 다운로드
!wget https://raw.githubusercontent.com/lodaju63/md/feat/dynamsoft/yolo_dynamsoft_colab.py

# 모델/비디오 파일은 GitHub에 올려두고 다운로드 (또는 Drive 사용)
```

### 3. 코드 파일 업로드

```python
# yolo_dynamsoft_colab.py 파일을 코랩에 업로드
from google.colab import files
uploaded = files.upload()  # yolo_dynamsoft_colab.py 선택
```

또는 GitHub에서 직접 가져오기:
```python
!wget https://raw.githubusercontent.com/lodaju63/md/feat/dynamsoft/yolo_dynamsoft_colab.py
```

## 🚀 실행 방법

### 기본 사용법

```python
# 모듈 import
from yolo_dynamsoft_colab import video_player_with_qr

# 비디오 파일 경로 설정
video_path = 'video.mp4'  # 업로드한 비디오 파일명

# 실행
video_player_with_qr(
    video_path=video_path,
    output_dir='results',
    show_preview=True,      # 프리뷰 표시 여부
    preview_interval=30      # 프리뷰 표시 간격 (프레임)
)
```

### Dynamsoft 라이선스 키 설정 (선택사항)

```python
import os

# 환경 변수로 라이선스 키 설정
os.environ['DYNAMSOFT_LICENSE_KEY'] = 'your_license_key_here'
```

## 📊 실행 예시

### 전체 예시 코드

```python
# 1. 패키지 설치
!pip install ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle
!apt-get -qq install -y fonts-nanum

# 2. 파일 업로드
from google.colab import files
print("모델 파일 업로드:")
uploaded = files.upload()  # model1.pt 선택
print("\n비디오 파일 업로드:")
uploaded = files.upload()  # video.mp4 선택

# 3. 코드 파일 가져오기
!wget https://raw.githubusercontent.com/lodaju63/md/feat/dynamsoft/yolo_dynamsoft_colab.py

# 4. 실행
from yolo_dynamsoft_colab import video_player_with_qr

video_player_with_qr(
    video_path='video.mp4',
    output_dir='results',
    show_preview=True,
    preview_interval=30
)
```

## 📁 결과 확인

### 작업 디렉토리 찾기

**코랩에서 작업하는 폴더는 `/content` 입니다!**

파일 브라우저에서:
1. 왼쪽 사이드바의 **📁 폴더 아이콘** 클릭
2. `content` 폴더 클릭 (bin, boot, datalab 등과 같은 레벨에 있음)
3. 여기가 작업 디렉토리입니다!

코드로 확인:
```python
import os

# 현재 작업 디렉토리 확인
print("현재 디렉토리:", os.getcwd())

# content 폴더의 파일 목록 확인
print("\n/content 폴더 내용:")
print(os.listdir('/content'))
```

### 결과 파일 위치

- **출력 비디오**: `/content/results/YYYYMMDD_HHMMSS/output_YYYYMMDD_HHMMSS.mp4`
- **로그 파일**: `/content/results/YYYYMMDD_HHMMSS/qr_detection_log_YYYYMMDD_HHMMSS.txt`

### 결과 다운로드

```python
from google.colab import files

# 결과 폴더 전체 다운로드
!zip -r results.zip results/
files.download('results.zip')

# 또는 개별 파일 다운로드
files.download('results/20250101_120000/output_20250101_120000.mp4')
```

## ⚙️ 주요 차이점 (로컬 vs 코랩)

| 기능 | 로컬 버전 | 코랩 버전 |
|------|----------|----------|
| 화면 표시 | `cv2.imshow()` | `matplotlib` / `IPython.display` |
| 키보드 입력 | ESC, SPACE 키 지원 | 미지원 (자동 실행) |
| 파일 경로 | 로컬 파일 시스템 | 코랩 파일 시스템 |
| 결과 확인 | 로컬 파일 탐색기 | 코랩 파일 브라우저 |

## 🔧 문제 해결

### 1. 모델 파일을 찾을 수 없음
```python
# 현재 디렉토리 확인
import os
print(os.listdir('.'))

# 모델 파일 경로 확인
if os.path.exists('model1.pt'):
    print("✅ 모델 파일 존재")
else:
    print("❌ 모델 파일 없음 - 업로드 필요")
```

### 2. 비디오 파일을 열 수 없음
```python
# 비디오 파일 확인
import cv2
cap = cv2.VideoCapture('video.mp4')
if cap.isOpened():
    print("✅ 비디오 파일 열기 성공")
    cap.release()
else:
    print("❌ 비디오 파일 열기 실패")
```

### 3. 메모리 부족
```python
# GPU 메모리 확인
!nvidia-smi

# 메모리 정리
import gc
gc.collect()
```

### 4. Dynamsoft 라이선스 오류
- 라이선스 키가 올바른지 확인
- 환경 변수 설정 확인
- Dynamsoft 패키지 재설치

### 5. 런타임 재시작 후 파일이 사라짐 ⚠️

**문제**: GPU 설정 변경 등으로 런타임을 재시작하면:
- `/content`의 **모든 파일**이 사라집니다
- **설치한 패키지**도 사라집니다 (`pip install`, `apt-get` 등)
- **모든 것을 다시 설치/업로드**해야 합니다

**해결 방법**:

#### 방법 A: Google Drive 사용 (권장)

```python
# 1. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. Drive에서 파일 복사
!cp /content/drive/MyDrive/model1.pt /content/
!cp /content/drive/MyDrive/sample_video3-1.mp4 /content/

# 3. 파일 확인
import os
print("현재 파일:", os.listdir('.'))
```

#### 방법 B: 파일 다시 업로드

```python
from google.colab import files

# 파일 다시 업로드
print("📁 model1.pt 업로드:")
files.upload()

print("📹 비디오 파일 업로드:")
files.upload()
```

#### 방법 C: 코드 파일은 GitHub에서 자동 다운로드

```python
# ⚠️ 런타임 재시작 후에는 모든 것을 다시 설치해야 합니다!

# 1. 패키지 재설치 (필수!)
!pip install -q ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle
!apt-get -qq install -y fonts-nanum

# 2. 코드 파일은 GitHub에서 자동으로 가져오기
!wget -q https://raw.githubusercontent.com/lodaju63/md/feat/dynamsoft/yolo_dynamsoft_colab.py

# 3. 파일 업로드 또는 Drive에서 복사
# (위의 방법 A 또는 B 참고)
```

**💡 팁**: 런타임 재시작 후 실행할 전체 코드를 하나의 셀에 모아두면 편리합니다!

## 📝 참고 사항

1. **처리 시간**: 코랩의 GPU/CPU 성능에 따라 처리 시간이 달라질 수 있습니다.
2. **파일 크기**: 큰 비디오 파일은 업로드/다운로드에 시간이 걸릴 수 있습니다.
3. **세션 시간**: 코랩 세션이 종료되면 파일이 삭제되므로, 중요한 결과는 다운로드하세요.
4. **프리뷰**: `show_preview=True`로 설정하면 일정 간격마다 프레임이 표시됩니다.

## ⚡ 성능 최적화 팁

### 1. GPU 사용 확인 및 설정

코랩에서 GPU를 사용하면 **훨씬 빠르게** 처리할 수 있습니다:

```python
# GPU 런타임 설정
# 메뉴: 런타임 > 런타임 유형 변경 > 하드웨어 가속기: GPU 선택

# ⚠️ 주의: GPU 설정 변경 시 런타임이 재시작되며 파일이 사라집니다!
# 해결: Google Drive 사용 (위의 "방법 1: Google Drive 사용" 참고)

# GPU 사용 여부 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU를 사용하려면: 런타임 > 런타임 유형 변경 > GPU 선택")
```

**⚠️ 중요**: 
- GPU 런타임이 설정되지 않으면 CPU로 실행되어 **로컬보다 느릴 수 있습니다!**
- **GPU 설정 변경 시 런타임이 재시작되며 `/content`의 모든 파일이 사라집니다!**
- **해결책**: Google Drive에 파일을 저장하고 마운트해서 사용하세요.

### 2. 프리뷰 비활성화 (최대 성능)

프리뷰를 끄면 **10-20% 더 빠르게** 실행됩니다:

```python
video_player_with_qr(
    video_path='sample_video3-1.mp4',
    output_dir='results',
    show_preview=False,  # 프리뷰 끄기 (성능 향상)
    verbose_log=False   # 상세 로그 끄기 (성능 향상)
)
```

### 3. 프리뷰 간격 늘리기

프리뷰를 보면서 실행하려면 간격을 늘리세요:

```python
video_player_with_qr(
    video_path='sample_video3-1.mp4',
    output_dir='results',
    show_preview=True,
    preview_interval=100,  # 100프레임마다 표시 (기본값: 30)
    verbose_log=False
)
```

### 4. 성능 비교

| 설정 | 예상 처리 시간 (1000프레임 기준) |
|------|--------------------------------|
| GPU + 프리뷰 OFF | **30-40초** (가장 빠름) |
| GPU + 프리뷰 ON (간격 100) | 40-50초 |
| GPU + 프리뷰 ON (간격 30) | 50-60초 |
| CPU + 프리뷰 OFF | 80-100초 (느림) |
| CPU + 프리뷰 ON | 100-120초 (가장 느림) |

### 5. 성능 문제 해결

**코랩이 로컬보다 느린 경우:**

1. **GPU 런타임 확인**: 런타임 > 런타임 유형 변경 > GPU 선택
2. **프리뷰 끄기**: `show_preview=False`
3. **로그 최소화**: `verbose_log=False`
4. **파일 I/O 최소화**: 작은 비디오 파일 사용

## 🎯 빠른 시작 템플릿

### 템플릿 A: Google Drive 사용 (권장 - 런타임 재시작 후에도 유지)

```python
# ============================================
# 구글 코랩 QR 탐지 빠른 시작 (Google Drive 사용)
# ============================================

# 1. 패키지 설치
!pip install -q ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle
!apt-get -qq install -y fonts-nanum

# 2. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 3. 코드 다운로드
!wget -q https://raw.githubusercontent.com/lodaju63/md/feat/dynamsoft/yolo_dynamsoft_colab.py

# 4. Drive에서 파일 복사 (최초 1회만 Drive에 업로드 필요)
#    - Google Drive 웹에서 직접 업로드하거나
#    - 코랩에서 업로드 후 Drive로 복사:
#      from google.colab import files
#      files.upload()  # model1.pt, video.mp4 업로드
#      !cp model1.pt /content/drive/MyDrive/
#      !cp sample_video3-1.mp4 /content/drive/MyDrive/

!cp /content/drive/MyDrive/model1.pt /content/
!cp /content/drive/MyDrive/sample_video3-1.mp4 /content/

# 5. 실행
from yolo_dynamsoft_colab import video_player_with_qr

video_player_with_qr(
    video_path='sample_video3-1.mp4',
    output_dir='results',
    show_preview=True,
    preview_interval=30
)
```

### 템플릿 B: 직접 업로드 (간단하지만 런타임 재시작 시 다시 업로드 필요)

```python
# ============================================
# 구글 코랩 QR 탐지 빠른 시작 (직접 업로드)
# ============================================

# 1. 패키지 설치
!pip install -q ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle
!apt-get -qq install -y fonts-nanum

# 2. 코드 다운로드
!wget -q https://raw.githubusercontent.com/lodaju63/md/feat/dynamsoft/yolo_dynamsoft_colab.py

# 3. 파일 업로드 (수동)
from google.colab import files
print("📁 model1.pt 업로드:")
files.upload()
print("\n📹 비디오 파일 업로드:")
files.upload()

# 4. 실행
from yolo_dynamsoft_colab import video_player_with_qr

video_player_with_qr(
    video_path='your_video.mp4',  # 업로드한 비디오 파일명
    output_dir='results',
    show_preview=True,
    preview_interval=30
)

# 5. 결과 다운로드
!zip -r results.zip results/
files.download('results.zip')
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
- 로그 파일 확인: `results/YYYYMMDD_HHMMSS/qr_detection_log_*.txt`
- 패키지 버전 확인: `!pip list | grep -E "ultralytics|opencv|dynamsoft"`
- 에러 메시지 전체 확인

