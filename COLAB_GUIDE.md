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

#### 방법 1: 코랩 파일 업로드 기능 사용
```python
from google.colab import files

# 모델 파일 업로드
uploaded = files.upload()  # model1.pt 선택

# 비디오 파일 업로드
uploaded = files.upload()  # 테스트할 비디오 파일 선택
```

#### 방법 2: Google Drive 사용
```python
from google.colab import drive
drive.mount('/content/drive')

# Drive에서 파일 복사
!cp /content/drive/MyDrive/model1.pt /content/
!cp /content/drive/MyDrive/video.mp4 /content/
```

#### 방법 3: GitHub에서 직접 다운로드
```python
# GitHub 저장소에서 파일 다운로드
!wget https://github.com/your-repo/model1.pt
!wget https://github.com/your-repo/video.mp4
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

### 결과 파일 위치

- **출력 비디오**: `results/YYYYMMDD_HHMMSS/output_YYYYMMDD_HHMMSS.mp4`
- **로그 파일**: `results/YYYYMMDD_HHMMSS/qr_detection_log_YYYYMMDD_HHMMSS.txt`

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

## 📝 참고 사항

1. **처리 시간**: 코랩의 GPU/CPU 성능에 따라 처리 시간이 달라질 수 있습니다.
2. **파일 크기**: 큰 비디오 파일은 업로드/다운로드에 시간이 걸릴 수 있습니다.
3. **세션 시간**: 코랩 세션이 종료되면 파일이 삭제되므로, 중요한 결과는 다운로드하세요.
4. **프리뷰**: `show_preview=True`로 설정하면 일정 간격마다 프레임이 표시됩니다.

## 🎯 빠른 시작 템플릿

```python
# ============================================
# 구글 코랩 QR 탐지 빠른 시작
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

