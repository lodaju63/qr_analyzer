# 이미지 전용 QR 코드 탐지 시스템 (img.py)

## 📋 개요

이미지만을 처리하는 전용 Streamlit 애플리케이션입니다. YOLO + U-Net/SegFormer + Transformer 방식을 지원하며, 다양한 전처리 옵션을 제공합니다.

## 🚀 주요 기능

### 1. 모델 지원
- ✅ **YOLO**: QR 코드 위치 탐지
- ✅ **Dynamsoft**: QR 코드 해독 (기본)
- 🔄 **Transformer Decoder**: QR 코드 해독 (딥러닝, 플레이스홀더)
- 🔄 **U-Net/SegFormer**: 이미지 복원/향상 (플레이스홀더)

### 2. 전처리 옵션

#### 1️⃣ 노이즈 제거
- **Bilateral Filter** (추천): 엣지 보존하면서 노이즈 제거
- **Gaussian Blur**: 일반 잡음 제거
- **Median Filter**: Salt-pepper 노이즈 제거

#### 2️⃣ 명암/조명 보정
- **CLAHE**: 지역적 명암비 조절 (금속 반사/그림자 환경 최적)
- **Gamma Correction**: 어두운 구역 밝게
- **Retinex (MSR)**: 복잡한 조명 조건에서 자연스러운 보정

#### 3️⃣ 반전 처리
- QR 코드가 어두운 배경 + 밝은 코드 형태일 때 반전

#### 4️⃣ 이진화
- **Adaptive Thresholding (Gaussian)**: 조도 불균일 환경에 적합
- **Adaptive Thresholding (Mean)**: 평균 기반 적응형 임계값
- **Otsu**: 균일 조명 환경에 빠르고 효과적

#### 5️⃣ 형태학적 연산
- **Closing**: 끊어진 QR 패턴 연결
- **Opening**: 노이즈 제거
- **Dilation**: 희미한 finder-pattern 보강

#### 6️⃣ Super Resolution
- 작은 QR 코드를 1.5~4배 확대

#### 7️⃣ Deblurring
- Lucy-Richardson 방식 (간단한 구현)
- 흔들린 영상, 이동 중 촬영에 유용

#### 8️⃣ 기하학적 보정
- **Rotation Correction**: 회전 각도 보정
- **Perspective Transform**: 비스듬한 QR을 정면으로 보정

### 3. UI 기능
- ✅ 원본 이미지와 전처리된 이미지를 나란히 표시
- ✅ 각 이미지의 탐지 및 해독 결과 비교
- ✅ 실시간 전처리 옵션 조정
- ✅ 결과 통계 (탐지 수, 해독 성공률)

## 📦 설치 및 실행

### 필요한 패키지
```bash
pip install streamlit opencv-python numpy pillow ultralytics
pip install dynamsoft-barcode-reader-bundle  # 선택적
```

### 실행 방법
```bash
streamlit run img.py
```

## 🎯 사용 방법

### 1. 모델 초기화
1. 사이드바에서 **"YOLO 모델 로드"** 버튼 클릭
2. **"Dynamsoft 초기화"** 버튼 클릭 (선택적)
3. Transformer 모델은 아직 구현되지 않음 (플레이스홀더)

### 2. 이미지 업로드
- 메인 영역에서 이미지 파일 업로드 (jpg, jpeg, png, bmp)

### 3. 전처리 옵션 설정
- 사이드바에서 원하는 전처리 옵션 활성화 및 파라미터 조정

### 4. 처리 실행
- **"🔄 이미지 처리 시작"** 버튼 클릭

### 5. 결과 확인
- 원본 이미지와 전처리된 이미지가 나란히 표시됨
- 각각의 탐지 및 해독 결과를 비교 가능

## ⚙️ 전처리 파이프라인 순서

1. 그레이스케일 변환 (선택)
2. 노이즈 제거
3. 명암/조명 보정
4. 반전 처리
5. Super Resolution
6. Deblurring
7. 이진화
8. 형태학적 연산
9. 기하학적 보정

## 🔧 추천 설정 (조선소 T-bar 환경)

```python
# 최적 파이프라인
1. Bilateral Filter (d=9, sigma_color=75, sigma_space=75)
2. CLAHE (clipLimit=2.0, tileSize=8x8)
3. Gamma Correction (gamma=1.0-1.2)
4. Adaptive Thresholding (Gaussian, blockSize=11, C=2)
5. Morphology (Closing → Opening)
```

## 📝 TODO (향후 구현)

- [ ] U-Net 모델 통합
- [ ] SegFormer 모델 통합
- [ ] Transformer Decoder 모델 통합
- [ ] Inpainting UI 추가 (마스크 선택)
- [ ] Perspective Transform 자동 탐지
- [ ] 다중 전처리 파이프라인 비교
- [ ] 결과 저장 기능

## 🐛 알려진 제한사항

1. **Transformer Decoder**: 현재 플레이스홀더만 구현됨
2. **U-Net/SegFormer**: 모델 로딩 및 추론 로직 미구현
3. **Inpainting**: 마스크 입력 UI 없음
4. **Perspective Transform**: 자동 QR finder pattern 탐지 미구현

## 💡 팁

- **처음 사용**: 기본 설정(노이즈 제거 + CLAHE)으로 시작
- **손상된 이미지**: Super Resolution + Deblurring 조합 시도
- **조도 불균일**: Adaptive Thresholding 필수
- **작은 QR**: Super Resolution로 확대 후 처리

## 📚 참고 자료

- [IMAGE_PROCESSING_COMPARISON.md](./IMAGE_PROCESSING_COMPARISON.md): 두 방식 비교 분석
- [yolo_dynamsoft.py](./yolo_dynamsoft.py): 기본 전처리 함수들

