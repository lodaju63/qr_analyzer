"""
QR 코드 전처리 모듈
조선소 T-Bar 제작 공정을 위한 QR 코드 전처리 엔진
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from qr_utils import PreprocessingType

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """이미지 전처리기"""
    
    def __init__(self):
        """전처리기 초기화"""
        # CLAHE 설정
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 이진화 설정
        self.binary_block_size = 11
        self.binary_c = 2
    
    def preprocess_image(self, image: np.ndarray, preprocessing_type: PreprocessingType) -> np.ndarray:
        """이미지 전처리"""
        if preprocessing_type == PreprocessingType.ORIGINAL:
            return image.copy()
        elif preprocessing_type == PreprocessingType.CLAHE:
            return self.apply_clahe(image)
        elif preprocessing_type == PreprocessingType.BINARY:
            return self.apply_binary(image)
        elif preprocessing_type == PreprocessingType.PIL_ENHANCE:
            return self.apply_pil_enhance(image)
        elif preprocessing_type == PreprocessingType.CENTER_CROP:
            return self.apply_center_crop(image)
        elif preprocessing_type == PreprocessingType.GAUSSIAN_BLUR:
            return self.apply_gaussian_blur(image)
        else:
            return image.copy()
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
        if len(image.shape) == 3:
            # 컬러 이미지인 경우 LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 그레이스케일 이미지
            return self.clahe.apply(image)
    
    def apply_binary(self, image: np.ndarray) -> np.ndarray:
        """적응적 이진화 적용"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 적응적 이진화
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, self.binary_block_size, self.binary_c
        )
        
        # 컬러 이미지로 변환 (3채널)
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary
    
    def apply_pil_enhance(self, image: np.ndarray) -> np.ndarray:
        """PIL을 사용한 이미지 향상"""
        try:
            from PIL import Image, ImageEnhance
            
            # OpenCV 이미지를 PIL로 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.5)
            
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            # PIL 이미지를 OpenCV로 변환
            if len(image.shape) == 3:
                return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            else:
                return np.array(enhanced)
                
        except ImportError:
            logger.warning("PIL이 설치되지 않았습니다. 원본 이미지를 반환합니다.")
            return image.copy()
    
    def apply_center_crop(self, image: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """중심 크롭 적용"""
        h, w = image.shape[:2]
        
        # 크롭할 크기 계산
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # 중심에서 크롭
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """가우시안 블러 적용"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_rotation(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """이미지 회전"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전된 이미지 생성
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated, center, angle
    
    def apply_scale(self, image: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, float]:
        """이미지 스케일링"""
        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return scaled, scale_factor

class PreprocessingPipeline:
    """전처리 파이프라인"""
    
    def __init__(self, mode: str = "optimized"):
        """
        Args:
            mode: "realtime", "optimized", "full"
        """
        self.mode = mode
        self.preprocessor = ImagePreprocessor()
        
        # 모드별 전처리 타입 설정
        if mode == "realtime":
            self.preprocessing_types = [PreprocessingType.ORIGINAL]
        elif mode == "optimized":
            self.preprocessing_types = [
                PreprocessingType.ORIGINAL,
                PreprocessingType.CLAHE,
                PreprocessingType.BINARY
            ]
        else:  # full
            self.preprocessing_types = [
                PreprocessingType.ORIGINAL,
                PreprocessingType.CLAHE,
                PreprocessingType.BINARY,
                PreprocessingType.PIL_ENHANCE,
                PreprocessingType.CENTER_CROP,
                PreprocessingType.GAUSSIAN_BLUR
            ]
    
    def process_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, PreprocessingType, Optional[float], Optional[float]]]:
        """이미지 전처리 실행"""
        processed_images = []
        
        for prep_type in self.preprocessing_types:
            # 기본 전처리
            processed = self.preprocessor.preprocess_image(image, prep_type)
            processed_images.append((processed, prep_type, None, None))
            
            # 추가 변형 (full 모드에서만)
            if self.mode == "full":
                # 회전 변형
                for angle in [90, 180, 270]:
                    rotated, center, rotation_angle = self.preprocessor.apply_rotation(processed, angle)
                    processed_images.append((rotated, prep_type, None, rotation_angle))
                
                # 스케일 변형
                for scale in [0.8, 1.2]:
                    scaled, scale_factor = self.preprocessor.apply_scale(processed, scale)
                    processed_images.append((scaled, prep_type, scale_factor, None))
        
        return processed_images

# 전역 파이프라인 인스턴스들
realtime_pipeline = PreprocessingPipeline("realtime")
optimized_pipeline = PreprocessingPipeline("optimized")
full_pipeline = PreprocessingPipeline("full")

def get_preprocessing_pipeline(mode: str = "optimized") -> PreprocessingPipeline:
    """전처리 파이프라인 가져오기"""
    if mode == "realtime":
        return realtime_pipeline
    elif mode == "optimized":
        return optimized_pipeline
    else:
        return full_pipeline

def preprocess_for_qr_detection(image: np.ndarray, 
                               preprocessing_types: List[PreprocessingType] = None) -> List[np.ndarray]:
    """QR 탐지를 위한 이미지 전처리"""
    if preprocessing_types is None:
        preprocessing_types = [
            PreprocessingType.ORIGINAL,
            PreprocessingType.CLAHE,
            PreprocessingType.BINARY
        ]
    
    preprocessor = ImagePreprocessor()
    processed_images = []
    
    for prep_type in preprocessing_types:
        processed = preprocessor.preprocess_image(image, prep_type)
        processed_images.append(processed)
    
    return processed_images

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """이미지 품질 향상"""
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced = clahe.apply(image)
    
    # 노이즈 제거
    enhanced = cv2.medianBlur(enhanced, 3)
    
    return enhanced

def create_multiscale_images(image: np.ndarray, scales: List[float] = None) -> List[Tuple[np.ndarray, float]]:
    """다중 스케일 이미지 생성"""
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    multiscale_images = []
    
    for scale in scales:
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        multiscale_images.append((scaled, scale))
    
    return multiscale_images

def create_rotated_images(image: np.ndarray, angles: List[float] = None) -> List[Tuple[np.ndarray, float]]:
    """회전된 이미지 생성"""
    if angles is None:
        angles = [0, 90, 180, 270]
    
    rotated_images = []
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    for angle in angles:
        if angle == 0:
            rotated_images.append((image.copy(), angle))
        else:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            rotated_images.append((rotated, angle))
    
    return rotated_images

