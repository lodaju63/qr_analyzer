"""
이미지 전용 QR 코드 탐지 시스템 (Streamlit)
YOLO + U-Net/SegFormer + Transformer 방식
다양한 전처리 옵션 제공
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import warnings
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import logging
import io
import zipfile

warnings.filterwarnings('ignore')

# Streamlit 로거 레벨 조정
logging.getLogger('streamlit').setLevel(logging.CRITICAL)

# 페이지 설정
st.set_page_config(
    page_title="이미지 QR 탐지 시스템",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 1. 전처리 함수들
# ============================================================================

def denoise_gaussian(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian Blur로 일반 잡음 제거"""
    if len(image.shape) == 3:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def denoise_median(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median Filter로 salt-pepper 노이즈 제거"""
    if len(image.shape) == 3:
        return cv2.medianBlur(image, kernel_size)
    else:
        return cv2.medianBlur(image, kernel_size)

def denoise_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral Filter - 엣지를 보존하면서 노이즈 제거"""
    if len(image.shape) == 3:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    else:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def enhance_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

def enhance_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma Correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_retinex(image: np.ndarray, sigma_list: List[float] = [15, 80, 250]) -> np.ndarray:
    """Multi-Scale Retinex (MSR)"""
    # 간단한 구현 (SSR)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    retinex = np.zeros_like(gray)
    for sigma in sigma_list:
        gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
        retinex += np.log10(gray + 1) - np.log10(gaussian + 1)
    
    retinex = retinex / len(sigma_list)
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    retinex = retinex.astype(np.uint8)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(retinex, cv2.COLOR_GRAY2BGR)
    return retinex

def invert_image(image: np.ndarray) -> np.ndarray:
    """이미지 반전"""
    return cv2.bitwise_not(image)

def adaptive_threshold_gaussian(image: np.ndarray, max_value: int = 255, 
                                block_size: int = 11, c: int = 2) -> np.ndarray:
    """Gaussian Adaptive Thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, block_size, c)

def adaptive_threshold_mean(image: np.ndarray, max_value: int = 255, 
                            block_size: int = 11, c: int = 2) -> np.ndarray:
    """Mean Adaptive Thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, block_size, c)

def threshold_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu Thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def morphology_closing(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Closing (팽창 → 침식) - 끊어진 패턴 연결"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def morphology_opening(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Opening (침식 → 팽창) - 노이즈 제거"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def morphology_dilation(image: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Dilation - 패턴 보강"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=iterations)

def correct_perspective(image: np.ndarray, src_points: Optional[np.ndarray] = None) -> np.ndarray:
    """Perspective Transform - QR 코드를 정면으로 보정"""
    h, w = image.shape[:2]
    
    if src_points is None:
        # 자동으로 QR finder pattern을 찾아서 보정하는 로직
        # 여기서는 간단히 원본 반환 (실제로는 QR 탐지 후 finder pattern 기반 변환)
        return image
    
    # 목적지 포인트 (정사각형)
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # Homography 계산
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    
    # Perspective 변환
    return cv2.warpPerspective(image, M, (w, h))

def correct_rotation(image: np.ndarray, angle: float = 0) -> np.ndarray:
    """Rotation Correction"""
    if angle == 0:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def super_resolution(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Super Resolution - 간단한 업스케일링 (실제로는 SRGAN 등을 사용)"""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def deblur_lucy_richardson(image: np.ndarray, iterations: int = 30, sigma: float = 1.5) -> np.ndarray:
    """Lucy-Richardson Deblurring (간단한 구현)"""
    # 실제로는 더 복잡한 구현이 필요
    # 여기서는 간단한 unsharp masking
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)
    return unsharp

def inpainting_telea(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Telea Inpainting"""
    if mask is None:
        return image
    
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def inpainting_ns(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Navier-Stokes Inpainting"""
    if mask is None:
        return image
    
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

# ============================================================================
# 2. 통합 전처리 파이프라인
# ============================================================================

def apply_preprocessing(image: np.ndarray, options: Dict) -> np.ndarray:
    """전처리 옵션에 따라 이미지 처리"""
    result = image.copy()
    
    # 1. 그레이스케일 변환 (필요시)
    if options.get('convert_grayscale', False):
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # 2. 노이즈 제거
    if options.get('denoise_enabled', False):
        denoise_method = options.get('denoise_method', 'bilateral')
        if denoise_method == 'gaussian':
            result = denoise_gaussian(result, options.get('gaussian_kernel', 5))
        elif denoise_method == 'median':
            result = denoise_median(result, options.get('median_kernel', 5))
        elif denoise_method == 'bilateral':
            result = denoise_bilateral(result, 
                                      options.get('bilateral_d', 9),
                                      options.get('bilateral_sigma_color', 75),
                                      options.get('bilateral_sigma_space', 75))
    
    # 3. 명암/조명 보정
    if options.get('enhancement_enabled', False):
        enhancement_methods = options.get('enhancement_methods', [])
        if 'clahe' in enhancement_methods:
            result = enhance_clahe(result, 
                                  options.get('clahe_clip_limit', 2.0),
                                  tuple(options.get('clahe_tile_size', [8, 8])))
        if 'gamma' in enhancement_methods:
            result = enhance_gamma(result, options.get('gamma_value', 1.0))
        if 'retinex' in enhancement_methods:
            result = enhance_retinex(result, options.get('retinex_sigma', [15, 80, 250]))
    
    # 4. 반전 처리
    if options.get('invert', False):
        result = invert_image(result)
    
    # 5. Super Resolution
    if options.get('super_resolution_enabled', False):
        result = super_resolution(result, options.get('sr_scale', 2.0))
    
    # 6. Deblurring
    if options.get('deblur_enabled', False):
        result = deblur_lucy_richardson(result, 
                                       options.get('deblur_iterations', 30),
                                       options.get('deblur_sigma', 1.5))
    
    # 7. 이진화
    if options.get('binarization_enabled', False):
        binarization_method = options.get('binarization_method', 'adaptive_gaussian')
        if binarization_method == 'adaptive_gaussian':
            result = adaptive_threshold_gaussian(result,
                                               options.get('adaptive_max_value', 255),
                                               options.get('adaptive_block_size', 11),
                                               options.get('adaptive_c', 2))
        elif binarization_method == 'adaptive_mean':
            result = adaptive_threshold_mean(result,
                                           options.get('adaptive_max_value', 255),
                                           options.get('adaptive_block_size', 11),
                                           options.get('adaptive_c', 2))
        elif binarization_method == 'otsu':
            result = threshold_otsu(result)
        
        # 이진화 후 BGR로 변환
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # 8. 형태학적 연산
    if options.get('morphology_enabled', False):
        morphology_ops = options.get('morphology_operations', [])
        kernel_size = options.get('morphology_kernel_size', 5)
        
        # 이진화된 이미지에서만 형태학적 연산
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result.copy()
        
        for op in morphology_ops:
            if op == 'closing':
                gray = morphology_closing(gray, kernel_size)
            elif op == 'opening':
                gray = morphology_opening(gray, kernel_size)
            elif op == 'dilation':
                gray = morphology_dilation(gray, kernel_size, 
                                         options.get('dilation_iterations', 1))
        
        if len(image.shape) == 3:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            result = gray
    
    # 9. 기하학적 보정
    if options.get('geometric_enabled', False):
        if options.get('rotation_angle', 0) != 0:
            result = correct_rotation(result, options.get('rotation_angle', 0))
        if options.get('perspective_correction', False):
            result = correct_perspective(result)
    
    return result

# ============================================================================
# 3. 딥러닝 모델 통합 (플레이스홀더)
# ============================================================================

def load_unet_model(model_path: Optional[str] = None):
    """U-Net 모델 로드 (플레이스홀더)"""
    # TODO: 실제 U-Net 모델 로드
    return None

def load_segformer_model(model_path: Optional[str] = None):
    """SegFormer 모델 로드 (플레이스홀더)"""
    # TODO: 실제 SegFormer 모델 로드
    return None

def enhance_with_unet(image: np.ndarray, model) -> np.ndarray:
    """U-Net으로 이미지 복원/향상"""
    if model is None:
        return image
    # TODO: U-Net 추론 로직
    return image

def enhance_with_segformer(image: np.ndarray, model) -> np.ndarray:
    """SegFormer로 이미지 복원/향상"""
    if model is None:
        return image
    # TODO: SegFormer 추론 로직
    return image

def load_transformer_decoder(model_path: Optional[str] = None):
    """Transformer Decoder 모델 로드 (플레이스홀더)"""
    # TODO: 실제 Transformer Decoder 모델 로드
    return None

def decode_with_transformer(image: np.ndarray, model) -> str:
    """Transformer Decoder로 QR 코드 해독"""
    if model is None:
        return ""
    # TODO: Transformer 추론 로직
    return ""

# ============================================================================
# 4. YOLO 탐지 및 해독
# ============================================================================

def detect_qr_with_yolo(image: np.ndarray, yolo_model, conf_threshold: float = 0.25) -> List[Dict]:
    """YOLO로 QR 코드 위치 탐지 (OBB 모델 지원)"""
    try:
        from ultralytics import YOLO
        results = yolo_model(image, conf=conf_threshold, verbose=False)
        result = results[0]
        
        detections = []
        h, w = image.shape[:2]
        
        # OBB 모델 처리 (우선순위 1)
        if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
            for i in range(len(result.obb)):
                try:
                    conf = float(result.obb.conf[i])
                    
                    # OBB의 xyxyxyxy 속성 사용 (4개 점 좌표 - 회전된 박스)
                    if hasattr(result.obb, 'xyxyxyxy') and result.obb.xyxyxyxy is not None and len(result.obb.xyxyxyxy) > i:
                        # OBB의 4개 점 좌표 가져오기 (GPU -> CPU -> Numpy)
                        obb_points = result.obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                        
                        # 경계 체크
                        obb_points[:, 0] = np.clip(obb_points[:, 0], 0, w)
                        obb_points[:, 1] = np.clip(obb_points[:, 1], 0, h)
                        
                        # axis-aligned 바운딩 박스 계산 (기존 호환성 유지)
                        x1 = int(np.min(obb_points[:, 0]))
                        y1 = int(np.min(obb_points[:, 1]))
                        x2 = int(np.max(obb_points[:, 0]))
                        y2 = int(np.max(obb_points[:, 1]))
                        
                        # 패딩 추가 (QR 코드 경계 확보)
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'obb_points': obb_points.tolist()  # OBB 좌표 저장 (시각화용)
                        })
                    # xyxy 속성 사용 (fallback)
                    elif hasattr(result.obb, 'xyxy') and result.obb.xyxy is not None and len(result.obb.xyxy) > i:
                        xyxy = result.obb.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # 패딩 추가 (QR 코드 경계 확보)
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                except Exception as e:
                    continue
        
        # 일반 detection 모델 처리 (우선순위 2)
        elif result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 패딩 추가 (QR 코드 경계 확보)
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        return detections
    except Exception as e:
        st.error(f"YOLO 탐지 오류: {e}")
        return []

def decode_qr_with_dynamsoft(image: np.ndarray, dbr_reader) -> Tuple[str, Optional[np.ndarray]]:
    """Dynamsoft로 QR 코드 해독"""
    try:
        from dynamsoft_barcode_reader_bundle import dbr as dbr_module
        
        if dbr_reader is None:
            return "", None
        
        # RGB 변환
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        captured_result = dbr_reader.capture(rgb_image, dbr_module.EnumImagePixelFormat.IPF_RGB_888)
        barcode_result = captured_result.get_decoded_barcodes_result()
        
        if barcode_result:
            items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
            if items and len(items) > 0:
                barcode_item = items[0]
                text = None
                if hasattr(barcode_item, 'get_text'):
                    text = barcode_item.get_text()
                elif hasattr(barcode_item, 'text'):
                    text = barcode_item.text
                
                # Quad 포인트 추출
                quad_xy = None
                try:
                    location = barcode_item.get_location() if hasattr(barcode_item, 'get_location') else None
                    if location:
                        result_points = location.result_points if hasattr(location, 'result_points') else None
                        if result_points:
                            quad_xy = [[int(p.x), int(p.y)] for p in result_points]
                except:
                    pass
                
                return text or "", quad_xy
        
        return "", None
    except Exception as e:
        st.warning(f"Dynamsoft 해독 오류: {e}")
        return "", None

# ============================================================================
# 5. 시각화 함수
# ============================================================================

def visualize_qr_results(image: np.ndarray, decodings: List[Dict]) -> np.ndarray:
    """QR 코드 탐지 결과를 이미지에 시각화 (OBB 모델 지원)"""
    display_image = image.copy()
    
    # 그레이스케일인 경우 BGR로 변환
    if len(display_image.shape) == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    
    for i, dec in enumerate(decodings):
        bbox = dec.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if dec.get('success') else (0, 0, 255)
        points = None
        
        # OBB 모델 좌표 우선 사용
        if 'obb_points' in dec and dec['obb_points'] and len(dec['obb_points']) == 4:
            obb_points = np.array(dec['obb_points'], dtype=np.int32)
            cv2.polylines(display_image, [obb_points.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
            points = obb_points
        
        # Quad 사용 (fallback)
        elif dec.get('quad') and len(dec['quad']) == 4:
            quad = np.array(dec['quad'])
            if len(quad) == 4:
                quad_array = np.array(quad)
                center = np.mean(quad_array, axis=0)
                angles = np.arctan2(quad_array[:, 1] - center[1], 
                                  quad_array[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                sorted_quad = quad_array[sorted_indices]
                cv2.polylines(display_image, [sorted_quad], True, color, 2)
                points = sorted_quad
        
        # bbox 사용 (최종 fallback)
        else:
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        # QR 번호 표시 (points가 있는 경우)
        if points is not None and len(points) > 0:
            track_id_text = f"#{i}"
            text_pos = (int(points[0][0]), int(points[0][1]) - 10)
            cv2.putText(display_image, track_id_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return display_image

# ============================================================================
# 6. 메인 처리 함수
# ============================================================================

def process_image(image: np.ndarray, preprocessing_options: Dict, 
                 yolo_model, dbr_reader, transformer_model,
                 use_deep_learning: bool = False) -> Dict:
    """이미지 처리 및 해독"""
    
    results = {
        'original_image': image.copy(),
        'original_image_visualized': None,  # 시각화된 원본
        'preprocessed_image': None,  # 전처리만 (시각화 안됨)
        'preprocessed_image_visualized': None,  # 전처리 + 시각화
        'original_detections': [],
        'preprocessed_detections': [],
        'original_decodings': [],
        'preprocessed_decodings': []
    }
    
    # YOLO 신뢰도 임계값
    conf_threshold = preprocessing_options.get('conf_threshold', 0.25)
    
    # 1. 원본 이미지 처리
    original_detections = detect_qr_with_yolo(image, yolo_model, conf_threshold)
    results['original_detections'] = original_detections
    
    original_decodings = []
    for det in original_detections:
        x1, y1, x2, y2 = det['bbox']
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:
            text, quad = decode_qr_with_dynamsoft(roi, dbr_reader)
            original_decodings.append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'text': text,
                'quad': quad,
                'success': len(text) > 0
            })
    results['original_decodings'] = original_decodings
    
    # 원본 이미지 시각화
    results['original_image_visualized'] = visualize_qr_results(
        results['original_image'], original_decodings
    )
    
    # 2. 딥러닝 향상 (선택적)
    enhanced_image = image.copy()
    if use_deep_learning:
        # U-Net 또는 SegFormer로 향상
        # TODO: 실제 모델 통합
        pass
    
    # 3. 전처리 적용
    preprocessed_image = apply_preprocessing(enhanced_image, preprocessing_options)
    results['preprocessed_image'] = preprocessed_image
    
    # 4. 전처리된 이미지 처리
    preprocessed_detections = detect_qr_with_yolo(preprocessed_image, yolo_model, conf_threshold)
    results['preprocessed_detections'] = preprocessed_detections
    
    preprocessed_decodings = []
    for det in preprocessed_detections:
        x1, y1, x2, y2 = det['bbox']
        roi = preprocessed_image[y1:y2, x1:x2]
        if roi.size > 0:
            # Dynamsoft 해독
            text, quad = decode_qr_with_dynamsoft(roi, dbr_reader)
            
            # Transformer 해독 (실패 시)
            if not text and transformer_model:
                text = decode_with_transformer(roi, transformer_model)
            
            preprocessed_decodings.append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'text': text,
                'quad': quad,
                'success': len(text) > 0,
                'obb_points': det.get('obb_points', None)  # OBB 모델 좌표 (시각화용)
            })
    results['preprocessed_decodings'] = preprocessed_decodings
    
    # 전처리 이미지 시각화
    if preprocessed_image is not None:
        results['preprocessed_image_visualized'] = visualize_qr_results(
            preprocessed_image, preprocessed_decodings
        )
    
    return results

# ============================================================================
# 7. Streamlit UI
# ============================================================================

def main():
    st.title("🖼️ 이미지 QR 코드 탐지 시스템")
    st.markdown("YOLO + U-Net/SegFormer + Transformer 방식")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'yolo_model' not in st.session_state:
        st.session_state.yolo_model = None
    if 'dbr_reader' not in st.session_state:
        st.session_state.dbr_reader = None
    if 'transformer_model' not in st.session_state:
        st.session_state.transformer_model = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # 사이드바 - 모델 초기화
    with st.sidebar:
        st.header("🔧 모델 초기화")
        
        # YOLO 모델
        if st.button("YOLO 모델 로드", width='stretch'):
            try:
                from ultralytics import YOLO
                model_path = os.environ.get('YOLO_MODEL_PATH', 'best.pt')  # 기본값: best.pt (OBB 모델)
                if os.path.exists(model_path):
                    with st.spinner("YOLO 모델 로딩 중..."):
                        st.session_state.yolo_model = YOLO(model_path)
                    st.success("✅ YOLO 모델 로드 완료")
                else:
                    st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            except Exception as e:
                st.error(f"YOLO 모델 로드 실패: {e}")
        
        # Dynamsoft 초기화
        if st.button("Dynamsoft 초기화", width='stretch'):
            try:
                from dynamsoft_barcode_reader_bundle import dbr, license, cvr
                license_key = os.environ.get('DYNAMSOFT_LICENSE_KEY', 
                                            't0085YQEAADYdcL2llMa8vH1Rtnun+43saE/kdAE7ZbIxMQGRMtSzVSZRI8vfOK4Ids52rjekwzh87yABFLraXw5Va1BV7NnBjI8m7qbw3kxOprI75ExJpw==')
                error = license.LicenseManager.init_license(license_key)
                if error[0] == 0:
                    st.session_state.dbr_reader = cvr.CaptureVisionRouter()
                    st.success("✅ Dynamsoft 초기화 완료")
                else:
                    st.warning(f"라이선스 오류: {error[1]}")
            except Exception as e:
                st.error(f"Dynamsoft 초기화 실패: {e}")
        
        # Transformer 모델 (선택적)
        st.info("💡 **Transformer 모델**: Dynamsoft 해독 실패 시 폴백으로 사용됩니다.")
        if st.button("Transformer 모델 로드 (선택적)", width='stretch'):
            st.info("⚠️ Transformer 모델은 아직 구현되지 않았습니다.")
            st.caption("💡 언제 로드하나요?")
            st.caption("  • Dynamsoft 해독 실패율이 높은 경우")
            st.caption("  • 심각하게 손상된 이미지가 많을 때")
            st.caption("  • 라이선스 비용 절감이 필요한 경우")
            st.caption("  • 자세한 내용: TRANSFORMER_USAGE_GUIDE.md 참고")
        
        st.markdown("---")
        
        # YOLO 탐지 설정
        st.header("🎯 YOLO 탐지 설정")
        conf_threshold = st.slider(
            "신뢰도 임계값 (Confidence Threshold)", 
            0.0, 1.0, 0.25, 0.01,
            help="YOLO가 QR 코드를 탐지하는 최소 신뢰도입니다.\n"
                 "• 값이 높으면: 더 확실한 QR만 탐지, 오탐지 감소, 하지만 일부 QR 놓칠 수 있음\n"
                 "• 값이 낮으면: 더 많은 QR 탐지 가능, 하지만 오탐지 증가\n"
                 "• 권장값: 0.25 (일반), 0.1 (약한 QR), 0.5 (정확도 우선)"
        )
        
        st.markdown("---")
        
        # 전처리 옵션 설정
        st.header("⚙️ 전처리 옵션")
        
        # 그레이스케일 변환
        convert_grayscale = st.checkbox(
            "그레이스케일 변환", 
            value=False,
            help="이미지를 흑백(그레이스케일)로 변환합니다.\n"
                 "• 장점: 처리 속도 향상, 일부 전처리 방법에 유리\n"
                 "• 단점: 색상 정보 손실\n"
                 "• 사용 시기: 색상이 중요하지 않은 경우"
        )
        
        # 1. 노이즈 제거
        st.subheader("1️⃣ 노이즈 제거")
        st.caption("철판의 녹, 스크래치, 먼지, 스파크 등 잡음을 제거합니다. QR 패턴은 보존합니다.")
        
        denoise_enabled = st.checkbox(
            "노이즈 제거 활성화", 
            value=False,
            help="QR 코드 탐지에 방해되는 노이즈를 제거합니다."
        )
        denoise_method = "bilateral"
        gaussian_kernel = 5
        median_kernel = 5
        bilateral_d = 9
        bilateral_sigma_color = 75
        bilateral_sigma_space = 75
        if denoise_enabled:
            denoise_method = st.selectbox(
                "노이즈 제거 방법", 
                ["bilateral", "gaussian", "median"],
                index=0,
                help="• Bilateral (추천): 엣지 보존하며 노이즈 제거 - QR 패턴 유지 최적\n"
                     "• Gaussian: 부드러운 블러 - 일반적인 잡음 제거\n"
                     "• Median: Salt-pepper 노이즈(작은 점들) 제거에 효과적"
            )
            if denoise_method == "gaussian":
                gaussian_kernel = st.slider(
                    "Gaussian Kernel Size", 
                    3, 15, 5, 2,
                    help="가우시안 블러 커널 크기 (홀수만 가능: 3, 5, 7, 9, 11, 13, 15)\n"
                         "• 값이 작으면: 약한 블러, 세부 정보 보존, 노이즈 일부 남음\n"
                         "• 값이 크면: 강한 블러, 노이즈 제거 강함, 하지만 QR 패턴이 흐려질 수 있음\n"
                         "• 권장값: 5 (일반), 3 (선명한 QR), 7-9 (노이즈 많음)"
                )
            elif denoise_method == "median":
                median_kernel = st.slider(
                    "Median Kernel Size", 
                    3, 15, 5, 2,
                    help="미디안 필터 커널 크기 (홀수만 가능)\n"
                         "• 값이 작으면: 작은 노이즈만 제거\n"
                         "• 값이 크면: 큰 노이즈도 제거하지만 QR 패턴 손상 가능\n"
                         "• 권장값: 5 (일반), 3 (작은 노이즈), 7 (큰 노이즈)"
                )
            elif denoise_method == "bilateral":
                bilateral_d = st.slider(
                    "Bilateral d (필터링 거리)", 
                    5, 15, 9, 2,
                    help="각 픽셀 주변에서 고려할 이웃 픽셀의 거리 (홀수만 가능)\n"
                         "• 값이 작으면: 작은 영역만 처리, 세밀한 노이즈 제거\n"
                         "• 값이 크면: 넓은 영역 처리, 큰 패턴 노이즈 제거\n"
                         "• 권장값: 9 (일반), 5 (세밀하게), 13 (큰 노이즈)"
                )
                bilateral_sigma_color = st.slider(
                    "Sigma Color (색상 차이 허용)", 
                    50, 150, 75, 5,
                    help="색상 값의 차이 허용 범위\n"
                         "• 값이 작으면: 비슷한 색상만 필터링, 엣지 보존 강함\n"
                         "• 값이 크면: 다른 색상도 평활화, 노이즈 제거 강함\n"
                         "• 권장값: 75 (일반), 50 (엣지 강조), 100-150 (강한 노이즈 제거)"
                )
                bilateral_sigma_space = st.slider(
                    "Sigma Space (공간 거리)", 
                    50, 150, 75, 5,
                    help="공간적 거리의 영향 범위\n"
                         "• 값이 작으면: 가까운 픽셀만 영향, 세밀한 처리\n"
                         "• 값이 크면: 멀리 있는 픽셀도 영향, 부드러운 처리\n"
                         "• 권장값: 75 (일반), 50 (세밀하게), 100-150 (부드럽게)"
                )
        
        # 2. 명암/조명 보정
        st.subheader("2️⃣ 명암/조명 보정")
        st.caption("금속 반사, 그림자, 불균일한 조명 등으로 인한 대비 문제를 해결합니다.")
        
        enhancement_enabled = st.checkbox(
            "명암 보정 활성화", 
            value=False,
            help="어두운 부분을 밝게, 밝은 부분을 조절하여 QR 코드 대비를 개선합니다."
        )
        enhancement_methods = []
        clahe_clip_limit = 2.0
        clahe_tile_size = [8, 8]
        gamma_value = 1.0
        retinex_sigma = "15,80,250"
        if enhancement_enabled:
            enhancement_methods = st.multiselect(
                "보정 방법 선택",
                ["clahe", "gamma", "retinex"],
                default=["clahe"],
                help="• CLAHE (추천): 지역별 명암 조절 - 금속 반사/그림자 환경에 최적\n"
                     "• Gamma: 전체적인 밝기 조절 - 어두운 이미지에 유리\n"
                     "• Retinex: 복잡한 조명 조건 보정 - 자연스러운 밝기 조절"
            )
            if "clahe" in enhancement_methods:
                clahe_clip_limit = st.slider(
                    "CLAHE Clip Limit", 
                    1.0, 5.0, 2.0, 0.1,
                    help="대비 제한 값 - 히스토그램 평활화 강도 조절\n"
                         "• 값이 낮으면 (1.0-2.0): 약한 대비 개선, 오탐지 감소, 자연스러움\n"
                         "• 값이 높으면 (3.0-5.0): 강한 대비 개선, QR 패턴 명확해짐, 하지만 오탐지 증가 가능\n"
                         "• 권장값: 2.0 (일반), 1.0-1.5 (오탐지 많을 때), 3.0-4.0 (대비 낮은 이미지)"
                )
                tile_size_val = st.slider(
                    "CLAHE Tile Size", 
                    4, 16, 8, 2,
                    help="지역 처리 타일 크기 (픽셀)\n"
                         "• 값이 작으면: 작은 영역별 처리, 세밀한 조절, 느림\n"
                         "• 값이 크면: 큰 영역별 처리, 빠름, 하지만 세밀함 떨어짐\n"
                         "• 권장값: 8 (일반), 4-6 (세밀하게), 12-16 (빠르게)"
                )
                clahe_tile_size = [tile_size_val, tile_size_val]
            if "gamma" in enhancement_methods:
                gamma_value = st.slider(
                    "Gamma 값", 
                    0.1, 3.0, 1.0, 0.1,
                    help="감마 보정 값 - 밝기 곡선 조절\n"
                         "• 값 < 1.0 (예: 0.5): 어두운 부분 밝게, 밝은 부분은 덜 변화\n"
                         "• 값 = 1.0: 변화 없음\n"
                         "• 값 > 1.0 (예: 1.5-2.0): 어두운 부분 더 어둡게, 밝은 부분 강조\n"
                         "• 권장값: 1.0 (기본), 0.5-0.8 (어두운 이미지), 1.2-1.5 (과다 노출 이미지)"
                )
            if "retinex" in enhancement_methods:
                retinex_sigma = st.text_input(
                    "Retinex Sigma (쉼표로 구분)", 
                    "15,80,250",
                    help="다중 스케일 Retinex의 시그마 값들 (작은값, 중간값, 큰값)\n"
                         "• 작은 값 (15): 작은 세부사항 보정\n"
                         "• 중간 값 (80): 중간 크기 조명 변화 보정\n"
                         "• 큰 값 (250): 큰 영역 조명 변화 보정\n"
                         "• 기본값: 15,80,250 (대부분의 경우 적합)"
                )
        
        # 3. 반전
        st.subheader("3️⃣ 반전 처리")
        st.caption("QR 코드가 어두운 배경에 밝은 코드 형태일 때 사용합니다.")
        
        invert = st.checkbox(
            "이미지 반전", 
            value=False,
            help="이미지 색상을 반전시킵니다 (검은색 ↔ 흰색)\n"
                 "• 사용 시기: QR 코드가 음화(네거티브) 형태일 때\n"
                 "• 효과: 밝은 배경 + 어두운 QR → 어두운 배경 + 밝은 QR로 변환"
        )
        
        # 4. 이진화
        st.subheader("4️⃣ 이진화")
        st.caption("이미지를 검은색/흰색 두 가지로만 나누어 QR 패턴을 명확하게 만듭니다.")
        
        binarization_enabled = st.checkbox(
            "이진화 활성화", 
            value=False,
            help="그레이스케일 이미지를 흑백(0 또는 255)으로 변환합니다.\n"
                 "• 장점: QR 패턴이 명확해짐, 해독률 향상\n"
                 "• 단점: 색상 정보 완전 손실\n"
                 "• 사용 시기: 조도 불균일, 대비 낮은 이미지"
        )
        binarization_method = "adaptive_gaussian"
        adaptive_block_size = 11
        adaptive_c = 2
        if binarization_enabled:
            binarization_method = st.selectbox(
                "이진화 방법",
                ["adaptive_gaussian", "adaptive_mean", "otsu"],
                index=0,
                help="• Adaptive Gaussian (추천): 조도 불균일 환경에 최적 - 지역별 임계값 계산\n"
                     "• Adaptive Mean: 평균 기반 적응형 - Gaussian보다 빠르지만 덜 정확\n"
                     "• Otsu: 전체 이미지 최적 임계값 - 조도 균일할 때 빠르고 효과적"
            )
            if "adaptive" in binarization_method:
                adaptive_block_size = st.slider(
                    "Block Size (지역 크기)", 
                    3, 21, 11, 2,
                    help="임계값을 계산할 지역의 크기 (홀수만 가능: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)\n"
                         "• 값이 작으면: 작은 영역별 처리, 세밀함, 노이즈에 민감\n"
                         "• 값이 크면: 큰 영역별 처리, 안정적, 하지만 세밀함 떨어짐\n"
                         "• 권장값: 11 (일반), 5-7 (세밀하게), 15-21 (큰 QR, 안정적)"
                )
                adaptive_c = st.slider(
                    "C 값 (상수)", 
                    -10, 10, 2, 1,
                    help="임계값에서 빼는 상수 값 - 밝기 조절\n"
                         "• 값이 높으면 (양수): 더 밝은 픽셀도 흰색으로, QR 패턴 두껍게\n"
                         "• 값이 낮으면 (음수): 더 어두운 픽셀도 검은색으로, QR 패턴 얇게\n"
                         "• 권장값: 2 (일반), 0-5 (밝은 QR), -5-0 (어두운 QR)"
                )
        
        # 5. 형태학적 연산
        st.subheader("5️⃣ 형태학적 연산")
        st.caption("QR 코드의 패턴을 연결하거나 노이즈를 제거하여 구조를 강화합니다.")
        
        morphology_enabled = st.checkbox(
            "형태학적 연산 활성화", 
            value=False,
            help="이진화된 이미지에서 QR 패턴의 구조를 개선합니다.\n"
                 "• Closing: 끊어진 선 연결\n"
                 "• Opening: 작은 노이즈 제거\n"
                 "• Dilation: 패턴 두껍게"
        )
        morphology_operations = []
        morphology_kernel_size = 5
        dilation_iterations = 1
        if morphology_enabled:
            morphology_operations = st.multiselect(
                "연산 선택",
                ["closing", "opening", "dilation"],
                default=["closing", "opening"],
                help="• Closing (팽창→침식): 끊어진 QR 패턴의 선을 연결합니다\n"
                     "• Opening (침식→팽창): QR 주변의 작은 노이즈를 제거합니다\n"
                     "• Dilation (팽창): 희미한 QR 패턴을 두껍게 만듭니다\n"
                     "• 권장 순서: Closing → Opening (일반), 또는 Dilation만 (희미한 QR)"
            )
            morphology_kernel_size = st.slider(
                "Kernel Size (커널 크기)", 
                3, 15, 5, 2,
                help="형태학적 연산에 사용할 커널(필터) 크기 (홀수만 가능)\n"
                     "• 값이 작으면: 작은 변화만 처리, 세밀함 유지\n"
                     "• 값이 크면: 큰 변화 처리, 패턴 크게 변화\n"
                     "• 권장값: 5 (일반), 3 (세밀하게), 7-9 (큰 변화)"
            )
            if "dilation" in morphology_operations:
                dilation_iterations = st.slider(
                    "Dilation Iterations (반복 횟수)", 
                    1, 5, 1,
                    help="Dilation을 반복할 횟수\n"
                         "• 값이 높으면: 더 두껍게, 희미한 QR도 강화\n"
                         "• 값이 낮으면: 약하게, 원본 유지\n"
                         "• 권장값: 1 (일반), 2-3 (희미한 QR)"
                )
        
        # 6. Super Resolution
        st.subheader("6️⃣ Super Resolution")
        st.caption("작거나 흐릿한 QR 코드를 확대하여 해독률을 향상시킵니다.")
        
        super_resolution_enabled = st.checkbox(
            "Super Resolution 활성화", 
            value=False,
            help="이미지의 해상도를 높입니다.\n"
                 "• 사용 시기: QR 코드가 작거나 멀리 있을 때\n"
                 "• 효과: QR 모듈(작은 정사각형)의 경계가 명확해짐"
        )
        sr_scale = 2.0
        if super_resolution_enabled:
            sr_scale = st.slider(
                "업스케일 비율", 
                1.5, 4.0, 2.0, 0.5,
                help="이미지를 몇 배로 확대할지 설정\n"
                     "• 값이 작으면 (1.5-2.0): 약간 확대, 처리 속도 빠름\n"
                     "• 값이 크면 (3.0-4.0): 많이 확대, QR 크게 보이지만 처리 느림\n"
                     "• 권장값: 2.0 (일반), 1.5 (약간 작은 QR), 3.0-4.0 (매우 작은 QR)"
            )
        
        # 7. Deblurring
        st.subheader("7️⃣ Deblurring")
        st.caption("흔들림이나 이동으로 인해 흐릿해진 QR 코드를 선명하게 만듭니다.")
        
        deblur_enabled = st.checkbox(
            "Deblurring 활성화", 
            value=False,
            help="블러(흐림) 현상을 제거하여 이미지를 선명하게 만듭니다.\n"
                 "• 사용 시기: 카메라 흔들림, 빠른 이동 중 촬영, 초점이 맞지 않을 때"
        )
        deblur_iterations = 30
        deblur_sigma = 1.5
        if deblur_enabled:
            deblur_iterations = st.slider(
                "Iterations (반복 횟수)", 
                10, 100, 30, 10,
                help="디블러링 알고리즘 반복 횟수\n"
                     "• 값이 낮으면: 빠른 처리, 약한 디블러\n"
                     "• 값이 높으면: 느린 처리, 강한 디블러, 하지만 과도하면 인공물 생성\n"
                     "• 권장값: 30 (일반), 10-20 (약간 흐림), 50-100 (심하게 흐림)"
            )
            deblur_sigma = st.slider(
                "Sigma (블러 강도 추정)", 
                0.5, 3.0, 1.5, 0.1,
                help="블러의 강도를 나타내는 값\n"
                     "• 값이 작으면: 작은 블러만 제거, 세부사항 보존\n"
                     "• 값이 크면: 큰 블러도 제거, 하지만 과도하면 인공물 생성\n"
                     "• 권장값: 1.5 (일반), 0.5-1.0 (약간 흐림), 2.0-3.0 (심하게 흐림)"
            )
        
        # 8. 기하학적 보정
        st.subheader("8️⃣ 기하학적 보정")
        st.caption("기울어지거나 회전된 QR 코드를 정면으로 펴서 해독률을 향상시킵니다.")
        
        geometric_enabled = st.checkbox(
            "기하학적 보정 활성화", 
            value=False,
            help="QR 코드의 기하학적 왜곡을 보정합니다.\n"
                 "• 회전 보정: 기울어진 QR을 바로 세움\n"
                 "• Perspective 보정: 비스듬하게 찍힌 QR을 정면으로 펴줌"
        )
        rotation_angle = 0
        perspective_correction = False
        if geometric_enabled:
            rotation_angle = st.slider(
                "회전 각도", 
                -180, 180, 0, 5,
                help="QR 코드를 회전시킬 각도 (도 단위)\n"
                     "• 양수: 시계 방향 회전\n"
                     "• 음수: 반시계 방향 회전\n"
                     "• 0: 회전 없음\n"
                     "• 사용 시기: QR 코드가 기울어져 있을 때"
            )
            perspective_correction = st.checkbox(
                "Perspective 보정", 
                value=False,
                help="비스듬하게 찍힌 QR 코드를 정면으로 보정합니다.\n"
                     "• 효과: 원근 왜곡 제거, QR 코드를 정사각형으로 만듦\n"
                     "• 사용 시기: QR 코드가 비스듬한 각도로 찍혔을 때\n"
                     "• 참고: 자동 탐지 기능은 아직 미구현, 수동 설정 필요"
            )
        
        # 딥러닝 향상
        st.markdown("---")
        st.subheader("🤖 딥러닝 향상")
        st.caption("딥러닝 모델을 사용하여 이미지를 복원하고 향상시킵니다. (현재 플레이스홀더)")
        
        use_deep_learning = st.checkbox(
            "U-Net/SegFormer 향상 사용", 
            value=False,
            help="딥러닝 기반 이미지 복원/향상 기능입니다.\n"
                 "• U-Net: 이미지 복원, 노이즈 제거, 디블러링\n"
                 "• SegFormer: 이미지 세그멘테이션 기반 향상\n"
                 "• 현재 상태: 플레이스홀더만 구현됨 (향후 구현 예정)\n"
                 "• 효과: 심각하게 손상된 이미지 복원 가능"
        )
        
        # 전처리 순서 안내
        st.markdown("---")
        with st.expander("ℹ️ 전처리 적용 순서", expanded=False):
            st.markdown("""
            전처리 옵션들은 다음 순서로 적용됩니다:
            
            1. **그레이스케일 변환** (선택)
            2. **노이즈 제거**
            3. **명암/조명 보정**
            4. **반전 처리**
            5. **Super Resolution**
            6. **Deblurring**
            7. **이진화**
            8. **형태학적 연산**
            9. **기하학적 보정**
            
            💡 **팁**: 순서를 고려하여 옵션을 활성화하세요!
            """)
        
        # 추천 설정
        with st.expander("💡 추천 설정 가이드", expanded=False):
            st.markdown("""
            ### 조선소 T-bar 환경 (철판 녹, 불균일 조명)
            1. ✅ 노이즈 제거: Bilateral Filter (d=9, sigma_color=75, sigma_space=75)
            2. ✅ 명암 보정: CLAHE (clipLimit=2.0, tileSize=8x8)
            3. ✅ 이진화: Adaptive Gaussian (blockSize=11, C=2)
            4. ✅ 형태학적: Closing → Opening
            
            ### 어두운 이미지
            1. ✅ 노이즈 제거: Bilateral Filter
            2. ✅ 명암 보정: CLAHE + Gamma (0.5-0.8)
            3. ✅ 이진화: Adaptive Gaussian
            
            ### 흐릿한 이미지
            1. ✅ Super Resolution (2.0-3.0배)
            2. ✅ Deblurring (iterations=30-50)
            
            ### 작은 QR 코드
            1. ✅ Super Resolution (3.0-4.0배)
            2. ✅ 노이즈 제거: Bilateral Filter
            """)
        
        # 전처리 옵션 정리
        preprocessing_options = {
            'convert_grayscale': convert_grayscale,
            'conf_threshold': conf_threshold,
            'denoise_enabled': denoise_enabled,
            'denoise_method': denoise_method if denoise_enabled else None,
            'gaussian_kernel': gaussian_kernel if denoise_enabled and denoise_method == "gaussian" else 5,
            'median_kernel': median_kernel if denoise_enabled and denoise_method == "median" else 5,
            'bilateral_d': bilateral_d if denoise_enabled and denoise_method == "bilateral" else 9,
            'bilateral_sigma_color': bilateral_sigma_color if denoise_enabled and denoise_method == "bilateral" else 75,
            'bilateral_sigma_space': bilateral_sigma_space if denoise_enabled and denoise_method == "bilateral" else 75,
            'enhancement_enabled': enhancement_enabled,
            'enhancement_methods': enhancement_methods if enhancement_enabled else [],
            'clahe_clip_limit': clahe_clip_limit if enhancement_enabled and "clahe" in enhancement_methods else 2.0,
            'clahe_tile_size': clahe_tile_size if enhancement_enabled and "clahe" in enhancement_methods else [8, 8],
            'gamma_value': gamma_value if enhancement_enabled and "gamma" in enhancement_methods else 1.0,
            'retinex_sigma': [float(s) for s in retinex_sigma.split(',')] if enhancement_enabled and "retinex" in enhancement_methods else [15.0, 80.0, 250.0],
            'invert': invert,
            'binarization_enabled': binarization_enabled,
            'binarization_method': binarization_method if binarization_enabled else None,
            'adaptive_block_size': adaptive_block_size if binarization_enabled and "adaptive" in binarization_method else 11,
            'adaptive_c': adaptive_c if binarization_enabled and "adaptive" in binarization_method else 2,
            'morphology_enabled': morphology_enabled,
            'morphology_operations': morphology_operations if morphology_enabled else [],
            'morphology_kernel_size': morphology_kernel_size if morphology_enabled else 5,
            'dilation_iterations': dilation_iterations if morphology_enabled and "dilation" in morphology_operations else 1,
            'super_resolution_enabled': super_resolution_enabled,
            'sr_scale': sr_scale if super_resolution_enabled else 2.0,
            'deblur_enabled': deblur_enabled,
            'deblur_iterations': deblur_iterations if deblur_enabled else 30,
            'deblur_sigma': deblur_sigma if deblur_enabled else 1.5,
            'geometric_enabled': geometric_enabled,
            'rotation_angle': rotation_angle if geometric_enabled else 0,
            'perspective_correction': perspective_correction if geometric_enabled else False,
        }
    
    # 메인 영역 - 이미지 업로드 및 처리
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        # 이미지 읽기
        image = np.array(Image.open(uploaded_file))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 처리 버튼
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 이미지 처리 시작", width='stretch', type="primary"):
                if st.session_state.yolo_model is None:
                    st.error("⚠️ 먼저 YOLO 모델을 로드하세요!")
                else:
                    with st.spinner("처리 중..."):
                        results = process_image(
                            image,
                            preprocessing_options,
                            st.session_state.yolo_model,
                            st.session_state.dbr_reader,
                            st.session_state.transformer_model,
                            use_deep_learning
                        )
                        st.session_state.results = results
                    st.success("✅ 처리 완료!")
        
        # 결과 표시
        if st.session_state.results:
            results = st.session_state.results
            
            # 원본 이미지와 전처리 이미지 나란히 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 원본 이미지")
                
                # 원본 이미지에 탐지 결과 표시 (시각화 함수 사용)
                if results['original_image_visualized'] is not None:
                    original_rgb = cv2.cvtColor(results['original_image_visualized'], cv2.COLOR_BGR2RGB)
                    st.image(original_rgb, width='stretch')
                else:
                    original_rgb = cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB)
                    st.image(original_rgb, width='stretch')
                
                # 원본 해독 결과
                st.markdown("**해독 결과:**")
                if results['original_decodings']:
                    for i, dec in enumerate(results['original_decodings']):
                        status = "✅" if dec['success'] else "❌"
                        st.text(f"{status} QR #{i+1}: {dec['text'] if dec['text'] else '해독 실패'} (신뢰도: {dec['confidence']:.2f})")
                else:
                    st.info("탐지된 QR 코드가 없습니다.")
            
            with col2:
                st.subheader("✨ 전처리된 이미지")
                
                if results['preprocessed_image_visualized'] is not None:
                    # 전처리 이미지에 탐지 결과 표시 (시각화 함수 사용)
                    preprocessed_vis = results['preprocessed_image_visualized']
                    # 그레이스케일인 경우 BGR로 변환
                    if len(preprocessed_vis.shape) == 2:
                        preprocessed_vis = cv2.cvtColor(preprocessed_vis, cv2.COLOR_GRAY2BGR)
                    preprocessed_rgb = cv2.cvtColor(preprocessed_vis, cv2.COLOR_BGR2RGB)
                    st.image(preprocessed_rgb, width='stretch')
                elif results['preprocessed_image'] is not None:
                    # 전처리만 있고 시각화가 없는 경우
                    preprocessed_clean = results['preprocessed_image']
                    if len(preprocessed_clean.shape) == 2:
                        preprocessed_clean = cv2.cvtColor(preprocessed_clean, cv2.COLOR_GRAY2BGR)
                    preprocessed_rgb = cv2.cvtColor(preprocessed_clean, cv2.COLOR_BGR2RGB)
                    st.image(preprocessed_rgb, width='stretch')
                else:
                    st.info("전처리된 이미지가 없습니다.")
                
                # 전처리 해독 결과 (항상 표시)
                if results['preprocessed_image'] is not None or results['preprocessed_image_visualized'] is not None:
                    st.markdown("**해독 결과:**")
                    if results['preprocessed_decodings']:
                        for i, dec in enumerate(results['preprocessed_decodings']):
                            status = "✅" if dec['success'] else "❌"
                            st.text(f"{status} QR #{i+1}: {dec['text'] if dec['text'] else '해독 실패'} (신뢰도: {dec['confidence']:.2f})")
                    else:
                        st.info("탐지된 QR 코드가 없습니다.")
            
            # 비교 통계
            st.markdown("---")
            st.subheader("📊 결과 비교")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("원본 탐지 수", len(results['original_detections']))
            with col2:
                st.metric("원본 해독 성공", sum(1 for d in results['original_decodings'] if d['success']))
            with col3:
                st.metric("전처리 탐지 수", len(results['preprocessed_detections']))
            with col4:
                st.metric("전처리 해독 성공", sum(1 for d in results['preprocessed_decodings'] if d['success']))
            
            # 다운로드 섹션
            st.markdown("---")
            st.subheader("💾 결과 다운로드")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                st.markdown("**📷 원본 이미지**")
                if results['original_image_visualized'] is not None:
                    # 시각화된 원본 이미지 다운로드
                    original_vis_rgb = cv2.cvtColor(results['original_image_visualized'], cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(original_vis_rgb)
                    buf = io.BytesIO()
                    pil_image.save(buf, format='JPEG', quality=95)
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ 원본 (시각화)",
                        data=buf.getvalue(),
                        file_name=f"original_visualized_{uploaded_file.name}",
                        mime="image/jpeg",
                        width='stretch',
                        help="탐지 결과가 표시된 원본 이미지"
                    )
            
            with download_col2:
                st.markdown("**✨ 전처리 이미지 (시각화)**")
                if results['preprocessed_image_visualized'] is not None:
                    # 시각화된 전처리 이미지 다운로드
                    preprocessed_vis = results['preprocessed_image_visualized']
                    # 그레이스케일인 경우 BGR로 변환
                    if len(preprocessed_vis.shape) == 2:
                        preprocessed_vis = cv2.cvtColor(preprocessed_vis, cv2.COLOR_GRAY2BGR)
                    preprocessed_vis_rgb = cv2.cvtColor(preprocessed_vis, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(preprocessed_vis_rgb)
                    buf = io.BytesIO()
                    pil_image.save(buf, format='JPEG', quality=95)
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ 전처리 (시각화)",
                        data=buf.getvalue(),
                        file_name=f"preprocessed_visualized_{uploaded_file.name}",
                        mime="image/jpeg",
                        width='stretch',
                        help="탐지 결과가 표시된 전처리 이미지"
                    )
            
            with download_col3:
                st.markdown("**✨ 전처리 이미지 (순수)**")
                if results['preprocessed_image'] is not None:
                    # 시각화 안된 순수 전처리 이미지 다운로드
                    preprocessed_clean = results['preprocessed_image']
                    # 그레이스케일인 경우 BGR로 변환
                    if len(preprocessed_clean.shape) == 2:
                        preprocessed_clean = cv2.cvtColor(preprocessed_clean, cv2.COLOR_GRAY2BGR)
                    preprocessed_clean_rgb = cv2.cvtColor(preprocessed_clean, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(preprocessed_clean_rgb)
                    buf = io.BytesIO()
                    pil_image.save(buf, format='JPEG', quality=95)
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ 전처리 (순수)",
                        data=buf.getvalue(),
                        file_name=f"preprocessed_clean_{uploaded_file.name}",
                        mime="image/jpeg",
                        width='stretch',
                        help="탐지 결과 표시 없이 전처리만 적용된 순수 이미지"
                    )
            
            # 전체 다운로드 (ZIP)
            if (results['original_image_visualized'] is not None or 
                results['preprocessed_image_visualized'] is not None or 
                results['preprocessed_image'] is not None):
                st.markdown("---")
                
                # ZIP 파일 생성
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # 원본 시각화
                    if results['original_image_visualized'] is not None:
                        original_vis_rgb = cv2.cvtColor(results['original_image_visualized'], cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(original_vis_rgb)
                        buf = io.BytesIO()
                        pil_image.save(buf, format='JPEG', quality=95)
                        buf.seek(0)
                        zip_file.writestr(f"original_visualized_{uploaded_file.name}", buf.getvalue())
                    
                    # 전처리 시각화
                    if results['preprocessed_image_visualized'] is not None:
                        preprocessed_vis = results['preprocessed_image_visualized']
                        if len(preprocessed_vis.shape) == 2:
                            preprocessed_vis = cv2.cvtColor(preprocessed_vis, cv2.COLOR_GRAY2BGR)
                        preprocessed_vis_rgb = cv2.cvtColor(preprocessed_vis, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(preprocessed_vis_rgb)
                        buf = io.BytesIO()
                        pil_image.save(buf, format='JPEG', quality=95)
                        buf.seek(0)
                        zip_file.writestr(f"preprocessed_visualized_{uploaded_file.name}", buf.getvalue())
                    
                    # 전처리 순수
                    if results['preprocessed_image'] is not None:
                        preprocessed_clean = results['preprocessed_image']
                        if len(preprocessed_clean.shape) == 2:
                            preprocessed_clean = cv2.cvtColor(preprocessed_clean, cv2.COLOR_GRAY2BGR)
                        preprocessed_clean_rgb = cv2.cvtColor(preprocessed_clean, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(preprocessed_clean_rgb)
                        buf = io.BytesIO()
                        pil_image.save(buf, format='JPEG', quality=95)
                        buf.seek(0)
                        zip_file.writestr(f"preprocessed_clean_{uploaded_file.name}", buf.getvalue())
                
                zip_buffer.seek(0)
                zip_filename = f"qr_results_{os.path.splitext(uploaded_file.name)[0]}.zip"
                st.download_button(
                    label="📦 모든 이미지 ZIP으로 다운로드",
                    data=zip_buffer.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip",
                    width='stretch',
                    help="원본(시각화), 전처리(시각화), 전처리(순수) 이미지를 모두 포함한 ZIP 파일"
                )

if __name__ == "__main__":
    main()

