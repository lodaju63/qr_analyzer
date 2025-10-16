"""
QR 코드 탐지를 위한 공통 유틸리티 모듈
조선소 T-Bar 제작 공정을 위한 QR 코드 인식 시스템
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# 로깅 설정
logger = logging.getLogger(__name__)

class DetectionMethod(Enum):
    """탐지 방법 열거형"""
    OPENCV = "opencv"
    PYZBAR = "pyzbar"
    QREADER = "qreader"

class PreprocessingType(Enum):
    """전처리 타입 열거형"""
    ORIGINAL = "original"
    CLAHE = "clahe"
    BINARY = "binary"
    PIL_ENHANCE = "pil_enhance"
    CENTER_CROP = "center_crop"
    GAUSSIAN_BLUR = "gaussian_blur"

@dataclass
class QRResult:
    """QR 코드 탐지 결과 데이터 클래스"""
    data: str
    method: DetectionMethod
    preprocessing: PreprocessingType
    confidence: float
    points: Optional[np.ndarray] = None
    bbox: Optional[List[int]] = None
    rect: Optional[object] = None
    detection_time: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class PerformanceStats:
    """성능 통계 데이터 클래스"""
    total_frames: int = 0
    total_detections: int = 0
    total_detection_time: float = 0.0
    avg_detection_time: float = 0.0
    current_fps: float = 0.0
    detection_rate: float = 0.0
    
    def update(self, detection_time: float = 0.0, detection_success: bool = False):
        """통계 업데이트"""
        self.total_frames += 1
        if detection_success:
            self.total_detections += 1
            self.total_detection_time += detection_time
            self.avg_detection_time = self.total_detection_time / self.total_detections
        
        self.detection_rate = self.total_detections / self.total_frames if self.total_frames > 0 else 0.0

class ImageQualityAnalyzer:
    """이미지 품질 분석기"""
    
    @staticmethod
    def analyze_brightness(image: np.ndarray) -> float:
        """이미지 밝기 분석"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    @staticmethod
    def analyze_contrast(image: np.ndarray) -> float:
        """이미지 대비 분석"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.std(gray)
    
    @staticmethod
    def analyze_sharpness(image: np.ndarray) -> float:
        """이미지 선명도 분석 (Laplacian variance)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def analyze_quality(image: np.ndarray) -> Dict[str, float]:
        """종합적인 이미지 품질 분석"""
        return {
            'brightness': ImageQualityAnalyzer.analyze_brightness(image),
            'contrast': ImageQualityAnalyzer.analyze_contrast(image),
            'sharpness': ImageQualityAnalyzer.analyze_sharpness(image)
        }

class CoordinateTransformer:
    """좌표 변환 유틸리티"""
    
    @staticmethod
    def scale_points(points: np.ndarray, scale_factor: float) -> np.ndarray:
        """점들을 스케일 변환"""
        if points is None:
            return None
        return points / scale_factor
    
    @staticmethod
    def scale_bbox(bbox: List[int], scale_factor: float) -> List[int]:
        """바운딩 박스를 스케일 변환"""
        if bbox is None or len(bbox) < 4:
            return None
        return [int(coord / scale_factor) for coord in bbox]
    
    @staticmethod
    def rotate_point_back(point: Tuple[float, float], center: Tuple[float, float], angle: float) -> Tuple[float, float]:
        """회전된 점을 원본으로 역변환"""
        x, y = point
        cx, cy = center
        
        # 각도를 라디안으로 변환
        angle_rad = np.radians(-angle)
        
        # 좌표를 중심점 기준으로 이동
        x_shifted = x - cx
        y_shifted = y - cy
        
        # 역회전 변환
        x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
        y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
        
        # 중심점을 다시 더해서 원본 좌표계로 복원
        x_original = x_rotated + cx
        y_original = y_rotated + cy
        
        return (x_original, y_original)
    
    @staticmethod
    def rotate_bbox_back(bbox: List[int], center: Tuple[float, float], angle: float, image_shape: Tuple[int, int]) -> List[int]:
        """회전된 바운딩 박스를 원본으로 역변환"""
        if not bbox or len(bbox) < 4:
            return None
        
        # bbox의 4개 모서리 좌표 계산
        x1, y1, x2, y2 = bbox
        corners = [
            (x1, y1),  # 좌상단
            (x2, y1),  # 우상단
            (x2, y2),  # 우하단
            (x1, y2)   # 좌하단
        ]
        
        # 각 모서리를 역회전
        rotated_corners = []
        for corner in corners:
            rotated_corner = CoordinateTransformer.rotate_point_back(corner, center, angle)
            rotated_corners.append(rotated_corner)
        
        # 역회전된 좌표에서 새로운 bbox 계산
        x_coords = [corner[0] for corner in rotated_corners]
        y_coords = [corner[1] for corner in rotated_corners]
        
        new_x1 = max(0, min(x_coords))
        new_y1 = max(0, min(y_coords))
        new_x2 = min(image_shape[1], max(x_coords))
        new_y2 = min(image_shape[0], max(y_coords))
        
        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

class DuplicateRemover:
    """중복 제거 유틸리티"""
    
    @staticmethod
    def is_same_qr_data(result1: QRResult, result2: QRResult) -> bool:
        """데이터 기반 중복 확인"""
        return result1.data == result2.data
    
    @staticmethod
    def is_same_qr_location(result1: QRResult, result2: QRResult, distance_threshold: float = 100.0) -> bool:
        """위치 기반 중복 확인"""
        if result1.data != result2.data:
            return False
        
        pos1 = DuplicateRemover._get_qr_position(result1)
        pos2 = DuplicateRemover._get_qr_position(result2)
        
        if pos1 is None or pos2 is None:
            return True  # 위치 정보가 없으면 데이터만으로 판단
        
        # 중심점 간 거리 계산
        center1 = pos1['center']
        center2 = pos2['center']
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < distance_threshold
    
    @staticmethod
    def _get_qr_position(result: QRResult) -> Optional[Dict]:
        """QR 결과에서 위치 정보 추출"""
        # OpenCV points
        if result.points is not None:
            points = result.points
            if isinstance(points, np.ndarray) and len(points) >= 4:
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])
                return {'center': (center_x, center_y), 'type': 'points'}
        
        # QReader bbox
        elif result.bbox is not None:
            bbox = result.bbox
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                return {'center': (center_x, center_y), 'type': 'bbox'}
        
        # PyZbar rect
        elif result.rect is not None:
            rect = result.rect
            center_x = rect.left + rect.width / 2
            center_y = rect.top + rect.height / 2
            return {'center': (center_x, center_y), 'type': 'rect'}
        
        return None
    
    @staticmethod
    def remove_duplicates(results: List[QRResult], method: str = "data") -> List[QRResult]:
        """중복 제거"""
        if method == "data":
            seen_data = set()
            unique_results = []
            for result in results:
                if result.data not in seen_data:
                    seen_data.add(result.data)
                    unique_results.append(result)
            return unique_results
        
        elif method == "location":
            unique_results = []
            for result in results:
                is_duplicate = False
                for existing_result in unique_results:
                    if DuplicateRemover.is_same_qr_location(result, existing_result):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_results.append(result)
            return unique_results
        
        return results

class VisualizationHelper:
    """시각화 도우미"""
    
    @staticmethod
    def draw_qr_result(image: np.ndarray, result: QRResult, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """QR 결과를 이미지에 그리기"""
        result_image = image.copy()
        
        # QR 코드 영역 그리기
        if result.points is not None and len(result.points) >= 4:
            points = result.points.astype(np.int32)
            cv2.polylines(result_image, [points], True, color, 2)
            
            # 텍스트 표시
            text = result.data[:20] + "..." if len(result.data) > 20 else result.data
            cv2.putText(result_image, text, (int(points[0][0]), int(points[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        elif result.bbox is not None and len(result.bbox) >= 4:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 텍스트 표시
            text = result.data[:20] + "..." if len(result.data) > 20 else result.data
            cv2.putText(result_image, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image
    
    @staticmethod
    def draw_multiple_qr_results(image: np.ndarray, results: List[QRResult]) -> np.ndarray:
        """여러 QR 결과를 이미지에 그리기"""
        result_image = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            result_image = VisualizationHelper.draw_qr_result(result_image, result, color)
        
        return result_image
    
    @staticmethod
    def draw_performance_info(image: np.ndarray, stats: PerformanceStats) -> np.ndarray:
        """성능 정보를 이미지에 그리기"""
        result_image = image.copy()
        
        # 성능 정보 텍스트
        info_lines = [
            f"FPS: {stats.current_fps:.1f}",
            f"Detections: {stats.total_detections}",
            f"Detection Rate: {stats.detection_rate:.1%}",
            f"Avg Time: {stats.avg_detection_time:.3f}s"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(result_image, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return result_image

class ConfigManager:
    """설정 관리자"""
    
    def __init__(self):
        self.config = {
            'detection': {
                'opencv_enabled': True,
                'pyzbar_enabled': True,
                'qreader_enabled': True,
                'detection_order': [DetectionMethod.OPENCV, DetectionMethod.PYZBAR, DetectionMethod.QREADER]
            },
            'preprocessing': {
                'clahe_enabled': True,
                'clahe_clip_limit': 2.0,
                'clahe_tile_size': (8, 8),
                'binary_enabled': True,
                'binary_block_size': 11,
                'binary_c': 2
            },
            'performance': {
                'max_detection_time': 1.0,  # 최대 탐지 시간 (초)
                'detection_interval': 5,    # N프레임마다 탐지
                'max_results_history': 100  # 최대 결과 히스토리
            },
            'visualization': {
                'show_confidence': True,
                'show_method': True,
                'show_timestamp': True,
                'line_thickness': 2,
                'font_scale': 0.5
            }
        }
    
    def get(self, key: str, default=None):
        """설정 값 가져오기"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value):
        """설정 값 설정하기"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def update(self, updates: Dict):
        """설정 업데이트"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)

# 전역 설정 관리자 인스턴스
config = ConfigManager()

