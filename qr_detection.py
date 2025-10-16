"""
QR 코드 탐지 모듈
조선소 T-Bar 제작 공정을 위한 QR 코드 인식 엔진
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import logging
from qr_utils import QRResult, DetectionMethod, PreprocessingType, DuplicateRemover, config

# PyZbar 및 QReader는 선택적 import
try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logging.warning("PyZbar를 사용할 수 없습니다. pip install pyzbar로 설치하세요.")

try:
    from qreader import QReader
    QREADER_AVAILABLE = True
except ImportError:
    QREADER_AVAILABLE = False
    logging.warning("QReader를 사용할 수 없습니다. pip install qreader로 설치하세요.")

logger = logging.getLogger(__name__)

class QRCodeDetector:
    """QR 코드 탐지기 기본 클래스"""
    
    def __init__(self):
        """탐지기 초기화"""
        self.opencv_detector = cv2.QRCodeDetector()
        
        # PyZbar 초기화
        if PYZBAR_AVAILABLE:
            self.pyzbar_available = True
        else:
            self.pyzbar_available = False
        
        # QReader 초기화
        if QREADER_AVAILABLE:
            try:
                self.qreader = QReader()
                self.qreader_available = True
                # 경고 메시지 숨기기
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning, module='qreader')
            except Exception as e:
                logger.warning(f"QReader 초기화 실패: {e}")
                self.qreader_available = False
        else:
            self.qreader_available = False
    
    def detect_opencv(self, image: np.ndarray) -> List[QRResult]:
        """OpenCV로 QR 코드 탐지"""
        results = []
        
        try:
            retval, decoded_info, points = self.opencv_detector.detectAndDecode(image)
            
            if retval and decoded_info is not None:
                if isinstance(decoded_info, str) and decoded_info:
                    result = QRResult(
                        data=decoded_info,
                        method=DetectionMethod.OPENCV,
                        preprocessing=PreprocessingType.ORIGINAL,
                        confidence=1.0,
                        points=points
                    )
                    results.append(result)
                
                elif isinstance(decoded_info, list) and len(decoded_info) > 0:
                    for i, info in enumerate(decoded_info):
                        if info:
                            result = QRResult(
                                data=info,
                                method=DetectionMethod.OPENCV,
                                preprocessing=PreprocessingType.ORIGINAL,
                                confidence=1.0,
                                points=points[i] if points is not None and i < len(points) else None
                            )
                            results.append(result)
        
        except Exception as e:
            logger.debug(f"OpenCV 탐지 오류: {e}")
        
        return results
    
    def detect_pyzbar(self, image: np.ndarray) -> List[QRResult]:
        """PyZbar로 QR 코드 탐지"""
        results = []
        
        if not self.pyzbar_available:
            return results
        
        try:
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pyzbar_results = pyzbar.decode(pil_image, symbols=[ZBarSymbol.QRCODE])
            
            for qr in pyzbar_results:
                data = qr.data.decode('utf-8')
                result = QRResult(
                    data=data,
                    method=DetectionMethod.PYZBAR,
                    preprocessing=PreprocessingType.ORIGINAL,
                    confidence=1.0,
                    rect=qr.rect
                )
                results.append(result)
        
        except Exception as e:
            logger.debug(f"PyZbar 탐지 오류: {e}")
        
        return results
    
    def detect_qreader(self, image: np.ndarray) -> List[QRResult]:
        """QReader로 QR 코드 탐지"""
        results = []
        
        if not self.qreader_available:
            return results
        
        try:
            # 1단계: QReader로 위치 탐지
            detections = self.qreader.detect(image)
            
            if detections and len(detections) > 0:
                for detection in detections:
                    try:
                        # 2단계: 탐지된 위치에서 텍스트 해독
                        decoded_text = self.qreader.decode(image, detection)
                        
                        if decoded_text:
                            # bbox 정보 추출
                            bbox = None
                            if isinstance(detection, dict) and 'bbox_xyxy' in detection:
                                bbox_xyxy = detection['bbox_xyxy']
                                if len(bbox_xyxy) >= 4:
                                    bbox = [int(bbox_xyxy[0]), int(bbox_xyxy[1]), 
                                           int(bbox_xyxy[2]), int(bbox_xyxy[3])]
                            
                            result = QRResult(
                                data=decoded_text,
                                method=DetectionMethod.QREADER,
                                preprocessing=PreprocessingType.ORIGINAL,
                                confidence=0.9,  # QReader는 confidence 정보가 제한적
                                bbox=bbox
                            )
                            results.append(result)
                    
                    except Exception as decode_error:
                        logger.debug(f"QReader 해독 오류: {decode_error}")
                        continue
        
        except Exception as e:
            logger.debug(f"QReader 탐지 오류: {e}")
        
        return results

class MultiMethodDetector:
    """다중 방법 QR 코드 탐지기"""
    
    def __init__(self, detection_order: List[DetectionMethod] = None):
        """
        Args:
            detection_order: 탐지 방법 순서
        """
        self.detector = QRCodeDetector()
        
        if detection_order is None:
            self.detection_order = config.get('detection.detection_order', [
                DetectionMethod.OPENCV,
                DetectionMethod.PYZBAR,
                DetectionMethod.QREADER
            ])
        else:
            self.detection_order = detection_order
    
    def detect_all_methods(self, image: np.ndarray) -> List[QRResult]:
        """모든 방법으로 QR 코드 탐지"""
        all_results = []
        
        for method in self.detection_order:
            if method == DetectionMethod.OPENCV:
                results = self.detector.detect_opencv(image)
            elif method == DetectionMethod.PYZBAR and self.detector.pyzbar_available:
                results = self.detector.detect_pyzbar(image)
            elif method == DetectionMethod.QREADER and self.detector.qreader_available:
                results = self.detector.detect_qreader(image)
            else:
                continue
            
            all_results.extend(results)
        
        return all_results
    
    def detect_fast(self, image: np.ndarray) -> List[QRResult]:
        """빠른 탐지 (OpenCV만 사용)"""
        return self.detector.detect_opencv(image)
    
    def detect_optimized(self, image: np.ndarray) -> List[QRResult]:
        """최적화된 탐지 (OpenCV + PyZbar)"""
        all_results = []
        
        # OpenCV 먼저 시도
        opencv_results = self.detector.detect_opencv(image)
        all_results.extend(opencv_results)
        
        # 결과가 없으면 PyZbar 시도
        if not opencv_results and self.detector.pyzbar_available:
            pyzbar_results = self.detector.detect_pyzbar(image)
            all_results.extend(pyzbar_results)
        
        return all_results

class RealtimeDetector:
    """실시간 QR 코드 탐지기"""
    
    def __init__(self, max_detection_time: float = 0.1):
        """
        Args:
            max_detection_time: 최대 탐지 시간 (초)
        """
        self.multi_detector = MultiMethodDetector()
        self.max_detection_time = max_detection_time
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'avg_detection_time': 0.0,
            'method_stats': {}
        }
    
    def detect_realtime(self, image: np.ndarray) -> Tuple[List[QRResult], float]:
        """실시간 QR 코드 탐지"""
        start_time = time.time()
        
        # 빠른 탐지 시도
        results = self.multi_detector.detect_fast(image)
        
        detection_time = time.time() - start_time
        
        # 통계 업데이트
        self.detection_stats['total_detections'] += 1
        if results:
            self.detection_stats['successful_detections'] += 1
        
        # 평균 탐지 시간 계산
        total_detections = self.detection_stats['total_detections']
        if total_detections > 0:
            current_avg = self.detection_stats['avg_detection_time']
            self.detection_stats['avg_detection_time'] = (
                (current_avg * (total_detections - 1) + detection_time) / total_detections
            )
        
        return results, detection_time
    
    def detect_with_timeout(self, image: np.ndarray) -> Tuple[List[QRResult], float]:
        """시간 제한이 있는 탐지"""
        start_time = time.time()
        results = []
        
        # OpenCV만 사용 (가장 빠름)
        opencv_results = self.multi_detector.detector.detect_opencv(image)
        results.extend(opencv_results)
        
        detection_time = time.time() - start_time
        
        # 시간이 남으면 PyZbar도 시도
        if not results and detection_time < self.max_detection_time:
            if self.multi_detector.detector.pyzbar_available:
                pyzbar_results = self.multi_detector.detector.detect_pyzbar(image)
                results.extend(pyzbar_results)
                detection_time = time.time() - start_time
        
        return results, detection_time
    
    def get_detection_stats(self) -> Dict:
        """탐지 통계 반환"""
        stats = self.detection_stats.copy()
        if stats['total_detections'] > 0:
            stats['success_rate'] = stats['successful_detections'] / stats['total_detections']
        else:
            stats['success_rate'] = 0.0
        return stats

class BatchDetector:
    """배치 QR 코드 탐지기"""
    
    def __init__(self):
        """배치 탐지기 초기화"""
        self.multi_detector = MultiMethodDetector()
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[QRResult]]:
        """여러 이미지 배치 탐지"""
        results = []
        
        for image in images:
            image_results = self.multi_detector.detect_all_methods(image)
            results.append(image_results)
        
        return results
    
    def detect_with_preprocessing(self, image: np.ndarray, preprocessing_types: List[PreprocessingType]) -> List[QRResult]:
        """전처리와 함께 탐지"""
        from qr_preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        all_results = []
        
        for prep_type in preprocessing_types:
            # 전처리 적용
            processed_image = preprocessor.preprocess_image(image, prep_type)
            
            # 탐지 실행
            results = self.multi_detector.detect_all_methods(processed_image)
            
            # 전처리 정보 업데이트
            for result in results:
                result.preprocessing = prep_type
            
            all_results.extend(results)
        
        # 중복 제거
        unique_results = DuplicateRemover.remove_duplicates(all_results, method="data")
        
        return unique_results

class DetectionPipeline:
    """탐지 파이프라인"""
    
    def __init__(self, mode: str = "optimized"):
        """
        Args:
            mode: "realtime", "optimized", "full"
        """
        self.mode = mode
        
        if mode == "realtime":
            self.detector = RealtimeDetector()
        elif mode == "optimized":
            self.multi_detector = MultiMethodDetector()
        else:  # full
            self.multi_detector = MultiMethodDetector()
            self.batch_detector = BatchDetector()
    
    def detect(self, image: np.ndarray) -> Tuple[List[QRResult], float]:
        """이미지에서 QR 코드 탐지"""
        start_time = time.time()
        
        if self.mode == "realtime":
            results, detection_time = self.detector.detect_realtime(image)
        elif self.mode == "optimized":
            results = self.multi_detector.detect_optimized(image)
            detection_time = time.time() - start_time
        else:  # full
            results = self.multi_detector.detect_all_methods(image)
            detection_time = time.time() - start_time
        
        return results, detection_time
    
    def detect_with_preprocessing(self, image: np.ndarray) -> Tuple[List[QRResult], float]:
        """전처리와 함께 탐지"""
        if self.mode == "realtime":
            # 실시간 모드에서는 전처리 없이 탐지
            return self.detect(image)
        
        from qr_preprocessing import get_preprocessing_pipeline
        
        start_time = time.time()
        all_results = []
        
        # 전처리 파이프라인 가져오기
        pipeline = get_preprocessing_pipeline(self.mode)
        processed_images = pipeline.process_image(image)
        
        # 각 전처리된 이미지에서 탐지
        for processed_image, prep_type, scale, angle in processed_images:
            if self.mode == "optimized":
                results = self.multi_detector.detect_optimized(processed_image)
            else:
                results = self.multi_detector.detect_all_methods(processed_image)
            
            # 결과에 전처리 정보 추가
            for result in results:
                result.preprocessing = prep_type
                if scale is not None:
                    # 좌표 변환 필요
                    pass
                if angle is not None:
                    # 좌표 변환 필요
                    pass
            
            all_results.extend(results)
        
        # 중복 제거
        unique_results = DuplicateRemover.remove_duplicates(all_results, method="data")
        detection_time = time.time() - start_time
        
        return unique_results, detection_time

# 전역 탐지 파이프라인 인스턴스들
realtime_detector = DetectionPipeline("realtime")
optimized_detector = DetectionPipeline("optimized")
full_detector = DetectionPipeline("full")

def get_detection_pipeline(mode: str = "optimized") -> DetectionPipeline:
    """탐지 파이프라인 가져오기"""
    if mode == "realtime":
        return realtime_detector
    elif mode == "optimized":
        return optimized_detector
    else:
        return full_detector

