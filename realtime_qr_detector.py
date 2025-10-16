"""
실시간 영상 QR 코드 탐지 모듈
조선소 T-Bar 제작 공정을 위한 고속 QR 코드 인식 시스템
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import threading
from collections import deque
import logging

# 모듈 import
from qr_utils import QRResult, PerformanceStats, DetectionMethod, PreprocessingType, config
from qr_detection import get_detection_pipeline
from qr_preprocessing import get_preprocessing_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeQRDetector:
    """실시간 영상 QR 코드 탐지기"""
    
    def __init__(self, 
                 camera_id: int = 0,
                 video_path: str = None,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 fps: int = 30,
                 detection_interval: int = 5):  # N프레임마다 탐지
        """
        Args:
            camera_id: 카메라 ID (0: 기본 카메라)
            video_path: 비디오 파일 경로 (None이면 카메라 사용)
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            fps: 초당 프레임 수
            detection_interval: 탐지 간격 (N프레임마다)
        """
        self.camera_id = camera_id
        self.video_path = video_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.detection_interval = detection_interval
        
        # 탐지 파이프라인 초기화
        self.detection_pipeline = get_detection_pipeline("realtime")
        self.preprocessing_pipeline = get_preprocessing_pipeline("realtime")
        
        # 성능 모니터링
        self.performance_stats = PerformanceStats()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # 결과 저장
        self.last_detection_results = []
        self.detection_history = deque(maxlen=config.get('performance.max_results_history', 100))
        
        # 스레드 제어
        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        
        # 카메라 초기화
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        logger.info(f"실시간 QR 탐지기 초기화 완료 - 해상도: {frame_width}x{frame_height}, FPS: {fps}")
    
    def initialize_camera(self) -> bool:
        """카메라 또는 비디오 파일 초기화"""
        try:
            if self.video_path:
                # 비디오 파일 사용
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    logger.error(f"비디오 파일을 열 수 없습니다: {self.video_path}")
                    return False
                
                # 비디오 정보 확인
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                logger.info(f"비디오 파일 로드 완료: {self.video_path}")
                logger.info(f"해상도: {actual_width}x{actual_height}, FPS: {actual_fps}, 총 프레임: {total_frames}")
            else:
                # 카메라 사용
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    logger.error(f"카메라 {self.camera_id}를 열 수 없습니다.")
                    return False
                
                # 카메라 설정
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # 실제 설정값 확인
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"카메라 설정 완료 - 해상도: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            return False
    
    def start_detection(self) -> bool:
        """실시간 탐지 시작"""
        if not self.initialize_camera():
            return False
        
        self.running = True
        
        # 카메라 캡처 스레드 시작
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # QR 탐지 스레드 시작
        self.detection_thread = threading.Thread(target=self._detect_qr_codes)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info("실시간 QR 탐지 시작")
        return True
    
    def stop_detection(self):
        """실시간 탐지 중지"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info("실시간 QR 탐지 중지")
    
    def _capture_frames(self):
        """프레임 캡처 스레드"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                self.performance_stats.total_frames += 1
            else:
                logger.warning("프레임 캡처 실패")
                time.sleep(0.01)
    
    def _detect_qr_codes(self):
        """QR 코드 탐지 스레드"""
        while self.running:
            if self.performance_stats.total_frames % self.detection_interval == 0:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        continue
                
                # QR 코드 탐지
                results, detection_time = self.detection_pipeline.detect(frame)
                
                # 성능 통계 업데이트
                self.performance_stats.update(detection_time, len(results) > 0)
                
                if results:
                    self.last_detection_results = results
                    
                    # 결과 저장
                    self.detection_history.append({
                        'timestamp': time.time(),
                        'results': results,
                        'detection_time': detection_time,
                        'frame_count': self.performance_stats.total_frames
                    })
                    
                    logger.info(f"QR 탐지 성공: {len(results)}개, 시간: {detection_time:.3f}s")
            
            time.sleep(0.01)  # CPU 사용량 조절
    
    def _detect_single_frame(self, frame) -> List[QRResult]:
        """단일 프레임에서 QR 코드 탐지 (빠른 버전)"""
        results, detection_time = self.detection_pipeline.detect(frame)
        return results
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 반환"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_detection_results(self) -> List[QRResult]:
        """최근 탐지 결과 반환"""
        return self.last_detection_results.copy()
    
    def get_performance_stats(self) -> PerformanceStats:
        """성능 통계 반환"""
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        
        if elapsed_time > 0:
            self.performance_stats.current_fps = self.fps_counter / elapsed_time
        else:
            self.performance_stats.current_fps = 0
        
        return self.performance_stats
    
    def visualize_results(self, frame: np.ndarray, results: List[QRResult]) -> np.ndarray:
        """탐지 결과 시각화"""
        from qr_utils import VisualizationHelper
        
        result_frame = frame.copy()
        
        for result in results:
            result_frame = VisualizationHelper.draw_qr_result(result_frame, result)
        
        return result_frame


def main():
    """메인 함수 - 실시간 QR 탐지 데모"""
    detector = RealtimeQRDetector(
        camera_id=0,
        frame_width=640,
        frame_height=480,
        fps=30,
        detection_interval=5
    )
    
    if not detector.start_detection():
        print("❌ 실시간 탐지 시작 실패")
        return
    
    print("🎥 실시간 QR 탐지 시작 (ESC 키로 종료)")
    
    try:
        while True:
            # 현재 프레임 가져오기
            frame = detector.get_current_frame()
            if frame is None:
                continue
            
            # 탐지 결과 가져오기
            results = detector.get_detection_results()
            
            # 결과 시각화
            if results:
                frame = detector.visualize_results(frame, results)
            
            # 성능 정보 표시
            from qr_utils import VisualizationHelper
            stats = detector.get_performance_stats()
            frame = VisualizationHelper.draw_performance_info(frame, stats)
            
            # 프레임 표시
            cv2.imshow('Realtime QR Detection', frame)
            
            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중지됨")
    
    finally:
        detector.stop_detection()
        cv2.destroyAllWindows()
        print("✅ 실시간 탐지 종료")


if __name__ == "__main__":
    main()
