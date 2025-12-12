"""
고성능 QR코드 영상 분석 데스크톱 앱 (PyQt6)
- YOLO 기반 QR 탐지
- Dynamsoft 기반 QR 해독
- QThread를 사용한 멀티스레딩 아키텍처
- PyQtGraph를 사용한 실시간 데이터 시각화
- Dark Theme + 반응형 UI + 전처리 옵션
"""

import sys
import os

# ==========================================
# PyInstaller --noconsole 에러 방지 코드
# YOLO가 화면(stdout)을 찾을 때 에러가 나지 않도록 가짜를 쥐어줌
# ==========================================
class NullWriter:
    """가짜 출력 스트림 (YOLO의 print 문제 해결)"""
    def write(self, text):
        pass
    
    def flush(self):
        pass
    
    @property
    def encoding(self):
        return "utf-8"  # YOLO가 인코딩을 물어볼 때 답변

# sys.stdout이 없으면(GUI 모드면) 가짜로 대체
if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()
# ==========================================
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QSplitter, QGroupBox, QGridLayout, QFrame, QHeaderView, QMessageBox,
    QScrollArea, QDialog, QCheckBox, QSlider, QLineEdit, QComboBox, QFormLayout,
    QDialogButtonBox, QStyleOptionSlider, QDoubleSpinBox, QSpinBox, QInputDialog
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QTimer, QSize
)
from PyQt6.QtGui import QImage, QPixmap, QFont

import pyqtgraph as pg
from pyqtgraph import PlotWidget, ScatterPlotItem

# ============================================================================
# 외부 라이브러리 import
# ============================================================================

# YOLO (지연 import - 모델 로드 시에만 import)
YOLO_AVAILABLE = False
YOLO = None

# Dynamsoft Barcode Reader
try:
    from dynamsoft_barcode_reader_bundle import dbr, license, cvr
    DBR_AVAILABLE = True
except ImportError:
    DBR_AVAILABLE = False
    print("⚠️ dynamsoft-barcode-reader-bundle을 설치하세요: pip install dynamsoft-barcode-reader-bundle")


# ============================================================================
# Custom Widgets
# ============================================================================

class NoWheelSlider(QSlider):
    """마우스 휠 비활성화된 슬라이더"""
    def wheelEvent(self, event):
        """마우스 휠 이벤트 무시"""
        event.ignore()


# ============================================================================
# 전처리 함수들 (img.py에서 가져옴)
# ============================================================================

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE 적용"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian Blur"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral Filter"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median Blur"""
    return cv2.medianBlur(image, kernel_size)

def apply_adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """Adaptive Thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, c)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary

def apply_morphology(image: np.ndarray, operation: str = 'closing', kernel_size: int = 5) -> np.ndarray:
    """형태학적 연산"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if operation == 'closing':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif operation == 'opening':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'dilation':
        result = cv2.dilate(gray, kernel, iterations=1)
    else:
        result = gray
    
    if len(image.shape) == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# ============================================================================
# 전처리 옵션 다이얼로그
# ============================================================================

class PreprocessingDialog(QDialog):
    """전처리 옵션 설정 다이얼로그"""
    
    def __init__(self, parent=None, current_options=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 전처리 옵션")
        self.setMinimumWidth(500)
        
        # 기본 옵션 (현재 옵션이 있으면 사용)
        if current_options:
            self.options = current_options.copy()
        else:
            self.options = {
                'use_clahe': False,
                'clahe_clip_limit': 2.0,
                'clahe_tile_size': 8,
                'use_denoise': False,
                'denoise_method': 'bilateral',
                'denoise_strength': 9,
                'use_threshold': False,
                'threshold_block_size': 11,
                'threshold_c': 2,
                'use_morphology': False,
                'morphology_operation': 'closing',
                'morphology_kernel_size': 5,
            }
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        # 1. CLAHE
        self.clahe_check = QCheckBox("CLAHE 대비 향상")
        self.clahe_check.setChecked(self.options.get('use_clahe', False))
        
        self.clahe_clip = QSlider(Qt.Orientation.Horizontal)
        self.clahe_clip.setRange(10, 50)
        self.clahe_clip.setValue(int(self.options.get('clahe_clip_limit', 2.0) * 10))
        self.clahe_clip.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.clahe_clip_label = QLabel(f"{self.options.get('clahe_clip_limit', 2.0):.1f}")
        self.clahe_clip.valueChanged.connect(lambda v: self.clahe_clip_label.setText(f"{v/10:.1f}"))
        
        self.clahe_tile = QSlider(Qt.Orientation.Horizontal)
        self.clahe_tile.setRange(4, 16)
        self.clahe_tile.setValue(self.options.get('clahe_tile_size', 8))
        self.clahe_tile.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.clahe_tile_label = QLabel(str(self.options.get('clahe_tile_size', 8)))
        self.clahe_tile.valueChanged.connect(lambda v: self.clahe_tile_label.setText(str(v)))
        
        clahe_layout = QVBoxLayout()
        clahe_layout.addWidget(self.clahe_check)
        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel("Clip Limit:"))
        clip_layout.addWidget(self.clahe_clip)
        clip_layout.addWidget(self.clahe_clip_label)
        clahe_layout.addLayout(clip_layout)
        tile_layout = QHBoxLayout()
        tile_layout.addWidget(QLabel("Tile Size:"))
        tile_layout.addWidget(self.clahe_tile)
        tile_layout.addWidget(self.clahe_tile_label)
        clahe_layout.addLayout(tile_layout)
        form.addRow("", clahe_layout)
        
        # 구분선
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        form.addRow(line1)
        
        # 2. 노이즈 제거
        self.denoise_check = QCheckBox("노이즈 제거")
        self.denoise_check.setChecked(self.options.get('use_denoise', False))
        
        self.denoise_method = QComboBox()
        self.denoise_method.addItems(['bilateral', 'gaussian', 'median'])
        method = self.options.get('denoise_method', 'bilateral')
        self.denoise_method.setCurrentText(method)
        
        self.denoise_strength = QSlider(Qt.Orientation.Horizontal)
        self.denoise_strength.setRange(3, 15)
        self.denoise_strength.setValue(self.options.get('denoise_strength', 9))
        self.denoise_strength.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.denoise_strength_label = QLabel(str(self.options.get('denoise_strength', 9)))
        self.denoise_strength.valueChanged.connect(lambda v: self.denoise_strength_label.setText(str(v)))
        
        denoise_layout = QVBoxLayout()
        denoise_layout.addWidget(self.denoise_check)
        denoise_layout.addWidget(QLabel("방법:"))
        denoise_layout.addWidget(self.denoise_method)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("강도:"))
        strength_layout.addWidget(self.denoise_strength)
        strength_layout.addWidget(self.denoise_strength_label)
        denoise_layout.addLayout(strength_layout)
        form.addRow("", denoise_layout)
        
        # 구분선
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        form.addRow(line2)
        
        # 3. 이진화
        self.threshold_check = QCheckBox("적응형 이진화")
        self.threshold_check.setChecked(self.options.get('use_threshold', False))
        
        self.threshold_block = QSlider(Qt.Orientation.Horizontal)
        self.threshold_block.setRange(3, 21)
        self.threshold_block.setValue(self.options.get('threshold_block_size', 11))
        self.threshold_block.setSingleStep(2)
        self.threshold_block.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_block_label = QLabel(str(self.options.get('threshold_block_size', 11)))
        self.threshold_block.valueChanged.connect(lambda v: self.threshold_block_label.setText(str(v if v % 2 == 1 else v + 1)))
        
        self.threshold_c = QSlider(Qt.Orientation.Horizontal)
        self.threshold_c.setRange(-10, 10)
        self.threshold_c.setValue(self.options.get('threshold_c', 2))
        self.threshold_c.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_c_label = QLabel(str(self.options.get('threshold_c', 2)))
        self.threshold_c.valueChanged.connect(lambda v: self.threshold_c_label.setText(str(v)))
        
        threshold_layout = QVBoxLayout()
        threshold_layout.addWidget(self.threshold_check)
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel("Block Size:"))
        block_layout.addWidget(self.threshold_block)
        block_layout.addWidget(self.threshold_block_label)
        threshold_layout.addLayout(block_layout)
        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("C 값:"))
        c_layout.addWidget(self.threshold_c)
        c_layout.addWidget(self.threshold_c_label)
        threshold_layout.addLayout(c_layout)
        form.addRow("", threshold_layout)
        
        # 구분선
        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        form.addRow(line3)
        
        # 4. 형태학적 연산
        self.morphology_check = QCheckBox("형태학적 연산")
        self.morphology_check.setChecked(self.options.get('use_morphology', False))
        
        self.morphology_operation = QComboBox()
        self.morphology_operation.addItems(['closing', 'opening', 'dilation'])
        operation = self.options.get('morphology_operation', 'closing')
        self.morphology_operation.setCurrentText(operation)
        
        self.morphology_kernel = QSlider(Qt.Orientation.Horizontal)
        self.morphology_kernel.setRange(3, 15)
        self.morphology_kernel.setValue(self.options.get('morphology_kernel_size', 5))
        self.morphology_kernel.setSingleStep(2)
        self.morphology_kernel.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.morphology_kernel_label = QLabel(str(self.options.get('morphology_kernel_size', 5)))
        self.morphology_kernel.valueChanged.connect(lambda v: self.morphology_kernel_label.setText(str(v if v % 2 == 1 else v + 1)))
        
        morphology_layout = QVBoxLayout()
        morphology_layout.addWidget(self.morphology_check)
        morphology_layout.addWidget(QLabel("연산:"))
        morphology_layout.addWidget(self.morphology_operation)
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size:"))
        kernel_layout.addWidget(self.morphology_kernel)
        kernel_layout.addWidget(self.morphology_kernel_label)
        morphology_layout.addLayout(kernel_layout)
        form.addRow("", morphology_layout)
        
        layout.addLayout(form)
        
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QCheckBox {
                color: #00ff00;
                font-weight: bold;
            }
            QLabel {
                color: #e0e0e0;
            }
            QSlider::groove:horizontal {
                background: #3e3e3e;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff00;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox {
                background-color: #2e2e2e;
                color: #e0e0e0;
                border: 1px solid #00ff00;
                padding: 5px;
            }
        """)
    
    def get_options(self):
        """옵션 가져오기"""
        block_val = self.threshold_block.value()
        block_val = block_val if block_val % 2 == 1 else block_val + 1
        
        morph_val = self.morphology_kernel.value()
        morph_val = morph_val if morph_val % 2 == 1 else morph_val + 1
        
        return {
            'use_clahe': self.clahe_check.isChecked(),
            'clahe_clip_limit': self.clahe_clip.value() / 10.0,
            'clahe_tile_size': self.clahe_tile.value(),
            'use_denoise': self.denoise_check.isChecked(),
            'denoise_method': self.denoise_method.currentText(),
            'denoise_strength': self.denoise_strength.value(),
            'use_threshold': self.threshold_check.isChecked(),
            'threshold_block_size': block_val,
            'threshold_c': self.threshold_c.value(),
            'use_morphology': self.morphology_check.isChecked(),
            'morphology_operation': self.morphology_operation.currentText(),
            'morphology_kernel_size': morph_val,
        }


# ============================================================================
# QThread Worker 클래스 (영상 처리 스레드)
# ============================================================================

class VideoProcessorWorker(QThread):
    """
    영상 처리를 담당하는 Worker Thread
    UI 스레드와 완전히 분리하여 고성능 처리 보장
    """
    # Signal 정의
    frame_processed = pyqtSignal(np.ndarray, np.ndarray, list, dict)  # (original_frame, preprocessed_frame, detections, metrics)
    progress_updated = pyqtSignal(int, int)  # (current_frame, total_frames)
    timeline_updated = pyqtSignal(int, int, float)  # (current_frame, total_frames, current_time)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.video_path: Optional[str] = None
        self.yolo_model = None
        self.dbr_reader = None
        self.is_running = False
        self.is_paused = False
        self.conf_threshold = 0.25
        self.display_mode = 'all'  # 'all', 'success', 'fail'
        self.preprocessing_options = {}
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.seek_to_frame = -1  # 시크할 프레임 번호 (-1이면 시크 안함)
        self.frame_interval = 1  # 프레임 간격 (1=모든 프레임 처리)
        
    def set_video(self, video_path: str):
        """비디오 파일 경로 설정"""
        self.video_path = video_path
        
    def set_model(self, yolo_model, dbr_reader):
        """YOLO 및 Dynamsoft 모델 설정"""
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        
    def set_conf_threshold(self, threshold: float):
        """YOLO 신뢰도 임계값 설정"""
        self.conf_threshold = threshold
        
    def set_display_mode(self, mode: str):
        """디스플레이 모드 설정"""
        self.display_mode = mode
    
    def set_preprocessing_options(self, options: Dict):
        """전처리 옵션 설정"""
        self.preprocessing_options = options
    
    def set_frame_interval(self, interval: int):
        """프레임 간격 설정"""
        self.frame_interval = max(1, interval)
        
    def pause(self):
        """일시정지"""
        self.is_paused = True
        
    def resume(self):
        """재개"""
        self.is_paused = False
        
    def stop(self):
        """정지"""
        self.is_running = False
    
    def seek_to(self, frame_number: int):
        """특정 프레임으로 이동"""
        self.seek_to_frame = frame_number
        
    def run(self):
        """메인 처리 루프 (별도 스레드에서 실행)"""
        print(">>> Worker thread RUN started!")  # 디버그
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                print(f">>> ERROR: Video path not found: {self.video_path}")  # 디버그
                self.error_occurred.emit("비디오 파일을 찾을 수 없습니다.")
                return
                
            if self.yolo_model is None:
                print(">>> ERROR: YOLO model is None!")  # 디버그
                self.error_occurred.emit("YOLO 모델이 로드되지 않았습니다.")
                return
            
            print(f">>> Opening video: {self.video_path}")  # 디버그
            # 비디오 열기
            cap = cv2.VideoCapture(self.video_path)
            print(f">>> Video opened: {cap.isOpened()}")  # 디버그
            if not cap.isOpened():
                print(">>> ERROR: Cannot open video!")  # 디버그
                self.error_occurred.emit("비디오 파일을 열 수 없습니다.")
                return
        except Exception as e:
            print(f">>> EXCEPTION in worker setup: {e}")  # 디버그
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Worker 초기화 오류: {e}")
            return
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # 원본 FPS 유지
        
        self.is_running = True
        self.cap = cap
        self.total_frames = total_frames
        self.current_frame_idx = 0
        
        try:
            frame_counter = 0  # 프레임 간격 카운터
            while self.is_running and cap.isOpened():
                # 일시정지 처리
                while self.is_paused:
                    time.sleep(0.1)
                    if not self.is_running:
                        break
                
                if not self.is_running:
                    break
                
                # 시크 처리
                if self.seek_to_frame >= 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_to_frame)
                    self.current_frame_idx = self.seek_to_frame
                    self.seek_to_frame = -1
                    frame_counter = 0  # 시크 후 카운터 리셋
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_idx = self.current_frame_idx
                
                # 프레임 간격 체크 (frame_interval마다 한 번만 처리)
                frame_counter += 1
                if frame_counter % self.frame_interval != 0:
                    # 타임라인만 업데이트하고 건너뛰기
                    current_time = frame_idx / fps if fps > 0 else 0
                    self.timeline_updated.emit(frame_idx, total_frames, current_time)
                    continue
                
                start_time = time.time()
                
                # 타임라인 정보 전송
                current_time = frame_idx / fps if fps > 0 else 0
                self.timeline_updated.emit(frame_idx, total_frames, current_time)
                
                # 원본 프레임 저장
                original_frame = frame.copy()
                
                # 전처리 적용
                preprocessed_frame = self._apply_preprocessing(frame.copy())
                
                # YOLO 탐지 (전처리된 프레임 사용)
                detections = self._detect_qr_codes(preprocessed_frame)
                
                # Dynamsoft 해독
                for det in detections:
                    self._decode_qr_code(preprocessed_frame, det)
                
                # 분석 지표 계산
                metrics = self._calculate_metrics(preprocessed_frame, detections)
                metrics['frame_idx'] = frame_idx
                metrics['frame_no'] = frame_idx  # on_frame_processed에서 사용
                metrics['total_frames'] = total_frames
                metrics['has_success'] = any(d.get('success', False) for d in detections)
                
                # 시각화된 프레임 생성 (원본과 전처리 모두)
                vis_original = self._visualize_frame(original_frame.copy(), detections)
                vis_preprocessed = self._visualize_frame(preprocessed_frame.copy(), detections)
                
                # Signal 발송
                self.frame_processed.emit(vis_original, vis_preprocessed, detections, metrics)
                self.progress_updated.emit(frame_idx, total_frames)
                
                # FPS 유지
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            self.error_occurred.emit(f"처리 중 오류 발생: {str(e)}")
        finally:
            cap.release()
            self.finished.emit()
    
    def _apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """전처리 적용"""
        result = frame.copy()
        opts = self.preprocessing_options
        
        if not opts:
            return result
        
        # CLAHE
        if opts.get('use_clahe', False):
            result = apply_clahe(result, opts.get('clahe_clip_limit', 2.0), opts.get('clahe_tile_size', 8))
        
        # 노이즈 제거
        if opts.get('use_denoise', False):
            method = opts.get('denoise_method', 'bilateral')
            strength = opts.get('denoise_strength', 9)
            if method == 'bilateral':
                result = apply_bilateral_filter(result, strength, 75, 75)
            elif method == 'gaussian':
                result = apply_gaussian_blur(result, strength)
            elif method == 'median':
                result = apply_median_blur(result, strength)
        
        # 이진화
        if opts.get('use_threshold', False):
            result = apply_adaptive_threshold(result, opts.get('threshold_block_size', 11), opts.get('threshold_c', 2))
        
        # 형태학적 연산
        if opts.get('use_morphology', False):
            result = apply_morphology(result, opts.get('morphology_operation', 'closing'), opts.get('morphology_kernel_size', 5))
        
        return result
            
    def _detect_qr_codes(self, frame: np.ndarray) -> List[Dict]:
        """YOLO로 QR 코드 탐지"""
        detections = []
        try:
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                h, w = frame.shape[:2]
                for box in result.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # 패딩 추가
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'text': '',
                        'quad': None,
                        'success': False,
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'area': (x2 - x1) * (y2 - y1)
                    })
        except Exception as e:
            print(f"YOLO 탐지 오류: {e}")
            
        return detections
    
    def _decode_qr_code(self, frame: np.ndarray, detection: Dict):
        """Dynamsoft로 QR 코드 해독"""
        if self.dbr_reader is None:
            print(f">>> [DECODE] ERROR: dbr_reader is None!")  # 디버그
            return
            
        try:
            print(f">>> [DECODE] Starting decode for bbox: {detection.get('bbox', 'no bbox')}")  # 디버그
            
            x1, y1, x2, y2 = detection['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                print(f">>> [DECODE] ERROR: ROI is empty!")  # 디버그
                return
            
            print(f">>> [DECODE] ROI shape: {roi.shape}, dtype: {roi.dtype}")  # 디버그
            
            # RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            print(f">>> [DECODE] RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")  # 디버그
            
            # Dynamsoft 해독
            print(f">>> [DECODE] Calling capture()...")  # 디버그
            captured_result = self.dbr_reader.capture(rgb_image, dbr.EnumImagePixelFormat.IPF_RGB_888)
            print(f">>> [DECODE] Capture returned: {captured_result}")  # 디버그
            
            # 방법 1: get_decoded_barcodes_result() 시도
            barcode_result = None
            items = None
            
            if hasattr(captured_result, 'get_decoded_barcodes_result'):
                barcode_result = captured_result.get_decoded_barcodes_result()
                print(f">>> [DECODE] get_decoded_barcodes_result(): {barcode_result}")  # 디버그
                
                if barcode_result:
                    items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
            
            # 방법 2: 직접 items 속성 접근 시도
            if not items and hasattr(captured_result, 'items'):
                items = captured_result.items
                print(f">>> [DECODE] Direct items access: {items}")  # 디버그
            
            # 방법 3: decoded_barcodes_result 속성 시도
            if not items and hasattr(captured_result, 'decoded_barcodes_result'):
                barcode_result = captured_result.decoded_barcodes_result
                print(f">>> [DECODE] decoded_barcodes_result property: {barcode_result}")  # 디버그
                if barcode_result:
                    items = barcode_result.items if hasattr(barcode_result, 'items') else None
            
            print(f">>> [DECODE] Final items: {items}, count: {len(items) if items else 0}")  # 디버그
            
            if items and len(items) > 0:
                barcode_item = items[0]
                print(f">>> [DECODE] Barcode item: {barcode_item}")  # 디버그
                
                # 텍스트 추출
                text = None
                if hasattr(barcode_item, 'get_text'):
                    text = barcode_item.get_text()
                elif hasattr(barcode_item, 'text'):
                    text = barcode_item.text
                
                print(f">>> [DECODE] Extracted text: {text}")  # 디버그
                
                # Quad 좌표 추출
                quad_xy = None
                try:
                    location = barcode_item.get_location() if hasattr(barcode_item, 'get_location') else None
                    if location:
                        result_points = location.result_points if hasattr(location, 'result_points') else None
                        if result_points:
                            quad_xy = [[int(p.x + x1), int(p.y + y1)] for p in result_points]
                except:
                    pass
                
                # Detection 업데이트
                detection['text'] = text or ''
                detection['quad'] = quad_xy
                detection['success'] = len(detection['text']) > 0
                
                print(f">>> [DECODE] SUCCESS! Text: '{text}', Success: {detection['success']}")  # 디버그
            else:
                print(f">>> [DECODE] FAIL: No items found")  # 디버그
                detection['text'] = ''
                detection['success'] = False
                    
        except Exception as e:
            print(f">>> [DECODE] EXCEPTION: {e}")  # 디버그
            import traceback
            traceback.print_exc()
            detection['text'] = ''
            detection['success'] = False
    
    def _calculate_metrics(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """분석 지표 계산"""
        metrics = {}
        
        # Blur Score (Laplacian Variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blur_score'] = blur_score
        
        # Brightness (평균 밝기)
        brightness = np.mean(gray)
        metrics['brightness'] = brightness
        
        # QR Box Size (평균)
        if detections:
            avg_area = np.mean([d['area'] for d in detections])
            metrics['qr_box_size'] = avg_area
        else:
            metrics['qr_box_size'] = 0
        
        # 인식 성공 여부
        metrics['has_success'] = any(d['success'] for d in detections)
        
        return metrics
    
    def _visualize_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """프레임에 QR 탐지 결과 시각화"""
        vis_frame = frame.copy()
        
        # 디스플레이 모드에 따른 필터링
        filtered_detections = detections
        if self.display_mode == 'success':
            filtered_detections = [d for d in detections if d['success']]
        elif self.display_mode == 'fail':
            filtered_detections = [d for d in detections if not d['success']]
        
        if not filtered_detections:
            # 탐지된 QR이 없을 때 "Searching..." 표시
            cv2.putText(vis_frame, "Searching...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # QR 코드 그리기
            for det in filtered_detections:
                color = (0, 255, 0) if det['success'] else (0, 0, 255)
                
                # Quad 사용 (우선)
                if det['quad'] and len(det['quad']) == 4:
                    quad = np.array(det['quad'], dtype=np.int32)
                    cv2.polylines(vis_frame, [quad], True, color, 2)
                else:
                    # BBox 사용
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # 텍스트 표시 (해독 성공 시)
                if det['success'] and det['text']:
                    x1, y1 = det['bbox'][:2]
                    cv2.putText(vis_frame, det['text'][:20], (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame


# ============================================================================
# 메인 윈도우 클래스
# ============================================================================

class QRAnalysisMainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        
        # 상태 변수
        self.yolo_model = None
        self.dbr_reader = None
        self.video_path = None
        self.worker = None
        self.preprocessing_options = {}
        
        # 데이터 버퍼 (실시간 그래프용)
        self.frame_indices = deque(maxlen=500)  # 최근 500 프레임
        self.success_history = deque(maxlen=500)
        self.blur_history = deque(maxlen=500)
        self.qr_size_history = deque(maxlen=500)
        self.heatmap_points = []  # 히트맵 포인트 (누적)
        
        # 통계
        self.total_frames_processed = 0
        self.total_success_frames = 0
        self.unique_qr_texts = set()
        self.current_fps = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps)
        self.fps_timer.start(1000)  # 1초마다 FPS 계산
        self.frame_count_for_fps = 0
        
        # 타임라인 제어
        self.total_video_frames = 0
        self.current_video_frame = 0
        self.is_seeking = False  # 시크바 드래그 중 여부
        
        # 로그 필터
        self.log_filter_mode = 'all'  # 'all', 'success', 'fail'
        self.all_log_entries = []  # 모든 로그 항목 저장 (필터링용)
        
        # UI 초기화
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("QR코드 영상 분석 시스템 - PyQt6")
        self.setGeometry(100, 100, 1800, 1000)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Splitter로 반응형 레이아웃
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 스크롤 내용 위젯
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        
        self.splitter.addWidget(scroll)
        
        # 스크롤 내용 레이아웃
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # 상단 컨트롤 버튼 (모델/영상 업로드만)
        control_layout = QHBoxLayout()
        
        self.btn_load_model = QPushButton("📦 모델 업로드")
        self.btn_load_model.setMinimumHeight(40)
        self.btn_load_model.clicked.connect(self.load_model)
        
        self.btn_load_video = QPushButton("🎬 영상 업로드")
        self.btn_load_video.setMinimumHeight(40)
        self.btn_load_video.clicked.connect(self.load_video)
        
        self.btn_reset = QPushButton("🔄 초기화")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #ff3838;
            }
        """)
        self.btn_reset.clicked.connect(self.reset_application)
        
        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.btn_load_video)
        control_layout.addWidget(self.btn_reset)
        control_layout.addStretch()
        
        content_layout.addLayout(control_layout)
        
        # 필터 버튼 (크기 축소) + 프레임 간격 설정
        filter_layout = QHBoxLayout()
        self.btn_show_all = QPushButton("전체")
        self.btn_show_all.setCheckable(True)
        self.btn_show_all.setChecked(True)
        self.btn_show_all.setMaximumWidth(60)
        self.btn_show_all.clicked.connect(lambda: self.set_display_mode('all'))
        
        self.btn_show_success = QPushButton("성공")
        self.btn_show_success.setCheckable(True)
        self.btn_show_success.setMaximumWidth(60)
        self.btn_show_success.clicked.connect(lambda: self.set_display_mode('success'))
        
        self.btn_show_fail = QPushButton("실패")
        self.btn_show_fail.setCheckable(True)
        self.btn_show_fail.setMaximumWidth(60)
        self.btn_show_fail.clicked.connect(lambda: self.set_display_mode('fail'))
        
        filter_layout.addWidget(QLabel("필터:"))
        filter_layout.addWidget(self.btn_show_all)
        filter_layout.addWidget(self.btn_show_success)
        filter_layout.addWidget(self.btn_show_fail)
        
        # 프레임 간격 설정
        filter_layout.addWidget(QLabel("  |  프레임 간격:"))
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 30)
        self.frame_interval_spin.setValue(1)
        self.frame_interval_spin.setSuffix(" 프레임")
        self.frame_interval_spin.setMaximumWidth(120)
        self.frame_interval_spin.setToolTip("처리할 프레임 간격 (1=모든 프레임, 2=2프레임마다 1번)")
        self.frame_interval_spin.valueChanged.connect(self.on_frame_interval_changed)
        filter_layout.addWidget(self.frame_interval_spin)
        
        filter_layout.addStretch()
        
        content_layout.addLayout(filter_layout)
        
        # 영상 플레이어 섹션 (컨트롤 버튼 + 타임라인 + 영상)
        video_section_layout = QVBoxLayout()
        
        # 영상 컨트롤 버튼 + 대시보드 (한 줄로 배치)
        video_control_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("▶️ 시작")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._debug_start_processing)
        
        self.btn_pause = QPushButton("⏸️ 일시정지")
        self.btn_pause.setMinimumHeight(40)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_processing)
        
        self.btn_stop = QPushButton("⏹️ 정지")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        
        # 타임라인 정보 라벨
        self.timeline_label = QLabel("00:00 / 00:00")
        self.timeline_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #00ff00;")
        
        video_control_layout.addWidget(self.btn_start)
        video_control_layout.addWidget(self.btn_pause)
        video_control_layout.addWidget(self.btn_stop)
        video_control_layout.addWidget(self.timeline_label)
        
        # 대시보드 (수평 배치)
        self._create_inline_dashboard(video_control_layout)
        
        # 히트맵/그래프 토글 버튼
        self.btn_heatmap = QPushButton("🗺️ 히트맵")
        self.btn_heatmap.setMinimumHeight(40)
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.clicked.connect(self.toggle_heatmap)
        
        self.btn_graphs = QPushButton("📈 그래프")
        self.btn_graphs.setMinimumHeight(40)
        self.btn_graphs.setCheckable(True)
        self.btn_graphs.clicked.connect(self.toggle_graphs)
        
        video_control_layout.addWidget(self.btn_heatmap)
        video_control_layout.addWidget(self.btn_graphs)
        video_control_layout.addStretch()
        
        video_section_layout.addLayout(video_control_layout)
        
        # 타임라인 시크바 (마우스 휠 비활성화)
        timeline_layout = QHBoxLayout()
        
        self.timeline_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_changed)
        self.timeline_slider.sliderPressed.connect(self.on_timeline_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self.on_timeline_slider_released)
        
        timeline_layout.addWidget(QLabel("⏮"))
        timeline_layout.addWidget(self.timeline_slider)
        timeline_layout.addWidget(QLabel("⏭"))
        
        video_section_layout.addLayout(timeline_layout)
        
        # 영상 플레이어 (원본 + 전처리)
        video_layout = QHBoxLayout()
        
        # 원본 영상
        original_video_group = QGroupBox("📹 원본 영상")
        original_video_layout = QVBoxLayout(original_video_group)
        self.original_video_label = QLabel()
        self.original_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_video_label.setMinimumSize(500, 375)  # 크기 증가
        self.original_video_label.setStyleSheet("QLabel { background-color: #1e1e1e; }")
        self.original_video_label.setText("원본 영상")
        original_video_layout.addWidget(self.original_video_label)
        video_layout.addWidget(original_video_group)
        
        # 전처리된 영상
        preprocessed_video_group = QGroupBox("✨ 전처리된 영상")
        preprocessed_video_layout = QVBoxLayout(preprocessed_video_group)
        self.preprocessed_video_label = QLabel()
        self.preprocessed_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preprocessed_video_label.setMinimumSize(500, 375)  # 크기 증가
        self.preprocessed_video_label.setStyleSheet("QLabel { background-color: #1e1e1e; }")
        self.preprocessed_video_label.setText("전처리된 영상")
        preprocessed_video_layout.addWidget(self.preprocessed_video_label)
        video_layout.addWidget(preprocessed_video_group)
        
        video_section_layout.addLayout(video_layout)
        
        content_layout.addLayout(video_section_layout)
        
        # 데이터 로그
        log_group = QGroupBox("📝 데이터 로그")
        log_layout = QVBoxLayout(log_group)
        
        # 로그 필터 버튼
        log_filter_layout = QHBoxLayout()
        log_filter_layout.addWidget(QLabel("로그 필터:"))
        
        self.btn_log_all = QPushButton("전체보기")
        self.btn_log_all.setCheckable(True)
        self.btn_log_all.setChecked(True)
        self.btn_log_all.clicked.connect(lambda: self.set_log_filter('all'))
        
        self.btn_log_success = QPushButton("성공만")
        self.btn_log_success.setCheckable(True)
        self.btn_log_success.clicked.connect(lambda: self.set_log_filter('success'))
        
        self.btn_log_fail = QPushButton("실패만")
        self.btn_log_fail.setCheckable(True)
        self.btn_log_fail.clicked.connect(lambda: self.set_log_filter('fail'))
        
        log_filter_layout.addWidget(self.btn_log_all)
        log_filter_layout.addWidget(self.btn_log_success)
        log_filter_layout.addWidget(self.btn_log_fail)
        log_filter_layout.addStretch()
        
        log_layout.addLayout(log_filter_layout)
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["Timestamp", "Frame No", "Decoded Data", "Status"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.setAlternatingRowColors(True)
        # 10줄 정도 보이도록 높이 설정 (헤더 + 10행 * 약 30px)
        self.log_table.setMinimumHeight(330)
        self.log_table.setMaximumHeight(330)
        
        log_layout.addWidget(self.log_table)
        content_layout.addWidget(log_group)
        
        # 히트맵 섹션 (토글 가능)
        self.heatmap_group = QGroupBox("🗺️ 공간 분포 히트맵")
        heatmap_layout = QVBoxLayout(self.heatmap_group)
        
        self.heatmap_widget = pg.PlotWidget()
        self.heatmap_widget.setBackground('#1e1e1e')
        self.heatmap_widget.setLabel('left', 'Y (px)')
        self.heatmap_widget.setLabel('bottom', 'X (px)')
        self.heatmap_widget.invertY(True)
        self.heatmap_widget.setMinimumHeight(300)
        self.heatmap_scatter = ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
        self.heatmap_widget.addItem(self.heatmap_scatter)
        
        heatmap_layout.addWidget(self.heatmap_widget)
        self.heatmap_group.hide()  # 초기에는 숨김
        content_layout.addWidget(self.heatmap_group)
        
        # 그래프 섹션 (토글 가능)
        self.graphs_group = QGroupBox("📈 실시간 분석 그래프")
        graphs_layout = QVBoxLayout(self.graphs_group)
        
        self.graph_success = pg.PlotWidget(title="인식 성공 여부")
        self.graph_success.setBackground('#1e1e1e')
        self.graph_success.setLabel('left', '성공 (1) / 실패 (0)')
        self.graph_success.setLabel('bottom', '프레임')
        self.graph_success.setYRange(0, 1.2)
        self.graph_success.setMinimumHeight(250)
        self.success_curve = self.graph_success.plot(pen=pg.mkPen(color=(0, 255, 0), width=2))
        
        self.graph_metrics = pg.PlotWidget(title="QR 크기 & Blur")
        self.graph_metrics.setBackground('#1e1e1e')
        self.graph_metrics.setLabel('left', '정규화된 값')
        self.graph_metrics.setLabel('bottom', '프레임')
        self.graph_metrics.setMinimumHeight(250)
        self.qr_size_curve = self.graph_metrics.plot(pen=pg.mkPen(color=(255, 255, 0), width=2), name='QR Size')
        self.blur_curve = self.graph_metrics.plot(pen=pg.mkPen(color=(0, 255, 255), width=2), name='Blur')
        self.graph_metrics.addLegend()
        
        graphs_layout.addWidget(self.graph_success)
        graphs_layout.addWidget(self.graph_metrics)
        self.graphs_group.hide()  # 초기에는 숨김
        content_layout.addWidget(self.graphs_group)
        
        # 사이드바 (전처리 옵션 패널) - 처음에는 숨김
        self.sidebar = self._create_preprocessing_sidebar()
        self.sidebar.hide()
        
        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 1)  # 메인 화면
        self.splitter.setStretchFactor(1, 0)  # 사이드바
        
        # 초기 사이드바 크기를 0으로 설정 (숨김)
        self.splitter.setSizes([self.width(), 0])
        
        main_layout.addWidget(self.splitter)
        
        # 메인 화면에 햄버거 버튼 추가 (처음에만 표시)
        self.btn_main_toggle = QPushButton("≡", scroll_content)
        self.btn_main_toggle.setFixedSize(50, 50)
        self.btn_main_toggle.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: #2e2e2e;
                color: #00ff00;
                border: 2px solid #00ff00;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #00ff00;
                color: #000000;
            }
        """)
        self.btn_main_toggle.clicked.connect(self.toggle_sidebar)
        self.btn_main_toggle.move(scroll_content.width() - 60, 10)
        self.btn_main_toggle.raise_()
    
    def _create_inline_dashboard(self, layout: QHBoxLayout):
        """인라인 대시보드 생성 (수평 배치)"""
        self.lbl_recognition_rate = QLabel("<b>인식률</b> <span style='color:#00ff00;'>0.0%</span>")
        self.lbl_fps = QLabel("<b>FPS</b> <span style='color:#ff00ff;'>0</span>")
        self.lbl_unique_qr = QLabel("<b>고유QR</b> <span style='color:#ffff00;'>0</span>")
        self.lbl_blur_score = QLabel("<b>Blur</b> <span style='color:#00ffff;'>0.0</span>")
        
        # 스타일 설정
        for lbl in [self.lbl_recognition_rate, self.lbl_fps, self.lbl_unique_qr, self.lbl_blur_score]:
            lbl.setStyleSheet("font-size: 11pt; padding: 5px; margin: 0px 5px;")
        
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.lbl_recognition_rate)
        layout.addWidget(self.lbl_fps)
        layout.addWidget(self.lbl_unique_qr)
        layout.addWidget(self.lbl_blur_score)
        layout.addWidget(QLabel("|"))
    
    def _create_preprocessing_sidebar(self) -> QWidget:
        """전처리 옵션 사이드바 생성"""
        sidebar = QWidget()
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(350)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 상단 헤더 (제목 + 닫기 버튼)
        header_layout = QHBoxLayout()
        
        # 제목
        title = QLabel("⚙️ 전처리 옵션")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00ff00;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 햄버거 메뉴 버튼 (사이드바 내부)
        self.btn_sidebar_toggle = QPushButton("≡")
        self.btn_sidebar_toggle.setFixedSize(50, 50)
        self.btn_sidebar_toggle.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: #2e2e2e;
                color: #00ff00;
                border: 2px solid #00ff00;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #00ff00;
                color: #000000;
            }
        """)
        self.btn_sidebar_toggle.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.btn_sidebar_toggle)
        
        layout.addLayout(header_layout)
        
        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        form = QVBoxLayout(scroll_content)
        
        # 1. CLAHE
        self.side_clahe_check = QCheckBox("CLAHE 대비 향상")
        form.addWidget(self.side_clahe_check)
        
        # Clip Limit (소숫점 입력)
        form.addWidget(QLabel("Clip Limit:"))
        clip_layout = QHBoxLayout()
        self.side_clahe_clip = QSlider(Qt.Orientation.Horizontal)
        self.side_clahe_clip.setRange(10, 50)
        self.side_clahe_clip.setValue(20)
        self.side_clahe_clip_spin = QDoubleSpinBox()
        self.side_clahe_clip_spin.setRange(1.0, 5.0)
        self.side_clahe_clip_spin.setSingleStep(0.1)
        self.side_clahe_clip_spin.setDecimals(1)
        self.side_clahe_clip_spin.setValue(2.0)
        self.side_clahe_clip_spin.setMaximumWidth(80)
        # 양방향 연동
        self.side_clahe_clip.valueChanged.connect(lambda v: self.side_clahe_clip_spin.setValue(v/10))
        self.side_clahe_clip_spin.valueChanged.connect(lambda v: self.side_clahe_clip.setValue(int(v*10)))
        clip_layout.addWidget(self.side_clahe_clip)
        clip_layout.addWidget(self.side_clahe_clip_spin)
        form.addLayout(clip_layout)
        
        # Tile Size (정수 입력)
        form.addWidget(QLabel("Tile Size:"))
        tile_layout = QHBoxLayout()
        self.side_clahe_tile = QSlider(Qt.Orientation.Horizontal)
        self.side_clahe_tile.setRange(4, 16)
        self.side_clahe_tile.setValue(8)
        self.side_clahe_tile_spin = QSpinBox()
        self.side_clahe_tile_spin.setRange(4, 16)
        self.side_clahe_tile_spin.setValue(8)
        self.side_clahe_tile_spin.setMaximumWidth(80)
        # 양방향 연동
        self.side_clahe_tile.valueChanged.connect(self.side_clahe_tile_spin.setValue)
        self.side_clahe_tile_spin.valueChanged.connect(self.side_clahe_tile.setValue)
        tile_layout.addWidget(self.side_clahe_tile)
        tile_layout.addWidget(self.side_clahe_tile_spin)
        form.addLayout(tile_layout)
        
        # 구분선
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        form.addWidget(line1)
        
        # 2. 노이즈 제거
        self.side_denoise_check = QCheckBox("노이즈 제거")
        form.addWidget(self.side_denoise_check)
        
        self.side_denoise_method = QComboBox()
        self.side_denoise_method.addItems(['bilateral', 'gaussian', 'median'])
        form.addWidget(QLabel("방법:"))
        form.addWidget(self.side_denoise_method)
        
        # 강도 (정수 입력)
        form.addWidget(QLabel("강도:"))
        strength_layout = QHBoxLayout()
        self.side_denoise_strength = QSlider(Qt.Orientation.Horizontal)
        self.side_denoise_strength.setRange(3, 15)
        self.side_denoise_strength.setValue(9)
        self.side_denoise_strength_spin = QSpinBox()
        self.side_denoise_strength_spin.setRange(3, 15)
        self.side_denoise_strength_spin.setValue(9)
        self.side_denoise_strength_spin.setMaximumWidth(80)
        # 양방향 연동
        self.side_denoise_strength.valueChanged.connect(self.side_denoise_strength_spin.setValue)
        self.side_denoise_strength_spin.valueChanged.connect(self.side_denoise_strength.setValue)
        strength_layout.addWidget(self.side_denoise_strength)
        strength_layout.addWidget(self.side_denoise_strength_spin)
        form.addLayout(strength_layout)
        
        # 구분선
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        form.addWidget(line2)
        
        # 3. 이진화
        self.side_threshold_check = QCheckBox("적응형 이진화")
        form.addWidget(self.side_threshold_check)
        
        # Block Size (홀수만, 정수 입력)
        form.addWidget(QLabel("Block Size (홀수):"))
        block_layout = QHBoxLayout()
        self.side_threshold_block = QSlider(Qt.Orientation.Horizontal)
        self.side_threshold_block.setRange(3, 21)
        self.side_threshold_block.setValue(11)
        self.side_threshold_block.setSingleStep(2)
        self.side_threshold_block_spin = QSpinBox()
        self.side_threshold_block_spin.setRange(3, 21)
        self.side_threshold_block_spin.setSingleStep(2)
        self.side_threshold_block_spin.setValue(11)
        self.side_threshold_block_spin.setMaximumWidth(80)
        # 양방향 연동 (홀수 강제)
        def sync_block_slider_to_spin(v):
            odd_v = v if v % 2 == 1 else v + 1
            self.side_threshold_block_spin.setValue(odd_v)
        def sync_block_spin_to_slider(v):
            odd_v = v if v % 2 == 1 else v + 1
            self.side_threshold_block.setValue(odd_v)
        self.side_threshold_block.valueChanged.connect(sync_block_slider_to_spin)
        self.side_threshold_block_spin.valueChanged.connect(sync_block_spin_to_slider)
        block_layout.addWidget(self.side_threshold_block)
        block_layout.addWidget(self.side_threshold_block_spin)
        form.addLayout(block_layout)
        
        # C 값 (소숫점 입력)
        form.addWidget(QLabel("C 값:"))
        c_layout = QHBoxLayout()
        self.side_threshold_c = QSlider(Qt.Orientation.Horizontal)
        self.side_threshold_c.setRange(-100, 100)
        self.side_threshold_c.setValue(20)
        self.side_threshold_c_spin = QDoubleSpinBox()
        self.side_threshold_c_spin.setRange(-10.0, 10.0)
        self.side_threshold_c_spin.setSingleStep(0.1)
        self.side_threshold_c_spin.setDecimals(1)
        self.side_threshold_c_spin.setValue(2.0)
        self.side_threshold_c_spin.setMaximumWidth(80)
        # 양방향 연동
        self.side_threshold_c.valueChanged.connect(lambda v: self.side_threshold_c_spin.setValue(v/10))
        self.side_threshold_c_spin.valueChanged.connect(lambda v: self.side_threshold_c.setValue(int(v*10)))
        c_layout.addWidget(self.side_threshold_c)
        c_layout.addWidget(self.side_threshold_c_spin)
        form.addLayout(c_layout)
        
        # 구분선
        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        form.addWidget(line3)
        
        # 4. 형태학적 연산
        self.side_morphology_check = QCheckBox("형태학적 연산")
        form.addWidget(self.side_morphology_check)
        
        self.side_morphology_operation = QComboBox()
        self.side_morphology_operation.addItems(['closing', 'opening', 'dilation'])
        form.addWidget(QLabel("연산:"))
        form.addWidget(self.side_morphology_operation)
        
        # Kernel Size (홀수만, 정수 입력)
        form.addWidget(QLabel("Kernel Size (홀수):"))
        kernel_layout = QHBoxLayout()
        self.side_morphology_kernel = QSlider(Qt.Orientation.Horizontal)
        self.side_morphology_kernel.setRange(3, 15)
        self.side_morphology_kernel.setValue(5)
        self.side_morphology_kernel.setSingleStep(2)
        self.side_morphology_kernel_spin = QSpinBox()
        self.side_morphology_kernel_spin.setRange(3, 15)
        self.side_morphology_kernel_spin.setSingleStep(2)
        self.side_morphology_kernel_spin.setValue(5)
        self.side_morphology_kernel_spin.setMaximumWidth(80)
        # 양방향 연동 (홀수 강제)
        def sync_kernel_slider_to_spin(v):
            odd_v = v if v % 2 == 1 else v + 1
            self.side_morphology_kernel_spin.setValue(odd_v)
        def sync_kernel_spin_to_slider(v):
            odd_v = v if v % 2 == 1 else v + 1
            self.side_morphology_kernel.setValue(odd_v)
        self.side_morphology_kernel.valueChanged.connect(sync_kernel_slider_to_spin)
        self.side_morphology_kernel_spin.valueChanged.connect(sync_kernel_spin_to_slider)
        kernel_layout.addWidget(self.side_morphology_kernel)
        kernel_layout.addWidget(self.side_morphology_kernel_spin)
        form.addLayout(kernel_layout)
        
        form.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # 적용 버튼
        apply_btn = QPushButton("✅ 적용")
        apply_btn.setMinimumHeight(40)
        apply_btn.clicked.connect(self.apply_sidebar_preprocessing)
        layout.addWidget(apply_btn)
        
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #252525;
            }
            QCheckBox {
                color: #00ff00;
                font-weight: bold;
            }
            QLabel {
                color: #e0e0e0;
            }
            QSlider::groove:horizontal {
                background: #3e3e3e;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff00;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox {
                background-color: #2e2e2e;
                color: #e0e0e0;
                border: 1px solid #00ff00;
                padding: 5px;
            }
        """)
        
        return sidebar
    
    def _create_metric_label(self, title: str, value: str, size: int = 24) -> QLabel:
        """지표 레이블 생성"""
        label = QLabel(f"<b>{title}</b><br><span style='font-size:{size}px; color:#00ff00;'>{value}</span>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        label.setMinimumHeight(60)
        return label
    
    def apply_dark_theme(self):
        """Dark Theme 적용"""
        dark_stylesheet = """
        QMainWindow {
            background-color: #121212;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Segoe UI', Arial;
            font-size: 11pt;
        }
        QScrollArea {
            background-color: #1e1e1e;
            border: none;
        }
        QGroupBox {
            border: 2px solid #00ff00;
            border-radius: 8px;
            margin-top: 10px;
            font-weight: bold;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;
            color: #00ff00;
        }
        QPushButton {
            background-color: #2e2e2e;
            color: #ffffff;
            border: 2px solid #00ff00;
            border-radius: 6px;
            padding: 8px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #00ff00;
            color: #000000;
        }
        QPushButton:pressed {
            background-color: #00aa00;
        }
        QPushButton:disabled {
            background-color: #1e1e1e;
            color: #666666;
            border-color: #666666;
        }
        QPushButton:checked {
            background-color: #00ff00;
            color: #000000;
        }
        QLabel {
            background-color: transparent;
            color: #e0e0e0;
        }
        QTableWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
            gridline-color: #333333;
            border: 1px solid #00ff00;
        }
        QHeaderView::section {
            background-color: #2e2e2e;
            color: #00ff00;
            padding: 5px;
            border: 1px solid #00ff00;
            font-weight: bold;
        }
        QTableWidget::item {
            padding: 5px;
        }
        QTableWidget::item:alternate {
            background-color: #252525;
        }
        """
        self.setStyleSheet(dark_stylesheet)
    
    # ============================================================================
    # 이벤트 핸들러
    # ============================================================================
    
    def load_model(self):
        """YOLO 모델 업로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO 모델 파일 선택", "", "YOLO Models (*.pt *.pth)"
        )
        
        if not file_path:
            return
        
        try:
            # YOLO 지연 import
            global YOLO, YOLO_AVAILABLE
            try:
                # PyInstaller 단일 exe 환경 지원
                if getattr(sys, 'frozen', False):
                    # PyInstaller로 패키징된 경우
                    bundle_dir = sys._MEIPASS
                    # ultralytics가 필요한 경로를 환경 변수로 설정
                    os.environ['TORCH_HOME'] = os.path.join(bundle_dir, 'torch')
                    os.environ['YOLO_CONFIG_DIR'] = os.path.join(bundle_dir, 'ultralytics', 'cfg')
                    # sys.path에 추가
                    if bundle_dir not in sys.path:
                        sys.path.insert(0, bundle_dir)
                
                from ultralytics import YOLO
                YOLO_AVAILABLE = True
            except Exception as e:
                QMessageBox.critical(self, "오류", f"ultralytics를 로드할 수 없습니다:\n{str(e)}\n\nPyTorch CPU 버전을 설치하세요:\npip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
                return
            
            self.yolo_model = YOLO(file_path)
            QMessageBox.information(self, "성공", f"YOLO 모델 로드 완료!\n{os.path.basename(file_path)}")
            
            # Dynamsoft 초기화
            if DBR_AVAILABLE and self.dbr_reader is None:
                try:
                    license_key = os.environ.get(
                        'DYNAMSOFT_LICENSE_KEY',
                        't0085YQEAADYdcL2llMa8vH1Rtnun+43saE/kdAE7ZbIxMQGRMtSzVSZRI8vfOK4Ids52rjekwzh87yABFLraXw5Va1BV7NnBjI8m7qbw3kxOprI75ExJpw=='
                    )
                    error = license.LicenseManager.init_license(license_key)
                    if error[0] == 0:
                        self.dbr_reader = cvr.CaptureVisionRouter()
                        QMessageBox.information(self, "성공", "Dynamsoft 초기화 완료!")
                    else:
                        QMessageBox.warning(self, "경고", f"Dynamsoft 라이선스 오류: {error[1]}")
                except Exception as e:
                    QMessageBox.warning(self, "경고", f"Dynamsoft 초기화 실패: {str(e)}")
            
            self._update_button_states()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"모델 로드 실패:\n{str(e)}")
    
    def load_video(self):
        """영상 파일 업로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if not file_path:
            return
        
        self.video_path = file_path
        QMessageBox.information(self, "성공", f"영상 로드 완료!\n{os.path.basename(file_path)}")
        self._update_button_states()
    
    def toggle_sidebar(self):
        """사이드바 토글 (전처리 옵션)"""
        if self.sidebar.isVisible():
            # 사이드바 닫기
            self.sidebar.hide()
            # Splitter 크기 조정 - 메인만 전체 사용
            total_width = self.splitter.width()
            self.splitter.setSizes([total_width, 0])
            # 메인 화면의 햄버거 버튼 표시
            self.btn_main_toggle.show()
        else:
            # 사이드바 열기
            self.sidebar.show()
            # Splitter 크기 조정 (반응형) - 메인 화면 축소
            total_width = self.splitter.width()
            sidebar_width = 320
            self.splitter.setSizes([total_width - sidebar_width, sidebar_width])
            # 메인 화면의 햄버거 버튼 숨김
            self.btn_main_toggle.hide()
            
            # 현재 옵션 값으로 UI 업데이트
            if self.preprocessing_options:
                self._update_sidebar_from_options()
    
    def resizeEvent(self, event):
        """윈도우 크기 변경 시 햄버거 버튼 위치 조정"""
        super().resizeEvent(event)
        # 메인 화면의 햄버거 버튼을 오른쪽 상단에 고정
        if hasattr(self, 'btn_main_toggle') and self.btn_main_toggle.isVisible():
            # 스크롤 영역의 viewport 너비 기준으로 위치 조정
            scroll_widget = self.splitter.widget(0)  # 스크롤 영역
            if scroll_widget and hasattr(scroll_widget, 'viewport'):
                viewport_width = scroll_widget.viewport().width()
                self.btn_main_toggle.move(viewport_width - 60, 10)
    
    def _update_sidebar_from_options(self):
        """사이드바를 현재 전처리 옵션 값으로 업데이트"""
        opts = self.preprocessing_options
        if not opts:
            return
        
        self.side_clahe_check.setChecked(opts.get('use_clahe', False))
        self.side_clahe_clip_spin.setValue(opts.get('clahe_clip_limit', 2.0))
        self.side_clahe_tile_spin.setValue(opts.get('clahe_tile_size', 8))
        
        self.side_denoise_check.setChecked(opts.get('use_denoise', False))
        self.side_denoise_method.setCurrentText(opts.get('denoise_method', 'bilateral'))
        self.side_denoise_strength_spin.setValue(opts.get('denoise_strength', 9))
        
        self.side_threshold_check.setChecked(opts.get('use_threshold', False))
        self.side_threshold_block_spin.setValue(opts.get('threshold_block_size', 11))
        self.side_threshold_c_spin.setValue(opts.get('threshold_c', 2.0))
        
        self.side_morphology_check.setChecked(opts.get('use_morphology', False))
        self.side_morphology_operation.setCurrentText(opts.get('morphology_operation', 'closing'))
        self.side_morphology_kernel_spin.setValue(opts.get('morphology_kernel_size', 5))
    
    def apply_sidebar_preprocessing(self):
        """사이드바 전처리 옵션 적용"""
        self.preprocessing_options = {
            'use_clahe': self.side_clahe_check.isChecked(),
            'clahe_clip_limit': self.side_clahe_clip_spin.value(),
            'clahe_tile_size': self.side_clahe_tile_spin.value(),
            'use_denoise': self.side_denoise_check.isChecked(),
            'denoise_method': self.side_denoise_method.currentText(),
            'denoise_strength': self.side_denoise_strength_spin.value(),
            'use_threshold': self.side_threshold_check.isChecked(),
            'threshold_block_size': self.side_threshold_block_spin.value(),
            'threshold_c': self.side_threshold_c_spin.value(),
            'use_morphology': self.side_morphology_check.isChecked(),
            'morphology_operation': self.side_morphology_operation.currentText(),
            'morphology_kernel_size': self.side_morphology_kernel_spin.value(),
        }
        
        QMessageBox.information(self, "성공", "전처리 옵션이 적용되었습니다!")
        
        # Worker에 전처리 옵션 전달
        if self.worker:
            self.worker.set_preprocessing_options(self.preprocessing_options)
    
    def toggle_heatmap(self):
        """히트맵 섹션 토글"""
        if self.heatmap_group.isVisible():
            self.heatmap_group.hide()
            self.btn_heatmap.setChecked(False)
        else:
            self.heatmap_group.show()
            self.btn_heatmap.setChecked(True)
    
    def toggle_graphs(self):
        """그래프 섹션 토글"""
        if self.graphs_group.isVisible():
            self.graphs_group.hide()
            self.btn_graphs.setChecked(False)
        else:
            self.graphs_group.show()
            self.btn_graphs.setChecked(True)
    
    def _debug_start_processing(self):
        """디버그용 시작 처리 래퍼"""
        print("\n" + "="*60)
        print("=== BUTTON CLICKED ===")
        print("="*60)
        import traceback
        traceback.print_stack()
        print("="*60 + "\n")
        self.start_processing()
    
    def start_processing(self):
        """영상 처리 시작"""
        print("\n" + "="*60)
        print("=== START PROCESSING CALLED ===")
        print("="*60)
        
        if not self.yolo_model or not self.video_path:
            print(">>> ERROR: Model or video not loaded!")
            QMessageBox.warning(self, "경고", "모델과 영상을 먼저 로드하세요.")
            return
        
        print(f">>> Model: {self.yolo_model}")
        print(f">>> Video: {self.video_path}")
        print(f">>> DBR Reader: {self.dbr_reader}")
        print(">>> Initializing data...")
        
        # 데이터 초기화
        self.frame_indices.clear()
        self.success_history.clear()
        self.blur_history.clear()
        self.qr_size_history.clear()
        self.heatmap_points.clear()
        self.total_frames_processed = 0
        self.total_success_frames = 0
        self.unique_qr_texts.clear()
        self.log_table.setRowCount(0)
        self.all_log_entries.clear()
        
        print("Creating worker thread...")  # 디버그용
        
        # Worker Thread 생성 및 시작
        self.worker = VideoProcessorWorker()
        self.worker.set_video(self.video_path)
        self.worker.set_model(self.yolo_model, self.dbr_reader)
        self.worker.set_preprocessing_options(self.preprocessing_options)
        self.worker.set_frame_interval(self.frame_interval_spin.value())
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.timeline_updated.connect(self.on_timeline_updated)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error_occurred.connect(self.on_error)
        print("Starting worker thread...")  # 디버그
        self.worker.start()
        print("Worker thread started!")  # 디버그
        
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.timeline_slider.setEnabled(True)
    
    def pause_processing(self):
        """일시정지/재개"""
        if self.worker and self.worker.is_running:
            if self.worker.is_paused:
                self.worker.resume()
                self.btn_pause.setText("⏸️ 일시정지")
            else:
                self.worker.pause()
                self.btn_pause.setText("▶️ 재개")
    
    def stop_processing(self):
        """정지"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸️ 일시정지")
        self.timeline_slider.setEnabled(False)
    
    def set_display_mode(self, mode: str):
        """디스플레이 모드 설정 (영상 표시)"""
        # 버튼 상태 업데이트
        self.btn_show_all.setChecked(mode == 'all')
        self.btn_show_success.setChecked(mode == 'success')
        self.btn_show_fail.setChecked(mode == 'fail')
        
        # Worker에 전달
        if self.worker:
            self.worker.set_display_mode(mode)
    
    def on_frame_interval_changed(self, value: int):
        """프레임 간격 변경"""
        if self.worker and self.worker.isRunning():
            self.worker.set_frame_interval(value)
    
    def set_log_filter(self, mode: str):
        """로그 필터 모드 설정"""
        # 버튼 상태 업데이트
        self.btn_log_all.setChecked(mode == 'all')
        self.btn_log_success.setChecked(mode == 'success')
        self.btn_log_fail.setChecked(mode == 'fail')
        
        self.log_filter_mode = mode
        self._refresh_log_table()
    
    def _on_frame_processed(self, original_frame: np.ndarray, preprocessed_frame: np.ndarray, 
                           detections: List[Dict], metrics: Dict):
        """프레임 처리 완료 시 호출 (시그널 핸들러)"""
        try:
            print(f">>> Frame processed! Frame: {metrics.get('frame_no', '?')}, QRs: {len(detections)}")  # 디버그
            self.on_frame_processed(original_frame, preprocessed_frame, detections, metrics)
        except Exception as e:
            print(f">>> EXCEPTION in _on_frame_processed: {e}")  # 디버그
            import traceback
            traceback.print_exc()
    
    def on_frame_processed(self, original_frame: np.ndarray, preprocessed_frame: np.ndarray, 
                          detections: List[Dict], metrics: Dict):
        """프레임 처리 완료 시 UI 업데이트"""
        # FPS 카운터
        self.frame_count_for_fps += 1
        
        # 통계 업데이트
        self.total_frames_processed += 1
        if metrics.get('has_success', False):
            self.total_success_frames += 1
        
        # 고유 QR 텍스트 저장
        for det in detections:
            if det['success'] and det['text']:
                self.unique_qr_texts.add(det['text'])
        
        # 영상 표시
        self._display_frame(self.original_video_label, original_frame)
        self._display_frame(self.preprocessed_video_label, preprocessed_frame)
        
        # 데이터 버퍼 업데이트
        frame_idx = metrics.get('frame_idx', self.total_frames_processed)
        self.frame_indices.append(frame_idx)
        self.success_history.append(1 if metrics.get('has_success', False) else 0)
        self.blur_history.append(metrics.get('blur_score', 0))
        self.qr_size_history.append(metrics.get('qr_box_size', 0))
        
        # 히트맵 포인트 추가
        for det in detections:
            if det['success']:
                self.heatmap_points.append(det['center'])
        
        # 그래프 업데이트
        self._update_graphs()
        
        # 대시보드 업데이트
        self._update_dashboard(metrics)
        
        # 로그 테이블 업데이트 (성공 및 실패 모두 기록)
        for det in detections:
            if det['success']:
                self._add_log_entry(frame_idx, det['text'], "✅ 성공")
            else:
                self._add_log_entry(frame_idx, "인식 실패", "❌ 실패")
    
    def on_processing_finished(self):
        """처리 완료"""
        QMessageBox.information(self, "완료", "영상 처리가 완료되었습니다!")
        self.stop_processing()
    
    def on_error(self, error_msg: str):
        """오류 발생"""
        print(f">>> ERROR SIGNAL RECEIVED: {error_msg}")  # 디버그
        QMessageBox.critical(self, "오류", error_msg)
        self.stop_processing()
    
    def on_timeline_updated(self, current_frame: int, total_frames: int, current_time: float):
        """타임라인 업데이트"""
        if not self.is_seeking:
            # 시크바 업데이트
            if total_frames > 0:
                progress = int((current_frame / total_frames) * 100)
                self.timeline_slider.setValue(progress)
            
            # 시간 라벨 업데이트
            total_time = (total_frames / 30.0) if total_frames > 0 else 0  # 임시로 30fps 가정
            current_minutes = int(current_time // 60)
            current_seconds = int(current_time % 60)
            total_minutes = int(total_time // 60)
            total_seconds = int(total_time % 60)
            
            self.timeline_label.setText(
                f"{current_minutes:02d}:{current_seconds:02d} / {total_minutes:02d}:{total_seconds:02d}"
            )
            
            # 내부 변수 업데이트
            self.total_video_frames = total_frames
            self.current_video_frame = current_frame
    
    def on_timeline_slider_pressed(self):
        """시크바 드래그 시작"""
        self.is_seeking = True
    
    def on_timeline_slider_released(self):
        """시크바 드래그 종료 - 실제 시크 수행"""
        self.is_seeking = False
        if self.worker and self.worker.is_running and self.total_video_frames > 0:
            # 슬라이더 값을 프레임 번호로 변환
            progress = self.timeline_slider.value()
            target_frame = int((progress / 100.0) * self.total_video_frames)
            self.worker.seek_to(target_frame)
    
    def on_timeline_slider_changed(self, value):
        """시크바 값 변경 - 드래그 중에는 시간만 업데이트"""
        if self.is_seeking and self.total_video_frames > 0:
            # 드래그 중에는 시간 라벨만 미리보기
            target_frame = int((value / 100.0) * self.total_video_frames)
            current_time = target_frame / 30.0  # 임시로 30fps 가정
            total_time = self.total_video_frames / 30.0
            
            current_minutes = int(current_time // 60)
            current_seconds = int(current_time % 60)
            total_minutes = int(total_time // 60)
            total_seconds = int(total_time % 60)
            
            self.timeline_label.setText(
                f"{current_minutes:02d}:{current_seconds:02d} / {total_minutes:02d}:{total_seconds:02d}"
            )
    
    def _refresh_log_table(self):
        """로그 테이블을 현재 필터에 맞게 새로고침"""
        self.log_table.setRowCount(0)
        
        for entry in self.all_log_entries:
            should_show = False
            if self.log_filter_mode == 'all':
                should_show = True
            elif self.log_filter_mode == 'success' and entry['is_success']:
                should_show = True
            elif self.log_filter_mode == 'fail' and not entry['is_success']:
                should_show = True
            
            if should_show:
                row_count = self.log_table.rowCount()
                self.log_table.insertRow(row_count)
                
                self.log_table.setItem(row_count, 0, QTableWidgetItem(entry['timestamp']))
                self.log_table.setItem(row_count, 1, QTableWidgetItem(str(entry['frame_no'])))
                self.log_table.setItem(row_count, 2, QTableWidgetItem(entry['decoded_data'][:50]))
                self.log_table.setItem(row_count, 3, QTableWidgetItem(entry['status']))
        
        # 자동 스크롤
        self.log_table.scrollToBottom()
    
    # ============================================================================
    # UI 업데이트 메서드
    # ============================================================================
    
    def _display_frame(self, label: QLabel, frame: np.ndarray):
        """영상 프레임 표시"""
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # QImage 생성
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 라벨 크기에 맞춰 스케일링
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
    
    def _update_graphs(self):
        """그래프 업데이트"""
        if not self.frame_indices:
            return
        
        x_data = list(self.frame_indices)
        
        # 그래프 1: 인식 성공 여부
        self.success_curve.setData(x_data, list(self.success_history))
        
        # 그래프 2: QR 크기 & Blur 점수 (정규화)
        if self.qr_size_history and max(self.qr_size_history) > 0:
            normalized_size = [s / max(self.qr_size_history) * 100 for s in self.qr_size_history]
        else:
            normalized_size = list(self.qr_size_history)
        
        if self.blur_history and max(self.blur_history) > 0:
            normalized_blur = [b / max(self.blur_history) * 100 for b in self.blur_history]
        else:
            normalized_blur = list(self.blur_history)
        
        self.qr_size_curve.setData(x_data, normalized_size)
        self.blur_curve.setData(x_data, normalized_blur)
        
        # 히트맵 업데이트
        if self.heatmap_points:
            points_array = np.array(self.heatmap_points)
            self.heatmap_scatter.setData(points_array[:, 0], points_array[:, 1])
    
    def _update_dashboard(self, metrics: Dict):
        """대시보드 업데이트 (인라인 버전)"""
        # 인식률
        if self.total_frames_processed > 0:
            recognition_rate = (self.total_success_frames / self.total_frames_processed) * 100
        else:
            recognition_rate = 0.0
        
        self.lbl_recognition_rate.setText(
            f"<b>인식률</b> <span style='color:#00ff00;'>{recognition_rate:.1f}%</span>"
        )
        
        # 고유 QR 개수
        self.lbl_unique_qr.setText(
            f"<b>고유QR</b> <span style='color:#ffff00;'>{len(self.unique_qr_texts)}</span>"
        )
        
        # Blur 점수
        blur_score = metrics.get('blur_score', 0)
        self.lbl_blur_score.setText(
            f"<b>Blur</b> <span style='color:#00ffff;'>{blur_score:.1f}</span>"
        )
    
    def _update_fps(self):
        """FPS 업데이트 (인라인 버전)"""
        self.current_fps = self.frame_count_for_fps
        self.frame_count_for_fps = 0
        
        self.lbl_fps.setText(
            f"<b>FPS</b> <span style='color:#ff00ff;'>{self.current_fps}</span>"
        )
    
    def _add_log_entry(self, frame_no: int, decoded_data: str, status: str):
        """로그 테이블에 항목 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 모든 로그 항목을 저장
        log_entry = {
            'timestamp': timestamp,
            'frame_no': frame_no,
            'decoded_data': decoded_data,
            'status': status,
            'is_success': '✅' in status
        }
        self.all_log_entries.append(log_entry)
        
        # 최대 1000개 항목 유지
        if len(self.all_log_entries) > 1000:
            self.all_log_entries.pop(0)
        
        # 현재 필터에 맞는 항목만 테이블에 추가
        should_show = False
        if self.log_filter_mode == 'all':
            should_show = True
        elif self.log_filter_mode == 'success' and log_entry['is_success']:
            should_show = True
        elif self.log_filter_mode == 'fail' and not log_entry['is_success']:
            should_show = True
        
        if should_show:
            row_count = self.log_table.rowCount()
            self.log_table.insertRow(row_count)
            
            self.log_table.setItem(row_count, 0, QTableWidgetItem(timestamp))
            self.log_table.setItem(row_count, 1, QTableWidgetItem(str(frame_no)))
            self.log_table.setItem(row_count, 2, QTableWidgetItem(decoded_data[:50]))
            self.log_table.setItem(row_count, 3, QTableWidgetItem(status))
            
            # 자동 스크롤
            self.log_table.scrollToBottom()
            
            # 최대 1000개 행 유지
            if self.log_table.rowCount() > 1000:
                self.log_table.removeRow(0)
    
    def reset_application(self):
        """애플리케이션 초기화"""
        reply = QMessageBox.question(
            self, 
            "초기화 확인",
            "모든 데이터와 설정을 초기화하시겠습니까?\n(처리 중인 영상은 중지됩니다)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Worker 중지
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()
            self.worker = None
            
            # 모델 및 영상 경로 초기화
            self.yolo_model = None
            self.video_path = None
            self.preprocessing_options = {}
            
            # 모든 데이터 초기화
            self.frame_indices.clear()
            self.success_history.clear()
            self.blur_history.clear()
            self.qr_size_history.clear()
            self.heatmap_points.clear()
            self.total_frames_processed = 0
            self.total_success_frames = 0
            self.unique_qr_texts.clear()
            self.all_log_entries.clear()
            self.total_video_frames = 0
            self.current_video_frame = 0
            
            # UI 초기화
            self.log_table.setRowCount(0)
            self.original_video_label.clear()
            self.original_video_label.setText("원본 영상")
            self.preprocessed_video_label.clear()
            self.preprocessed_video_label.setText("전처리된 영상")
            self.timeline_slider.setValue(0)
            self.timeline_label.setText("00:00 / 00:00")
            
            # 그래프 초기화
            self.success_curve.setData([], [])
            self.qr_size_curve.setData([], [])
            self.blur_curve.setData([], [])
            self.heatmap_scatter.setData([], [])
            
            # 대시보드 초기화 (인라인 버전)
            self.lbl_recognition_rate.setText("<b>인식률</b> <span style='color:#00ff00;'>0.0%</span>")
            self.lbl_fps.setText("<b>FPS</b> <span style='color:#ff00ff;'>0</span>")
            self.lbl_unique_qr.setText("<b>고유QR</b> <span style='color:#ffff00;'>0</span>")
            self.lbl_blur_score.setText("<b>Blur</b> <span style='color:#00ffff;'>0.0</span>")
            
            # 히트맵/그래프 숨김
            self.heatmap_group.hide()
            self.graphs_group.hide()
            self.btn_heatmap.setChecked(False)
            self.btn_graphs.setChecked(False)
            
            # 필터 초기화
            self.log_filter_mode = 'all'
            self.btn_log_all.setChecked(True)
            self.btn_log_success.setChecked(False)
            self.btn_log_fail.setChecked(False)
            
            self.btn_show_all.setChecked(True)
            self.btn_show_success.setChecked(False)
            self.btn_show_fail.setChecked(False)
            
            # 버튼 상태 초기화
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.btn_pause.setText("⏸️ 일시정지")
            self.timeline_slider.setEnabled(False)
            
            QMessageBox.information(self, "완료", "초기화가 완료되었습니다!")
    
    def _update_button_states(self):
        """버튼 상태 업데이트"""
        can_start = self.yolo_model is not None and self.video_path is not None
        self.btn_start.setEnabled(can_start)


# ============================================================================
# 메인 함수
# ============================================================================

class LoginDialog(QDialog):
    """로그인 다이얼로그"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR 분석 시스템 - 로그인")
        self.setModal(True)
        self.setFixedSize(350, 200)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # 제목
        title = QLabel("🔐 QR 영상 분석 시스템")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; color: #00ff00;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 패스워드 입력
        pwd_layout = QHBoxLayout()
        pwd_layout.addWidget(QLabel("패스워드:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("패스워드를 입력하세요")
        self.password_input.returnPressed.connect(self.check_password)
        pwd_layout.addWidget(self.password_input)
        layout.addLayout(pwd_layout)
        
        # 버튼
        btn_layout = QHBoxLayout()
        self.btn_login = QPushButton("로그인")
        self.btn_login.clicked.connect(self.check_password)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_login)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        # 시도 횟수
        self.attempts = 0
        self.max_attempts = 3
        
        # 스타일
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 2px solid #00ff00;
                padding: 8px;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #00ff00;
                color: #000000;
                border: none;
                padding: 10px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
            QPushButton:pressed {
                background-color: #009900;
            }
        """)
    
    def check_password(self):
        """패스워드 확인"""
        password = self.password_input.text()
        correct_password = "2017112166"
        
        if password == correct_password:
            self.accept()
        else:
            self.attempts += 1
            remaining = self.max_attempts - self.attempts
            
            if remaining > 0:
                QMessageBox.warning(
                    self,
                    "로그인 실패",
                    f"패스워드가 틀렸습니다.\n남은 시도 횟수: {remaining}회"
                )
                self.password_input.clear()
                self.password_input.setFocus()
            else:
                QMessageBox.critical(
                    self,
                    "접근 거부",
                    "로그인 시도 횟수를 초과했습니다.\n프로그램을 종료합니다."
                )
                self.reject()


# 전역 플래그: main()이 이미 실행 중인지 확인
_app_started = False

def main():
    global _app_started
    
    print("\n" + "="*60)
    print("=== MAIN() FUNCTION CALLED ===")
    print(f">>> _app_started flag: {_app_started}")
    print("="*60)
    import traceback
    traceback.print_stack()
    print("="*60 + "\n")
    
    # 이미 실행 중이면 종료
    if _app_started:
        print(">>> main() already running! Ignoring duplicate call.")
        return
    
    _app_started = True
    print(">>> Setting _app_started = True")
    
    app = QApplication(sys.argv)
    
    # 폰트 설정
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # 로그인 다이얼로그 표시
    print(">>> Creating LoginDialog...")
    login = LoginDialog()
    print(">>> Showing LoginDialog...")
    if login.exec() == QDialog.DialogCode.Accepted:
        # 로그인 성공 시 메인 윈도우 실행
        window = QRAnalysisMainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        # 로그인 실패 시 종료
        sys.exit(0)


if __name__ == '__main__':
    # Windows PyInstaller 지원
    from multiprocessing import freeze_support
    freeze_support()
    
    print(">>> __main__ block executing...")
    main()
