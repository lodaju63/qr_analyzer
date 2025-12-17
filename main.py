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
import queue
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QSplitter, QGroupBox, QGridLayout, QFrame, QHeaderView, QMessageBox,
    QScrollArea, QDialog, QCheckBox, QSlider, QLineEdit, QComboBox, QFormLayout,
    QDialogButtonBox, QStyleOptionSlider, QDoubleSpinBox, QSpinBox, QInputDialog,
    QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QTimer, QObject, QMutex, QMutexLocker, QPointF
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QWheelEvent, QMouseEvent

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


class ROIVideoLabel(QLabel):
    """ROI 그리기를 지원하는 비디오 레이블"""
    roi_changed = pyqtSignal(int, int, int, int)  # (x1, y1, x2, y2)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_mode = False
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.original_pixmap = None
        self.actual_frame_size = None  # 실제 프레임 크기 (h, w) - 원본 영상 해상도
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def set_actual_frame_size(self, height: int, width: int):
        """실제 프레임 크기 설정 (원본 영상 해상도)"""
        self.actual_frame_size = (height, width)
    
    def set_roi_mode(self, enabled: bool):
        """ROI 그리기 모드 활성화/비활성화"""
        self.roi_mode = enabled
        if not enabled:
            self.roi_rect = None
            self.roi_start = None
            self.roi_end = None
            if self.original_pixmap:
                self.setPixmap(self.original_pixmap)
        self.update()
    
    def clear_roi(self):
        """ROI 영역 초기화"""
        self.roi_rect = None
        self.roi_start = None
        self.roi_end = None
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap)
        self.update()
    
    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """현재 ROI 영역 반환 (x1, y1, x2, y2)"""
        return self.roi_rect
    
    def setPixmap(self, pixmap: QPixmap):
        """픽스맵 설정 (원본 저장)"""
        self.original_pixmap = pixmap
        if self.roi_mode and self.roi_rect:
            # ROI가 있으면 그려서 표시
            self._draw_roi_on_pixmap(pixmap.copy())
        else:
            super().setPixmap(pixmap)
    
    def _draw_roi_on_pixmap(self, pixmap: QPixmap):
        """픽스맵에 ROI 사각형 그리기"""
        if not self.roi_rect:
            super().setPixmap(pixmap)
            return
        
        from PyQt6.QtGui import QPainter, QPen, QColor
        
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 0), 2)  # 초록색, 두께 2
        painter.setPen(pen)
        
        x1, y1, x2, y2 = self.roi_rect
        # QLabel의 실제 이미지 영역 계산
        label_size = self.size()
        pixmap_size = pixmap.size()
        
        # 중앙 정렬된 픽스맵의 실제 위치 계산
        if pixmap_size.width() > 0 and pixmap_size.height() > 0:
            scale_x = pixmap_size.width() / label_size.width() if label_size.width() > 0 else 1
            scale_y = pixmap_size.height() / label_size.height() if label_size.height() > 0 else 1
            
            # 마우스 좌표를 픽스맵 좌표로 변환
            px1 = int(x1 * scale_x)
            py1 = int(y1 * scale_y)
            px2 = int(x2 * scale_x)
            py2 = int(y2 * scale_y)
            
            painter.drawRect(px1, py1, px2 - px1, py2 - py1)
        
        painter.end()
        super().setPixmap(pixmap)
    
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        if not self.roi_mode or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        
        self.drawing = True
        pos = event.position().toPoint()
        self.roi_start = (pos.x(), pos.y())
        self.roi_end = self.roi_start
    
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if not self.roi_mode:
            super().mouseMoveEvent(event)
            return
        
        pos = event.position().toPoint()
        
        if self.drawing:
            self.roi_end = (pos.x(), pos.y())
            # 실시간으로 ROI 그리기
            if self.original_pixmap:
                temp_pixmap = self.original_pixmap.copy()
                self._draw_temp_roi(temp_pixmap)
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트"""
        if not self.roi_mode or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        
        if self.drawing:
            self.drawing = False
            pos = event.position().toPoint()
            self.roi_end = (pos.x(), pos.y())
            
            # ROI 영역 계산
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            
            # 최소 크기 체크 (10x10 픽셀 이상)
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.roi_rect = (x1, y1, x2, y2)
                # 원본 프레임 좌표로 변환하여 전송
                self._emit_roi_in_frame_coords()
                # ROI 그려진 픽스맵 표시
                if self.original_pixmap:
                    self._draw_roi_on_pixmap(self.original_pixmap.copy())
            else:
                self.roi_rect = None
                if self.original_pixmap:
                    self.setPixmap(self.original_pixmap)
    
    def _draw_temp_roi(self, pixmap: QPixmap):
        """임시 ROI 사각형 그리기 (드래그 중)"""
        if not self.roi_start or not self.roi_end:
            super().setPixmap(pixmap)
            return
        
        from PyQt6.QtGui import QPainter, QPen, QColor
        
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 0), 2)  # 초록색, 두께 2
        painter.setPen(pen)
        
        label_size = self.size()
        pixmap_size = pixmap.size()
        
        if pixmap_size.width() > 0 and pixmap_size.height() > 0:
            scale_x = pixmap_size.width() / label_size.width() if label_size.width() > 0 else 1
            scale_y = pixmap_size.height() / label_size.height() if label_size.height() > 0 else 1
            
            px1 = int(self.roi_start[0] * scale_x)
            py1 = int(self.roi_start[1] * scale_y)
            px2 = int(self.roi_end[0] * scale_x)
            py2 = int(self.roi_end[1] * scale_y)
            
            painter.drawRect(px1, py1, px2 - px1, py2 - py1)
        
        painter.end()
        super().setPixmap(pixmap)
    
    def _emit_roi_in_frame_coords(self):
        """ROI를 원본 프레임 좌표로 변환하여 시그널 전송"""
        if not self.roi_rect or not self.original_pixmap:
            return
        
        if not self.actual_frame_size:
            # 실제 프레임 크기가 설정되지 않았으면 픽스맵 크기 사용 (하위 호환성)
            pixmap_size = self.original_pixmap.size()
            actual_h, actual_w = pixmap_size.height(), pixmap_size.width()
        else:
            # 실제 프레임 크기 사용 (원본 영상 해상도)
            actual_h, actual_w = self.actual_frame_size
        
        label_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            return
        
        # QLabel에 표시된 픽스맵의 실제 크기 계산 (중앙 정렬 고려)
        label_w = label_size.width()
        label_h = label_size.height()
        pixmap_w = pixmap_size.width()  # 스케일링된 픽스맵 크기
        pixmap_h = pixmap_size.height()  # 스케일링된 픽스맵 크기
        
        # 중앙 정렬 오프셋 계산
        if pixmap_w / pixmap_h > label_w / label_h:
            # 픽스맵이 더 넓음 (상하 여백)
            scale = label_h / pixmap_h
            offset_x = (label_w - pixmap_w * scale) / 2
            offset_y = 0
        else:
            # 픽스맵이 더 높음 (좌우 여백)
            scale = label_w / pixmap_w
            offset_x = 0
            offset_y = (label_h - pixmap_h * scale) / 2
        
        # ROI 좌표를 스케일링된 픽스맵 좌표로 변환
        x1, y1, x2, y2 = self.roi_rect
        pixmap_x1 = max(0, int((x1 - offset_x) / scale))
        pixmap_y1 = max(0, int((y1 - offset_y) / scale))
        pixmap_x2 = min(pixmap_w, int((x2 - offset_x) / scale))
        pixmap_y2 = min(pixmap_h, int((y2 - offset_y) / scale))
        
        # 스케일링된 픽스맵 좌표를 실제 프레임 좌표로 변환
        # 픽스맵 크기와 실제 프레임 크기의 비율 계산
        scale_to_actual_w = actual_w / pixmap_w if pixmap_w > 0 else 1
        scale_to_actual_h = actual_h / pixmap_h if pixmap_h > 0 else 1
        
        frame_x1 = max(0, int(pixmap_x1 * scale_to_actual_w))
        frame_y1 = max(0, int(pixmap_y1 * scale_to_actual_h))
        frame_x2 = min(actual_w, int(pixmap_x2 * scale_to_actual_w))
        frame_y2 = min(actual_h, int(pixmap_y2 * scale_to_actual_h))
        
        self.roi_changed.emit(frame_x1, frame_y1, frame_x2, frame_y2)


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
    elif operation == 'erosion':
        result = cv2.erode(gray, kernel, iterations=1)
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


# ============================================================================
# 비동기 멀티스레딩 아키텍처 (끊김 없는 재생)
# ============================================================================

class AnalysisWorker(QThread):
    """
    분석 전담 Worker - 백그라운드에서 YOLO + Dynamsoft 해독 수행
    큐에서 프레임을 받아서 분석하고 결과를 반환
    """
    result_ready = pyqtSignal(int, list, dict)  # (frame_idx, detections, metrics)

    def __init__(self, input_queue, yolo_model, dbr_reader=None, conf_threshold=0.25, 
                 resnet_model=None, resnet_class_names=None, resnet_preprocess=None, processing_mode='decode',
                 unet_model=None):
        super().__init__()
        self.input_queue = input_queue
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        self.resnet_model = resnet_model
        self.resnet_class_names = resnet_class_names or []
        self.resnet_preprocess = resnet_preprocess
        self.processing_mode = processing_mode  # 'decode' 또는 'classify'
        self.conf_threshold = conf_threshold
        self.preprocessing_options = {}
        self.unet_model = unet_model  # UNet 복원 모델
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.running = True
        self._unet_warning_shown = False  # UNet 경고 한 번만 표시하기 위한 플래그

    def update_options(self, options):
        """전처리 옵션 업데이트"""
        self.preprocessing_options = options
    
    def update_conf_threshold(self, threshold):
        """YOLO 신뢰도 임계값 업데이트"""
        self.conf_threshold = threshold
    
    def set_roi(self, roi_rect: Optional[Tuple[int, int, int, int]]):
        """ROI 영역 설정"""
        self.roi_rect = roi_rect

    def stop(self):
        """분석 스레드 정지"""
        self.running = False

    def run(self):
        """메인 분석 루프 (백그라운드에서 계속 실행)"""
        while self.running:
            try:
                # 큐에서 최신 프레임 꺼내기 (timeout 0.1초)
                frame, frame_idx, fps, total_frames = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # --- [무거운 분석 작업 수행] ---
            try:
                # 0. ROI 적용 (ROI가 설정되어 있으면 프레임 크롭)
                roi_offset_x = 0
                roi_offset_y = 0
                roi_scale_factor = 1.0  # 리사이징 스케일 팩터
                if self.roi_rect:
                    x1, y1, x2, y2 = self.roi_rect
                    h, w = frame.shape[:2]
                    # 경계 체크
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    roi_offset_x = x1
                    roi_offset_y = y1
                    original_frame_size = frame.shape[:2]
                    frame = frame[y1:y2, x1:x2]
                    if frame.size == 0:
                        continue  # 유효하지 않은 ROI
                    
                    # 크롭된 프레임이 너무 작으면 최소 크기로 리사이징 (YOLO 성능 향상)
                    h_crop, w_crop = frame.shape[:2]
                    min_size = 640  # YOLO 최적 크기
                    scale_factor = 1.0
                    
                    if h_crop < min_size or w_crop < min_size:
                        # 비율 유지하면서 최소 크기로 리사이징
                        if h_crop < w_crop:
                            scale_factor = min_size / h_crop
                        else:
                            scale_factor = min_size / w_crop
                        
                        new_h = int(h_crop * scale_factor)
                        new_w = int(w_crop * scale_factor)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        # 리사이징 후 좌표 변환을 위해 스케일 저장
                        roi_scale_factor = scale_factor
                    else:
                        roi_scale_factor = 1.0
                    
                # 1. 전처리 (UNet 복원 포함)
                processed_frame = self._apply_preprocessing(frame)

                # 2. YOLO 탐지 (듀얼 패스: 원본 + 전처리)
                detections_orig = self._detect_qr_codes(frame)  # 크롭된 원본 프레임으로 탐지
                detections_prep = self._detect_qr_codes(processed_frame)  # 크롭된 전처리 프레임으로 탐지
                
                # 3. 결과 합치기 및 중복 제거
                all_detections = detections_orig + detections_prep
                detections = self._merge_detections(all_detections)

                # 4. 해독 또는 분류 (크롭된 프레임 좌표로 수행)
                if self.processing_mode == 'classify' and self.resnet_model:
                    # ResNet 분류 모드
                    for det in detections:
                        # 원본 프레임에서 먼저 시도
                        if not det.get('success', False):
                            self._classify_with_resnet(frame, det)
                        # 실패하면 전처리 프레임에서 시도
                        if not det.get('success', False):
                            self._classify_with_resnet(processed_frame, det)
                else:
                    # Dynamsoft 해독 모드
                    for det in detections:
                        # 원본 프레임에서 먼저 시도
                        if not det.get('success', False):
                            self._decode_qr_code(frame, det)
                        # 실패하면 전처리 프레임에서 시도
                        if not det.get('success', False):
                            self._decode_qr_code(processed_frame, det)
                
                # ROI가 있으면 탐지 결과 좌표를 원본 프레임 좌표로 변환 (해독 후 변환)
                if self.roi_rect:
                    for det in detections:
                        # bbox 좌표 변환 (리사이징 고려)
                        if 'bbox' in det:
                            x1, y1, x2, y2 = det['bbox']
                            # 리사이징된 경우 원래 크기로 변환
                            if roi_scale_factor != 1.0:
                                x1 = int(x1 / roi_scale_factor)
                                y1 = int(y1 / roi_scale_factor)
                                x2 = int(x2 / roi_scale_factor)
                                y2 = int(y2 / roi_scale_factor)
                            # 원본 프레임 좌표로 변환
                            det['bbox'] = [x1 + roi_offset_x, y1 + roi_offset_y, 
                                          x2 + roi_offset_x, y2 + roi_offset_y]
                        # center 좌표 변환
                        if 'center' in det:
                            cx, cy = det['center']
                            if roi_scale_factor != 1.0:
                                cx = int(cx / roi_scale_factor)
                                cy = int(cy / roi_scale_factor)
                            det['center'] = (cx + roi_offset_x, cy + roi_offset_y)
                        # quad 좌표 변환
                        if 'quad' in det and det['quad']:
                            quad = []
                            for px, py in det['quad']:
                                if roi_scale_factor != 1.0:
                                    px = int(px / roi_scale_factor)
                                    py = int(py / roi_scale_factor)
                                quad.append([px + roi_offset_x, py + roi_offset_y])
                            det['quad'] = quad

                # 4. 분석 지표 계산
                metrics = self._calculate_metrics(processed_frame, detections)
                metrics['frame_idx'] = frame_idx
                metrics['frame_no'] = frame_idx
                metrics['total_frames'] = total_frames
                metrics['has_success'] = any(d.get('success', False) for d in detections)
                metrics['fps'] = fps

                # 6. 결과 전송
                self.result_ready.emit(frame_idx, detections, metrics)
            except Exception as e:
                pass

            # 큐 작업 완료 알림
            self.input_queue.task_done()

    def _apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """전처리 적용"""
        result = frame.copy()
        opts = self.preprocessing_options
        
        if not opts:
            return result
        
        # UNet 복원 적용 (가장 먼저 적용)
        if opts.get('use_unet_restore', False) and self.unet_model:
            result = self._apply_unet_restore(result)
        
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
    
    def _apply_unet_restore(self, frame: np.ndarray) -> np.ndarray:
        """UNet 모델을 사용하여 이미지 복원"""
        if self.unet_model is None:
            return frame
        
        try:
            import torch
            from torchvision import transforms
            from PIL import Image
            
            device = torch.device("cpu")
            
            # 모델이 딕셔너리 형태인 경우 (state_dict만 있는 경우)
            if isinstance(self.unet_model, dict):
                # 경고를 한 번만 표시
                if not self._unet_warning_shown:
                    print("⚠️ UNet 모델 구조가 정의되지 않았습니다. state_dict만 있는 경우 모델 클래스를 먼저 정의해야 합니다.")
                    print("   UNet 복원 적용이 비활성화됩니다.")
                    self._unet_warning_shown = True
                # 전처리 옵션에서 use_unet_restore를 False로 설정
                if self.preprocessing_options.get('use_unet_restore', False):
                    self.preprocessing_options['use_unet_restore'] = False
                return frame
            
            # 모델을 eval 모드로 설정
            model = self.unet_model
            if hasattr(model, 'eval'):
                model.eval()
            model = model.to(device)
            
            # 이미지 전처리
            # OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    # BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
            else:
                # 그레이스케일인 경우 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # PIL Image로 변환
            pil_image = Image.fromarray(frame_rgb)
            
            # 원본 크기 저장
            original_size = pil_image.size  # (width, height)
            
            # 모델 입력 크기에 맞게 리사이즈 (일반적으로 256x256 또는 512x512)
            # UNet 모델의 입력 크기를 확인하거나 기본값 사용
            input_size = 256  # 기본값, 필요시 조정
            pil_image_resized = pil_image.resize((input_size, input_size), Image.Resampling.LANCZOS)
            
            # 텐서로 변환
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            input_tensor = transform(pil_image_resized).unsqueeze(0).to(device)
            
            # 추론
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # 출력 텐서를 이미지로 변환
            output_tensor = output_tensor.squeeze(0).cpu()
            
            # 정규화 (모델 출력이 0-1 범위라고 가정)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # numpy 배열로 변환
            output_np = output_tensor.permute(1, 2, 0).numpy()
            output_np = (output_np * 255).astype(np.uint8)
            
            # PIL Image로 변환 후 원본 크기로 리사이즈
            output_pil = Image.fromarray(output_np)
            output_pil = output_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            # numpy 배열로 변환
            output_array = np.array(output_pil)
            
            # RGB를 BGR로 변환 (OpenCV 형식)
            if len(output_array.shape) == 3:
                output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
            
            return output_array
            
        except Exception as e:
            print(f"⚠️ UNet 복원 적용 중 오류 발생: {str(e)}")
            return frame
    
    def _detect_qr_codes(self, frame: np.ndarray) -> List[Dict]:
        """YOLO로 QR 코드 탐지"""
        detections = []
        try:
            # 프레임 크기 확인 (너무 작으면 탐지하지 않음)
            h, w = frame.shape[:2]
            if h < 32 or w < 32:
                # 너무 작은 프레임은 탐지하지 않음
                return detections
            
            # YOLO 탐지 수행
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
            result = results[0]
            
            # 디버깅: YOLO 결과 확인
            if result.boxes is not None and len(result.boxes) > 0:
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
            pass
            
        return detections
    
    def _merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """중복 탐지 결과 병합 (NMS 유사 로직)"""
        if not detections:
            return []
        
        # 신뢰도가 높은 순으로 정렬
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        merged = []
        
        for det in sorted_detections:
            is_duplicate = False
            bx1, by1, bx2, by2 = det['bbox']
            b_center = det['center']
            b_area = det['area']
            
            # 이미 추가된 박스와 비교
            for existing in merged:
                ex1, ey1, ex2, ey2 = existing['bbox']
                e_center = existing['center']
                e_area = existing['area']
                
                # 중심점 거리 계산
                center_dist = np.sqrt((b_center[0] - e_center[0])**2 + (b_center[1] - e_center[1])**2)
                
                # 중심점이 가까우면 (박스 크기의 30% 이내) 중복으로 간주
                threshold = min(b_area, e_area) ** 0.5 * 0.3
                
                if center_dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        return merged
    
    def _decode_qr_code(self, frame: np.ndarray, detection: Dict):
        """Dynamsoft로 QR 코드 해독"""
        if self.dbr_reader is None:
            return
            
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # 프레임 크기 확인 및 경계 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # Dynamsoft 해독
            captured_result = self.dbr_reader.capture(rgb_image, dbr.EnumImagePixelFormat.IPF_RGB_888)
            
            # 결과 추출
            barcode_result = None
            items = None
            
            if hasattr(captured_result, 'get_decoded_barcodes_result'):
                barcode_result = captured_result.get_decoded_barcodes_result()
                if barcode_result:
                    items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
            
            if not items and hasattr(captured_result, 'items'):
                items = captured_result.items
            
            if not items and hasattr(captured_result, 'decoded_barcodes_result'):
                barcode_result = captured_result.decoded_barcodes_result
                if barcode_result:
                    items = barcode_result.items if hasattr(barcode_result, 'items') else None
            
            if items and len(items) > 0:
                barcode_item = items[0]
                
                # 텍스트 추출
                text = None
                if hasattr(barcode_item, 'get_text'):
                    text = barcode_item.get_text()
                elif hasattr(barcode_item, 'text'):
                    text = barcode_item.text
                
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
            else:
                detection['text'] = ''
                detection['success'] = False
                    
        except Exception as e:
            detection['text'] = ''
            detection['success'] = False
    
    def _classify_with_resnet(self, frame: np.ndarray, detection: Dict):
        """ResNet으로 QR 코드 분류"""
        if self.resnet_model is None or self.resnet_preprocess is None:
            return
            
        try:
            import torch
            from PIL import Image
            
            x1, y1, x2, y2 = detection['bbox']
            
            # 프레임 크기 확인 및 경계 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # 이미지 크기 체크
            if x2 - x1 < 10 or y2 - y1 < 10:
                return
            
            # BGR to RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # PIL Image로 변환 및 전처리
            pil_img = Image.fromarray(rgb_image)
            input_tensor = self.resnet_preprocess(pil_img).unsqueeze(0)
            
            # ResNet 분류
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)
                label = self.resnet_class_names[idx] if idx < len(self.resnet_class_names) else f"Class_{idx}"
                score = conf.item() * 100
            
            # Detection 업데이트
            detection['text'] = label
            detection['success'] = True
            detection['confidence'] = score  # ResNet 신뢰도 저장
                    
        except Exception as e:
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


class VideoPlayThread(QThread):
    """
    영상 재생 전담 Thread - 영상을 끊김 없이 30 FPS로 재생
    분석 결과를 기다리지 않고 계속 프레임을 송출
    **핵심: 모든 프레임을 표시하되, 분석 요청만 프레임 간격에 맞춤**
    """
    # UI로 보낼 최종 이미지 (원본 + 박스 그려진 것)
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, list, dict)
    timeline_updated = pyqtSignal(int, int, float)  # (current_frame, total_frames, current_time)
    progress_updated = pyqtSignal(int, int)  # (current_frame, total_frames)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, analysis_queue):
        super().__init__()
        self.video_path = video_path
        self.analysis_queue = analysis_queue  # 분석가에게 줄 우체통
        self.running = True
        self.paused = False
        self.seek_request = -1
        self.frame_interval = 1  # 분석 프레임 간격 (1=모든 프레임 분석)
        self.display_mode = 'all'
        
        # 최신 분석 결과 저장소 - QMutex로 스레드 안전성 확보
        self.result_mutex = QMutex()
        self.latest_detections = []
        self.latest_metrics = {}
        self.latest_result_frame_idx = -1
        self.preprocessing_options = {}  # 전처리 옵션 저장
        
        self.cap = None
        self.total_frames = 0
        self.fps = 30

    def update_latest_result(self, frame_idx, detections, metrics):
        """분석가가 결과를 던져주면 여기서 받아서 저장 (스레드 안전)"""
        with QMutexLocker(self.result_mutex):
            self.latest_detections = detections.copy() if detections else []
            self.latest_metrics = metrics.copy() if metrics else {}
            self.latest_result_frame_idx = frame_idx

    def set_display_mode(self, mode: str):
        """디스플레이 모드 설정"""
        self.display_mode = mode
    
    def set_frame_interval(self, interval: int):
        """분석 프레임 간격 설정"""
        self.frame_interval = max(1, interval)

    def set_preprocessing_options(self, options: Dict):
        """전처리 옵션 설정"""
        self.preprocessing_options = options

    def pause(self):
        """일시정지"""
        self.paused = True

    def resume(self):
        """재개"""
        self.paused = False

    def stop(self):
        """정지"""
        self.running = False

    def seek(self, frame_no):
        """특정 프레임으로 이동"""
        self.seek_request = frame_no

    def run(self):
        """메인 재생 루프 (영상을 끊김 없이 송출) - 모든 프레임 표시"""
        try:
            if not os.path.exists(self.video_path):
                self.error_occurred.emit("비디오 파일을 찾을 수 없습니다.")
                return

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("비디오 파일을 열 수 없습니다.")
                return

            self.cap = cap
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = 1.0 / self.fps
            
            frame_idx = 0
            analysis_frame_counter = 0  # 분석 요청용 카운터

            while self.running and cap.isOpened():
                # Seek 처리
                # Seek 처리
                if self.seek_request >= 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_request)
                    frame_idx = self.seek_request
                    self.seek_request = -1
                    # 시크하면 이전 결과 지우기 (Mutex로 보호)
                    with QMutexLocker(self.result_mutex):
                        self.latest_detections = []
                        self.latest_result_frame_idx = -1
                    analysis_frame_counter = 0

                # 일시정지
                if self.paused:
                    self.msleep(100)
                    continue

                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                analysis_frame_counter += 1

                # 타임라인 업데이트 (모든 프레임마다)
                current_time = frame_idx / self.fps if self.fps > 0 else 0
                self.timeline_updated.emit(frame_idx, self.total_frames, current_time)
                
                # --- [핵심: 분석 요청만 프레임 간격에 맞춤] ---
                # frame_interval마다 한 번씩, 그리고 큐가 비어있을 때만 넣음
                if analysis_frame_counter % self.frame_interval == 0 and self.analysis_queue.empty():
                    try:
                        # 프레임 복사본을 넘겨야 원본 훼손 방지
                        self.analysis_queue.put_nowait((frame.copy(), frame_idx, self.fps, self.total_frames))
                    except queue.Full:
                        pass  # 이미 분석 중이면 패스 (자동 프레임 드롭)

                # --- [시각화: 모든 프레임마다 화면 갱신] ---
                # 조건문 밖에서 무조건 실행 -> 모든 프레임 표시
                original_frame = frame.copy()
                
                # 전처리 적용
                preprocessed_frame = self._apply_preprocessing(frame.copy())
                
                # 최신 분석 결과 가져오기 (Mutex로 보호)
                with QMutexLocker(self.result_mutex):
                    current_detections = self.latest_detections.copy() if self.latest_detections else []
                    current_metrics = self.latest_metrics.copy() if self.latest_metrics else {}
                
                # 박스 그리기
                vis_original = self._visualize_frame(original_frame.copy(), current_detections)
                vis_preprocessed = self._visualize_frame(preprocessed_frame.copy(), current_detections)

                # UI로 전송 (분석 안 기다림 -> 30FPS 유지됨)
                self.frame_ready.emit(vis_original, vis_preprocessed, current_detections, current_metrics)
                self.progress_updated.emit(frame_idx, self.total_frames)

                # FPS 유지 (정밀한 대기)
                elapsed = time.time() - start_time
                delay = max(0, frame_interval - elapsed)
                time.sleep(delay)

        except Exception as e:
            self.error_occurred.emit(f"재생 중 오류 발생: {str(e)}")
        finally:
            if cap:
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
        
    def _visualize_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """프레임에 QR 탐지 결과 시각화"""
        vis_frame = frame.copy()
        
        # 디스플레이 모드에 따른 필터링
        filtered_detections = detections
        if self.display_mode == 'success':
            filtered_detections = [d for d in detections if d.get('success', False)]
        elif self.display_mode == 'fail':
            filtered_detections = [d for d in detections if not d.get('success', False)]
        
        if not filtered_detections:
            # 탐지된 QR이 없을 때 "Searching..." 표시
            cv2.putText(vis_frame, "Searching...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # QR 코드 그리기
            for det in filtered_detections:
                color = (0, 255, 0) if det.get('success', False) else (0, 0, 255)
                
                # Quad 사용 (우선)
                if det.get('quad') and len(det['quad']) == 4:
                    quad = np.array(det['quad'], dtype=np.int32)
                    cv2.polylines(vis_frame, [quad], True, color, 2)
                else:
                    # BBox 사용
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # 텍스트 표시 (해독 성공 시)
                if det.get('success') and det.get('text'):
                    x1, y1 = det['bbox'][:2]
                    cv2.putText(vis_frame, det['text'][:20], (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame


class VideoManager(QObject):
    """
    비동기 영상 처리 매니저
    VideoPlayThread와 AnalysisWorker를 관리하고 동기화
    """
    # 외부로 전달할 시그널
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, list, dict)
    timeline_updated = pyqtSignal(int, int, float)
    progress_updated = pyqtSignal(int, int)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, yolo_model, dbr_reader=None, resnet_model=None, resnet_class_names=None, resnet_preprocess=None, processing_mode='decode', unet_model=None):
        super().__init__()
        # 통신용 큐 (크기 1 = 최신 프레임만 유지, 자동 프레임 드롭)
        self.queue = queue.Queue(maxsize=1)
        
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        self.resnet_model = resnet_model
        self.resnet_class_names = resnet_class_names or []
        self.resnet_preprocess = resnet_preprocess
        self.processing_mode = processing_mode  # 'decode' 또는 'classify'
        self.unet_model = unet_model  # UNet 복원 모델
        self.roi_rect = None  # (x1, y1, x2, y2)
        
        self.play_thread = None
        self.analysis_thread = None
        self.is_running = False
        self.is_paused = False

    def start(self, video_path, preprocessing_options=None, conf_threshold=0.25, frame_interval=1):
        """비동기 처리 시작"""
        # 1. 분석 스레드 생성 및 시작
        self.analysis_thread = AnalysisWorker(
            self.queue, self.yolo_model, self.dbr_reader, conf_threshold,
            self.resnet_model, self.resnet_class_names, self.resnet_preprocess, self.processing_mode,
            self.unet_model
        )
        if preprocessing_options:
            self.analysis_thread.update_options(preprocessing_options)
        # ROI 설정 (self.roi_rect가 설정되어 있으면 전달)
        if self.roi_rect:
            self.analysis_thread.set_roi(self.roi_rect)
        self.analysis_thread.start()

        # 2. 재생 스레드 생성
        self.play_thread = VideoPlayThread(video_path, self.queue)
        self.play_thread.set_frame_interval(frame_interval)
        if preprocessing_options:
            self.play_thread.set_preprocessing_options(preprocessing_options)
        
        # 3. 신호 연결 (분석 결과 -> 플레이어에게 전달) - DirectConnection으로 스레드 안전성 확보
        self.analysis_thread.result_ready.connect(self.play_thread.update_latest_result, Qt.ConnectionType.DirectConnection)
        
        # 4. 플레이어 시그널을 외부로 중계
        self.play_thread.frame_ready.connect(self.frame_ready.emit)
        self.play_thread.timeline_updated.connect(self.timeline_updated.emit)
        self.play_thread.progress_updated.connect(self.progress_updated.emit)
        self.play_thread.finished.connect(self._on_finished)
        self.play_thread.error_occurred.connect(self.error_occurred.emit)
        
        # 5. 재생 시작
        self.play_thread.start()
        self.is_running = True
        self.is_paused = False

    def pause(self):
        """일시정지"""
        if self.play_thread:
            self.play_thread.pause()
            self.is_paused = True

    def resume(self):
        """재개"""
        if self.play_thread:
            self.play_thread.resume()
            self.is_paused = False

    def stop(self):
        """정지"""
        self.is_running = False
        
        # 종료 순서 중요: 재생 -> 분석
        if self.play_thread:
            self.play_thread.stop()
            self.play_thread.wait()
        
        if self.analysis_thread:
            self.analysis_thread.stop()
            self.analysis_thread.wait()
    
    def set_roi(self, roi_rect: Optional[Tuple[int, int, int, int]]):
        """ROI 영역 설정"""
        self.roi_rect = roi_rect
        if self.analysis_thread:
            self.analysis_thread.set_roi(roi_rect)
    
    def wait(self):
        """스레드 종료 대기 (QThread.wait()와 호환성을 위한 메서드)"""
        # stop()에서 이미 모든 스레드의 wait()를 호출했으므로
        # 여기서는 추가 작업이 필요 없음
        pass

    def seek_to(self, frame_number: int):
        """특정 프레임으로 이동"""
        if self.play_thread:
            self.play_thread.seek(frame_number)

    def set_display_mode(self, mode: str):
        """디스플레이 모드 설정"""
        if self.play_thread:
            self.play_thread.set_display_mode(mode)

    def set_frame_interval(self, interval: int):
        """프레임 간격 설정"""
        if self.play_thread:
            self.play_thread.set_frame_interval(interval)

    def set_preprocessing_options(self, options: Dict):
        """전처리 옵션 설정"""
        # AnalysisWorker에 전처리 옵션 전달 (분석용)
        if self.analysis_thread:
            self.analysis_thread.update_options(options)
        
        # VideoPlayThread에 전처리 옵션 전달 (화면 표시용)
        if self.play_thread:
            self.play_thread.set_preprocessing_options(options)

    def set_conf_threshold(self, threshold: float):
        """YOLO 신뢰도 임계값 설정"""
        if self.analysis_thread:
            self.analysis_thread.update_conf_threshold(threshold)

    def _on_finished(self):
        """재생 완료 시 호출"""
        self.is_running = False
        self.finished.emit()


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
        self.resnet_model = None
        self.resnet_class_names = []
        self.resnet_preprocess = None
        self.unet_model = None  # UNet 복원 모델
        self.processing_mode = 'decode'  # 'decode' 또는 'classify'
        self.is_running = False
        self.is_paused = False
        self.conf_threshold = 0.25
        self.display_mode = 'all'  # 'all', 'success', 'fail'
        self.preprocessing_options = {}
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.seek_to_frame = -1  # 시크할 프레임 번호 (-1이면 시크 안함)
        self.frame_interval = 1  # 프레임 간격 (1=모든 프레임 처리)
        self._unet_warning_shown = False  # UNet 경고 한 번만 표시하기 위한 플래그
    
    def set_roi(self, roi_rect: Optional[Tuple[int, int, int, int]]):
        """ROI 영역 설정"""
        self.roi_rect = roi_rect
        
    def set_video(self, video_path: str):
        """비디오 파일 경로 설정"""
        self.video_path = video_path
        
    def set_model(self, yolo_model, dbr_reader=None, resnet_model=None, resnet_class_names=None, resnet_preprocess=None, unet_model=None):
        """YOLO 및 Dynamsoft/ResNet/UNet 모델 설정"""
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        self.resnet_model = resnet_model
        self.resnet_class_names = resnet_class_names or []
        self.resnet_preprocess = resnet_preprocess
        self.unet_model = unet_model
    
    def set_processing_mode(self, mode: str):
        """처리 모드 설정"""
        self.processing_mode = mode
        
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
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.error_occurred.emit("비디오 파일을 찾을 수 없습니다.")
                return
                
            if self.yolo_model is None:
                self.error_occurred.emit("YOLO 모델이 로드되지 않았습니다.")
                return
            
            # 비디오 열기
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("비디오 파일을 열 수 없습니다.")
                return
        except Exception as e:
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
                
                # ROI 적용 (ROI가 설정되어 있으면 프레임 크롭)
                roi_offset_x = 0
                roi_offset_y = 0
                roi_scale_factor = 1.0  # 리사이징 스케일 팩터
                if self.roi_rect:
                    x1, y1, x2, y2 = self.roi_rect
                    h, w = frame.shape[:2]
                    # 경계 체크
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    roi_offset_x = x1
                    roi_offset_y = y1
                    original_frame_size = frame.shape[:2]
                    frame = frame[y1:y2, x1:x2]
                    if frame.size == 0:
                        continue  # 유효하지 않은 ROI
                    
                    # 크롭된 프레임이 너무 작으면 최소 크기로 리사이징 (YOLO 성능 향상)
                    h_crop, w_crop = frame.shape[:2]
                    min_size = 640  # YOLO 최적 크기
                    scale_factor = 1.0
                    
                    if h_crop < min_size or w_crop < min_size:
                        # 비율 유지하면서 최소 크기로 리사이징
                        if h_crop < w_crop:
                            scale_factor = min_size / h_crop
                        else:
                            scale_factor = min_size / w_crop
                        
                        new_h = int(h_crop * scale_factor)
                        new_w = int(w_crop * scale_factor)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        # 리사이징 후 좌표 변환을 위해 스케일 저장
                        roi_scale_factor = scale_factor
                    else:
                        roi_scale_factor = 1.0
                    
                # 전처리 적용
                preprocessed_frame = self._apply_preprocessing(frame.copy())
                
                # YOLO 탐지 (듀얼 패스: 원본 + 전처리 - AnalysisWorker와 동일하게)
                detections_orig = self._detect_qr_codes(frame)  # 크롭된 원본 프레임으로 탐지
                detections_prep = self._detect_qr_codes(preprocessed_frame)  # 크롭된 전처리 프레임으로 탐지
                
                # 결과 합치기 및 중복 제거
                all_detections = detections_orig + detections_prep
                detections = self._merge_detections(all_detections)
                
                # 분석 모드에 따라 분류 또는 해독
                if self.processing_mode == 'classify' and self.resnet_model:
                    for det in detections:
                        # 원본 프레임에서 먼저 시도
                        if not det.get('success', False):
                            self._classify_with_resnet(frame, det)
                        # 실패하면 전처리 프레임에서 시도
                        if not det.get('success', False):
                            self._classify_with_resnet(preprocessed_frame, det)
                else:
                    # Dynamsoft 해독 (크롭된 프레임 좌표로 해독)
                    for det in detections:
                        # 원본 프레임에서 먼저 시도
                        if not det.get('success', False):
                            self._decode_qr_code(frame, det)
                        # 실패하면 전처리 프레임에서 시도
                        if not det.get('success', False):
                            self._decode_qr_code(preprocessed_frame, det)
                
                # ROI가 있으면 탐지 결과 좌표를 원본 프레임 좌표로 변환 (해독 후 변환)
                if self.roi_rect:
                    for det in detections:
                        # bbox 좌표 변환 (리사이징 고려)
                        if 'bbox' in det:
                            x1, y1, x2, y2 = det['bbox']
                            # 리사이징된 경우 원래 크기로 변환
                            if roi_scale_factor != 1.0:
                                x1 = int(x1 / roi_scale_factor)
                                y1 = int(y1 / roi_scale_factor)
                                x2 = int(x2 / roi_scale_factor)
                                y2 = int(y2 / roi_scale_factor)
                            # 원본 프레임 좌표로 변환
                            det['bbox'] = [x1 + roi_offset_x, y1 + roi_offset_y, 
                                          x2 + roi_offset_x, y2 + roi_offset_y]
                        # center 좌표 변환
                        if 'center' in det:
                            cx, cy = det['center']
                            if roi_scale_factor != 1.0:
                                cx = int(cx / roi_scale_factor)
                                cy = int(cy / roi_scale_factor)
                            det['center'] = (cx + roi_offset_x, cy + roi_offset_y)
                        # quad 좌표 변환
                        if 'quad' in det and det['quad']:
                            quad = []
                            for px, py in det['quad']:
                                if roi_scale_factor != 1.0:
                                    px = int(px / roi_scale_factor)
                                    py = int(py / roi_scale_factor)
                                quad.append([px + roi_offset_x, py + roi_offset_y])
                            det['quad'] = quad
                
                # 분석 지표 계산
                metrics = self._calculate_metrics(preprocessed_frame, detections)
                metrics['frame_idx'] = frame_idx
                metrics['frame_no'] = frame_idx  # on_frame_processed에서 사용
                metrics['total_frames'] = total_frames
                metrics['has_success'] = any(d.get('success', False) for d in detections)
                
                # 시각화된 프레임 생성 (원본과 전처리 모두)
                # 원본 프레임에 전처리 적용 (시각화용)
                original_preprocessed = self._apply_preprocessing(original_frame.copy())
                vis_original = self._visualize_frame(original_frame.copy(), detections)
                vis_preprocessed = self._visualize_frame(original_preprocessed, detections)
                
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
            if cap:
                cap.release()
            self.finished.emit()
    
    def _apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """전처리 적용"""
        result = frame.copy()
        opts = self.preprocessing_options
        
        if not opts:
            return result
        
        # UNet 복원 적용 (가장 먼저 적용)
        if opts.get('use_unet_restore', False) and self.unet_model:
            result = self._apply_unet_restore(result)
        
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
    
    def _apply_unet_restore(self, frame: np.ndarray) -> np.ndarray:
        """UNet 모델을 사용하여 이미지 복원"""
        if self.unet_model is None:
            return frame
        
        try:
            import torch
            from torchvision import transforms
            from PIL import Image
            
            device = torch.device("cpu")
            
            # 모델이 딕셔너리 형태인 경우 (state_dict만 있는 경우)
            if isinstance(self.unet_model, dict):
                # 경고를 한 번만 표시
                if not self._unet_warning_shown:
                    print("⚠️ UNet 모델 구조가 정의되지 않았습니다. state_dict만 있는 경우 모델 클래스를 먼저 정의해야 합니다.")
                    print("   UNet 복원 적용이 비활성화됩니다.")
                    self._unet_warning_shown = True
                # 전처리 옵션에서 use_unet_restore를 False로 설정
                if self.preprocessing_options.get('use_unet_restore', False):
                    self.preprocessing_options['use_unet_restore'] = False
                return frame
            
            # 모델을 eval 모드로 설정
            model = self.unet_model
            if hasattr(model, 'eval'):
                model.eval()
            model = model.to(device)
            
            # 이미지 전처리
            # OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    # BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
            else:
                # 그레이스케일인 경우 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # PIL Image로 변환
            pil_image = Image.fromarray(frame_rgb)
            
            # 원본 크기 저장
            original_size = pil_image.size  # (width, height)
            
            # 모델 입력 크기에 맞게 리사이즈 (일반적으로 256x256 또는 512x512)
            # UNet 모델의 입력 크기를 확인하거나 기본값 사용
            input_size = 256  # 기본값, 필요시 조정
            pil_image_resized = pil_image.resize((input_size, input_size), Image.Resampling.LANCZOS)
            
            # 텐서로 변환
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            input_tensor = transform(pil_image_resized).unsqueeze(0).to(device)
            
            # 추론
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # 출력 텐서를 이미지로 변환
            output_tensor = output_tensor.squeeze(0).cpu()
            
            # 정규화 (모델 출력이 0-1 범위라고 가정)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # numpy 배열로 변환
            output_np = output_tensor.permute(1, 2, 0).numpy()
            output_np = (output_np * 255).astype(np.uint8)
            
            # PIL Image로 변환 후 원본 크기로 리사이즈
            output_pil = Image.fromarray(output_np)
            output_pil = output_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            # numpy 배열로 변환
            output_array = np.array(output_pil)
            
            # RGB를 BGR로 변환 (OpenCV 형식)
            if len(output_array.shape) == 3:
                output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
            
            return output_array
            
        except Exception as e:
            print(f"⚠️ UNet 복원 적용 중 오류 발생: {str(e)}")
            return frame
            
    def _detect_qr_codes(self, frame: np.ndarray) -> List[Dict]:
        """YOLO로 QR 코드 탐지"""
        detections = []
        try:
            # 프레임 크기 확인 (너무 작으면 경고)
            h, w = frame.shape[:2]
            if h < 32 or w < 32:
                # 너무 작은 프레임은 탐지하지 않음
                return detections
            
            # YOLO 탐지 수행
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
            result = results[0]
            
            # 디버깅: YOLO 결과 확인
            if result.boxes is not None and len(result.boxes) > 0:
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
            pass
            
        return detections
    
    def _merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """중복 탐지 결과 병합 (NMS 유사 로직)"""
        if not detections:
            return []
        
        # 신뢰도가 높은 순으로 정렬
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        merged = []
        
        for det in sorted_detections:
            is_duplicate = False
            bx1, by1, bx2, by2 = det['bbox']
            b_center = det.get('center', [(bx1 + bx2) // 2, (by1 + by2) // 2])
            b_area = det.get('area', (bx2 - bx1) * (by2 - by1))
            
            # 이미 추가된 박스와 비교
            for existing in merged:
                ex1, ey1, ex2, ey2 = existing['bbox']
                e_center = existing.get('center', [(ex1 + ex2) // 2, (ey1 + ey2) // 2])
                e_area = existing.get('area', (ex2 - ex1) * (ey2 - ey1))
                
                # 중심점 거리 계산
                center_dist = np.sqrt((b_center[0] - e_center[0])**2 + (b_center[1] - e_center[1])**2)
                
                # 중심점이 가까우면 (박스 크기의 30% 이내) 중복으로 간주
                threshold = min(b_area, e_area) ** 0.5 * 0.3
                
                if center_dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        return merged
    
    def _classify_with_resnet(self, frame: np.ndarray, detection: Dict):
        """ResNet으로 QR 코드 분류"""
        if self.resnet_model is None or self.resnet_preprocess is None:
            return
            
        try:
            import torch
            from PIL import Image
            
            x1, y1, x2, y2 = detection['bbox']
            
            # 프레임 크기 확인 및 경계 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # 이미지 크기 체크
            if x2 - x1 < 10 or y2 - y1 < 10:
                return
            
            # BGR to RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # PIL Image로 변환 및 전처리
            pil_img = Image.fromarray(rgb_image)
            input_tensor = self.resnet_preprocess(pil_img).unsqueeze(0)
            
            # ResNet 분류
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)
                label = self.resnet_class_names[idx] if idx < len(self.resnet_class_names) else f"Class_{idx}"
                score = conf.item() * 100
            
            # Detection 업데이트
            detection['text'] = label
            detection['success'] = True
            detection['confidence'] = score  # ResNet 신뢰도 저장
                    
        except Exception as e:
            detection['text'] = ''
            detection['success'] = False
    
    def _decode_qr_code(self, frame: np.ndarray, detection: Dict):
        """Dynamsoft로 QR 코드 해독"""
        if self.dbr_reader is None:
            return
            
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # 프레임 크기 확인 및 경계 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # Dynamsoft 해독
            captured_result = self.dbr_reader.capture(rgb_image, dbr.EnumImagePixelFormat.IPF_RGB_888)
            
            # 방법 1: get_decoded_barcodes_result() 시도
            barcode_result = None
            items = None
            
            if hasattr(captured_result, 'get_decoded_barcodes_result'):
                barcode_result = captured_result.get_decoded_barcodes_result()
                if barcode_result:
                    items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
            
            # 방법 2: 직접 items 속성 접근 시도
            if not items and hasattr(captured_result, 'items'):
                items = captured_result.items
            
            # 방법 3: decoded_barcodes_result 속성 시도
            if not items and hasattr(captured_result, 'decoded_barcodes_result'):
                barcode_result = captured_result.decoded_barcodes_result
                if barcode_result:
                    items = barcode_result.items if hasattr(barcode_result, 'items') else None
            
            if items and len(items) > 0:
                barcode_item = items[0]
                
                # 텍스트 추출
                text = None
                if hasattr(barcode_item, 'get_text'):
                    text = barcode_item.get_text()
                elif hasattr(barcode_item, 'text'):
                    text = barcode_item.text
                
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
            else:
                detection['text'] = ''
                detection['success'] = False
                    
        except Exception as e:
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
# 웹캠 모드 전용 창
# ============================================================================

class WebcamAIWorker(QThread):
    """웹캠용 AI 워커 (base.py의 AIWorker와 유사)"""
    result_ready = pyqtSignal(list)  # 결과 전송 시그널

    def __init__(self, frame_queue, yolo_model, dbr_reader=None, resnet_model=None, resnet_class_names=None, resnet_preprocess=None, conf_threshold=0.25):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.conf_threshold = conf_threshold
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        self.resnet_model = resnet_model
        self.resnet_class_names = resnet_class_names or []
        self.resnet_preprocess = resnet_preprocess
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.webcam_mode = 'resnet' if resnet_model else 'dynamsoft'
    
    def set_roi(self, roi_rect: Optional[Tuple[int, int, int, int]]):
        """ROI 영역 설정"""
        self.roi_rect = roi_rect

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            results = []
            
            # 원본 프레임 저장 (ResNet/Dynamsoft 분류 시 사용)
            original_frame = frame.copy()
            
            # ROI 적용 (ROI가 설정되어 있으면 프레임 크롭)
            roi_offset_x = 0
            roi_offset_y = 0
            cropped_frame = frame
            if self.roi_rect:
                x1, y1, x2, y2 = self.roi_rect
                h, w = frame.shape[:2]
                # 경계 체크 (원본 프레임 경계 내에서)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                roi_offset_x = x1
                roi_offset_y = y1
                # ROI 영역 크롭 (경계 포함)
                cropped_frame = frame[y1:y2, x1:x2]
                if cropped_frame.size == 0:
                    continue  # 유효하지 않은 ROI

            # YOLO 탐지 (크롭된 프레임에서)
            detections = []
            if self.yolo_model is not None:
                try:
                    yolo_results = self.yolo_model(cropped_frame, conf=self.conf_threshold, verbose=False)
                    result = yolo_results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        h, w = cropped_frame.shape[:2]
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # 평소처럼 pad 추가 (ROI 모드와 일반 모드 동일)
                            pad = 20
                            # 크롭된 프레임 내에서 pad 추가 (경계를 넘지 않도록)
                            x1_padded = max(0, x1 - pad)
                            y1_padded = max(0, y1 - pad)
                            x2_padded = min(w, x2 + pad)
                            y2_padded = min(h, y2 + pad)
                            
                            # 유효한 탐지 박스인지 확인 (크롭된 프레임 경계 내에 있고, 최소 크기 이상)
                            if (x1_padded >= 0 and y1_padded >= 0 and 
                                x2_padded <= w and y2_padded <= h and 
                                x2_padded > x1_padded and y2_padded > y1_padded):
                                detections.append({
                                    'bbox': [x1_padded, y1_padded, x2_padded, y2_padded],
                                    'confidence': conf
                                })
                except Exception as e:
                    pass
            
            # ResNet 모드인 경우 ResNet 분류, 아니면 Dynamsoft 해독
            if self.webcam_mode == 'resnet' and self.resnet_model is not None and len(detections) > 0:
                for det in detections:
                    # ROI 내부 좌표
                    x1_local, y1_local, x2_local, y2_local = det['bbox']
                    # 원본 프레임 좌표로 변환
                    x1 = x1_local + roi_offset_x
                    y1 = y1_local + roi_offset_y
                    x2 = x2_local + roi_offset_x
                    y2 = y2_local + roi_offset_y
                    
                    # 원본 프레임에서 ROI 추출
                    roi = original_frame[y1:y2, x1:x2]
                    
                    if roi.size == 0:
                        continue
                    
                    # 이미지 크기 체크
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        results.append([x1, y1, x2, y2, "", 0.0])
                        continue
                    
                    try:
                        import torch
                        from PIL import Image
                        
                        # BGR to RGB 변환
                        if len(roi.shape) == 3 and roi.shape[2] == 3:
                            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                        
                        # PIL Image로 변환 및 전처리
                        pil_img = Image.fromarray(rgb_image)
                        input_tensor = self.resnet_preprocess(pil_img).unsqueeze(0)
                        
                        # ResNet 분류
                        with torch.no_grad():
                            outputs = self.resnet_model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs[0], dim=0)
                            conf, idx = torch.max(probs, 0)
                            label = self.resnet_class_names[idx] if idx < len(self.resnet_class_names) else f"Class_{idx}"
                            confidence_score = conf.item() * 100  # 백분율로 변환
                        
                        results.append([x1, y1, x2, y2, str(label), confidence_score])
                    except Exception as e:
                        results.append([x1, y1, x2, y2, "", 0.0])
            elif self.dbr_reader is not None and len(detections) > 0:
                for det in detections:
                    # ROI 내부 좌표
                    x1_local, y1_local, x2_local, y2_local = det['bbox']
                    # 원본 프레임 좌표로 변환
                    x1 = x1_local + roi_offset_x
                    y1 = y1_local + roi_offset_y
                    x2 = x2_local + roi_offset_x
                    y2 = y2_local + roi_offset_y
                    
                    # 원본 프레임에서 ROI 추출
                    roi = original_frame[y1:y2, x1:x2]
                    
                    if roi.size == 0:
                        continue
                    
                    if len(roi.shape) == 3 and roi.shape[2] == 3:
                        rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                    
                    try:
                        captured_result = self.dbr_reader.capture(rgb_image, dbr.EnumImagePixelFormat.IPF_RGB_888)
                        
                        barcode_result = None
                        items = None
                        
                        if hasattr(captured_result, 'get_decoded_barcodes_result'):
                            barcode_result = captured_result.get_decoded_barcodes_result()
                            if barcode_result:
                                items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
                        
                        if not items and hasattr(captured_result, 'items'):
                            items = captured_result.items
                        
                        if not items and hasattr(captured_result, 'decoded_barcodes_result'):
                            barcode_result = captured_result.decoded_barcodes_result
                            if barcode_result:
                                items = barcode_result.items if hasattr(barcode_result, 'items') else None
                        
                        decoded_text = ""
                        if items and len(items) > 0:
                            barcode_item = items[0]
                            
                            if hasattr(barcode_item, 'get_text'):
                                try:
                                    decoded_text = barcode_item.get_text()
                                except:
                                    pass
                            
                            if not decoded_text and hasattr(barcode_item, 'text'):
                                try:
                                    decoded_text = barcode_item.text
                                except:
                                    pass
                            
                            if not decoded_text and hasattr(barcode_item, 'getBarcodeText'):
                                try:
                                    decoded_text = barcode_item.getBarcodeText()
                                except:
                                    pass
                        
                        if decoded_text:
                            results.append([x1, y1, x2, y2, str(decoded_text)])
                        else:
                            results.append([x1, y1, x2, y2, ""])
                            
                    except Exception as e:
                        results.append([x1, y1, x2, y2, "", 0.0])
            elif len(detections) > 0:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    results.append([x1, y1, x2, y2, "", 0.0])

            self.result_ready.emit(results)
            self.frame_queue.task_done()

    def stop(self):
        self.running = False
        self.wait()


class WebcamVideoPlayer(QThread):
    """웹캠용 비디오 플레이어"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # 초기 하드웨어 설정값을 UI로 보내는 신호
    initial_settings_signal = pyqtSignal(dict)

    def __init__(self, frame_queue, camera_source=0):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.latest_ai_results = []
        # UI에서 들어온 변경 요청을 담을 큐
        self.command_queue = queue.Queue()
        self.cap = None
        self.resolution_queue = queue.Queue()  # 해상도 변경 요청 큐
        self.current_width = 640
        self.current_height = 480
        self.camera_source = camera_source  # 0 또는 IP 웹캠 URL
        self.is_ip_camera = isinstance(camera_source, str) and not camera_source.isdigit()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2.0  # 재연결 시도 간격 (초)

    def update_ai_results(self, results):
        """AI 워커가 결과를 보내면 여기서 업데이트"""
        self.latest_ai_results = results

    def update_setting(self, prop_id, value):
        """UI에서 설정을 바꿀 때 호출"""
        self.command_queue.put((prop_id, value))

    def _open_camera(self):
        """카메라 열기 (재연결 지원)"""
        try:
            if isinstance(self.camera_source, str) and self.camera_source.isdigit():
                camera_idx = int(self.camera_source)
            elif isinstance(self.camera_source, int):
                camera_idx = self.camera_source
            else:
                # URL인 경우 (IP 웹캠)
                camera_idx = self.camera_source
            
            # 기존 연결이 있으면 해제
            if self.cap is not None:
                self.cap.release()
            
            # IP 웹캠인 경우 타임아웃 설정
            if self.is_ip_camera:
                # OpenCV 백엔드를 명시적으로 지정 (IP 카메라의 경우)
                # FFMPEG 백엔드가 없으면 기본 백엔드 사용
                try:
                    self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_FFMPEG)
                except:
                    # FFMPEG 백엔드가 없으면 기본 백엔드 사용
                    self.cap = cv2.VideoCapture(camera_idx)
                
                # 타임아웃 설정 (밀리초 단위) - 지원되는 경우에만 설정
                try:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5초 타임아웃
                except:
                    pass
                try:
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)  # 읽기 타임아웃 3초
                except:
                    pass
            else:
                self.cap = cv2.VideoCapture(camera_idx)
            
            # 연결 확인
            if self.cap.isOpened():
                # 연결 성공 시 재시도 카운터 리셋
                self.reconnect_attempts = 0
                return True
            else:
                return False
        except Exception as e:
            print(f"카메라 열기 오류: {e}")
            return False
    
    def run(self):
        # 1. 카메라 열기 (재시도 로직 포함)
        while self.running:
            if self._open_camera():
                break
            
            # 연결 실패 시 재시도
            self.reconnect_attempts += 1
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                print(f"⚠️ 카메라 연결 실패: 최대 재시도 횟수({self.max_reconnect_attempts}) 초과")
                # UI에 오류 신호 전송 (필요시)
                return
            
            print(f"카메라 연결 재시도 중... ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
            time.sleep(self.reconnect_delay)
        
        if not self.cap or not self.cap.isOpened():
            return

        # 초기 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_height)

        # 2. 현재 하드웨어의 설정값을 읽어옴 (덮어쓰기 전!)
        # 이 값들이 바로 '내 기본 하드웨어 설정'이 됩니다.
        current_settings = {}
        try:
            current_settings = {
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                'auto_exposure': self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                'gain': self.cap.get(cv2.CAP_PROP_GAIN),
                'focus': self.cap.get(cv2.CAP_PROP_FOCUS),
                'autofocus': self.cap.get(cv2.CAP_PROP_AUTOFOCUS),
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            # 실제 설정된 해상도 저장
            self.current_width = current_settings['width']
            self.current_height = current_settings['height']
        except Exception as e:
            pass
        
        # 3. 읽은 값을 UI로 전송 -> 슬라이더 초기화 및 '기본값'으로 저장됨
        self.initial_settings_signal.emit(current_settings)

        fps = 30
        frame_interval = 1.0 / fps

        while self.running and self.cap.isOpened():
            start_time = time.time()

            # 4. UI에서 변경 요청이 있으면 적용
            try:
                while not self.command_queue.empty():
                    prop_id, value = self.command_queue.get_nowait()
                    self.cap.set(prop_id, value)
            except queue.Empty:
                pass
            except Exception as e:
                pass

            ret, frame = self.cap.read()
            if not ret:
                # IP 웹캠인 경우 연결 끊김 가능성 체크
                if self.is_ip_camera:
                    # 카메라가 열려있는지 확인
                    if not self.cap.isOpened():
                        print("⚠️ 카메라 연결이 끊어졌습니다. 재연결 시도 중...")
                        # 재연결 시도
                        self.reconnect_attempts = 0
                        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
                            time.sleep(self.reconnect_delay)
                            if self._open_camera():
                                print("✅ 카메라 재연결 성공")
                                break
                            self.reconnect_attempts += 1
                            print(f"재연결 시도 중... ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                        
                        if not self.cap or not self.cap.isOpened():
                            print("❌ 카메라 재연결 실패")
                            break
                        continue
                else:
                    # 로컬 웹캠인 경우 짧은 대기
                    time.sleep(0.01)
                    continue

            # AI에게 일감 던지기
            if self.frame_queue.empty():
                self.frame_queue.put(frame.copy())

            # 시각화
            for res in self.latest_ai_results:
                if len(res) >= 6:
                    x1, y1, x2, y2, text, confidence = res
                else:
                    x1, y1, x2, y2, text = res
                color = (0, 255, 0) if text else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if text:
                    cv2.putText(frame, text[:30], (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # UI로 전송
            self.change_pixmap_signal.emit(frame)

            # FPS 유지
            processing_time = time.time() - start_time
            delay = max(0, frame_interval - processing_time)
            time.sleep(delay)

        # 웹캠 모드 종료 시 원래 설정으로 복원
        if self.cap is not None and self.cap.isOpened():
            # 원래 설정 복원을 위한 신호 전송 (UI에서 처리)
            pass
        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()
    
    def restore_original_settings(self):
        """원래 설정으로 복원 (종료 시 호출)"""
        if self.cap is not None and self.cap.isOpened():
            # 원래 설정 복원은 UI에서 처리
            pass


class WebcamWindow(QMainWindow):
    """웹캠 모드 전용 창"""
    
    def __init__(self, yolo_model, dbr_reader=None, resnet_model=None, resnet_class_names=None, resnet_preprocess=None):
        super().__init__()
        self.setWindowTitle("웹캠 QR 분석")
        self.resize(1280, 900)
        # 창 크기 조정 가능하도록 설정 (기본값은 1280x900이지만 사용자가 조정 가능)
        self.setMinimumSize(800, 600)  # 최소 크기 설정
        
        # 모델 저장
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        self.resnet_model = resnet_model
        self.resnet_class_names = resnet_class_names or []
        self.resnet_preprocess = resnet_preprocess
        self.webcam_mode = 'resnet' if resnet_model else 'dynamsoft'  # 웹캠 모드: 'dynamsoft' 또는 'resnet'

        # 로그 관련 변수
        self.frame_counter = 0
        self.log_filter_mode = 'all'  # 'all', 'success', 'fail'
        self.all_log_entries = []  # 모든 로그 항목 저장 (필터링용)
        
        # 스레드 변수 초기화
        self.ai_worker = None
        self.player = None
        self.frame_queue = queue.Queue(maxsize=1)
        
        # ROI 관련 변수
        self.roi_mode = False
        self.roi_rect = None  # (x1, y1, x2, y2)
        
        # 좌표 변환을 위한 실제 프레임 크기 저장
        self.actual_frame_size = None  # (height, width) - 원본 영상 해상도

        # UI 구성 - 전체를 스크롤 가능하게 만들기
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.central_widget = QWidget()
        scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(scroll_area)
        
        self.layout = QVBoxLayout(self.central_widget)

        # 비디오 레이블 (ROI 지원)
        self.video_label = ROIVideoLabel()
        self.video_label.setText("웹캠 연결 중...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.roi_changed.connect(self.on_roi_changed)
        self.layout.addWidget(self.video_label, 1)  # stretch factor 1로 설정하여 확장 가능하게

        # 카메라 소스 설정 섹션
        camera_group = QGroupBox("📹 카메라 설정")
        camera_layout = QHBoxLayout(camera_group)
        
        camera_layout.addWidget(QLabel("카메라 소스:"))
        self.camera_source_input = QLineEdit()
        self.camera_source_input.setPlaceholderText("0 (로컬 웹캠) 또는 http://IP:포트/video")
        self.camera_source_input.setText("0")  # 기본값: 로컬 웹캠
        camera_layout.addWidget(self.camera_source_input)
        
        self.btn_connect_camera = QPushButton("연결")
        self.btn_connect_camera.clicked.connect(self._reconnect_camera)
        camera_layout.addWidget(self.btn_connect_camera)
        
        camera_layout.addStretch()
        
        # ROI 모드 토글 버튼
        self.btn_roi_mode = QPushButton("🎯 ROI 모드")
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.setChecked(False)
        self.btn_roi_mode.clicked.connect(self.toggle_roi_mode)
        camera_layout.addWidget(self.btn_roi_mode)
        
        self.layout.addWidget(camera_group)

        # 로그 섹션
        log_group = QGroupBox("📋 데이터 로그")
        log_layout = QVBoxLayout(log_group)
        
        # 로그 필터 버튼
        log_filter_layout = QHBoxLayout()
        self.btn_log_all = QPushButton("전체")
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
        
        # 로그 테이블
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels(["Timestamp", "Frame No", "Decoded Data", "Status", "Confidence"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.setAlternatingRowColors(True)
        # Vertical header 클릭 비활성화 (홀수 행 선택 버그 방지)
        self.log_table.verticalHeader().setSectionsClickable(False)
        self.log_table.verticalHeader().setDefaultSectionSize(25)
        # 행 선택 모드 설정
        self.log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.log_table.setMinimumHeight(200)
        self.log_table.setMaximumHeight(200)
        
        log_layout.addWidget(self.log_table)
        self.layout.addWidget(log_group)

        # 초기 연결 (UI 생성 완료 후)
        self._reconnect_camera()

    def _get_camera_source(self):
        """카메라 소스 가져오기"""
        text = self.camera_source_input.text().strip()
        if not text:
            return 0
        # 숫자만 있으면 int로 변환, 아니면 URL로 사용
        if text.isdigit():
            return int(text)
        return text

    def _reconnect_camera(self):
        """카메라 재연결"""
        # 기존 스레드가 있으면 종료
        if self.player:
            self.player.stop()
            self.player.wait()
        if self.ai_worker:
            self.ai_worker.stop()
            self.ai_worker.wait()
        
        # 큐 초기화
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        # 새로운 카메라 소스로 스레드 생성
        camera_source = self._get_camera_source()
        if self.webcam_mode == 'resnet' and self.resnet_model:
            self.ai_worker = WebcamAIWorker(
                self.frame_queue, self.yolo_model, None,
                self.resnet_model, self.resnet_class_names, self.resnet_preprocess
            )
        else:
            self.ai_worker = WebcamAIWorker(self.frame_queue, self.yolo_model, self.dbr_reader)
        self.player = WebcamVideoPlayer(self.frame_queue, camera_source)
        
        # 시그널 연결
        self.player.change_pixmap_signal.connect(self.update_image)
        self.ai_worker.result_ready.connect(self.on_ai_result)
        # 플레이어에도 결과 전달 (시각화용)
        self.ai_worker.result_ready.connect(self.player.update_ai_results)
        
        # ROI 설정 (이미 ROI가 있으면 전달)
        if self.roi_rect and self.ai_worker:
            self.ai_worker.set_roi(self.roi_rect)
        
        # 시작
        self.ai_worker.start()
        self.player.start()
        
        # IP 웹캠인 경우 연결 상태 확인을 위한 짧은 대기
        if isinstance(camera_source, str) and not camera_source.isdigit():
            QMessageBox.information(
                self, 
                "연결", 
                f"IP 웹캠 연결 시도 중...\n소스: {camera_source}\n\n연결이 실패하면 자동으로 재시도합니다."
            )
        else:
            QMessageBox.information(self, "연결", f"카메라 연결 시도 중...\n소스: {camera_source}")
    
    def init_sliders(self, settings):
        """플레이어가 보낸 초기 하드웨어 값으로 슬라이더 세팅"""
        # 1. 초기값 백업 (리셋 버튼용)
        self.default_hw_settings = settings.copy()
        

        # 2. 슬라이더와 SpinBox에 값 적용 (신호 차단 후 적용하여 불필요한 set 방지)
        self.brightness_slider.blockSignals(True)
        self.brightness_spin.blockSignals(True)
        self.exposure_slider.blockSignals(True)
        self.exposure_spin.blockSignals(True)
        self.gain_slider.blockSignals(True)
        self.gain_spin.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        self.contrast_spin.blockSignals(True)
        self.focus_slider.blockSignals(True)
        self.focus_spin.blockSignals(True)
        self.exposure_auto_check.blockSignals(True)
        self.focus_auto_check.blockSignals(True)

        # 값 적용 (None이거나 범위를 벗어날 수 있으므로 예외처리 필요)
        try:
            # 해상도 설정
            width = settings.get('width', 640)
            height = settings.get('height', 480)
            resolution_text = f"{width}x{height}"
            # 현재 해상도에 맞는 항목 찾기
            resolution_map = {
                "320x240": 0,
                "640x480": 1,
                "800x600": 2,
                "1024x768": 3,
                "1280x720": 4,
                "1280x1024": 5,
                "1920x1080": 6
            }
            # 가장 가까운 해상도 찾기
            closest_idx = 1  # 기본값: 640x480
            min_diff = float('inf')
            for res_text, idx in resolution_map.items():
                w, h = map(int, res_text.split('x'))
                diff = abs(w - width) + abs(h - height)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = idx
            self.resolution_combo.blockSignals(True)
            self.resolution_combo.setCurrentIndex(closest_idx)
            self.resolution_combo.blockSignals(False)
            
            # 밝기 (0-255)
            brightness = settings.get('brightness', -1)
            if brightness != -1 and 0 <= brightness <= 255:
                self.brightness_slider.setValue(int(brightness * 10))
                self.brightness_spin.setValue(brightness)
            else:
                self.brightness_spin.setValue(0.0)
            
            # 노출 (-13 ~ 1)
            exposure = settings.get('exposure', 0)
            auto_exposure = settings.get('auto_exposure', 1.0)
            if auto_exposure == 1.0 or auto_exposure == 0.75:  # 자동 모드
                self.exposure_auto_check.setChecked(True)
                self.exposure_slider.setEnabled(False)
                self.exposure_spin.setEnabled(False)
                self.exposure_spin.setValue(0.0)
            else:
                self.exposure_auto_check.setChecked(False)
                self.exposure_slider.setEnabled(True)
                self.exposure_spin.setEnabled(True)
                if -13 <= exposure <= 1:
                    self.exposure_slider.setValue(int(exposure * 10))
                    self.exposure_spin.setValue(exposure)
                else:
                    self.exposure_spin.setValue(0.0)
            
            # 게인 (0-100)
            gain = settings.get('gain', -1)
            if gain != -1 and 0 <= gain <= 100:
                self.gain_slider.setValue(int(gain * 10))
                self.gain_spin.setValue(gain)
            else:
                self.gain_spin.setValue(0.0)
            
            # 대비 (0-255)
            contrast = settings.get('contrast', -1)
            if contrast != -1 and 0 <= contrast <= 255:
                self.contrast_slider.setValue(int(contrast * 10))
                self.contrast_spin.setValue(contrast)
            else:
                self.contrast_spin.setValue(0.0)
            
            # 초점 (0-250)
            focus = settings.get('focus', -1)
            autofocus = settings.get('autofocus', 0)
            if autofocus == 1:  # 자동 초점
                self.focus_auto_check.setChecked(True)
                self.focus_slider.setEnabled(False)
                self.focus_spin.setEnabled(False)
                self.focus_spin.setValue(0.0)
            else:
                self.focus_auto_check.setChecked(False)
                self.focus_slider.setEnabled(True)
                self.focus_spin.setEnabled(True)
                if focus != -1 and 0 <= focus <= 250:
                    self.focus_slider.setValue(int(focus * 10))
                    self.focus_spin.setValue(focus)
                else:
                    self.focus_spin.setValue(0.0)
                    
        except Exception as e:
            pass

        self.brightness_slider.blockSignals(False)
        self.brightness_spin.blockSignals(False)
        self.exposure_slider.blockSignals(False)
        self.exposure_spin.blockSignals(False)
        self.gain_slider.blockSignals(False)
        self.gain_spin.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        self.contrast_spin.blockSignals(False)
        self.focus_slider.blockSignals(False)
        self.focus_spin.blockSignals(False)
        self.exposure_auto_check.blockSignals(False)
        self.focus_auto_check.blockSignals(False)
    
    def reset_to_default(self):
        """처음 저장해둔 기본값으로 되돌리기"""
        if not self.default_hw_settings:
            return
        
        s = self.default_hw_settings
        
        # 슬라이더와 SpinBox 값 변경 -> valueChanged 신호 발생 -> Player가 카메라 설정 변경
        self.brightness_slider.blockSignals(True)
        self.brightness_spin.blockSignals(True)
        self.exposure_slider.blockSignals(True)
        self.exposure_spin.blockSignals(True)
        self.gain_slider.blockSignals(True)
        self.gain_spin.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        self.contrast_spin.blockSignals(True)
        self.focus_slider.blockSignals(True)
        self.focus_spin.blockSignals(True)
        self.exposure_auto_check.blockSignals(True)
        self.focus_auto_check.blockSignals(True)
        
        try:
            # 밝기
            brightness = s.get('brightness', -1)
            if brightness != -1 and 0 <= brightness <= 255:
                self.brightness_slider.setValue(int(brightness * 10))
                self.brightness_spin.setValue(brightness)
                self.player.update_setting(cv2.CAP_PROP_BRIGHTNESS, brightness)
            
            # 노출
            auto_exposure = s.get('auto_exposure', 1.0)
            exposure = s.get('exposure', 0)
            if auto_exposure == 1.0 or auto_exposure == 0.75:
                self.exposure_auto_check.setChecked(True)
                self.exposure_slider.setEnabled(False)
                self.exposure_spin.setEnabled(False)
                self.exposure_spin.setValue(0.0)
                self.player.update_setting(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            else:
                self.exposure_auto_check.setChecked(False)
                self.exposure_slider.setEnabled(True)
                self.exposure_spin.setEnabled(True)
                if -13 <= exposure <= 1:
                    self.exposure_slider.setValue(int(exposure * 10))
                    self.exposure_spin.setValue(exposure)
                    self.player.update_setting(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    self.player.update_setting(cv2.CAP_PROP_EXPOSURE, exposure)
            
            # 게인
            gain = s.get('gain', -1)
            if gain != -1 and 0 <= gain <= 100:
                self.gain_slider.setValue(int(gain * 10))
                self.gain_spin.setValue(gain)
                self.player.update_setting(cv2.CAP_PROP_GAIN, gain)
            
            # 대비
            contrast = s.get('contrast', -1)
            if contrast != -1 and 0 <= contrast <= 255:
                self.contrast_slider.setValue(int(contrast * 10))
                self.contrast_spin.setValue(contrast)
                self.player.update_setting(cv2.CAP_PROP_CONTRAST, contrast)
            
            # 초점
            autofocus = s.get('autofocus', 0)
            focus = s.get('focus', -1)
            if autofocus == 1:
                self.focus_auto_check.setChecked(True)
                self.focus_slider.setEnabled(False)
                self.focus_spin.setEnabled(False)
                self.focus_spin.setValue(0.0)
                self.player.update_setting(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                self.focus_auto_check.setChecked(False)
                self.focus_slider.setEnabled(True)
                self.focus_spin.setEnabled(True)
                if focus != -1 and 0 <= focus <= 250:
                    self.focus_slider.setValue(int(focus * 10))
                    self.focus_spin.setValue(focus)
                    self.player.update_setting(cv2.CAP_PROP_AUTOFOCUS, 0)
                    self.player.update_setting(cv2.CAP_PROP_FOCUS, focus)
            
            # 해상도 복원
            width = s.get('width', 640)
            height = s.get('height', 480)
            self.player.change_resolution(width, height)
            # 콤보박스도 업데이트
            resolution_text = f"{width}x{height}"
            resolution_map = {
                "320x240": 0,
                "640x480": 1,
                "800x600": 2,
                "1024x768": 3,
                "1280x720": 4,
                "1280x1024": 5,
                "1920x1080": 6
            }
            closest_idx = 1
            min_diff = float('inf')
            for res_text, idx in resolution_map.items():
                w, h = map(int, res_text.split('x'))
                diff = abs(w - width) + abs(h - height)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = idx
            self.resolution_combo.blockSignals(True)
            self.resolution_combo.setCurrentIndex(closest_idx)
            self.resolution_combo.blockSignals(False)
        except Exception as e:
            pass
        
        self.brightness_slider.blockSignals(False)
        self.brightness_spin.blockSignals(False)
        self.exposure_slider.blockSignals(False)
        self.exposure_spin.blockSignals(False)
        self.gain_slider.blockSignals(False)
        self.gain_spin.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        self.contrast_spin.blockSignals(False)
        self.focus_slider.blockSignals(False)
        self.focus_spin.blockSignals(False)
        self.exposure_auto_check.blockSignals(False)
        self.focus_auto_check.blockSignals(False)
    
    def on_resolution_changed(self, text):
        """해상도 변경 핸들러"""
        # 텍스트에서 해상도 추출 (예: "640x480 (VGA)" -> 640, 480)
        try:
            resolution_part = text.split(' ')[0]  # "640x480" 부분만 추출
            width, height = map(int, resolution_part.split('x'))
            self.player.change_resolution(width, height)
        except Exception as e:
            pass
    
    def on_brightness_slider_changed(self, value):
        """밝기 슬라이더 변경 핸들러"""
        brightness = value / 10.0
        self.brightness_spin.blockSignals(True)
        self.brightness_spin.setValue(brightness)
        self.brightness_spin.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_BRIGHTNESS, brightness)
    
    def on_brightness_spin_changed(self, value):
        """밝기 SpinBox 변경 핸들러"""
        self.brightness_slider.blockSignals(True)
        self.brightness_slider.setValue(int(value * 10))
        self.brightness_slider.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_BRIGHTNESS, value)
    
    def on_exposure_slider_changed(self, value):
        """노출 슬라이더 변경 핸들러"""
        exposure = value / 10.0
        self.exposure_spin.blockSignals(True)
        self.exposure_spin.setValue(exposure)
        self.exposure_spin.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_EXPOSURE, exposure)
    
    def on_exposure_spin_changed(self, value):
        """노출 SpinBox 변경 핸들러"""
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(int(value * 10))
        self.exposure_slider.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_EXPOSURE, value)
    
    def on_exposure_auto_changed(self, checked):
        """자동 노출 체크박스 변경 핸들러"""
        if checked:
            self.exposure_slider.setEnabled(False)
            self.exposure_spin.setEnabled(False)
            self.player.update_setting(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        else:
            self.exposure_slider.setEnabled(True)
            self.exposure_spin.setEnabled(True)
            self.player.update_setting(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 모드
            # 현재 SpinBox 값 적용
            self.on_exposure_spin_changed(self.exposure_spin.value())
    
    def on_gain_slider_changed(self, value):
        """게인 슬라이더 변경 핸들러"""
        gain = value / 10.0
        self.gain_spin.blockSignals(True)
        self.gain_spin.setValue(gain)
        self.gain_spin.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_GAIN, gain)
    
    def on_gain_spin_changed(self, value):
        """게인 SpinBox 변경 핸들러"""
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(value * 10))
        self.gain_slider.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_GAIN, value)
    
    def on_contrast_slider_changed(self, value):
        """대비 슬라이더 변경 핸들러"""
        contrast = value / 10.0
        self.contrast_spin.blockSignals(True)
        self.contrast_spin.setValue(contrast)
        self.contrast_spin.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_CONTRAST, contrast)
    
    def on_contrast_spin_changed(self, value):
        """대비 SpinBox 변경 핸들러"""
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(int(value * 10))
        self.contrast_slider.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_CONTRAST, value)
    
    def on_focus_slider_changed(self, value):
        """초점 슬라이더 변경 핸들러"""
        focus = value / 10.0
        self.focus_spin.blockSignals(True)
        self.focus_spin.setValue(focus)
        self.focus_spin.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_FOCUS, focus)
    
    def on_focus_spin_changed(self, value):
        """초점 SpinBox 변경 핸들러"""
        self.focus_slider.blockSignals(True)
        self.focus_slider.setValue(int(value * 10))
        self.focus_slider.blockSignals(False)
        self.player.update_setting(cv2.CAP_PROP_FOCUS, value)
    
    def on_focus_auto_changed(self, checked):
        """자동 초점 체크박스 변경 핸들러"""
        if checked:
            self.focus_slider.setEnabled(False)
            self.focus_spin.setEnabled(False)
            self.player.update_setting(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self.focus_slider.setEnabled(True)
            self.focus_spin.setEnabled(True)
            self.player.update_setting(cv2.CAP_PROP_AUTOFOCUS, 0)
            # 현재 SpinBox 값 적용
            self.on_focus_spin_changed(self.focus_spin.value())
    
    def on_ai_result(self, results):
        """AI 결과 처리 및 로그 추가"""
        self.frame_counter += 1
        self.player.update_ai_results(results)
        
        # 결과를 로그에 추가
        if results:
            for res in results:
                if len(res) >= 6:
                    x1, y1, x2, y2, text, confidence = res
                else:
                    x1, y1, x2, y2, text = res
                    confidence = 0.0
                
                if text:
                    self._add_log_entry(self.frame_counter, text, "✅ 성공", confidence)
                else:
                    self._add_log_entry(self.frame_counter, "인식 실패", "❌ 실패", 0.0)
    
    def toggle_roi_mode(self):
        """ROI 모드 토글"""
        self.roi_mode = self.btn_roi_mode.isChecked()
        self.video_label.set_roi_mode(self.roi_mode)
        
        if not self.roi_mode:
            # ROI 모드 비활성화 시 ROI 초기화
            self.roi_rect = None
            self.video_label.clear_roi()
            # AI worker에 ROI 초기화 알림
            if self.ai_worker:
                self.ai_worker.set_roi(None)
        else:
            # ROI 모드 활성화 시 안내 메시지
            QMessageBox.information(
                self,
                "ROI 모드 활성화",
                "웹캠 화면에서 마우스로 드래그하여 관심 영역(ROI)을 그려주세요.\n"
                "ROI가 설정되면 해당 영역만 집중적으로 탐지/분류합니다."
            )
    
    def on_roi_changed(self, x1: int, y1: int, x2: int, y2: int):
        """ROI 영역 변경 시 호출 (화면 좌표를 실제 프레임 좌표로 변환)"""
        # ROIVideoLabel._emit_roi_in_frame_coords()에서 이미 변환된 좌표를 받음
        # 하지만 추가 검증을 위해 실제 프레임 크기와 비교
        
        # 실제 프레임 크기가 설정되어 있으면 경계 체크
        if self.actual_frame_size:
            actual_h, actual_w = self.actual_frame_size
            
            # 경계 체크 및 클리핑
            x1 = max(0, min(x1, actual_w - 1))
            y1 = max(0, min(y1, actual_h - 1))
            x2 = max(x1 + 1, min(x2, actual_w))
            y2 = max(y1 + 1, min(y2, actual_h))
        
        # 실행 중이어도 ROI 좌표를 실시간으로 갱신
        # AI Worker는 다음 프레임을 처리할 때 바뀐 좌표를 가져다 쓰면 됩니다.
        self.roi_rect = (x1, y1, x2, y2)
        # AI worker에 ROI 업데이트 알림 (실시간 반영)
        if self.ai_worker:
            self.ai_worker.set_roi(self.roi_rect)
    
    def set_log_filter(self, mode: str):
        """로그 필터 설정"""
        self.log_filter_mode = mode
        self.btn_log_all.setChecked(mode == 'all')
        self.btn_log_success.setChecked(mode == 'success')
        self.btn_log_fail.setChecked(mode == 'fail')
        self._refresh_log_table()
    
    def _refresh_log_table(self):
        """로그 테이블 새로고침 (필터 적용)"""
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
                
                # 데이터 정리 (모든 공백 제거 - 앞뒤 공백 및 내부 불필요한 공백)
                # 저장된 데이터도 다시 한 번 정리 (이전에 저장된 데이터에 공백이 있을 수 있음)
                # 먼저 공백 제거 후 슬라이싱 (슬라이싱 후 공백 제거하면 문제 발생 가능)
                decoded_data_raw = ' '.join(str(entry.get('decoded_data', '')).split())
                decoded_data = decoded_data_raw[:50] if len(decoded_data_raw) > 50 else decoded_data_raw
                # 슬라이싱 후에도 다시 한 번 공백 제거 (안전장치)
                decoded_data = ' '.join(decoded_data.split())
                
                timestamp = ' '.join(str(entry.get('timestamp', '')).split())
                frame_no = ' '.join(str(entry.get('frame_no', '')).split())
                status = ' '.join(str(entry.get('status', '')).split())
                confidence = ' '.join(str(entry.get('confidence', '-')).split())
                
                # 모든 값에서 앞뒤 공백 완전 제거
                timestamp = timestamp.strip()
                frame_no = frame_no.strip()
                decoded_data = decoded_data.strip()
                status = status.strip()
                confidence = confidence.strip()
                
                # QTableWidgetItem 생성 및 텍스트 정렬 설정 (모든 행에 동일한 설정 적용)
                # 빈 문자열이 아닌 실제 값으로 생성하여 공백 문제 방지
                item0 = QTableWidgetItem(str(timestamp).strip() if timestamp else "")
                item0.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 0, item0)
                
                # Frame No - 공백 완전 제거 후 설정
                frame_no_str = str(frame_no).strip() if frame_no else ""
                item1 = QTableWidgetItem(frame_no_str)
                item1.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 1, item1)
                
                item2 = QTableWidgetItem(str(decoded_data).strip() if decoded_data else "")
                item2.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 2, item2)
                
                item3 = QTableWidgetItem(str(status).strip() if status else "")
                item3.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 3, item3)
                
                item4 = QTableWidgetItem(str(confidence).strip() if confidence else "-")
                item4.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 4, item4)
        
        self.log_table.scrollToBottom()
    
    def _add_log_entry(self, frame_no: int, decoded_data: str, status: str, confidence: float = 0.0):
        """로그 테이블에 항목 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 데이터 정리 (저장 시점부터 공백 제거)
        timestamp_clean = ' '.join(str(timestamp).split())
        frame_no_clean = ' '.join(str(frame_no).split())
        decoded_data_clean = ' '.join(str(decoded_data).split())
        status_clean = ' '.join(str(status).split())
        confidence_str = f"{confidence:.2f}" if confidence > 0 else "-"
        
        # 모든 로그 항목을 저장 (정리된 데이터로 저장)
        log_entry = {
            'timestamp': timestamp_clean,
            'frame_no': frame_no_clean,
            'decoded_data': decoded_data_clean,
            'status': status_clean,
            'confidence': confidence_str,
            'is_success': '✅' in status_clean
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
            
            # 이미 정리된 변수 사용 (timestamp_clean, frame_no_clean 등)
            decoded_data_display = decoded_data_clean[:50] if len(decoded_data_clean) > 50 else decoded_data_clean
            
            # QTableWidgetItem 생성 및 텍스트 정렬 설정
            item0 = QTableWidgetItem(timestamp_clean)
            item0.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 0, item0)
            
            item1 = QTableWidgetItem(frame_no_clean)
            item1.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 1, item1)
            
            item2 = QTableWidgetItem(decoded_data_display)
            item2.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 2, item2)
            
            item3 = QTableWidgetItem(status_clean)
            item3.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 3, item3)
            
            item4 = QTableWidgetItem(confidence_str)
            item4.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 4, item4)
            
            # 자동 스크롤
            self.log_table.scrollToBottom()
            
            # 최대 1000개 행 유지
            if self.log_table.rowCount() > 1000:
                self.log_table.removeRow(0)

    def update_image(self, cv_img):
        """OpenCV 이미지를 PyQt 이미지로 변환하여 표시"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 실제 프레임 크기 저장 (좌표 변환을 위해 필수!)
        self.actual_frame_size = (h, w)  # (height, width)
        
        # ROIVideoLabel인 경우 실제 프레임 크기 설정 (좌표 변환을 위해 필수!)
        if isinstance(self.video_label, ROIVideoLabel):
            self.video_label.set_actual_frame_size(h, w)  # 실제 프레임 크기 (원본 해상도)
        
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        if label_width > 0 and label_height > 0:
            scaled_image = convert_to_Qt_format.scaled(
                label_width, label_height, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            p = QPixmap.fromImage(scaled_image)
        else:
            p = QPixmap.fromImage(convert_to_Qt_format)
        
        return p

    def closeEvent(self, event):
        # 스레드 종료
        if self.player:
            self.player.stop()
        if self.ai_worker:
            self.ai_worker.stop()
        # 메인 윈도우의 참조 정리
        if hasattr(self, 'parent_window') and self.parent_window:
            self.parent_window.webcam_window = None
        event.accept()


# ============================================================================
# 프레임 집중 분석 창
# ============================================================================

class ZoomableGraphicsView(QGraphicsView):
    """줌/팬 기능이 있는 이미지 뷰어"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
    
    def set_image(self, image: np.ndarray, preserve_zoom: bool = False):
        """이미지 설정"""
        # 현재 줌 상태 저장
        current_zoom = self.zoom_factor if preserve_zoom else None
        current_transform = self.transform() if preserve_zoom else None
        
        # OpenCV 이미지를 QPixmap으로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 기존 아이템 제거
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        
        # 새 아이템 추가
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        # 줌 상태 복원 또는 리셋
        if preserve_zoom and current_zoom is not None and current_transform is not None:
            # 줌 상태 복원
            self.setTransform(current_transform)
        else:
            # 초기 줌 리셋
            self.reset_zoom()
    
    def reset_zoom(self):
        """줌 리셋"""
        self.zoom_factor = 1.0
        self.resetTransform()
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠로 줌 인/아웃"""
        # 줌 인/아웃
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            # 줌 인
            if self.zoom_factor * zoom_in_factor <= self.max_zoom:
                self.scale(zoom_in_factor, zoom_in_factor)
                self.zoom_factor *= zoom_in_factor
        else:
            # 줌 아웃
            if self.zoom_factor * zoom_out_factor >= self.min_zoom:
                self.scale(zoom_out_factor, zoom_out_factor)
                self.zoom_factor *= zoom_out_factor


class FrameAnalysisWindow(QMainWindow):
    """프레임 집중 분석 창"""
    
    def __init__(self, frame: np.ndarray, yolo_model, dbr_reader=None, resnet_model=None, resnet_class_names=None, resnet_preprocess=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍 프레임 집중 분석")
        self.resize(1400, 900)
        
        self.original_frame = frame.copy()
        self.current_frame = frame.copy()
        self.yolo_model = yolo_model
        self.dbr_reader = dbr_reader
        self.resnet_model = resnet_model
        self.resnet_class_names = resnet_class_names or []
        self.resnet_preprocess = resnet_preprocess
        self.qr_counter = 0  # QR 번호 카운터
        self.processing_mode = 'classify' if resnet_model else 'decode'
        
        self.init_ui()
        self.apply_dark_theme()
        
        # 초기 이미지 표시
        self.image_viewer.set_image(self.current_frame)
    
    def init_ui(self):
        """UI 초기화"""
        # 스크롤 가능한 중앙 위젯
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        
        main_layout = QVBoxLayout(central_widget)
        
        # 상단: 이미지 뷰어와 전처리 패널
        top_layout = QHBoxLayout()
        
        # 왼쪽: 이미지 뷰어
        viewer_group = QGroupBox("🖼️ 이미지 뷰어 (마우스 휠: 줌, 드래그: 이동)")
        viewer_layout = QVBoxLayout(viewer_group)
        
        self.image_viewer = ZoomableGraphicsView()
        self.image_viewer.setMinimumSize(800, 600)
        viewer_layout.addWidget(self.image_viewer)
        
        # 줌 컨트롤 버튼
        zoom_layout = QHBoxLayout()
        btn_reset_zoom = QPushButton("🔍 줌 리셋")
        btn_reset_zoom.clicked.connect(self.image_viewer.reset_zoom)
        zoom_layout.addWidget(btn_reset_zoom)
        zoom_layout.addStretch()
        viewer_layout.addLayout(zoom_layout)
        
        top_layout.addWidget(viewer_group, 2)
        
        # 오른쪽: 전처리 패널
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        
        # 전처리 옵션 그룹
        preprocess_group = QGroupBox("⚙️ 전처리 옵션")
        preprocess_layout = QVBoxLayout(preprocess_group)
        preprocess_scroll = QScrollArea()
        preprocess_scroll.setWidgetResizable(True)
        preprocess_scroll.setMinimumHeight(600)
        
        preprocess_content = QWidget()
        preprocess_content_layout = QVBoxLayout(preprocess_content)
        
        # CLAHE
        clahe_group = QGroupBox("CLAHE (대비 향상)")
        clahe_layout = QVBoxLayout(clahe_group)
        self.clahe_check = QCheckBox("사용")
        clahe_layout.addWidget(self.clahe_check)
        
        clahe_clip_layout = QHBoxLayout()
        clahe_clip_layout.addWidget(QLabel("Clip Limit:"))
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.1, 10.0)
        self.clahe_clip_spin.setValue(2.0)
        self.clahe_clip_spin.setSingleStep(0.1)
        clahe_clip_layout.addWidget(self.clahe_clip_spin)
        clahe_layout.addLayout(clahe_clip_layout)
        
        clahe_tile_layout = QHBoxLayout()
        clahe_tile_layout.addWidget(QLabel("Tile Size:"))
        self.clahe_tile_spin = QSpinBox()
        self.clahe_tile_spin.setRange(2, 32)
        self.clahe_tile_spin.setValue(8)
        clahe_tile_layout.addWidget(self.clahe_tile_spin)
        clahe_layout.addLayout(clahe_tile_layout)
        
        preprocess_content_layout.addWidget(clahe_group)
        
        # 노이즈 제거
        denoise_group = QGroupBox("노이즈 제거")
        denoise_layout = QVBoxLayout(denoise_group)
        self.denoise_check = QCheckBox("사용")
        denoise_layout.addWidget(self.denoise_check)
        
        denoise_method_layout = QHBoxLayout()
        denoise_method_layout.addWidget(QLabel("방법:"))
        self.denoise_method = QComboBox()
        self.denoise_method.addItems(['bilateral', 'gaussian', 'median'])
        denoise_method_layout.addWidget(self.denoise_method)
        denoise_layout.addLayout(denoise_method_layout)
        
        denoise_strength_layout = QHBoxLayout()
        denoise_strength_layout.addWidget(QLabel("강도:"))
        self.denoise_strength_spin = QSpinBox()
        self.denoise_strength_spin.setRange(1, 50)
        self.denoise_strength_spin.setValue(9)
        denoise_strength_layout.addWidget(self.denoise_strength_spin)
        denoise_layout.addLayout(denoise_strength_layout)
        
        # Bilateral 전용 옵션
        bilateral_group = QGroupBox("Bilateral Filter 옵션")
        bilateral_layout = QVBoxLayout(bilateral_group)
        sigma_color_layout = QHBoxLayout()
        sigma_color_layout.addWidget(QLabel("Sigma Color:"))
        self.bilateral_sigma_color = QDoubleSpinBox()
        self.bilateral_sigma_color.setRange(1.0, 200.0)
        self.bilateral_sigma_color.setValue(75.0)
        sigma_color_layout.addWidget(self.bilateral_sigma_color)
        bilateral_layout.addLayout(sigma_color_layout)
        
        sigma_space_layout = QHBoxLayout()
        sigma_space_layout.addWidget(QLabel("Sigma Space:"))
        self.bilateral_sigma_space = QDoubleSpinBox()
        self.bilateral_sigma_space.setRange(1.0, 200.0)
        self.bilateral_sigma_space.setValue(75.0)
        sigma_space_layout.addWidget(self.bilateral_sigma_space)
        bilateral_layout.addLayout(sigma_space_layout)
        denoise_layout.addWidget(bilateral_group)
        
        preprocess_content_layout.addWidget(denoise_group)
        
        # 이진화
        threshold_group = QGroupBox("이진화 (Adaptive Threshold)")
        threshold_layout = QVBoxLayout(threshold_group)
        self.threshold_check = QCheckBox("사용")
        threshold_layout.addWidget(self.threshold_check)
        
        threshold_block_layout = QHBoxLayout()
        threshold_block_layout.addWidget(QLabel("Block Size (홀수):"))
        self.threshold_block_spin = QSpinBox()
        self.threshold_block_spin.setRange(3, 51)
        self.threshold_block_spin.setValue(11)
        self.threshold_block_spin.setSingleStep(2)
        threshold_block_layout.addWidget(self.threshold_block_spin)
        threshold_layout.addLayout(threshold_block_layout)
        
        threshold_c_layout = QHBoxLayout()
        threshold_c_layout.addWidget(QLabel("C 값:"))
        self.threshold_c_spin = QSpinBox()
        self.threshold_c_spin.setRange(-50, 50)
        self.threshold_c_spin.setValue(2)
        threshold_c_layout.addWidget(self.threshold_c_spin)
        threshold_layout.addLayout(threshold_c_layout)
        
        preprocess_content_layout.addWidget(threshold_group)
        
        # 형태학적 연산
        morphology_group = QGroupBox("형태학적 연산")
        morphology_layout = QVBoxLayout(morphology_group)
        self.morphology_check = QCheckBox("사용")
        morphology_layout.addWidget(self.morphology_check)
        
        morphology_op_layout = QHBoxLayout()
        morphology_op_layout.addWidget(QLabel("연산:"))
        self.morphology_operation = QComboBox()
        self.morphology_operation.addItems(['closing', 'opening', 'dilation', 'erosion'])
        morphology_op_layout.addWidget(self.morphology_operation)
        morphology_layout.addLayout(morphology_op_layout)
        
        morphology_kernel_layout = QHBoxLayout()
        morphology_kernel_layout.addWidget(QLabel("Kernel Size (홀수):"))
        self.morphology_kernel_spin = QSpinBox()
        self.morphology_kernel_spin.setRange(3, 31)
        self.morphology_kernel_spin.setValue(5)
        self.morphology_kernel_spin.setSingleStep(2)
        morphology_kernel_layout.addWidget(self.morphology_kernel_spin)
        morphology_layout.addLayout(morphology_kernel_layout)
        
        preprocess_content_layout.addWidget(morphology_group)
        
        # 가우시안 블러 (추가 옵션)
        blur_group = QGroupBox("가우시안 블러")
        blur_layout = QVBoxLayout(blur_group)
        self.blur_check = QCheckBox("사용")
        blur_layout.addWidget(self.blur_check)
        
        blur_kernel_layout = QHBoxLayout()
        blur_kernel_layout.addWidget(QLabel("Kernel Size (홀수):"))
        self.blur_kernel_spin = QSpinBox()
        self.blur_kernel_spin.setRange(3, 51)
        self.blur_kernel_spin.setValue(5)
        self.blur_kernel_spin.setSingleStep(2)
        blur_kernel_layout.addWidget(self.blur_kernel_spin)
        blur_layout.addLayout(blur_kernel_layout)
        
        preprocess_content_layout.addWidget(blur_group)
        
        preprocess_content_layout.addStretch()
        preprocess_scroll.setWidget(preprocess_content)
        preprocess_layout.addWidget(preprocess_scroll)
        
        right_layout.addWidget(preprocess_group)
        
        # 분석 버튼
        analyze_layout = QVBoxLayout()
        btn_analyze = QPushButton("🔍 분석 시작")
        btn_analyze.setMinimumHeight(50)
        btn_analyze.clicked.connect(self.analyze_frame)
        analyze_layout.addWidget(btn_analyze)
        
        btn_reset = QPushButton("↺ 원본으로 리셋")
        btn_reset.clicked.connect(self.reset_to_original)
        analyze_layout.addWidget(btn_reset)
        
        right_layout.addLayout(analyze_layout)
        
        top_layout.addWidget(right_panel, 1)
        main_layout.addLayout(top_layout)
        
        # 하단: 로그 테이블
        log_group = QGroupBox("📋 분석 로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["QR 번호", "해독 정보", "신뢰도", "상태"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.verticalHeader().setSectionsClickable(False)
        self.log_table.verticalHeader().setDefaultSectionSize(25)
        self.log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.log_table.setMinimumHeight(200)
        
        log_layout.addWidget(self.log_table)
        main_layout.addWidget(log_group)
    
    def apply_dark_theme(self):
        """다크 테마 적용"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #00ff00;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2e2e2e;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: #00ff00;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2e2e2e;
                color: #00ff00;
                border: 1px solid #00ff00;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
            }
            QPushButton:pressed {
                background-color: #1e1e1e;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2e2e2e;
                color: #e0e0e0;
                border: 1px solid #00ff00;
                padding: 3px;
            }
            QComboBox {
                background-color: #2e2e2e;
                color: #e0e0e0;
                border: 1px solid #00ff00;
                padding: 3px;
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
            }
            QTableWidget::item {
                padding: 5px;
                color: #e0e0e0;
            }
            QTableWidget::item:alternate {
                background-color: #252525;
                color: #e0e0e0;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #404040;
            }
        """)
    
    def get_preprocessing_options(self):
        """전처리 옵션 가져오기"""
        block_val = self.threshold_block_spin.value()
        if block_val % 2 == 0:
            block_val += 1
        
        morph_val = self.morphology_kernel_spin.value()
        if morph_val % 2 == 0:
            morph_val += 1
        
        blur_val = self.blur_kernel_spin.value()
        if blur_val % 2 == 0:
            blur_val += 1
        
        return {
            'use_clahe': self.clahe_check.isChecked(),
            'clahe_clip_limit': self.clahe_clip_spin.value(),
            'clahe_tile_size': self.clahe_tile_spin.value(),
            'use_denoise': self.denoise_check.isChecked(),
            'denoise_method': self.denoise_method.currentText(),
            'denoise_strength': self.denoise_strength_spin.value(),
            'bilateral_sigma_color': self.bilateral_sigma_color.value(),
            'bilateral_sigma_space': self.bilateral_sigma_space.value(),
            'use_threshold': self.threshold_check.isChecked(),
            'threshold_block_size': block_val,
            'threshold_c': self.threshold_c_spin.value(),
            'use_morphology': self.morphology_check.isChecked(),
            'morphology_operation': self.morphology_operation.currentText(),
            'morphology_kernel_size': morph_val,
            'use_blur': self.blur_check.isChecked(),
            'blur_kernel_size': blur_val,
        }
    
    def apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """전처리 적용"""
        result = frame.copy()
        opts = self.get_preprocessing_options()
        
        # 가우시안 블러
        if opts.get('use_blur', False):
            kernel_size = opts.get('blur_kernel_size', 5)
            result = apply_gaussian_blur(result, kernel_size)
        
        # CLAHE
        if opts.get('use_clahe', False):
            result = apply_clahe(result, opts.get('clahe_clip_limit', 2.0), opts.get('clahe_tile_size', 8))
        
        # 노이즈 제거
        if opts.get('use_denoise', False):
            method = opts.get('denoise_method', 'bilateral')
            strength = opts.get('denoise_strength', 9)
            if method == 'bilateral':
                sigma_color = opts.get('bilateral_sigma_color', 75.0)
                sigma_space = opts.get('bilateral_sigma_space', 75.0)
                result = apply_bilateral_filter(result, strength, sigma_color, sigma_space)
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
    
    def reset_to_original(self):
        """원본 프레임으로 리셋"""
        self.current_frame = self.original_frame.copy()
        self.image_viewer.set_image(self.current_frame, preserve_zoom=True)
    
    def _detect_qr_codes(self, frame: np.ndarray) -> list:
        """YOLO로 QR 코드 탐지"""
        detections = []
        if self.yolo_model:
            try:
                results = self.yolo_model(frame, conf=0.25, verbose=False)
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    h, w = frame.shape[:2]
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'success': False,
                            'text': '',
                            'qr_number': 0
                        })
            except Exception as e:
                pass
        return detections
    
    def _merge_detections(self, detections: list) -> list:
        """중복 탐지 결과 병합"""
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 이미 사용된 탐지인지 확인
            is_duplicate = False
            for i, merged_det in enumerate(merged):
                m_x1, m_y1, m_x2, m_y2 = merged_det['bbox']
                m_center_x = (m_x1 + m_x2) / 2
                m_center_y = (m_y1 + m_y2) / 2
                
                # 중심점 거리 계산
                dist = ((center_x - m_center_x) ** 2 + (center_y - m_center_y) ** 2) ** 0.5
                
                # 거리가 임계값 이하면 중복으로 간주
                threshold = 50
                if dist < threshold:
                    is_duplicate = True
                    # 신뢰도가 높은 것으로 업데이트
                    if det['confidence'] > merged_det['confidence']:
                        merged[i] = det
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        return merged
    
    def _parse_dbr_result(self, result):
        """Dynamsoft 결과 파싱 헬퍼 (Worker와 동일한 안전한 로직)"""
        try:
            barcode_result = None
            items = None
            
            # 1. get 메서드 시도
            if hasattr(result, 'get_decoded_barcodes_result'):
                barcode_result = result.get_decoded_barcodes_result()
                if barcode_result:
                    items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
            
            # 2. items 직접 접근 시도
            if not items and hasattr(result, 'items'):
                items = result.items
            
            # 3. 속성 접근 시도
            if not items and hasattr(result, 'decoded_barcodes_result'):
                barcode_result = result.decoded_barcodes_result
                if barcode_result:
                    items = barcode_result.items if hasattr(barcode_result, 'items') else None
            
            # 결과 텍스트 추출
            if items and len(items) > 0:
                barcode_item = items[0]
                if hasattr(barcode_item, 'get_text'):
                    return barcode_item.get_text()
                elif hasattr(barcode_item, 'text'):
                    return barcode_item.text
                
        except Exception as e:
            print(f"파싱 오류: {e}")
            pass
            
        return None
    
    def _decode_qr_code(self, frame: np.ndarray, detection: dict):
        """Dynamsoft로 QR 코드 해독"""
        if self.dbr_reader is None:
            return
            
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # 프레임 크기 확인 및 경계 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # Dynamsoft 해독 (dbr은 이미 전역적으로 import됨)
            captured_result = self.dbr_reader.capture(rgb_image, dbr.EnumImagePixelFormat.IPF_RGB_888)
            
            # 결과 파싱 (개선된 파서 사용)
            decoded_text = self._parse_dbr_result(captured_result)
            
            if decoded_text:
                detection['text'] = decoded_text
                detection['success'] = True
            else:
                detection['text'] = ''
                detection['success'] = False
                    
        except Exception as e:
            # 에러 원인을 터미널에 출력하여 확인
            print(f"[ERROR] 해독 중 오류 발생: {e}")
            detection['text'] = ''
            detection['success'] = False
    
    def _classify_with_resnet(self, frame: np.ndarray, detection: dict):
        """ResNet으로 QR 코드 분류"""
        if self.resnet_model is None or self.resnet_preprocess is None:
            return
            
        try:
            import torch
            from PIL import Image
            
            x1, y1, x2, y2 = detection['bbox']
            
            # 프레임 크기 확인 및 경계 체크
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return
            
            # 이미지 크기 체크
            if x2 - x1 < 10 or y2 - y1 < 10:
                return
            
            # BGR to RGB 변환
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # PIL Image로 변환 및 전처리
            pil_img = Image.fromarray(rgb_image)
            input_tensor = self.resnet_preprocess(pil_img).unsqueeze(0)
            
            # ResNet 분류
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)
                label = self.resnet_class_names[idx] if idx < len(self.resnet_class_names) else f"Class_{idx}"
                score = conf.item() * 100
            
            # Detection 업데이트
            detection['text'] = label
            detection['success'] = True
            detection['confidence'] = score  # ResNet 신뢰도 저장
                    
        except Exception as e:
            detection['text'] = ''
            detection['success'] = False
    
    def analyze_frame(self):
        """프레임 분석"""
        # 로그 테이블 초기화
        self.log_table.setRowCount(0)
        self.qr_counter = 0
        
        # 전처리 적용
        processed_frame = self.apply_preprocessing(self.original_frame.copy())
        self.current_frame = processed_frame.copy()
        
        # 이미지 업데이트 (줌 상태 유지)
        self.image_viewer.set_image(processed_frame, preserve_zoom=True)
        
        # YOLO 탐지 (듀얼 패스: 원본 + 전처리)
        detections_orig = self._detect_qr_codes(self.original_frame)
        detections_prep = self._detect_qr_codes(processed_frame)
        
        # 결과 합치기 및 중복 제거
        all_detections = detections_orig + detections_prep
        detections = self._merge_detections(all_detections)
        
        # 분석 모드에 따라 분류 또는 해독
        decoded_count = 0
        if self.processing_mode == 'classify' and self.resnet_model:
            if len(detections) > 0:
                for det in detections:
                    # 원본 프레임에서 먼저 시도
                    if not det.get('success', False):
                        self._classify_with_resnet(self.original_frame, det)
                    # 실패하면 전처리 프레임에서 시도
                    if not det.get('success', False):
                        self._classify_with_resnet(processed_frame, det)
                    
                    if det.get('success', False):
                        decoded_count += 1
                    
                    # QR 번호 부여
                    self.qr_counter += 1
                    det['qr_number'] = self.qr_counter
                    
                    # 로그에 추가
                    self._add_log_entry(det)
        elif self.dbr_reader and len(detections) > 0:
            for det in detections:
                # 원본 프레임에서 먼저 시도
                if not det.get('success', False):
                    self._decode_qr_code(self.original_frame, det)
                # 실패하면 전처리 프레임에서 시도
                if not det.get('success', False):
                    self._decode_qr_code(processed_frame, det)
                
                if det.get('success', False):
                    decoded_count += 1
                
                # QR 번호 부여
                self.qr_counter += 1
                det['qr_number'] = self.qr_counter
                
                # 로그에 추가
                self._add_log_entry(det)
        
        # 탐지 결과 시각화 (QR 번호만 표시, 해독 성공 여부에 따라 색상 구분)
        if len(detections) > 0:
            vis_frame = processed_frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                # 해독 성공: 초록색, 실패: 빨간색
                color = (0, 255, 0) if det.get('success', False) else (0, 0, 255)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # QR 번호 표시 (모든 경우)
                qr_num = det.get('qr_number', 0)
                if qr_num > 0:
                    cv2.putText(vis_frame, f"QR-{qr_num}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            self.image_viewer.set_image(vis_frame, preserve_zoom=True)
            self.current_frame = vis_frame
    
    def _add_log_entry(self, det: dict):
        """로그 테이블에 항목 추가"""
        row_count = self.log_table.rowCount()
        self.log_table.insertRow(row_count)
        
        qr_number = det.get('qr_number', 0)
        decoded_text = det.get('text', '') if det.get('success', False) else '해독 실패'
        confidence = det.get('confidence', 0.0)
        status = "✅ 성공" if det.get('success', False) else "❌ 실패"
        
        item0 = QTableWidgetItem(f"QR-{qr_number}")
        item0.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.log_table.setItem(row_count, 0, item0)
        
        item1 = QTableWidgetItem(decoded_text[:50] if len(decoded_text) > 50 else decoded_text)
        item1.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.log_table.setItem(row_count, 1, item1)
        
        item2 = QTableWidgetItem(f"{confidence:.2f}")
        item2.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.log_table.setItem(row_count, 2, item2)
        
        item3 = QTableWidgetItem(status)
        item3.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.log_table.setItem(row_count, 3, item3)
        
        self.log_table.scrollToBottom()


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
        self.resnet_model = None
        self.resnet_class_names = ['2025', '2101', '2102', '2103', '2104', '2105']  # 기본 클래스 이름
        self.unet_model = None  # UNet 복원 모델
        self.video_path = None
        self.worker = None
        self.preprocessing_options = {}
        self.processing_mode = 'sync'  # 'sync', 'async', 'webcam'
        self.analysis_mode = 'decode'  # 'decode' (Dynamsoft) 또는 'classify' (ResNet)
        self.webcam_window = None  # 웹캠 창 참조
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
        
        # ROI 관련 변수
        self.roi_mode = False
        self.roi_rect = None  # (x1, y1, x2, y2) 원본 프레임 좌표
        
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
        
        self.btn_load_model = QPushButton("📦 YOLO 모델")
        self.btn_load_model.setMinimumHeight(40)
        self.btn_load_model.clicked.connect(self.load_model)

        self.btn_load_resnet = QPushButton("🤖 ResNet 모델")
        self.btn_load_resnet.setMinimumHeight(40)
        self.btn_load_resnet.clicked.connect(self.load_resnet_model)

        self.btn_load_unet = QPushButton("🔄 UNet 모델")
        self.btn_load_unet.setMinimumHeight(40)
        self.btn_load_unet.clicked.connect(self.load_unet_model)

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
        control_layout.addWidget(self.btn_load_resnet)
        control_layout.addWidget(self.btn_load_unet)
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
        # 처리 모드 선택
        filter_layout.addWidget(QLabel("  |  모드:"))
        self.btn_mode_sync = QPushButton("동기")
        self.btn_mode_sync.setCheckable(True)
        self.btn_mode_sync.setChecked(True)
        self.btn_mode_sync.setMinimumWidth(70)
        self.btn_mode_sync.clicked.connect(lambda: self.set_processing_mode('sync'))
        
        self.btn_mode_async = QPushButton("비동기")
        self.btn_mode_async.setCheckable(True)
        self.btn_mode_async.setMinimumWidth(70)
        self.btn_mode_async.clicked.connect(lambda: self.set_processing_mode('async'))
        
        self.btn_mode_webcam = QPushButton("웹캠")
        self.btn_mode_webcam.setCheckable(True)
        self.btn_mode_webcam.setMinimumWidth(70)
        self.btn_mode_webcam.clicked.connect(lambda: self.set_processing_mode('webcam'))

        filter_layout.addWidget(self.btn_mode_sync)
        filter_layout.addWidget(self.btn_mode_async)
        filter_layout.addWidget(self.btn_mode_webcam)
        
        # 분석 모드 선택 (분류모드/해독모드)
        filter_layout.addWidget(QLabel("  |  분석:"))
        self.btn_analysis_classify = QPushButton("분류모드")
        self.btn_analysis_classify.setCheckable(True)
        self.btn_analysis_classify.setChecked(False)
        self.btn_analysis_classify.setMinimumWidth(80)
        self.btn_analysis_classify.clicked.connect(lambda: self.set_analysis_mode('classify'))
        
        self.btn_analysis_decode = QPushButton("해독모드")
        self.btn_analysis_decode.setCheckable(True)
        self.btn_analysis_decode.setChecked(True)
        self.btn_analysis_decode.setMinimumWidth(80)
        self.btn_analysis_decode.clicked.connect(lambda: self.set_analysis_mode('decode'))
        
        filter_layout.addWidget(self.btn_analysis_classify)
        filter_layout.addWidget(self.btn_analysis_decode)
        filter_layout.addStretch()
        
        content_layout.addLayout(filter_layout)
        
        # 영상 플레이어 섹션 (컨트롤 버튼 + 타임라인 + 영상)
        video_section_layout = QVBoxLayout()
        
        # 영상 컨트롤 버튼 + 대시보드 (한 줄로 배치)
        video_control_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("▶️ 시작")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)
        
        self.btn_pause = QPushButton("⏸️ 일시정지")
        self.btn_pause.setMinimumHeight(40)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_processing)
        
        self.btn_stop = QPushButton("⏹️ 정지")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.stop_processing(show_message=True))
        
        # 타임라인 정보 라벨
        self.timeline_label = QLabel("00:00 / 00:00")
        self.timeline_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #00ff00;")
        
        video_control_layout.addWidget(self.btn_start)
        video_control_layout.addWidget(self.btn_pause)
        video_control_layout.addWidget(self.btn_stop)
        video_control_layout.addWidget(self.timeline_label)
        
        # 대시보드 (수평 배치)
        self._create_inline_dashboard(video_control_layout)
        
        # ROI 모드 토글 버튼
        self.btn_roi_mode = QPushButton("🎯 ROI 모드")
        self.btn_roi_mode.setMinimumHeight(40)
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.setToolTip("ROI 모드를 활성화하면 마우스로 관심 영역을 그릴 수 있습니다")
        self.btn_roi_mode.clicked.connect(self.toggle_roi_mode)
        
        # 히트맵/그래프 토글 버튼
        self.btn_heatmap = QPushButton("🗺️ 히트맵")
        self.btn_heatmap.setMinimumHeight(40)
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.clicked.connect(self.toggle_heatmap)
        
        self.btn_graphs = QPushButton("📈 그래프")
        self.btn_graphs.setMinimumHeight(40)
        self.btn_graphs.setCheckable(True)
        self.btn_graphs.clicked.connect(self.toggle_graphs)
        
        video_control_layout.addWidget(self.btn_roi_mode)
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
        
        # 원본 영상 (ROI 그리기 지원)
        original_video_group = QGroupBox("📹 원본 영상")
        original_video_layout = QVBoxLayout(original_video_group)
        self.original_video_label = ROIVideoLabel()
        self.original_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_video_label.setMinimumSize(500, 375)  # 크기 증가
        self.original_video_label.setStyleSheet("QLabel { background-color: #1e1e1e; }")
        self.original_video_label.setText("원본 영상")
        self.original_video_label.roi_changed.connect(self.on_roi_changed)
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
        self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels(["Timestamp", "Frame No", "Decoded Data", "Status", "Confidence"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.setAlternatingRowColors(True)
        # Vertical header 클릭 비활성화 (홀수 행 선택 버그 방지)
        self.log_table.verticalHeader().setSectionsClickable(False)
        self.log_table.verticalHeader().setDefaultSectionSize(25)
        # 행 선택 모드 설정
        self.log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        # 10줄 정도 보이도록 높이 설정 (헤더 + 10행 * 약 30px)
        self.log_table.setMinimumHeight(330)
        self.log_table.setMaximumHeight(330)
        # 더블클릭 이벤트 연결
        self.log_table.itemDoubleClicked.connect(self.on_log_table_double_clicked)

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
        
        # 구분선
        line4 = QFrame()
        line4.setFrameShape(QFrame.Shape.HLine)
        form.addWidget(line4)
        
        # 5. UNet 복원 적용
        self.side_unet_restore_check = QCheckBox("🔄 UNet 복원 적용")
        self.side_unet_restore_check.setToolTip("분류모드/해독모드에서 복원 적용 시 UNet 모델을 먼저 적용합니다.")
        form.addWidget(self.side_unet_restore_check)
        
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
            color: #e0e0e0;
        }
        QTableWidget::item:alternate {
            background-color: #252525;
            color: #e0e0e0;
            padding: 5px;
        }
        QTableWidget::item:selected {
            background-color: #404040;
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
                
                # OpenCV setNumThreads 호환성 체크 및 설정
                try:
                    if hasattr(cv2, 'setNumThreads'):
                        cv2.setNumThreads(0)  # 멀티스레딩 비활성화 (ultralytics와의 충돌 방지)
                except:
                    pass
                
                from ultralytics import YOLO
                YOLO_AVAILABLE = True
            except Exception as e:
                error_msg = str(e)
                if 'setNumThreads' in error_msg:
                    QMessageBox.critical(self, "오류", f"OpenCV 버전 호환성 문제입니다.\n\n오류: {error_msg}\n\n해결 방법:\npip install --upgrade opencv-python")
                else:
                    QMessageBox.critical(self, "오류", f"ultralytics를 로드할 수 없습니다:\n{error_msg}\n\nPyTorch CPU 버전을 설치하세요:\npip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
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
    
    def load_resnet_model(self):
        """ResNet 모델 업로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ResNet 모델 파일 선택", "", "PyTorch Models (*.pth *.pt)"
        )
        
        if not file_path:
            return
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            device = torch.device("cpu")
            
            # 먼저 체크포인트를 로드해서 클래스 개수 및 클래스 이름 확인
            checkpoint = torch.load(file_path, map_location=device)
            
            # 체크포인트에서 클래스 이름 찾기
            class_names_from_model = None
            if isinstance(checkpoint, dict):
                # 다양한 키 이름으로 클래스 이름 찾기
                if 'class_names' in checkpoint:
                    class_names_from_model = checkpoint['class_names']
                elif 'classes' in checkpoint:
                    class_names_from_model = checkpoint['classes']
                elif 'labels' in checkpoint:
                    class_names_from_model = checkpoint['labels']
                elif 'CLASS_NAMES' in checkpoint:
                    class_names_from_model = checkpoint['CLASS_NAMES']
            
            # state_dict에서 fc 레이어의 출력 크기 확인
            num_classes = None
            if isinstance(checkpoint, dict):
                if 'fc.weight' in checkpoint:
                    num_classes = checkpoint['fc.weight'].shape[0]
                elif 'state_dict' in checkpoint:
                    if 'fc.weight' in checkpoint['state_dict']:
                        num_classes = checkpoint['state_dict']['fc.weight'].shape[0]
                elif 'model' in checkpoint:
                    if 'fc.weight' in checkpoint['model']:
                        num_classes = checkpoint['model']['fc.weight'].shape[0]
            else:
                # checkpoint가 state_dict 자체인 경우
                if 'fc.weight' in checkpoint:
                    num_classes = checkpoint['fc.weight'].shape[0]
            
            if num_classes is None:
                # 클래스 개수를 찾을 수 없으면 기본값 사용
                num_classes = len(self.resnet_class_names)
                QMessageBox.warning(self, "경고", f"모델에서 클래스 개수를 자동으로 감지할 수 없습니다.\n기본값 {num_classes}개를 사용합니다.")
            else:
                # 클래스 이름 처리
                if class_names_from_model is not None:
                    # 모델에서 클래스 이름을 찾은 경우
                    if isinstance(class_names_from_model, (list, tuple)) and len(class_names_from_model) == num_classes:
                        self.resnet_class_names = list(class_names_from_model)
                    else:
                        # 클래스 이름 형식이 맞지 않으면 사용자에게 입력받기
                        class_names_from_model = None
                
                if class_names_from_model is None:
                    # 모델에 클래스 이름이 없거나 형식이 맞지 않으면 사용자에게 입력받기
                    text, ok = QInputDialog.getText(
                        self, 
                        "클래스 이름 입력", 
                        f"모델에 클래스 이름이 저장되어 있지 않습니다.\n\n{num_classes}개 클래스의 이름을 쉼표(,)로 구분하여 입력하세요:\n예: 2025,2101,2102,2103",
                        text=",".join(self.resnet_class_names[:num_classes]) if len(self.resnet_class_names) >= num_classes else ",".join([f"Class_{i}" for i in range(num_classes)])
                    )
                    
                    if ok and text.strip():
                        # 사용자 입력 파싱
                        input_names = [name.strip() for name in text.split(',') if name.strip()]
                        if len(input_names) == num_classes:
                            self.resnet_class_names = input_names
                        else:
                            QMessageBox.warning(self, "경고", f"입력한 클래스 개수({len(input_names)})가 모델의 클래스 개수({num_classes})와 일치하지 않습니다.\n기본 이름을 사용합니다.")
                            self.resnet_class_names = input_names[:num_classes] if len(input_names) > num_classes else input_names + [f"Class_{i}" for i in range(len(input_names), num_classes)]
                    else:
                        # 사용자가 취소하거나 빈 입력을 한 경우
                        if num_classes <= len(self.resnet_class_names):
                            self.resnet_class_names = self.resnet_class_names[:num_classes]
                        else:
                            self.resnet_class_names = self.resnet_class_names + [f"Class_{i}" for i in range(len(self.resnet_class_names), num_classes)]
            
            # ResNet18 모델 생성 (클래스 개수에 맞게)
            classifier = models.resnet18(weights=None)
            classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
            
            # 모델 로드
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                classifier.load_state_dict(checkpoint['model'])
            else:
                classifier.load_state_dict(checkpoint)
            
            classifier.eval()
            
            self.resnet_model = classifier
            self.resnet_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            QMessageBox.information(self, "성공", f"ResNet 모델 로드 완료!\n{os.path.basename(file_path)}\n클래스 개수: {num_classes}\n클래스: {', '.join(self.resnet_class_names)}")
            self._update_button_states()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"ResNet 모델 로드 실패:\n{str(e)}")
    
    def load_unet_model(self):
        """UNet 복원 모델 업로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "UNet 모델 파일 선택", "", "PyTorch Models (*.pth *.pt)"
        )
        
        if not file_path:
            return
        
        try:
            import torch
            import torch.nn as nn
            
            device = torch.device("cpu")
            
            # 체크포인트 로드
            checkpoint = torch.load(file_path, map_location=device)
            
            # 모델 구조 추론을 위한 시도
            # 일반적인 UNet 구조를 가정하거나, 사용자가 모델 클래스를 정의했을 수 있음
            # 여기서는 간단하게 state_dict만 로드하는 방식 사용
            
            # 체크포인트에서 모델 정보 확인
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # 모델 객체가 저장된 경우
                    self.unet_model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # state_dict만 있는 경우 - 사용자에게 모델 구조를 물어봐야 할 수도 있지만
                    # 일단 경고 메시지 표시
                    QMessageBox.warning(
                        self, 
                        "주의", 
                        "모델 구조 정보가 없습니다.\n"
                        "state_dict만 있는 경우, 모델 클래스를 먼저 정의해야 합니다.\n"
                        "일단 state_dict를 저장해두고, 모델 적용 시 사용하겠습니다."
                    )
                    # state_dict를 저장
                    self.unet_model = {'state_dict': checkpoint['state_dict']}
                else:
                    # state_dict 자체인 경우
                    self.unet_model = {'state_dict': checkpoint}
            else:
                # 모델 객체 자체인 경우
                self.unet_model = checkpoint
            
            # 모델을 eval 모드로 설정 (모델 객체인 경우)
            if hasattr(self.unet_model, 'eval'):
                self.unet_model.eval()
            
            QMessageBox.information(self, "성공", f"UNet 모델 로드 완료!\n{os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"UNet 모델 로드 실패:\n{str(e)}")
    
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
        
        self.side_unet_restore_check.setChecked(opts.get('use_unet_restore', False))
    
    def apply_sidebar_preprocessing(self):
        """사이드바 전처리 옵션 적용"""
        # UNet 복원 적용 체크 시 모델 확인
        use_unet_restore = self.side_unet_restore_check.isChecked()
        if use_unet_restore and not self.unet_model:
            QMessageBox.warning(self, "경고", "UNet 복원 적용을 사용하려면 먼저 UNet 모델을 로드하세요.")
            self.side_unet_restore_check.setChecked(False)
            use_unet_restore = False
        
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
            'use_unet_restore': use_unet_restore,
        }
        
        QMessageBox.information(self, "성공", "전처리 옵션이 적용되었습니다!")
        
        # Worker에 전처리 옵션 전달
        if self.worker:
            self.worker.set_preprocessing_options(self.preprocessing_options)
    
    def toggle_roi_mode(self):
        """ROI 모드 토글"""
        self.roi_mode = self.btn_roi_mode.isChecked()
        self.original_video_label.set_roi_mode(self.roi_mode)
        
        if not self.roi_mode:
            # ROI 모드 비활성화 시 ROI 초기화
            self.roi_rect = None
            self.original_video_label.clear_roi()
            # 워커들에게 ROI 초기화 알림
            self._update_roi_to_workers()
        else:
            # ROI 모드 활성화 시 안내 메시지
            QMessageBox.information(
                self,
                "ROI 모드 활성화",
                "원본 영상 화면에서 마우스로 드래그하여 관심 영역(ROI)을 그려주세요.\n"
                "ROI가 설정되면 해당 영역만 집중적으로 탐지/해독합니다."
            )
    
    def on_roi_changed(self, x1: int, y1: int, x2: int, y2: int):
        """ROI 영역 변경 시 호출"""
        # 처리 중이면 ROI 변경 불가
        if hasattr(self, 'worker') and self.worker:
            is_running = getattr(self.worker, 'is_running', False) or (hasattr(self.worker, 'isRunning') and self.worker.isRunning())
            if is_running:
                QMessageBox.warning(
                    self,
                    "ROI 변경 불가",
                    "영상 처리가 진행 중입니다.\nROI를 변경하려면 먼저 정지하세요."
                )
                # ROI를 이전 값으로 복원 (그림은 유지)
                # clear_roi()를 호출하지 않고, 이전 ROI를 다시 설정
                if hasattr(self, 'roi_rect') and self.roi_rect:
                    # 이전 ROI 좌표를 다시 설정 (그림 복원)
                    prev_x1, prev_y1, prev_x2, prev_y2 = self.roi_rect
                    if hasattr(self, 'original_video_label') and self.original_video_label.original_pixmap:
                        # 이전 ROI를 다시 그리기
                        temp_pixmap = self.original_video_label.original_pixmap.copy()
                        self.original_video_label._draw_roi_on_pixmap(temp_pixmap)
                return
        
        self.roi_rect = (x1, y1, x2, y2)
        # 워커들에게 ROI 업데이트 알림
        self._update_roi_to_workers()
    
    def _update_roi_to_workers(self):
        """모든 워커에게 ROI 업데이트"""
        # 동기/비동기 모드 모두 self.worker를 통해 접근
        if hasattr(self, 'worker') and self.worker:
            if hasattr(self.worker, 'set_roi'):
                self.worker.set_roi(self.roi_rect)
        
        # 웹캠 모드
        if hasattr(self, 'webcam_window') and self.webcam_window:
            if hasattr(self.webcam_window, 'ai_worker') and self.webcam_window.ai_worker:
                if hasattr(self.webcam_window.ai_worker, 'set_roi'):
                    self.webcam_window.ai_worker.set_roi(self.roi_rect)
    
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
    
    def start_processing(self):
        """영상 처리 시작"""
        if not self.yolo_model or not self.video_path:
            QMessageBox.warning(self, "경고", "모델과 영상을 먼저 로드하세요.")
            return
        
        # 분류모드 체크
        if self.analysis_mode == 'classify' and not self.resnet_model:
            QMessageBox.warning(self, "경고", "분류모드를 사용하려면 ResNet 모델을 먼저 로드하세요.")
            return
        
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
        self._finished_message_shown = False

        # 모드에 따라 다른 Worker 사용
        if self.processing_mode == 'async':
            # 분석 모드에 따라 모델 전달
            if self.analysis_mode == 'classify':
                self.worker = VideoManager(
                    self.yolo_model, None,
                    self.resnet_model, self.resnet_class_names, self.resnet_preprocess, 'classify',
                    self.unet_model
                )
            else:
                self.worker = VideoManager(self.yolo_model, self.dbr_reader, None, None, None, 'decode', self.unet_model)
            
            # 시그널 연결
            self.worker.frame_ready.connect(self._on_frame_processed)
            self.worker.timeline_updated.connect(self.on_timeline_updated)
            self.worker.progress_updated.connect(lambda cur, total: None)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.error_occurred.connect(self.on_error)
            
            # ROI 설정 (비동기 모드)
            if self.roi_rect:
                self.worker.set_roi(self.roi_rect)
            
            # 비동기 처리 시작
            self.worker.start(
                video_path=self.video_path,
                preprocessing_options=self.preprocessing_options,
                conf_threshold=0.25,
                frame_interval=self.frame_interval_spin.value()
            )
        else:
            self.worker = VideoProcessorWorker()
            self.worker.set_video(self.video_path)
            
            # 분석 모드에 따라 모델 전달
            if self.analysis_mode == 'classify':
                self.worker.set_model(self.yolo_model, None, self.resnet_model, self.resnet_class_names, self.resnet_preprocess, self.unet_model)
            else:
                self.worker.set_model(self.yolo_model, self.dbr_reader, None, None, None, self.unet_model)
            
            self.worker.set_preprocessing_options(self.preprocessing_options)
            self.worker.set_frame_interval(self.frame_interval_spin.value())
            self.worker.set_processing_mode(self.analysis_mode)  # 'decode' 또는 'classify'
            
            # ROI 설정 (동기 모드)
            if self.roi_rect:
                self.worker.set_roi(self.roi_rect)
            
            self.worker.frame_processed.connect(self._on_frame_processed)
            self.worker.timeline_updated.connect(self.on_timeline_updated)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.error_occurred.connect(self.on_error)
            self.worker.start()
        
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
    
    def stop_processing(self, show_message=False):
        """정지"""
        # 정지 버튼을 눌렀을 때는 완료 메시지가 표시되지 않도록 플래그 설정
        if show_message:
            self._finished_message_shown = True
        
        if self.worker:
            self.worker.stop()
            # wait() 메서드가 있는 경우에만 호출 (동기/비동기 모두 호환)
            if hasattr(self.worker, 'wait'):
                self.worker.wait()
            
            # worker 정지 후 is_running을 False로 설정하여 모드 변경 가능하도록 함
            if hasattr(self.worker, 'is_running'):
                self.worker.is_running = False
        
        # 정지 메시지 표시 (정지 버튼을 눌렀을 때만) - 비동기 모드에서도 표시
        if show_message:
            QMessageBox.information(self, "정지", "정지되었습니다.")
        
        # 버튼 활성화 (항상 실행 - 정지 버튼을 눌렀을 때와 영상 완료 시 모두)
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

    def set_analysis_mode(self, mode: str):
        """분석 모드 설정 (분류모드/해독모드)"""
        if mode == 'classify':
            if not self.resnet_model:
                QMessageBox.warning(self, "경고", "먼저 ResNet 모델을 로드하세요.")
                self.btn_analysis_classify.setChecked(False)
                self.btn_analysis_decode.setChecked(True)
                return
            self.analysis_mode = 'classify'
            self.btn_analysis_classify.setChecked(True)
            self.btn_analysis_decode.setChecked(False)
            QMessageBox.information(self, "분류모드", "분류모드로 변경되었습니다.\nYOLO로 탐지한 QR 코드를 ResNet 모델로 분류합니다.")
        else:
            self.analysis_mode = 'decode'
            self.btn_analysis_classify.setChecked(False)
            self.btn_analysis_decode.setChecked(True)
            QMessageBox.information(self, "해독모드", "해독모드로 변경되었습니다.\nYOLO로 탐지한 QR 코드를 Dynamsoft로 해독합니다.")
        
        self._update_button_states()
    
    def set_processing_mode(self, mode: str):
        """처리 모드 설정"""
        # 웹캠 모드인 경우 새로운 창 열기
        if mode == 'webcam':
            self._open_webcam_window()
            # 버튼 상태는 웹캠 모드로 설정하지 않음 (별도 창이므로)
            return
        
        # 처리 중인지 확인
        if self.worker:
            is_running = getattr(self.worker, 'is_running', False) or (hasattr(self.worker, 'isRunning') and self.worker.isRunning())
            if is_running:
                QMessageBox.warning(self, "모드 변경 불가", "영상 처리가 진행 중입니다.\n정지 또는 완료 후 모드를 변경할 수 있습니다.")
                # 이전 모드로 버튼 상태 복원
                if self.processing_mode == 'sync':
                    self.btn_mode_sync.setChecked(True)
                    self.btn_mode_async.setChecked(False)
                    self.btn_mode_webcam.setChecked(False)
                elif self.processing_mode == 'async':
                    self.btn_mode_sync.setChecked(False)
                    self.btn_mode_async.setChecked(True)
                    self.btn_mode_webcam.setChecked(False)
                return
        
        # 모드 변경 확인
        mode_name = "동기 모드" if mode == 'sync' else "비동기 모드"
        desc = "동기 모드: 안정적이지만 프레임 간격에 따라 영상이 끊길 수 있습니다." if mode == 'sync' else "비동기 모드: 영상은 부드럽지만 박스가 약간 지연될 수 있습니다."
        reply = QMessageBox.question(
            self, 
            "모드 변경", 
            f"{mode_name}로 변경하시겠습니까?\n\n{desc}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            # 변경 취소 - 이전 모드로 버튼 상태 복원
            if self.processing_mode == 'sync':
                self.btn_mode_sync.setChecked(True)
                self.btn_mode_async.setChecked(False)
                self.btn_mode_webcam.setChecked(False)
            elif self.processing_mode == 'async':
                self.btn_mode_sync.setChecked(False)
                self.btn_mode_async.setChecked(True)
                self.btn_mode_webcam.setChecked(False)
            return
        
        # 모드 변경 실행
        self.processing_mode = mode
        if mode == 'sync':
            self.btn_mode_sync.setChecked(True)
            self.btn_mode_async.setChecked(False)
            self.btn_mode_webcam.setChecked(False)
        elif mode == 'async':
            self.btn_mode_sync.setChecked(False)
            self.btn_mode_async.setChecked(True)
            self.btn_mode_webcam.setChecked(False)
        elif mode == 'webcam':
            self.btn_mode_sync.setChecked(False)
            self.btn_mode_async.setChecked(False)
            self.btn_mode_webcam.setChecked(True)
    
    def _open_webcam_window(self):
        """웹캠 모드 창 열기"""
        # 이미 열려있으면 포커스만 이동
        if self.webcam_window and self.webcam_window.isVisible():
            self.webcam_window.raise_()
            self.webcam_window.activateWindow()
            return
        
        # 모델이 로드되어 있는지 확인
        if not self.yolo_model:
            QMessageBox.warning(self, "경고", "먼저 YOLO 모델을 로드하세요.")
            self.btn_mode_webcam.setChecked(False)
            return
        
        # 웹캠 창 생성 및 표시 (분석 모드에 따라 모델 전달)
        if self.analysis_mode == 'classify' and self.resnet_model:
            self.webcam_window = WebcamWindow(
                self.yolo_model, None,
                self.resnet_model, self.resnet_class_names, self.resnet_preprocess
            )
        else:
            self.webcam_window = WebcamWindow(self.yolo_model, self.dbr_reader)
        self.webcam_window.parent_window = self  # 참조 저장
        self.webcam_window.show()
        # 웹캠 모드는 별도 창이므로 버튼 체크 해제
        self.btn_mode_webcam.setChecked(False)
        
    def on_processing_finished(self):
        """처리 완료"""
        # 중복 호출 방지 (정지 버튼을 눌렀을 때는 메시지 표시 안 함)
        if hasattr(self, '_finished_message_shown') and self._finished_message_shown:
            # 이미 정지된 상태면 버튼만 활성화 (정지 버튼을 눌렀을 때)
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.btn_pause.setText("⏸️ 일시정지")
            self.timeline_slider.setEnabled(False)
            
            # worker 정지 후 is_running을 False로 설정하여 모드 변경 가능하도록 함
            if self.worker:
                if hasattr(self.worker, 'is_running'):
                    self.worker.is_running = False
            return
        
        # 완료 메시지 표시
        QMessageBox.information(self, "완료", "영상 처리가 완료되었습니다!")
        self._finished_message_shown = True
        
        # 정지 처리 및 버튼 활성화
        # worker는 이미 종료되었으므로 stop() 호출 없이 버튼만 활성화
        if self.worker:
            # worker 정지 후 is_running을 False로 설정하여 모드 변경 가능하도록 함
            if hasattr(self.worker, 'is_running'):
                self.worker.is_running = False
            
            # 동기 모드에서는 stop() 호출
            if self.processing_mode == 'sync':
                self.worker.stop()
                self.worker.wait()
        
        # 버튼 활성화 (항상 실행 - 영상 완료 시) - 비동기 모드에서도 확실히 활성화
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸️ 일시정지")
        self.timeline_slider.setEnabled(False)
        
    def on_frame_interval_changed(self, value: int):
        """프레임 간격 변경"""
        if self.worker:
            # VideoManager는 is_running 속성 사용, VideoProcessorWorker는 isRunning() 메서드 사용
            is_running = getattr(self.worker, 'is_running', False) or (hasattr(self.worker, 'isRunning') and self.worker.isRunning())
            if is_running:
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
            self.on_frame_processed(original_frame, preprocessed_frame, detections, metrics)
        except Exception as e:
            pass
    
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
            confidence = det.get('confidence', 0.0)
            if det['success']:
                self._add_log_entry(frame_idx, det['text'], "✅ 성공", confidence)
            else:
                self._add_log_entry(frame_idx, "인식 실패", "❌ 실패", confidence)
    
    def on_processing_finished(self):
        """처리 완료"""
        # 중복 호출 방지 (정지 버튼을 눌렀을 때는 메시지 표시 안 함)
        if hasattr(self, '_finished_message_shown') and self._finished_message_shown:
            # 이미 정지된 상태면 버튼만 활성화
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.btn_pause.setText("⏸️ 일시정지")
            self.timeline_slider.setEnabled(False)
            return
        
        # 완료 메시지 표시
        QMessageBox.information(self, "완료", "영상 처리가 완료되었습니다!")
        self._finished_message_shown = True
        
        # 정지 처리 및 버튼 활성화
        # worker는 이미 종료되었으므로 stop() 호출 없이 버튼만 활성화
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸️ 일시정지")
        self.timeline_slider.setEnabled(False)

    def on_error(self, error_msg: str):
        """오류 발생"""
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
                
                # 데이터 정리 (모든 공백 제거 - 앞뒤 공백 및 내부 불필요한 공백)
                # 저장된 데이터도 다시 한 번 정리 (이전에 저장된 데이터에 공백이 있을 수 있음)
                # 먼저 공백 제거 후 슬라이싱 (슬라이싱 후 공백 제거하면 문제 발생 가능)
                decoded_data_raw = ' '.join(str(entry.get('decoded_data', '')).split())
                decoded_data = decoded_data_raw[:50] if len(decoded_data_raw) > 50 else decoded_data_raw
                # 슬라이싱 후에도 다시 한 번 공백 제거 (안전장치)
                decoded_data = ' '.join(decoded_data.split())
                
                timestamp = ' '.join(str(entry.get('timestamp', '')).split())
                frame_no = ' '.join(str(entry.get('frame_no', '')).split())
                status = ' '.join(str(entry.get('status', '')).split())
                
                # 모든 값에서 앞뒤 공백 완전 제거
                timestamp = timestamp.strip()
                frame_no = frame_no.strip()
                decoded_data = decoded_data.strip()
                status = status.strip()
                
                # QTableWidgetItem 생성 및 텍스트 정렬 설정 (모든 행에 동일한 설정 적용)
                # 빈 문자열이 아닌 실제 값으로 생성하여 공백 문제 방지
                item0 = QTableWidgetItem(str(timestamp).strip() if timestamp else "")
                item0.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 0, item0)
                
                # Frame No - 공백 완전 제거 후 설정
                frame_no_str = str(frame_no).strip() if frame_no else ""
                item1 = QTableWidgetItem(frame_no_str)
                item1.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 1, item1)
                
                item2 = QTableWidgetItem(str(decoded_data).strip() if decoded_data else "")
                item2.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 2, item2)
                
                item3 = QTableWidgetItem(str(status).strip() if status else "")
                item3.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.log_table.setItem(row_count, 3, item3)
        
        # 자동 스크롤
        self.log_table.scrollToBottom()
    
    def on_log_table_double_clicked(self, item):
        """로그 테이블 더블클릭 시 프레임 분석 창 열기"""
        if not self.video_path or not self.yolo_model:
            QMessageBox.warning(self, "경고", "영상과 모델을 먼저 로드하세요.")
            return
        
        # 선택된 행의 프레임 번호 가져오기
        row = item.row()
        frame_no_item = self.log_table.item(row, 1)  # Frame No 컬럼
        if not frame_no_item:
            return
        
        try:
            frame_no = int(frame_no_item.text().strip())
        except ValueError:
            QMessageBox.warning(self, "오류", "유효하지 않은 프레임 번호입니다.")
            return
        
        # 비디오에서 해당 프레임 읽기
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "오류", "비디오 파일을 열 수 없습니다.")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.warning(self, "오류", f"프레임 {frame_no}를 읽을 수 없습니다.")
            return
        
        # FrameAnalysisWindow 열기 (분석 모드에 따라 모델 전달)
        if self.analysis_mode == 'classify' and self.resnet_model:
            analysis_window = FrameAnalysisWindow(
                frame, self.yolo_model, None, 
                self.resnet_model, self.resnet_class_names, self.resnet_preprocess, self
            )
        else:
            analysis_window = FrameAnalysisWindow(frame, self.yolo_model, self.dbr_reader, None, None, None, self)
        analysis_window.show()
    
    # ============================================================================
    # UI 업데이트 메서드
    # ============================================================================
    
    def _display_frame(self, label: QLabel, frame: np.ndarray):
        """영상 프레임 표시"""
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # QImage 생성 - 메모리 안전성을 위해 copy() 사용
        # rgb_frame.data는 포인터이므로 함수 종료 후 메모리가 해제될 수 있음
        rgb_frame_copy = rgb_frame.copy()
        q_image = QImage(rgb_frame_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # 라벨 크기에 맞춰 스케일링
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)
        
        # ROIVideoLabel인 경우 실제 프레임 크기 설정
        if isinstance(label, ROIVideoLabel):
            label.set_actual_frame_size(h, w)  # 실제 프레임 크기 (원본 해상도)
    
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
    
    def _add_log_entry(self, frame_no: int, decoded_data: str, status: str, confidence: float = 0.0):
        """로그 테이블에 항목 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 데이터 정리 (저장 시점부터 공백 제거)
        timestamp_clean = ' '.join(str(timestamp).split())
        frame_no_clean = ' '.join(str(frame_no).split())
        decoded_data_clean = ' '.join(str(decoded_data).split())
        status_clean = ' '.join(str(status).split())
        confidence_str = f"{confidence:.2f}" if confidence > 0 else "-"
        
        # 모든 로그 항목을 저장 (정리된 데이터로 저장)
        log_entry = {
            'timestamp': timestamp_clean,
            'frame_no': frame_no_clean,
            'decoded_data': decoded_data_clean,
            'status': status_clean,
            'confidence': confidence_str,
            'is_success': '✅' in status_clean
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
            
            # 이미 정리된 변수 사용 (timestamp_clean, frame_no_clean 등)
            decoded_data_display = decoded_data_clean[:50] if len(decoded_data_clean) > 50 else decoded_data_clean
            
            # QTableWidgetItem 생성 및 텍스트 정렬 설정
            item0 = QTableWidgetItem(timestamp_clean)
            item0.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 0, item0)
            
            item1 = QTableWidgetItem(frame_no_clean)
            item1.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 1, item1)
            
            item2 = QTableWidgetItem(decoded_data_display)
            item2.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 2, item2)
            
            item3 = QTableWidgetItem(status_clean)
            item3.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 3, item3)
            
            item4 = QTableWidgetItem(confidence_str)
            item4.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row_count, 4, item4)
            
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
            self.dbr_reader = None
            self.resnet_model = None
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
        # 분류모드인 경우 ResNet 모델도 필요
        if self.analysis_mode == 'classify':
            can_start = self.yolo_model is not None and self.resnet_model is not None and self.video_path is not None
        else:
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
    
    # 이미 실행 중이면 종료
    if _app_started:
        return
    
    _app_started = True
    
    app = QApplication(sys.argv)
    
    # 폰트 설정
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # 로그인 다이얼로그 표시
    login = LoginDialog()
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
    
    main()
