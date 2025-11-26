"""
영상 플레이어 + 실시간 QR 탐지 (병렬 처리)
[최종 최적화]: YOLO ROI 리스트를 먼저 필터링하여 중복 스레드 생성을 방지
[성능 개선]: YOLO 탐지 + Dynamsoft 해독
[일직선 움직임 최적화]: 조선소 T-bar 제작 공정 등 일직선 움직임 QR 코드 추적 최적화
"""

import cv2
import time
import os
import sys
import platform
import numpy as np
import threading
import queue
from queue import Queue, Empty

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# Dynamsoft Barcode Reader import (dynamsoft-barcode-reader-bundle v11)
try:
    from dynamsoft_barcode_reader_bundle import dbr, license, cvr
    DBR_AVAILABLE = True
    DBR_VERSION = "bundle_v11"
except ImportError:
    # 이전 버전 (dbr) 시도
    try:
        from dbr import BarcodeReader, BarcodeReaderError
        DBR_AVAILABLE = True
        DBR_VERSION = "dbr_legacy"
    except ImportError:
        DBR_AVAILABLE = False
        DBR_VERSION = None
        print("⚠️ Dynamsoft Barcode Reader를 사용할 수 없습니다. pip install dynamsoft-barcode-reader-bundle로 설치하세요.")

# YOLO 모델 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics를 사용할 수 없습니다. pip install ultralytics로 설치하세요.")

# PIL import (한글 폰트 지원용)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL을 사용할 수 없습니다. pip install Pillow로 설치하세요.")

def _process_decoded_text(decoded_text):
    """디코딩된 텍스트 처리 (특수 문자 및 인코딩 처리)"""
    if not decoded_text:
        return None
    
    # 특수 문자 처리
    decoded_text = decoded_text.replace('–', '-').replace('—', '-')
    
    # 한글 인코딩 처리
    try:
        if isinstance(decoded_text, bytes):
            decoded_text = decoded_text.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_text = decoded_text.decode('cp949')
        except:
            decoded_text = str(decoded_text)
    
    return decoded_text

def yolo_detect_qr_locations(model, frame, conf_threshold=0.25):
    """YOLO 모델로 QR 코드 위치 빠르게 탐지"""
    try:
        results = model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        
        locations = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 패딩 추가 (QR 코드 경계 확보)
                pad = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                locations.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        return locations
    except Exception as e:
        return []

# -----------------------------------------------------------------
# ★★★★★ IoU 계산 함수들 ★★★★★
# -----------------------------------------------------------------
def calculate_iou(bbox1, bbox2):
    """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 교집합 영역 계산
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 교집합이 없는 경우
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 각 박스의 면적
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 합집합 영역
    union = area1 + area2 - intersection
    
    # IoU 계산
    iou = intersection / union if union > 0 else 0.0
    return iou

# -----------------------------------------------------------------
# ★★★★★ 새로운 최적화 함수 ★★★★★
#
# YOLO ROI 리스트를 필터링하는 함수
# -----------------------------------------------------------------
def filter_overlapping_yolo_rois(locations, iou_threshold=0.5):
    """
    YOLO가 반환한 ROI 리스트에서 겹치는 ROI를 제거 (NMS와 유사)
    """
    if not locations:
        return []
    
    # 신뢰도(confidence) 기준으로 정렬 (높은 것이 우선)
    locations.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered_locations = []
    for location in locations:
        is_overlapping = False
        bbox1 = location['bbox']
        
        for filtered in filtered_locations:
            bbox2 = filtered['bbox']
            # calculate_iou는 (x1, y1, x2, y2) 포맷을 사용
            iou = calculate_iou(bbox1, bbox2)
            
            if iou > iou_threshold:
                is_overlapping = True
                break
        
        if not is_overlapping:
            filtered_locations.append(location)
            
    return filtered_locations

# -----------------------------------------------------------------
# ★★★★★ 핵심 수정 사항 ★★★★★
#
# `process_frame_with_yolo`가 `filter_overlapping_yolo_rois`를
# 호출하도록 수정합니다.
# -----------------------------------------------------------------
def process_frame_with_yolo(frame, yolo_model, conf_threshold=0.25):
    """YOLO로 빠르게 위치만 탐지 (해독 제거, 비동기 해독으로 분리)
    
    Args:
        frame: 입력 프레임
        yolo_model: YOLO 모델
        conf_threshold: YOLO 신뢰도 임계값
    
    Returns:
        filtered_locations: 필터링된 QR 위치 리스트 [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
    """
    # 1단계: YOLO로 빠르게 QR 코드 위치 탐지
    if yolo_model is not None:
        qr_locations = yolo_detect_qr_locations(yolo_model, frame, conf_threshold)
        
        # ★★★★★ 새로운 최적화 단계 ★★★★★
        # 겹치는 ROI를 먼저 제거
        filtered_locations = filter_overlapping_yolo_rois(qr_locations, iou_threshold=0.5)
        
        # ROI 필터링 결과 (필요시 주석 해제)
        # if len(qr_locations) > len(filtered_locations):
        #     print(f"    ⚡ ROI 필터링: {len(qr_locations)}개 -> {len(filtered_locations)}개")
        
        return filtered_locations
    
    return []

def create_single_frame(frame):
    """원본 프레임만 사용"""
    return frame, [1.0]


def get_platform_font_paths():
    """플랫폼별 한글 폰트 경로 반환"""
    system = platform.system()
    font_paths = []
    
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",   # 굴림
            "C:/Windows/Fonts/batang.ttc",  # 바탕
            "C:/Windows/Fonts/arial.ttf"    # Arial (fallback)
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # Apple SD Gothic Neo
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # Apple Gothic
            "/Library/Fonts/AppleGothic.ttf",  # Apple Gothic (alternative)
            "/System/Library/Fonts/Helvetica.ttc",  # Helvetica (fallback)
        ]
    elif system == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # 나눔고딕
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans (fallback)
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Liberation Sans
        ]
    
    # 프로젝트 내 폰트 경로도 추가 (모든 플랫폼)
    project_font_path = "data/font/NanumGothic.ttf"
    if os.path.exists(project_font_path):
        font_paths.insert(0, project_font_path)  # 최우선
    
    return font_paths

def put_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """OpenCV 이미지에 한글 텍스트를 그리는 함수 (플랫폼 자동 감지)"""
    if not PIL_AVAILABLE:
        # PIL이 없으면 OpenCV 기본 폰트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    
    try:
        # OpenCV 이미지를 PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 플랫폼별 폰트 경로 가져오기
        font_paths = get_platform_font_paths()
        
        font = None
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except:
                continue
        
        # 폰트를 찾지 못한 경우 기본 폰트 사용
        if font is None:
            font = ImageFont.load_default()
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color)
        
        # PIL 이미지를 OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        # 오류 발생 시 OpenCV 기본 폰트로 fallback
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img


def is_center_in_bbox(center_x, center_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
    """중심점이 사각형 안에 있는지 확인"""
    return bbox_x1 <= center_x <= bbox_x2 and bbox_y1 <= center_y <= bbox_y2


def calculate_center_distance(bbox1, bbox2):
    """두 바운딩 박스의 중심점 간 거리 계산"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 중심점 계산
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    # 유클리드 거리
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    # 박스의 대각선 길이로 정규화 (큰 박스 기준)
    diag1 = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
    diag2 = np.sqrt((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)
    max_diag = max(diag1, diag2)
    
    # 정규화된 거리 (0~1 사이)
    normalized_distance = distance / max_diag if max_diag > 0 else float('inf')
    
    return normalized_distance

def get_qr_center_and_bbox(detection):
    """QR의 중심점과 사각형 좌표를 반환 - quad_xy 우선 사용"""
    # quad_xy가 있으면 가장 정확한 중심점과 사각형 사용
    if 'quad_xy' in detection:
        quad = detection['quad_xy']
        if quad is not None and len(quad) == 4:
            quad_array = np.array(quad)
            center = np.mean(quad_array, axis=0)
            # quad_xy의 바운딩 박스 계산
            x_coords = quad_array[:, 0]
            y_coords = quad_array[:, 1]
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)
            return center[0], center[1], x1, y1, x2, y2
    
    # polygon_xy가 있으면 사용
    if 'polygon_xy' in detection:
        polygon = detection['polygon_xy']
        if polygon is not None and len(polygon) >= 4:
            polygon_array = np.array(polygon)
            center = np.mean(polygon_array, axis=0)
            x_coords = polygon_array[:, 0]
            y_coords = polygon_array[:, 1]
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)
            return center[0], center[1], x1, y1, x2, y2
    
    # bbox_xyxy 사용 (fallback)
    elif 'bbox_xyxy' in detection:
        bbox = detection['bbox_xyxy']
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y, x1, y1, x2, y2
    
    # cxcy+wh 사용 (최종 fallback)
    elif 'cxcy' in detection and 'wh' in detection:
        cx, cy = detection['cxcy']
        w, h = detection['wh']
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        return cx, cy, x1, y1, x2, y2
    
    return None, None, None, None, None, None

def extract_bounding_box(detection, image_width=None, image_height=None):
    """
    Detection 결과에서 우선순위에 따라 Bounding Box 추출
    
    우선순위:
    1. polygon_xy 또는 quad_xy (가장 정확)
    2. cxcy + wh (좋은 대안)
    3. bbox_xyxyn (정규화된 좌표)
    4. bbox_xyxy (기본)
    """
    # 🥇 1순위: polygon_xy 또는 quad_xy (가장 정확)
    for key in ['polygon_xy', 'quad_xy']:
        if key in detection:
            points = detection[key]
            if len(points) >= 4:
                # 모든 점의 x, y 좌표 추출
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                
                # Bounding Box 계산
                x1 = min(x_coords)
                y1 = min(y_coords)
                x2 = max(x_coords)
                y2 = max(y_coords)
                
                return [x1, y1, x2, y2], f"📍 {key} 기반"
    
    # 🥈 2순위: cxcy + wh (좋은 대안)
    if 'cxcy' in detection and 'wh' in detection:
        cx, cy = detection['cxcy']
        w, h = detection['wh']
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return [x1, y1, x2, y2], f"📍 cxcy+wh 기반 (중심: {cx:.1f},{cy:.1f}, 크기: {w:.1f}x{h:.1f})"
    
    # 🥉 3순위: bbox_xyxyn (정규화된 좌표)
    if 'bbox_xyxyn' in detection and image_width and image_height:
        bbox_norm = detection['bbox_xyxyn']
        x1 = bbox_norm[0] * image_width
        y1 = bbox_norm[1] * image_height
        x2 = bbox_norm[2] * image_width
        y2 = bbox_norm[3] * image_height
        
        return [x1, y1, x2, y2], f"📍 정규화 좌표 기반"
    
    # 4순위: bbox_xyxy (기본)
    if 'bbox_xyxy' in detection:
        return detection['bbox_xyxy'], f"📍 bbox_xyxy 기반"
    
    return None, "⚠️ 위치 정보 없음"


# -----------------------------------------------------------------
# ★★★★★ 프레임 간 추적 기능 ★★★★★
# -----------------------------------------------------------------
class QRTrack:
    """단일 QR 코드 추적 정보"""
    def __init__(self, track_id, qr_data, frame_number):
        self.track_id = track_id
        self.qr_data = qr_data  # {'text': str, 'detection': dict, 'method': str, 'success': bool}
        self.frame_number = frame_number
        self.last_seen_frame = frame_number
        self.missed_frames = 0
        self.history = []  # 위치 이력 [(x1, y1, x2, y2), ...]
        
        # 위치 정보 추출
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr_data['detection'])
        if center_x is not None:
            self.bbox = (x1, y1, x2, y2)
            self.center = (center_x, center_y)
            self.history.append(self.bbox)
        else:
            self.bbox = None
            self.center = None
    
    def update(self, qr_data, frame_number):
        """추적 정보 업데이트"""
        self.qr_data = qr_data
        self.frame_number = frame_number
        self.last_seen_frame = frame_number
        self.missed_frames = 0
        
        # 위치 정보 업데이트
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr_data['detection'])
        if center_x is not None:
            self.bbox = (x1, y1, x2, y2)
            self.center = (center_x, center_y)
            self.history.append(self.bbox)
            # 최근 10개만 유지
            if len(self.history) > 10:
                self.history.pop(0)
    
    def predict_position(self):
        """이전 위치 기반으로 다음 위치 예측 (일직선 움직임 가정, 개선된 선형 예측)"""
        if self.bbox is None:
            return None
        
        if len(self.history) < 2:
            # 이력이 부족하면 현재 위치 반환
            return self.bbox
        
        # 최근 N개 위치로 평균 속도 계산 (일직선 움직임 가정)
        # 최대 5개 프레임의 이력을 사용하여 더 안정적인 속도 계산
        num_frames = min(len(self.history), 5)
        recent_history = self.history[-num_frames:]
        
        # 각 프레임의 중심점 계산
        centers = []
        for bbox in recent_history:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append((center_x, center_y))
        
        # 일직선 움직임 검증: 최근 위치들이 일직선에 가까운지 확인
        # (x, y) 좌표들의 분산을 계산하여 일직선 정도 측정
        if len(centers) >= 3:
            # 최소 제곱법으로 직선 피팅
            x_coords = [c[0] for c in centers]
            y_coords = [c[1] for c in centers]
            
            # 선형 회귀로 방향 벡터 계산
            n = len(centers)
            sum_x = sum(x_coords)
            sum_y = sum(y_coords)
            sum_xy = sum(x * y for x, y in zip(x_coords, y_coords))
            sum_x2 = sum(x * x for x in x_coords)
            
            # 기울기 계산 (y = ax + b 형태)
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) > 1e-6:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                # 일직선 정도: 각 점이 직선에서 얼마나 벗어나는지 측정
                intercept = (sum_y - slope * sum_x) / n
                # 평균 거리 오차 계산
                distances = [abs(y - (slope * x + intercept)) for x, y in zip(x_coords, y_coords)]
                avg_distance_error = sum(distances) / len(distances)
                
                # 일직선 움직임이면 (오차가 작으면) 평균 속도 사용
                if avg_distance_error < 10:  # 10픽셀 이내 오차면 일직선으로 간주
                    # 평균 속도 계산 (첫 위치와 마지막 위치)
                    first_center = centers[0]
                    last_center = centers[-1]
                    total_vx = (last_center[0] - first_center[0]) / (num_frames - 1)
                    total_vy = (last_center[1] - first_center[1]) / (num_frames - 1)
                else:
                    # 일직선이 아니면 최근 2개만 사용
                    first_center = centers[-2]
                    last_center = centers[-1]
                    total_vx = last_center[0] - first_center[0]
                    total_vy = last_center[1] - first_center[1]
            else:
                # 수직선에 가까운 경우
                first_center = centers[-2]
                last_center = centers[-1]
                total_vx = last_center[0] - first_center[0]
                total_vy = last_center[1] - first_center[1]
        else:
            # 이력이 부족하면 최근 2개만 사용
            first_center = centers[-2]
            last_center = centers[-1]
            total_vx = last_center[0] - first_center[0]
            total_vy = last_center[1] - first_center[1]
        
        # 현재 위치
        curr_bbox = self.history[-1]
        curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
        curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
        
        # missed_frames를 고려하여 예측 거리 조정
        frames_to_predict = self.missed_frames + 1
        predicted_center_x = curr_center_x + total_vx * frames_to_predict
        predicted_center_y = curr_center_y + total_vy * frames_to_predict
        
        # 박스 크기 유지
        box_width = curr_bbox[2] - curr_bbox[0]
        box_height = curr_bbox[3] - curr_bbox[1]
        
        # 예측 위치
        predicted_bbox = (
            int(predicted_center_x - box_width / 2),
            int(predicted_center_y - box_height / 2),
            int(predicted_center_x + box_width / 2),
            int(predicted_center_y + box_height / 2)
        )
        
        return predicted_bbox


class QRTracker:
    """QR 코드 프레임 간 추적 관리자 (일직선 움직임 최적화)"""
    def __init__(self, max_missed_frames=10, iou_threshold=0.15, center_dist_threshold=1.2, 
                 linear_motion_boost=True):
        """
        Args:
            max_missed_frames: 추적을 유지할 최대 실패 프레임 수 (일직선 움직임이므로 증가)
            iou_threshold: 매칭을 위한 최소 IoU 값 (일직선 움직임이므로 낮춤)
            center_dist_threshold: 중심점 거리 임계값 (정규화된 거리, 일직선 움직임이므로 증가)
            linear_motion_boost: 일직선 움직임 가정 시 예측 위치 기반 매칭 강화 여부
        """
        self.tracks = {}  # {track_id: QRTrack}
        self.next_track_id = 0
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.center_dist_threshold = center_dist_threshold
        self.linear_motion_boost = linear_motion_boost
    
    def update(self, detected_qrs, frame_number):
        """
        탐지된 QR 코드들과 추적 중인 QR 코드들을 매칭하여 업데이트
        
        Args:
            detected_qrs: 탐지된 QR 코드 리스트 [{'text': str, 'detection': dict, ...}, ...]
            frame_number: 현재 프레임 번호
        
        Returns:
            추적된 QR 코드 리스트 (탐지된 것 + 추적만 유지되는 것)
        """
        # 1. 탐지된 QR 코드들의 bbox 추출
        detected_bboxes = []
        for qr in detected_qrs:
            center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr['detection'])
            if center_x is not None:
                detected_bboxes.append({
                    'qr': qr,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
        
        # 2. 활성 추적 목록 (missed_frames가 임계값 이하인 것들)
        active_tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.missed_frames <= self.max_missed_frames
        }
        
        # 3. 탐지된 QR과 추적 중인 QR 매칭 (개선된 알고리즘)
        # ★★★★★ 개선: 예측 위치, 텍스트 매칭, 복합 점수 사용 ★★★★★
        matched_detections = set()
        matched_tracks = set()
        
        # 매칭 점수 계산 (모든 조합)
        match_scores = []  # [(track_id, detection_idx, score, iou, center_dist, text_match), ...]
        
        for track_id, track in active_tracks.items():
            if track.bbox is None:
                continue
            
            # 예측 위치 계산 (missed_frames가 있으면 예측 위치 사용)
            if track.missed_frames > 0:
                predicted_bbox = track.predict_position()
                if predicted_bbox is not None:
                    track_bbox = predicted_bbox
                else:
                    track_bbox = track.bbox
            else:
                track_bbox = track.bbox
            
            track_center = track.center
            track_text = track.qr_data.get('text', '')
            
            for idx, det in enumerate(detected_bboxes):
                # IoU 계산
                iou = calculate_iou(track_bbox, det['bbox'])
                
                # 중심점 거리 계산
                center_dist = calculate_center_distance(track_bbox, det['bbox'])
                
                # 텍스트 매칭 확인
                det_text = det['qr'].get('text', '')
                text_match = (track_text != '' and det_text != '' and track_text == det_text)
                
                # 예측 위치 기반 거리 계산 (일직선 움직임 가정 시 중요)
                predicted_bbox = track.predict_position()
                pred_center_dist = None
                if predicted_bbox is not None and self.linear_motion_boost:
                    pred_center_dist = calculate_center_distance(predicted_bbox, det['bbox'])
                
                # 동적 임계값 (missed_frames가 많을수록 낮춤, 일직선 움직임이므로 더 관대하게)
                dynamic_iou_threshold = self.iou_threshold * (1.0 - track.missed_frames * 0.05)
                dynamic_iou_threshold = max(0.05, dynamic_iou_threshold)  # 최소 0.05 (더 낮춤)
                dynamic_center_dist_threshold = self.center_dist_threshold * (1.0 + track.missed_frames * 0.1)
                dynamic_center_dist_threshold = min(2.0, dynamic_center_dist_threshold)  # 최대 2.0
                
                # 매칭 조건: IoU 또는 중심점 거리 또는 텍스트 매칭 또는 예측 위치 기반 거리
                # 일직선 움직임이므로 예측 위치 기반 매칭도 허용
                matches = False
                if text_match:
                    matches = True  # 텍스트 매칭은 항상 허용
                elif iou >= dynamic_iou_threshold:
                    matches = True
                elif center_dist <= dynamic_center_dist_threshold:
                    matches = True
                elif pred_center_dist is not None and pred_center_dist <= dynamic_center_dist_threshold * 1.2:
                    # 예측 위치 기반 매칭 (일직선 움직임 가정 시)
                    matches = True
                
                if matches:
                    # 복합 점수 계산 (높을수록 좋은 매칭)
                    # 텍스트 매칭이 있으면 매우 높은 점수
                    if text_match:
                        score = 1000.0 + iou * 100  # 텍스트 매칭 우선
                    elif pred_center_dist is not None and self.linear_motion_boost:
                        # 일직선 움직임 가정 시 예측 위치 기반 점수 (가중치 증가)
                        score = iou * 100 + (1.0 - center_dist) * 50 + (1.0 - pred_center_dist) * 100
                    else:
                        # IoU와 중심점 거리를 조합한 점수
                        score = iou * 100 + (1.0 - center_dist) * 50
                    
                    match_scores.append((track_id, idx, score, iou, center_dist, text_match, pred_center_dist))
        
        # 점수 순으로 정렬 (높은 점수 우선)
        match_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 최적 매칭 수행 (greedy 방식이지만 점수 순으로 처리)
        for match_data in match_scores:
            if len(match_data) == 7:
                track_id, detection_idx, score, iou, center_dist, text_match, pred_center_dist = match_data
            else:
                track_id, detection_idx, score, iou, center_dist, text_match = match_data[:6]
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            
            # 매칭 성공: 추적 업데이트
            track = active_tracks[track_id]
            det = detected_bboxes[detection_idx]
            track.update(det['qr'], frame_number)
            matched_detections.add(detection_idx)
            matched_tracks.add(track_id)
        
        # 4. 매칭되지 않은 탐지는 새로운 추적 생성
        for idx, det in enumerate(detected_bboxes):
            if idx not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                new_track = QRTrack(track_id, det['qr'], frame_number)
                self.tracks[track_id] = new_track
        
        # 5. 매칭되지 않은 추적은 missed_frames 증가
        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks:
                track.missed_frames += 1
                track.frame_number = frame_number
        
        # 6. 추적 결과 반환 (탐지된 것 + 추적만 유지되는 것)
        tracked_qrs = []
        
        # 탐지된 QR (매칭된 것) - 개선된 방식
        # 매칭 정보를 저장해두고 사용
        detection_to_track = {}  # {detection_idx: track_id}
        for match_data in match_scores:
            if len(match_data) >= 2:
                track_id = match_data[0]
                detection_idx = match_data[1]
                if track_id in matched_tracks and detection_idx in matched_detections:
                    if detection_idx not in detection_to_track:
                        detection_to_track[detection_idx] = track_id
        
        for idx, det in enumerate(detected_bboxes):
            if idx in matched_detections and idx in detection_to_track:
                track_id = detection_to_track[idx]
                track = active_tracks[track_id]
                tracked_qrs.append({
                    **track.qr_data,
                    'track_id': track_id,
                    'tracked': True
                })
        
        # 추적만 유지되는 QR (탐지되지 않았지만 추적 유지)
        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks and track.missed_frames > 0:
                # 예측 위치로 업데이트
                predicted_bbox = track.predict_position()
                if predicted_bbox is not None:
                    # qr_data 복사 및 예측 위치로 업데이트
                    tracked_qr = track.qr_data.copy()
                    tracked_qr['track_id'] = track_id
                    tracked_qr['tracked'] = True
                    tracked_qr['predicted'] = True
                    tracked_qr['missed_frames'] = track.missed_frames
                    
                    # detection에 예측 위치 추가
                    if 'detection' in tracked_qr:
                        tracked_qr['detection'] = tracked_qr['detection'].copy()
                        tracked_qr['detection']['bbox_xyxy'] = list(predicted_bbox)
                    
                    tracked_qrs.append(tracked_qr)
        
        # 7. 오래된 추적 제거
        tracks_to_remove = [
            tid for tid, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for tid in tracks_to_remove:
            del self.tracks[tid]
        
        return tracked_qrs
    
    def get_active_track_count(self):
        """활성 추적 개수 반환"""
        return len([t for t in self.tracks.values() if t.missed_frames <= self.max_missed_frames])


def video_player_with_qr(video_path, output_dir="video_player_results"):
    """영상 플레이어 + 실시간 QR 탐지"""
    
    # 🕐 전체 실행 시간 측정 시작
    total_start_time = time.time()
    
    # 결과 폴더: 실행마다 고유 하위 폴더 사용 (Windows 파일 잠김 이슈 회피)
    import shutil
    import datetime
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_run_dir, exist_ok=True)
    
    # 로그 파일 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_run_dir, f"qr_detection_log_{timestamp}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log_print(message):
        """콘솔 출력과 파일 저장을 동시에"""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    # 플랫폼 정보 로깅
    system = platform.system()
    system_version = platform.version()
    machine = platform.machine()
    processor = platform.processor()
    python_version = sys.version.split()[0]
    
    log_print(f"🖥️  플랫폼 정보:")
    log_print(f"  OS: {system} {system_version}")
    log_print(f"  아키텍처: {machine}")
    if processor:
        log_print(f"  프로세서: {processor}")
    log_print(f"  Python: {python_version}")
    log_print(f"  OpenCV: {cv2.__version__}")
    
    log_print(f"\n📁 결과 폴더 생성: {output_run_dir}")
    log_print(f"📁 로그 파일: {log_file_path}")
    
    # QR 탐지기 초기화
    detector = cv2.QRCodeDetector()
    
    # YOLO 모델 초기화
    yolo_model = None
    use_yolo_mode = True  # YOLO 모드 사용 여부
    
    if YOLO_AVAILABLE and use_yolo_mode:
        try:
            model_path = 'model1.pt'
            if os.path.exists(model_path):
                yolo_model = YOLO(model_path)
                log_print("✅ YOLO 모델 초기화 완료 (빠른 탐지 모드)")
                log_print(f"   모델: {model_path}")
            else:
                log_print(f"⚠️ YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
                log_print("   기존 방식으로 전환합니다.")
                use_yolo_mode = False
        except Exception as e:
            log_print(f"❌ YOLO 모델 초기화 실패: {e}")
            log_print("   기존 방식으로 전환합니다.")
            use_yolo_mode = False
    else:
        use_yolo_mode = False
    
    # Dynamsoft Barcode Reader 초기화 (기본 해독기)
    dbr_reader = None
    log_print(f"🔍 Dynamsoft 초기화 시작: DBR_AVAILABLE={DBR_AVAILABLE}, DBR_VERSION={DBR_VERSION}")
    if DBR_AVAILABLE:
        try:
            # 라이선스 키 확인 (환경 변수 또는 직접 입력)
            license_key = os.environ.get('DYNAMSOFT_LICENSE_KEY', '')
            if not license_key:
                # 환경 변수에 없으면 직접 입력 (사용자가 설정)
                license_key = 't0085YQEAADYdcL2llMa8vH1Rtnun+43saE/kdAE7ZbIxMQGRMtSzVSZRI8vfOK4Ids52rjekwzh87yABFLraXw5Va1BV7NnBjI8m7qbw3kxOprI75ExJpw=='
            
            if license_key:
                log_print(f"🔑 Dynamsoft 라이선스 키 확인: {license_key[:30]}... (길이: {len(license_key)})")
                
                if DBR_VERSION == "bundle_v11":
                    # Bundle v11 API 사용
                    error = license.LicenseManager.init_license(license_key)
                    if error[0] != 0:
                        log_print(f"⚠️ Dynamsoft 라이선스 초기화 실패: {error[1]}")
                    else:
                        # CaptureVisionRouter 생성 및 설정
                        dbr_reader = cvr.CaptureVisionRouter()
                        from dynamsoft_barcode_reader_bundle import EnumPresetTemplate
                        error_code, error_msg, settings = dbr_reader.get_simplified_settings(EnumPresetTemplate.PT_DEFAULT)
                        if error_code == 0 and settings:
                            # QR 코드만 탐지하도록 설정
                            barcode_settings = settings.barcode_settings
                            if barcode_settings:
                                barcode_settings.barcode_format_ids = dbr.EnumBarcodeFormat.BF_QR_CODE
                                # 성능 최적화 설정 (해독률 향상)
                                # 최대 바코드 개수 증가
                                if hasattr(barcode_settings, 'expected_barcodes_count'):
                                    barcode_settings.expected_barcodes_count = 10
                                # 해독 시도 횟수 증가
                                if hasattr(barcode_settings, 'deblur_level'):
                                    barcode_settings.deblur_level = 9  # 최대 디블러 레벨
                            dbr_reader.update_settings(EnumPresetTemplate.PT_DEFAULT, settings)
                        log_print("✅ Dynamsoft Barcode Reader 초기화 완료 (기본 해독기, Bundle v11)")
                else:
                    # 이전 버전 API (현재는 사용 안 함)
                    log_print("⚠️ 이전 버전 Dynamsoft API는 지원하지 않습니다. bundle_v11을 사용하세요.")
            else:
                log_print("⚠️ Dynamsoft 라이선스 키가 설정되지 않았습니다.")
        except Exception as e:
            log_print(f"❌ Dynamsoft 초기화 실패: {e}")
            import traceback
            log_print(traceback.format_exc())
            dbr_reader = None
    
    log_print(f"📊 사용 가능한 탐지기:")
    log_print(f"  - YOLO 모델: {'✅ (빠른 탐지 모드)' if yolo_model else '❌'}")
    log_print(f"  - Dynamsoft: {'✅ (기본 해독기)' if dbr_reader else '❌'}")
    log_print(f"  - PIL (한글폰트): {'✅' if PIL_AVAILABLE else '❌'}")
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log_print(f"\n📹 비디오 정보:")
    log_print(f"  파일: {video_path}")
    log_print(f"  해상도: {width}x{height}")
    log_print(f"  FPS: {fps:.2f}")
    log_print(f"  총 프레임: {total_frames}")
    log_print(f"  길이: {total_frames/fps:.2f}초")
    
    # 해상도 조정 (화면에 맞게)
    display_width = 1280
    display_height = 720
    
    if width > display_width:
        scale = display_width / width
        display_width = int(width * scale)
        display_height = int(height * scale)
    
    # ★★★★★ 영상 저장을 위한 VideoWriter 초기화 (플랫폼별 코덱 자동 선택) ★★★★★
    system = platform.system()
    
    # 플랫폼별 최적 코덱 우선순위
    if system == "Windows":
        codec_options = [
            ('mp4v', 'mp4'),  # Windows에서 가장 호환성 좋음
            ('XVID', 'avi'),
            ('MJPG', 'avi'),
        ]
    elif system == "Darwin":  # macOS
        codec_options = [
            ('avc1', 'mp4'),  # H.264 (macOS에서 가장 호환성 좋음)
            ('mp4v', 'mp4'),
            ('MJPG', 'mov'),
        ]
    else:  # Linux
        codec_options = [
            ('mp4v', 'mp4'),
            ('XVID', 'avi'),
            ('MJPG', 'avi'),
        ]
    
    output_video_path = None
    out_video = None
    
    for codec, ext in codec_options:
        try:
            output_video_path = os.path.join(output_run_dir, f"output_{run_id}.{ext}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (display_width, display_height))
            if out_video.isOpened():
                log_print(f"📹 출력 영상 파일: {output_video_path} (코덱: {codec}, 해상도: {display_width}x{display_height}, FPS: {fps:.2f})")
                break
            else:
                out_video.release()
                out_video = None
        except Exception as e:
            if out_video:
                out_video.release()
            out_video = None
            continue
    
    if out_video is None or not out_video.isOpened():
        log_print(f"❌ 모든 코덱 시도 실패. 영상 저장이 불가능할 수 있습니다.")
        log_print(f"   시도한 코덱: {[c[0] for c in codec_options]}")
        # 기본값으로 mp4v 시도
        output_video_path = os.path.join(output_run_dir, f"output_{run_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (display_width, display_height))
    
    log_print(f"  화면 해상도: {display_width}x{display_height}")
    log_print(f"\n🎬 영상 재생 시작!")
    log_print(f"  - ESC 키: 종료")
    log_print(f"  - SPACE 키: 일시정지/재생")
    log_print(f"  - S 키: 현재 프레임 저장")
    
    # 재생 제어 변수
    paused = False
    frame_count = 0
    detected_count = 0
    start_time = time.time()
    
    # FPS 계산용
    fps_counter = 0
    fps_start_time = time.time()
    
    # ★★★★★ 프레임 간 추적 기능 초기화 ★★★★★
    # 개선된 매칭: IoU 임계값 낮춤 (0.2), 중심점 거리 임계값 추가
    # 일직선 움직임 최적화: max_missed_frames 증가, 임계값 조정
    qr_tracker = QRTracker(max_missed_frames=10, iou_threshold=0.15, center_dist_threshold=1.2, 
                           linear_motion_boost=True)
    use_tracking = True  # 추적 기능 사용 여부
    base_detection_interval = 1  # 기본 탐지 간격 (모든 프레임 탐지)
    max_detection_interval = 1  # 최대 탐지 간격 (모든 프레임 탐지)
    
    # 탐지 간격 설정 (모든 프레임 탐지)
    detection_interval = base_detection_interval
    last_detection_frame = 0
    
    # ★★★★★ 비동기 해독 워커 스레드 초기화 ★★★★★
    decode_queue = None
    decode_results = {}  # {track_id: {'text': str, 'quad_xy': list, 'decode_bbox': list, 'decode_method': str}}
    decode_worker_thread = None
    stop_decode_worker = None
    decode_lock = threading.Lock()
    
    # Dynamsoft가 있으면 해독 워커 시작
    if dbr_reader is not None:
        decode_queue = Queue(maxsize=10)
        stop_decode_worker = threading.Event()
        
        def decode_worker():
            """백그라운드에서 해독 수행하는 워커 스레드 (Dynamsoft만 사용)"""
            log_count = 0
            previous_decode_results = {}  # 이전 프레임 해독 결과 저장 (깜빡임 방지)
            while not stop_decode_worker.is_set():
                try:
                    item = decode_queue.get(timeout=0.1)
                    if item is None:
                        return
                    
                    if len(item) == 5:
                        track_id, roi, bbox, roi_offset, frame_num = item  # roi_offset: (roi_x1, roi_y1), frame_num: 프레임 번호
                    else:
                        track_id, roi, bbox, roi_offset = item  # roi_offset: (roi_x1, roi_y1)
                        frame_num = None  # 프레임 번호가 없으면 None
                    decoded_text = None
                    quad_xy = None
                    used_dbr = False  # Dynamsoft 사용 여부 추적
                    decode_method_detail = None  # 성공한 방법 상세 정보
                    
                    try:
                        # Dynamsoft로 해독 시도
                        if dbr_reader is not None:
                            try:
                                # 전처리 적용 (해독률 향상)
                                # 1. 그레이스케일 변환
                                if len(roi.shape) == 3:
                                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                else:
                                    roi_gray = roi.copy()
                                
                                # 2. CLAHE 적용 (어두운 환경 대비 개선)
                                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                                roi_enhanced = clahe.apply(roi_gray)
                                
                                # 3. 정규화 (대비 끌어올림)
                                roi_norm = cv2.normalize(roi_enhanced, None, 0, 255, cv2.NORM_MINMAX)
                                
                                # 4. 흰색 테두리 추가 (일반 QR 코드용: 검은색 QR, 밝은 배경)
                                rh, rw = roi_norm.shape
                                border_size = 20
                                white_canvas = np.full((rh + border_size*2, rw + border_size*2), 255, dtype=np.uint8)
                                white_canvas[border_size:border_size+rh, border_size:border_size+rw] = roi_norm
                                
                                # 5. RGB로 변환 (Dynamsoft는 RGB를 사용)
                                roi_rgb = cv2.cvtColor(white_canvas, cv2.COLOR_GRAY2RGB)
                                
                                # Bundle v11 API 사용
                                if DBR_VERSION == "bundle_v11":
                                    from dynamsoft_barcode_reader_bundle import dbr as dbr_module
                                    
                                    items = None
                                    
                                    # 방법 1: 원본 이미지로 시도 (일반 QR 코드용)
                                    captured_result = dbr_reader.capture(roi_rgb, dbr_module.EnumImagePixelFormat.IPF_RGB_888)
                                    barcode_result = captured_result.get_decoded_barcodes_result()
                                    if barcode_result:
                                        items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
                                        if items and len(items) > 0:
                                            decode_method_detail = "원본(흰테두리)"
                                    
                                    # 방법 2: 원본 실패 시, 반전 이미지로 시도 (하얀색 QR 코드용)
                                    if not items or len(items) == 0:
                                        # 반전 이미지 전처리 (하얀색 QR 코드용)
                                        roi_inverted_gray = cv2.bitwise_not(roi_norm)
                                        # 검은색 테두리 추가 (하얀색 QR 코드는 어두운 배경에 밝은 QR)
                                        black_canvas = np.full((rh + border_size*2, rw + border_size*2), 0, dtype=np.uint8)
                                        black_canvas[border_size:border_size+rh, border_size:border_size+rw] = roi_inverted_gray
                                        roi_rgb_inverted = cv2.cvtColor(black_canvas, cv2.COLOR_GRAY2RGB)
                                        
                                        captured_result_inverted = dbr_reader.capture(roi_rgb_inverted, dbr_module.EnumImagePixelFormat.IPF_RGB_888)
                                        barcode_result_inverted = captured_result_inverted.get_decoded_barcodes_result()
                                        if barcode_result_inverted:
                                            items = barcode_result_inverted.get_items() if hasattr(barcode_result_inverted, 'get_items') else None
                                            if items and len(items) > 0:
                                                decode_method_detail = "반전(정규화후,검은테두리)"
                                    
                                    if items and len(items) > 0:
                                        # 첫 번째 바코드 사용
                                        barcode_item = items[0]
                                        
                                        # text 속성 확인
                                        text = None
                                        if hasattr(barcode_item, 'get_text'):
                                            text = barcode_item.get_text()
                                        elif hasattr(barcode_item, 'text'):
                                            text = barcode_item.text
                                        elif hasattr(barcode_item, 'barcode_text'):
                                            text = barcode_item.barcode_text
                                        
                                        if text:
                                            decoded_text = text
                                            used_dbr = True  # Dynamsoft 성공
                                            
                                            # 성공한 방법 로그 출력 (처음 몇 번만)
                                            if log_count < 10 and decode_method_detail:
                                                log_print(f"✅ Dynamsoft 해독 성공 [T{track_id}] - 방법: {decode_method_detail}")
                                                log_count += 1
                                            
                                            # quad_xy 추출
                                            try:
                                                location = None
                                                if hasattr(barcode_item, 'get_location'):
                                                    location = barcode_item.get_location()
                                                elif hasattr(barcode_item, 'location'):
                                                    location = barcode_item.location
                                                
                                                if location:
                                                    result_points = None
                                                    if hasattr(location, 'result_points'):
                                                        result_points = location.result_points
                                                    elif hasattr(location, 'points'):
                                                        result_points = location.points
                                                    elif hasattr(location, 'get_result_points'):
                                                        result_points = location.get_result_points()
                                                    
                                                    if result_points:
                                                        roi_x1, roi_y1 = roi_offset
                                                        quad_xy = []
                                                        for point in result_points:
                                                            abs_x = roi_x1 + int(point.x)
                                                            abs_y = roi_y1 + int(point.y)
                                                            quad_xy.append([abs_x, abs_y])
                                            except Exception as loc_e:
                                                # 위치 추출 실패는 무시
                                                pass
                                            
                                            # 텍스트 처리
                                            decoded_text = _process_decoded_text(decoded_text)
                            except Exception as e:
                                # Dynamsoft 실패
                                if log_count < 5:
                                    log_print(f"⚠️ Dynamsoft 해독 실패 [T{track_id}]: {str(e)[:100]}")
                                    log_count += 1
                                decoded_text = None
                        
                        # Dynamsoft 실패 시 로그 출력 (디버깅용)
                        if decoded_text is None and dbr_reader is not None:
                            if log_count < 5:
                                log_print(f"⚠️ Dynamsoft 해독 실패 [T{track_id}]: 바코드를 찾지 못했습니다.")
                                log_count += 1
                        
                        # 결과 저장
                        if decoded_text:
                            # quad_xy가 없으면 bbox 기반으로 생성
                            if quad_xy is None:
                                x1, y1, x2, y2 = bbox
                                quad_xy = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            
                            # 해독 방법 결정
                            decode_method = "Dynamsoft"
                            
                            with decode_lock:
                                decode_results[track_id] = {
                                    'text': decoded_text,
                                    'quad_xy': quad_xy,
                                    'decode_bbox': list(bbox),
                                    'decode_method': decode_method,  # 해독 방법 저장
                                    'decode_method_detail': decode_method_detail,  # 성공한 방법 상세 정보
                                    'frame': frame_num if frame_num is not None else 0  # 프레임 번호 저장 (깜빡임 방지용)
                                }
                                # 이전 프레임 결과에도 저장 (깜빡임 방지)
                                previous_decode_results[track_id] = decode_results[track_id].copy()
                            if log_count < 10:
                                log_print(f"✅ 해독 성공 [T{track_id}] ({decode_method}): {decoded_text[:50]}")
                                log_count += 1
                    except Exception as e:
                        if log_count < 3 and track_id <= 3:
                            log_print(f"⚠️ 해독 실패 [T{track_id}]: {str(e)[:50]}")
                            log_count += 1
                        pass
                    
                    decode_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    log_print(f"해독 워커 오류: {e}")
                    if 'item' in locals() and item:
                        decode_queue.task_done()
        
        decode_worker_thread = threading.Thread(target=decode_worker, daemon=True)
        decode_worker_thread.start()
        if dbr_reader is not None:
            log_print("✅ 비동기 해독 워커 스레드 시작 (Dynamsoft만 사용)")
        else:
            log_print("⚠️ 비동기 해독 워커 스레드 시작 (해독기 없음 - Dynamsoft 초기화 실패)")
    
    # 통계 변수
    success_count = 0
    failed_count = 0
    tracking_stats = {
        'total_tracks': 0,
        'active_tracks': 0,
        'predicted_frames': 0
    }
    
    # 방법별 성공률 추적 (YOLO 모드에 따라 최적화)
    if use_yolo_mode and yolo_model is not None:
        # YOLO 모드: YOLO + Dynamsoft 통계만 초기화
        method_stats = {
            "YOLO": 0,
            "YOLO+Dynamsoft": 0
        }
        method_detection_count = {
            "YOLO": 0,
            "YOLO+Dynamsoft": 0
        }
        method_unique_detection_count = {
            "YOLO": 0,
            "YOLO+Dynamsoft": 0
        }
        # Dynamsoft 방법별 상세 통계
        dynamsoft_method_stats = {
            "원본(흰테두리)": 0,
            "반전(정규화후,검은테두리)": 0
        }
    else:
        # 비-YOLO 모드는 지원하지 않음
        method_stats = {"YOLO+Dynamsoft": 0}
        method_detection_count = {"YOLO": 0}
        method_unique_detection_count = {"YOLO": 0}
    
    # 모든 방법에서 찾은 QR 코드들을 저장 (중복 제거용)
    all_detected_qrs = []
    
    # 현재 프레임용 변수
    current_success = 0
    current_failed = 0
    
    ret = True  # 초기화
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n📺 영상 재생 완료!")
                    # 영상이 끝나면 즉시 루프 종료
                    break
                
                frame_count += 1
            
            # 해상도 조정 (화면 표시용)
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # QR 코드 탐지 (성능 최적화)
            detected = False
            detected_text = ""
            detection_method = ""
            points = None
            
            # 다중 QR 시각화를 위한 리스트
            all_qr_visualizations = []  # [{"points": [...], "text": "...", "method": "...", "success": bool}, ...]
            unique_qrs = []  # 초기화
            
            # ★★★★★ 모든 프레임에서 탐지 (원본 속도 최적화) ★★★★★
            # 탐지 간격 체크 (모든 프레임 탐지)
            should_detect = (frame_count - last_detection_frame) >= detection_interval
            
            # ★★★★★ 추적 모드: 탐지하지 않는 프레임에서도 추적 결과 사용 ★★★★★
            if use_tracking and not should_detect:
                # 추적만 사용 (탐지 없이)
                tracked_qrs = []
                for track_id, track in qr_tracker.tracks.items():
                    if track.missed_frames <= qr_tracker.max_missed_frames:
                        # 예측 위치로 업데이트
                        predicted_bbox = track.predict_position()
                        if predicted_bbox is not None:
                            tracked_qr = track.qr_data.copy()
                            tracked_qr['track_id'] = track_id
                            tracked_qr['tracked'] = True
                            tracked_qr['predicted'] = True
                            tracked_qr['missed_frames'] = track.missed_frames
                            
                            # detection에 예측 위치 추가
                            if 'detection' in tracked_qr:
                                tracked_qr['detection'] = tracked_qr['detection'].copy()
                                tracked_qr['detection']['bbox_xyxy'] = list(predicted_bbox)
                            
                            tracked_qrs.append(tracked_qr)
                
                # 추적 결과를 unique_qrs로 설정
                if tracked_qrs:
                    # 해독 결과 확인 및 업데이트
                    for qr in tracked_qrs:
                        track_id = qr.get('track_id')
                        if track_id is not None and decode_results is not None:
                            with decode_lock:
                                if track_id in decode_results:
                                    decode_result = decode_results[track_id]
                                    if decode_result.get('text'):
                                        qr['text'] = decode_result['text']
                                        qr['success'] = True
                                        # 실제 사용된 방법으로 업데이트 (YOLO + 해독 방법)
                                        decode_method = decode_result.get('decode_method', 'Unknown')
                                        qr['method'] = f"YOLO+{decode_method}"
                                    if 'detection' in qr and decode_result.get('quad_xy'):
                                        # quad_xy를 현재 추적 위치에 맞춰서 변환
                                        current_bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                                        decode_bbox = decode_result.get('decode_bbox')
                                        
                                        if current_bbox is not None and len(current_bbox) == 4 and \
                                           decode_bbox is not None and len(decode_bbox) == 4:
                                            # 해독 시점의 bbox와 현재 추적 bbox의 차이 계산
                                            decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                            curr_x1, curr_y1, curr_x2, curr_y2 = map(int, current_bbox)
                                            
                                            # 중심점 이동량 계산
                                            decode_cx = (decode_x1 + decode_x2) / 2
                                            decode_cy = (decode_y1 + decode_y2) / 2
                                            curr_cx = (curr_x1 + curr_x2) / 2
                                            curr_cy = (curr_y1 + curr_y2) / 2
                                            
                                            dx = curr_cx - decode_cx
                                            dy = curr_cy - decode_cy
                                            
                                            # quad_xy를 현재 추적 위치에 맞춰서 이동
                                            quad_xy_original = decode_result['quad_xy']
                                            quad_xy_transformed = []
                                            for qx, qy in quad_xy_original:
                                                quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                            qr['detection']['quad_xy'] = quad_xy_transformed
                                        else:
                                            # bbox 정보가 없으면 원본 quad_xy 사용
                                            qr['detection']['quad_xy'] = decode_result['quad_xy']
                    
                    unique_qrs = tracked_qrs
                    detected = True
                    detected_text = tracked_qrs[0].get('text', '')
                    detection_method = tracked_qrs[0].get('method', '')
                    
                    # 추적 결과 시각화 준비
                    all_qr_visualizations = []
                    for qr in tracked_qrs:
                        detection = qr.get('detection')
                        if detection is None:
                            continue
                        
                        # quad_xy 우선 사용, 없으면 bbox_xyxy 사용
                        qr_points = None
                        if 'quad_xy' in detection and detection['quad_xy'] is not None:
                            quad = detection['quad_xy']
                            if len(quad) == 4:
                                quad_array = np.array(quad)
                                center = np.mean(quad_array, axis=0)
                                angles = np.arctan2(quad_array[:, 1] - center[1], quad_array[:, 0] - center[0])
                                sorted_indices = np.argsort(angles)
                                sorted_quad = quad_array[sorted_indices]
                                qr_points = np.array([sorted_quad], dtype=np.float32)
                        
                        if qr_points is None and 'bbox_xyxy' in detection:
                            bbox = detection['bbox_xyxy']
                            x1, y1, x2, y2 = bbox
                            qr_points = np.array([[
                                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                            ]], dtype=np.float32)
                        
                        if qr_points is not None:
                            all_qr_visualizations.append({
                                "points": qr_points,
                                "text": qr.get('text', ''),
                                "method": qr.get('method', ''),
                                "success": qr.get('success', False),
                                "track_id": qr.get('track_id', None)  # QR 번호용
                            })
            
            # 프레임 스킵/처리 콘솔 출력 제거
            
            if should_detect:
                # 현재 프레임용 변수 초기화
                current_success = 0
                current_failed = 0
                
                try:
                    # 🚀 병렬 처리로 모든 QR 탐지 방법 동시 실행
                    start_time = time.time()
                    
                    # 원본 프레임만 사용
                    single_frame, scales = create_single_frame(frame)
                    
                    # YOLO 모드 사용 여부에 따라 처리 방식 선택
                    if use_yolo_mode and yolo_model is not None:
                        # 🚀 YOLO 기반 빠른 탐지만 수행 (해독은 비동기로 분리)
                        filtered_locations = process_frame_with_yolo(single_frame, yolo_model, conf_threshold=0.25)
                        
                        # 탐지 결과를 추적 형식으로 변환
                        detected_qrs = []
                        for i, location in enumerate(filtered_locations):
                            x1, y1, x2, y2 = location['bbox']
                            qr_data = {
                                'bbox': location['bbox'],
                                'confidence': location['confidence'],
                                'text': '',  # 아직 해독 안됨
                                'detection': {
                                    'bbox_xyxy': location['bbox'],
                                    'quad_xy': None  # 해독 후 업데이트
                                },
                                'method': 'YOLO',
                                'success': False
                            }
                            detected_qrs.append(qr_data)
                        
                        unique_qrs = detected_qrs
                    else:
                        # 비-YOLO 모드는 지원하지 않음 (Dynamsoft만 사용)
                        log_print("⚠️ 비-YOLO 모드는 지원하지 않습니다. YOLO 모드만 사용 가능합니다.")
                        unique_qrs = []
                    
                    # ★★★★★ 추적 기능 적용 ★★★★★
                    if use_tracking:
                        # 추적 업데이트
                        tracked_qrs = qr_tracker.update(unique_qrs, frame_count)
                        unique_qrs = tracked_qrs
                        
                        # 추적 통계 업데이트 (시각화 없이 내부적으로만 사용)
                        active_count = qr_tracker.get_active_track_count()
                        tracking_stats['active_tracks'] = max(tracking_stats['active_tracks'], active_count)
                        tracking_stats['total_tracks'] = max(tracking_stats['total_tracks'], qr_tracker.next_track_id)
                        
                        # 예측된 프레임 수 카운트
                        predicted_count = sum(1 for qr in tracked_qrs if qr.get('predicted', False))
                        if predicted_count > 0:
                            tracking_stats['predicted_frames'] += predicted_count
                        
                        # ★★★★★ 비동기 해독 큐에 추가 ★★★★★
                        if decode_queue is not None and dbr_reader is not None:
                            for tracked_qr in tracked_qrs:
                                track_id = tracked_qr.get('track_id')
                                if track_id is not None:
                                    # 이미 해독된 것은 스킵 (하지만 quad_xy는 업데이트)
                                    with decode_lock:
                                        if track_id in decode_results:
                                            # 해독 결과 업데이트
                                            decode_result = decode_results[track_id]
                                            tracked_qr['text'] = decode_result['text']
                                            tracked_qr['success'] = True
                                            # 실제 사용된 방법으로 업데이트 (YOLO + 해독 방법)
                                            decode_method = decode_result.get('decode_method', 'Unknown')
                                            tracked_qr['method'] = f"YOLO+{decode_method}"
                                            if 'detection' in tracked_qr and decode_result.get('quad_xy'):
                                                # quad_xy를 현재 추적 위치에 맞춰서 변환
                                                current_bbox = tracked_qr.get('bbox', tracked_qr.get('detection', {}).get('bbox_xyxy'))
                                                decode_bbox = decode_result.get('decode_bbox')
                                                
                                                if current_bbox is not None and len(current_bbox) == 4 and \
                                                   decode_bbox is not None and len(decode_bbox) == 4:
                                                    # 해독 시점의 bbox와 현재 추적 bbox의 차이 계산
                                                    decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                                    curr_x1, curr_y1, curr_x2, curr_y2 = map(int, current_bbox)
                                                    
                                                    # 중심점 이동량 계산
                                                    decode_cx = (decode_x1 + decode_x2) / 2
                                                    decode_cy = (decode_y1 + decode_y2) / 2
                                                    curr_cx = (curr_x1 + curr_x2) / 2
                                                    curr_cy = (curr_y1 + curr_y2) / 2
                                                    
                                                    dx = curr_cx - decode_cx
                                                    dy = curr_cy - decode_cy
                                                    
                                                    # quad_xy를 현재 추적 위치에 맞춰서 이동
                                                    quad_xy_original = decode_result['quad_xy']
                                                    quad_xy_transformed = []
                                                    for qx, qy in quad_xy_original:
                                                        quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                                    tracked_qr['detection']['quad_xy'] = quad_xy_transformed
                                                else:
                                                    # bbox 정보가 없으면 원본 quad_xy 사용
                                                    tracked_qr['detection']['quad_xy'] = decode_result['quad_xy']
                                            continue
                                    
                                    # ROI 추출하여 해독 큐에 추가
                                    bbox = tracked_qr.get('bbox', tracked_qr.get('detection', {}).get('bbox_xyxy'))
                                    if bbox is not None and len(bbox) == 4:
                                        x1, y1, x2, y2 = map(int, bbox)
                                        roi = frame[y1:y2, x1:x2]
                                        if roi.size > 0:
                                            try:
                                                decode_queue.put_nowait((track_id, roi, bbox, (x1, y1)))
                                            except:
                                                # 큐가 가득 차면 스킵
                                                pass
                
                    # 전체 탐지 개수 업데이트 (테스트용 상세 통계)
                    # YOLO 모드에서는 results가 없으므로 스킵 (비-YOLO 모드는 지원하지 않음)
                    
                    parallel_time = time.time() - start_time
                    
                    # ★★★★★ 프레임별 상세 로그 출력 ★★★★★
                    log_print(f"\n📹 프레임 {frame_count}/{total_frames} (처리 시간: {parallel_time*1000:.1f}ms)")
                    
                    # 결과 처리 및 통계 업데이트
                    if unique_qrs:
                        log_print(f"  🔍 탐지: {len(unique_qrs)}개 QR 코드 발견")
                        
                        for idx, qr in enumerate(unique_qrs):
                            # 메타데이터는 건너뛰기
                            if isinstance(qr, dict) and 'meta' in qr:
                                continue
                            
                            track_id = qr.get('track_id', 'N/A')
                            bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy', []))
                            conf = qr.get('confidence', 0.0)
                            text = qr.get('text', '')
                            method = qr.get('method', 'Unknown')
                            success = qr.get('success', False)
                            
                            # 프레임별 상세 로그
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                bbox_str = f"bbox=({x1}, {y1}, {x2}, {y2})"
                            else:
                                bbox_str = "bbox=N/A"
                            
                            status = "✅ 성공" if success else "❌ 실패"
                            log_print(f"    QR[{idx}] T{track_id}: {status} | {bbox_str} | conf={conf:.3f} | method={method}")
                            if text:
                                text_short = text[:50] + "..." if len(text) > 50 else text
                                log_print(f"      텍스트: {text_short}")
                            
                            # 성공/실패 통계
                            if success:
                                current_success += 1
                            else:
                                current_failed += 1
                            
                            # 방법명으로 통계 업데이트 (단순화: QReader 제거로 복잡한 로직 불필요)
                            method_name = qr['method']
                            original_method = method_name  # 이미 정리된 형태로 사용
                            
                            # method_stats에 존재하는 키인지 확인
                            if original_method in method_stats:
                                method_stats[original_method] += 1
                            
                            # 테스트용 상세 통계 업데이트
                            if original_method in method_detection_count:
                                method_detection_count[original_method] += 1
                            if original_method in method_unique_detection_count:
                                method_unique_detection_count[original_method] += 1
                            
                            # 시각화 데이터 추가 - ★★★ `quad_xy`가 다시 정밀해짐
                            qr_points = None
                            detection = qr.get('detection')
                            
                            # detection이 없으면 건너뛰기
                            if detection is None:
                                continue
                            
                            # 해독 결과 확인 및 quad_xy 업데이트 (탐지 프레임에서도)
                            track_id = qr.get('track_id')
                            if track_id is not None and decode_results is not None:
                                with decode_lock:
                                    if track_id in decode_results:
                                        decode_result = decode_results[track_id]
                                        # 해독 결과로 텍스트와 방법 업데이트
                                        if decode_result.get('text'):
                                            qr['text'] = decode_result['text']
                                            qr['success'] = True
                                            # 실제 사용된 방법으로 업데이트 (YOLO + 해독 방법)
                                            decode_method = decode_result.get('decode_method', 'Unknown')
                                            qr['method'] = f"YOLO+{decode_method}"
                                            
                                            # Dynamsoft 방법별 상세 통계 업데이트 (한 번만 카운트)
                                            if decode_method == 'Dynamsoft' and track_id in decode_results:
                                                method_detail = decode_result.get('decode_method_detail')
                                                if method_detail and method_detail in dynamsoft_method_stats:
                                                    dynamsoft_method_stats[method_detail] += 1
                                        
                                        if 'detection' in qr and decode_result.get('quad_xy'):
                                            # quad_xy를 현재 추적 위치에 맞춰서 변환
                                            current_bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                                            decode_bbox = decode_result.get('decode_bbox')
                                            
                                            if current_bbox is not None and len(current_bbox) == 4 and \
                                               decode_bbox is not None and len(decode_bbox) == 4:
                                                # 해독 시점의 bbox와 현재 추적 bbox의 차이 계산
                                                decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                                curr_x1, curr_y1, curr_x2, curr_y2 = map(int, current_bbox)
                                                
                                                # 중심점 이동량 계산
                                                decode_cx = (decode_x1 + decode_x2) / 2
                                                decode_cy = (decode_y1 + decode_y2) / 2
                                                curr_cx = (curr_x1 + curr_x2) / 2
                                                curr_cy = (curr_y1 + curr_y2) / 2
                                                
                                                dx = curr_cx - decode_cx
                                                dy = curr_cy - decode_cy
                                                
                                                # quad_xy를 현재 추적 위치에 맞춰서 이동
                                                quad_xy_original = decode_result['quad_xy']
                                                quad_xy_transformed = []
                                                for qx, qy in quad_xy_original:
                                                    quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                                qr['detection']['quad_xy'] = quad_xy_transformed
                                            else:
                                                # bbox 정보가 없으면 원본 quad_xy 사용
                                                qr['detection']['quad_xy'] = decode_result['quad_xy']
                            
                            # Dynamsoft 결과 처리 - quad_xy로 정확한 기울어진 형태 사용
                            if 'quad_xy' in detection and detection['quad_xy'] is not None:
                                # quad_xy 사용 (기울어진 사각형의 4개 꼭짓점)
                                quad = detection['quad_xy']
                                if len(quad) == 4:
                                    # 4개 점을 사각형 순서로 정렬 (왼쪽위→오른쪽위→오른쪽아래→왼쪽아래)
                                    quad_array = np.array(quad)
                                    # 중심점 계산
                                    center = np.mean(quad_array, axis=0)
                                    # 각 점의 각도 계산 (중심점 기준)
                                    angles = np.arctan2(quad_array[:, 1] - center[1], quad_array[:, 0] - center[0])
                                    # 각도 순으로 정렬
                                    sorted_indices = np.argsort(angles)
                                    sorted_quad = quad_array[sorted_indices]
                                    qr_points = np.array([sorted_quad], dtype=np.float32)
                            
                            elif 'bbox_xyxy' in detection:
                                # 축 정렬 바운딩 박스 (fallback)
                                bbox = detection['bbox_xyxy']
                                x1, y1, x2, y2 = bbox
                                qr_points = np.array([[
                                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                                ]], dtype=np.float32)
                            
                            elif 'cxcy' in detection and 'wh' in detection:
                                # 중심점+크기 (fallback)
                                cx, cy = detection['cxcy']
                                w, h = detection['wh']
                                x1, y1 = cx - w/2, cy - h/2
                                x2, y2 = cx + w/2, cy + h/2
                                qr_points = np.array([[
                                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                                ]], dtype=np.float32)
                            
                            else:
                                # extract_bounding_box 함수 사용 (최종 fallback)
                                bbox, method_info = extract_bounding_box(detection, frame.shape[1], frame.shape[0])
                                if bbox is not None:
                                    x1, y1, x2, y2 = bbox
                                    qr_points = np.array([[
                                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                                    ]], dtype=np.float32)
                            
                            if qr_points is not None:
                                all_qr_visualizations.append({
                                    "points": qr_points,
                                    "text": qr['text'],
                                    "method": qr['method'],
                                    "success": qr['success'],  # qr의 실제 success 값 사용
                                    "track_id": qr.get('track_id', None)  # QR 번호용
                                })
                    
                    # 첫 번째 성공한 결과를 메인 시각화용으로 설정
                    if unique_qrs:
                        detected = True
                        detected_text = unique_qrs[0]['text']
                        detection_method = unique_qrs[0]['method']
                    
                    
                   
                    
                    # 처리한 프레임은 항상 업데이트 (QR 발견 여부와 관계없이)
                    last_detection_frame = frame_count
                    
                    if detected:
                        detected_count += 1
                        
                        # 통계는 최종 결과에서만 출력
                        
                        # 다중 QR 코드 영역을 화면에 표시
                        if all_qr_visualizations:
                            # 원본 좌표를 화면 좌표로 변환하는 스케일
                            scale_x = display_width / width
                            scale_y = display_height / height
                            
                            try:
                                for j, qr_viz in enumerate(all_qr_visualizations):
                                    try:
                                        points = qr_viz["points"]
                                        qr_text = qr_viz["text"]
                                        qr_method = qr_viz["method"]
                                        qr_success = qr_viz["success"]
                                        
                                        # points 형태 확인 및 변환
                                        if len(points.shape) == 3 and points.shape[1] == 4:
                                            # (1, 4, 2) 형태인 경우
                                            points_2d = points[0]  # (4, 2)로 변환
                                        elif len(points.shape) == 2 and points.shape[0] == 4:
                                            # (4, 2) 형태인 경우
                                            points_2d = points
                                        else:
                                            points_2d = points.reshape(-1, 2) if points.size > 0 else None
                                        
                                        if points_2d is not None and len(points_2d) >= 4:
                                            # 원본 좌표를 화면 좌표로 변환
                                            display_points = points_2d.copy()
                                            display_points[:, 0] *= scale_x
                                            display_points[:, 1] *= scale_y
                                            display_points = display_points.astype(np.int32)
                                            
                                            # 해독 실패 시 빨간 박스, 성공 시 스케일별 색상
                                            if not qr_success or "실패" in qr_text or "실패" in qr_method:
                                                box_color = (0, 0, 255)  # 빨간색 (BGR)
                                                text_color = (0, 0, 255)  # 빨간색
                                            else:
                                                # 성공 시 초록색
                                                box_color = (0, 255, 0)  # 초록색 (BGR)
                                                text_color = (0, 255, 0)
                                            
                                            # QR 박스 그리기 (일반 선)
                                            cv2.polylines(display_frame, [display_points], True, box_color, 2)
                                            
                                            # 텍스트 표시 (하이픈 문자 정리)
                                            display_text = qr_text[:30] + "..." if len(qr_text) > 30 else qr_text
                                            # OpenCV putText에서 문제가 되는 특수 문자들을 표준 하이픈으로 변경
                                            display_text = display_text.replace('–', '-').replace('—', '-').replace('−', '-')
                                            display_text = display_text.replace('？', '?').replace('！', '!').replace('，', ',')
                                            
                                            # QR 번호 추가
                                            track_id = qr_viz.get('track_id')
                                            if track_id is not None:
                                                display_text = f"#{track_id} {display_text}"
                                            
                                            text_pos = (int(display_points[0][0]), int(display_points[0][1]) - 15 - (j * 20))
                                            cv2.putText(display_frame, display_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                                            
                                            # 탐지 방법 표시 (첫 번째 QR만, 한글 폰트 사용)
                                            if j == 0:
                                                method_text = f"Method: {qr_method}"
                                                display_frame = put_korean_text(display_frame, method_text, (10, 25), font_size=16, color=text_color)
                                        else:
                                            pass  # points_2d 변환 실패 (콘솔 출력 제거)
                                    except Exception as e:
                                        log_print(f"    ❌ 개별 QR 시각화 오류: {e}")
                                        import traceback
                                        log_print(traceback.format_exc())
                            except Exception as e:
                                print(f"    ❌ 시각화 오류: {e}")
                                # 기본 시각화 (폰트 크기 줄임, 하이픈 문자 정리)
                                text = detected_text[:30] + "..." if len(detected_text) > 30 else detected_text
                                text = text.replace('–', '-').replace('—', '-').replace('−', '-')
                                text = text.replace('？', '?').replace('！', '!').replace('，', ',')
                                cv2.putText(display_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                method_text = f"Method: {detection_method}"
                                display_frame = put_korean_text(display_frame, method_text, (10, 25), font_size=16, color=(0, 255, 0))
                    else:
                        # 탐지되지 않은 프레임 로그
                        log_print(f"  ⚠️ 탐지된 QR 코드 없음")
                        # 시각화 데이터가 없을 때 기본 시각화
                        text = detected_text[:30] + "..." if len(detected_text) > 30 else detected_text
                        text = text.replace('–', '-').replace('—', '-').replace('−', '-')
                        text = text.replace('？', '?').replace('！', '!').replace('，', ',')
                        cv2.putText(display_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        method_text = f"Method: {detection_method}"
                        display_frame = put_korean_text(display_frame, method_text, (10, 25), font_size=16, color=(0, 255, 0))
                    
                    # 결과 통계 업데이트
                    if "실패" in detected_text or "실패" in detection_method:
                        failed_count += 1
                    else:
                        success_count += 1
                
                except Exception as e:
                    log_print(f"  ❌ 프레임 {frame_count} 처리 오류: {e}")
                    import traceback
                    log_print(traceback.format_exc())
            
            # 성능 정보 표시
            fps_counter += 1
            if fps_counter % 30 == 0:  # 30프레임마다 FPS 계산
                elapsed = time.time() - fps_start_time
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                
                # 성능 정보 텍스트
                info_text = f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames} | QR: {detected_count}"
                cv2.putText(display_frame, info_text, (10, display_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 일시정지 상태 표시
            if paused:
                cv2.putText(display_frame, "PAUSED - Press SPACE to resume", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # ★★★★★ 영상에 프레임 저장 ★★★★★
            if out_video.isOpened():
                out_video.write(display_frame)
            
            # 화면에 표시
            cv2.imshow("Video Player + QR Detection", display_frame)
            
            # 키 입력 처리 (영상이 끝나지 않았을 때만)
            if paused or (not paused and ret):
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1
            
            if key == 27:  # ESC 키
                print("\n🛑 사용자가 종료했습니다.")
                break
            elif key == ord(' '):  # SPACE 키
                paused = not paused
                if paused:
                    print("⏸️  일시정지")
                else:
                    print("▶️  재생")
            elif key == ord('s'):  # S 키
                # 현재 프레임 저장 (시각화된 상태로)
                save_path = os.path.join(output_run_dir, f"screenshot_{frame_count:06d}.jpg")
                cv2.imwrite(save_path, display_frame)
                print(f"📷 스크린샷 저장: {save_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Ctrl+C로 종료되었습니다.")
    
    # 정리
    # 해독 워커 스레드 종료
    if stop_decode_worker is not None:
        stop_decode_worker.set()
        if decode_queue is not None:
            try:
                decode_queue.put(None, timeout=0.1)  # 워커 스레드 종료 신호
            except:
                pass
        if decode_worker_thread is not None:
            decode_worker_thread.join(timeout=1.0)  # 타임아웃 단축
            if decode_worker_thread.is_alive():
                log_print("⚠️ 해독 워커 스레드가 타임아웃 내에 종료되지 않았습니다.")
    
    # ★★★★★ 영상 저장 종료 ★★★★★
    if out_video.isOpened():
        out_video.release()
        log_print(f"✅ 영상 저장 완료: {output_video_path}")
    
    # 리소스 정리
    cap.release()
    
    # 창 닫기 (여러 번 시도)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # 창 닫기 이벤트 처리
    cv2.destroyAllWindows()  # 한 번 더 시도
    
    # 총 실행 시간 계산 (로그 파일 닫기 전에)
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 결과 요약 계산
    elapsed = time.time() - start_time
    # 🕐 전체 실행 시간은 이미 위에서 계산됨
    
    # 로그 파일에 결과 통계 기록 (로그 파일 닫기 전에)
    log_print(f"\n📊 결과 통계!")
    log_print(f"  총 프레임: {total_frames}")
    log_print(f"  재생 시간: {elapsed:.1f}초")
    log_print(f"  🚀 총 실행 시간: {total_execution_time:.1f}초 (병렬 처리)")
    log_print(f"  탐지된 QR 코드: {detected_count}개")
    log_print(f"  고유 QR 코드: {detected_count}개 (중복 제거 후)")
    log_print(f"  탐지율: {detected_count/frame_count*100:.1f}%" if frame_count > 0 else "  탐지율: 0.0%")
    log_print(f"  ✅ 성공: {success_count}개")
    log_print(f"  ❌ 실패: {failed_count}개")
    log_print(f"  결과 저장: {output_run_dir}/")
    log_print(f"  📹 출력 영상: {output_video_path}")
    
    log_print(f"\n🎯 방법별 성공률:")
    total_method_success = sum(method_stats.values())
    if total_method_success > 0:
        for method, count in method_stats.items():
            if count > 0:  # 0인 항목은 출력하지 않음 (최적화)
                percentage = (count / total_method_success) * 100
                log_print(f"  {method}: {count}개 ({percentage:.1f}%)")
    else:
        log_print("  탐지된 QR 코드가 없습니다.")
    
    log_print(f"\n📊 테스트용 상세 통계:")
    # 0이 아닌 방법만 출력 (최적화)
    non_zero_detections = {m: c for m, c in method_detection_count.items() if c > 0}
    if non_zero_detections:
        log_print(f"  방법별 탐지 개수:")
        for method, count in non_zero_detections.items():
            log_print(f"    {method}: {count}개")
        
        log_print(f"  방법별 성공률 (탐지 대비):")
        for method in non_zero_detections.keys():
            detected = method_detection_count[method]
            success = method_stats.get(method, 0)
            if detected > 0:
                success_rate = (success / detected) * 100
                log_print(f"    {method}: {success}/{detected} ({success_rate:.1f}%)")
    else:
        log_print("  탐지된 방법이 없습니다.")
    
    # Dynamsoft 방법별 상세 통계 출력
    if use_yolo_mode and yolo_model is not None and 'dynamsoft_method_stats' in locals():
        log_print(f"\n🔍 Dynamsoft 방법별 상세 통계:")
        total_dynamsoft = sum(dynamsoft_method_stats.values())
        if total_dynamsoft > 0:
            for method_detail, count in dynamsoft_method_stats.items():
                if count > 0:
                    percentage = (count / total_dynamsoft) * 100
                    log_print(f"  {method_detail}: {count}개 ({percentage:.1f}%)")
        else:
            log_print("  Dynamsoft 해독 성공이 없습니다.")
    
    # 추적 통계는 내부적으로만 사용 (시각화 제거)
    
    # 로그 파일에 총 실행 시간 기록
    log_print(f"\n" + "=" * 60)
    log_print(f"⏱️  총 실행 시간: {total_execution_time:.2f}초 ({total_execution_time/60:.2f}분)")
    if frame_count > 0:
        avg_time_per_frame = total_execution_time / frame_count
        log_print(f"   평균 프레임 처리 시간: {avg_time_per_frame*1000:.2f}ms/프레임")
    log_print(f"=" * 60)
    log_print(f"\n📝 로그 파일 저장 완료: {log_file_path}")
    log_file.close()
    
    # 콘솔에도 결과 출력
    print(f"\n📊 결과 통계!")
    print(f"  총 프레임: {total_frames}")
    print(f"  재생 시간: {elapsed:.1f}초")
    print(f"  🚀 총 실행 시간: {total_execution_time:.1f}초 (병렬 처리)")
    print(f"  탐지된 QR 코드: {detected_count}개")
    print(f"  고유 QR 코드: {detected_count}개 (중복 제거 후)")
    print(f"  탐지율: {detected_count/frame_count*100:.1f}%" if frame_count > 0 else "  탐지율: 0.0%")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {failed_count}개")
    print(f"  결과 저장: {output_run_dir}/")
    print(f"  📹 출력 영상: {output_video_path}")
    
    print(f"\n🎯 방법별 성공률:")
    total_method_success = sum(method_stats.values())
    if total_method_success > 0:
        for method, count in method_stats.items():
            if count > 0:  # 0인 항목은 출력하지 않음 (최적화)
                percentage = (count / total_method_success) * 100
                print(f"  {method}: {count}개 ({percentage:.1f}%)")
    else:
        print("  탐지된 QR 코드가 없습니다.")
    
    print(f"\n📊 테스트용 상세 통계:")
    # 0이 아닌 방법만 출력 (최적화)
    non_zero_detections = {m: c for m, c in method_detection_count.items() if c > 0}
    if non_zero_detections:
        print(f"  방법별 탐지 개수:")
        for method, count in non_zero_detections.items():
            print(f"    {method}: {count}개")
        
        print(f"  방법별 성공률 (탐지 대비):")
        for method in non_zero_detections.keys():
            detected = method_detection_count[method]
            success = method_stats.get(method, 0)
            if detected > 0:
                success_rate = (success / detected) * 100
                print(f"    {method}: {success}/{detected} ({success_rate:.1f}%)")
    else:
        print("  탐지된 방법이 없습니다.")
    
    # Dynamsoft 방법별 상세 통계 출력
    if use_yolo_mode and yolo_model is not None and 'dynamsoft_method_stats' in locals():
        print(f"\n🔍 Dynamsoft 방법별 상세 통계:")
        total_dynamsoft = sum(dynamsoft_method_stats.values())
        if total_dynamsoft > 0:
            for method_detail, count in dynamsoft_method_stats.items():
                if count > 0:
                    percentage = (count / total_dynamsoft) * 100
                    print(f"  {method_detail}: {count}개 ({percentage:.1f}%)")
        else:
            print("  Dynamsoft 해독 성공이 없습니다.")
    
    # 추적 통계는 내부적으로만 사용 (시각화 제거)
    
    # 총 실행 시간 (마지막에 강조 표시)
    print(f"\n" + "=" * 60)
    print(f"⏱️  총 실행 시간: {total_execution_time:.2f}초 ({total_execution_time/60:.2f}분)")
    if frame_count > 0:
        avg_time_per_frame = total_execution_time / frame_count
        print(f"   평균 프레임 처리 시간: {avg_time_per_frame*1000:.2f}ms/프레임")
    print(f"=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python video_synch.py <비디오_파일_경로>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_player_with_qr(video_path)