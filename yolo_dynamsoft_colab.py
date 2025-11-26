"""
구글 코랩용: 영상 플레이어 + 실시간 QR 탐지 (병렬 처리)
[코랩 최적화]: GUI 제거, matplotlib/IPython.display 사용
[성능 개선]: YOLO 탐지 + Dynamsoft 해독
[일직선 움직임 최적화]: 조선소 T-bar 제작 공정 등 일직선 움직임 QR 코드 추적 최적화
"""

import cv2
import time
import os
import sys
import numpy as np
import threading
import queue
from queue import Queue, Empty
from IPython.display import display, Image, clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# 코랩 환경 확인
IN_COLAB = 'google.colab' in sys.modules

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
            iou = calculate_iou(bbox1, bbox2)
            
            if iou > iou_threshold:
                is_overlapping = True
                break
        
        if not is_overlapping:
            filtered_locations.append(location)
            
    return filtered_locations

def process_frame_with_yolo(frame, yolo_model, conf_threshold=0.25):
    """YOLO로 빠르게 위치만 탐지 (해독 제거, 비동기 해독으로 분리)"""
    if yolo_model is not None:
        qr_locations = yolo_detect_qr_locations(yolo_model, frame, conf_threshold)
        filtered_locations = filter_overlapping_yolo_rois(qr_locations, iou_threshold=0.5)
        return filtered_locations
    
    return []

def create_single_frame(frame):
    """원본 프레임만 사용"""
    return frame, [1.0]

def get_platform_font_paths():
    """플랫폼별 한글 폰트 경로 반환 (코랩용)"""
    font_paths = []
    
    # 코랩에서는 나눔 폰트 사용
    if IN_COLAB:
        # 코랩에서 나눔 폰트 설치 경로
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        # 로컬 환경
        import platform
        system = platform.system()
        if system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/gulim.ttc",
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ]
        else:  # Linux
            font_paths = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            ]
    
    # 프로젝트 내 폰트 경로도 추가
    project_font_path = "data/font/NanumGothic.ttf"
    if os.path.exists(project_font_path):
        font_paths.insert(0, project_font_path)
    
    return font_paths

def put_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """OpenCV 이미지에 한글 텍스트를 그리는 함수 (코랩 호환)"""
    if not PIL_AVAILABLE:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        font_paths = get_platform_font_paths()
        
        font = None
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        draw.text(position, text, font=font, fill=color)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

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
    """QR의 중심점과 사각형 좌표를 반환"""
    if 'quad_xy' in detection:
        quad = detection['quad_xy']
        if quad is not None and len(quad) == 4:
            quad_array = np.array(quad)
            center = np.mean(quad_array, axis=0)
            x_coords = quad_array[:, 0]
            y_coords = quad_array[:, 1]
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)
            return center[0], center[1], x1, y1, x2, y2
    
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
    
    elif 'bbox_xyxy' in detection:
        bbox = detection['bbox_xyxy']
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y, x1, y1, x2, y2
    
    return None, None, None, None, None, None

# QRTrack, QRTracker 클래스는 원본과 동일 (간소화 버전)
class QRTrack:
    """단일 QR 코드 추적 정보"""
    def __init__(self, track_id, qr_data, frame_number):
        self.track_id = track_id
        self.qr_data = qr_data
        self.frame_number = frame_number
        self.last_seen_frame = frame_number
        self.missed_frames = 0
        self.history = []
        
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr_data['detection'])
        if center_x is not None:
            self.bbox = (x1, y1, x2, y2)
            self.center = (center_x, center_y)
            self.history.append(self.bbox)
        else:
            self.bbox = None
            self.center = None
    
    def update(self, qr_data, frame_number):
        self.qr_data = qr_data
        self.frame_number = frame_number
        self.last_seen_frame = frame_number
        self.missed_frames = 0
        
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr_data['detection'])
        if center_x is not None:
            self.bbox = (x1, y1, x2, y2)
            self.center = (center_x, center_y)
            self.history.append(self.bbox)
            if len(self.history) > 10:
                self.history.pop(0)
    
    def predict_position(self):
        """이전 위치 기반으로 다음 위치 예측"""
        if self.bbox is None or len(self.history) < 2:
            return self.bbox
        
        num_frames = min(len(self.history), 5)
        recent_history = self.history[-num_frames:]
        
        centers = []
        for bbox in recent_history:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append((center_x, center_y))
        
        if len(centers) >= 2:
            first_center = centers[-2]
            last_center = centers[-1]
            total_vx = last_center[0] - first_center[0]
            total_vy = last_center[1] - first_center[1]
        else:
            return self.bbox
        
        curr_bbox = self.history[-1]
        curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
        curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
        
        frames_to_predict = self.missed_frames + 1
        predicted_center_x = curr_center_x + total_vx * frames_to_predict
        predicted_center_y = curr_center_y + total_vy * frames_to_predict
        
        box_width = curr_bbox[2] - curr_bbox[0]
        box_height = curr_bbox[3] - curr_bbox[1]
        
        predicted_bbox = (
            int(predicted_center_x - box_width / 2),
            int(predicted_center_y - box_height / 2),
            int(predicted_center_x + box_width / 2),
            int(predicted_center_y + box_height / 2)
        )
        
        return predicted_bbox

class QRTracker:
    """QR 코드 프레임 간 추적 관리자"""
    def __init__(self, max_missed_frames=10, iou_threshold=0.15, center_dist_threshold=1.2, 
                 linear_motion_boost=True):
        self.tracks = {}
        self.next_track_id = 0
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.center_dist_threshold = center_dist_threshold
        self.linear_motion_boost = linear_motion_boost
    
    def update(self, detected_qrs, frame_number):
        detected_bboxes = []
        for qr in detected_qrs:
            center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr['detection'])
            if center_x is not None:
                detected_bboxes.append({
                    'qr': qr,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
        
        active_tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.missed_frames <= self.max_missed_frames
        }
        
        matched_detections = set()
        matched_tracks = set()
        match_scores = []
        
        for track_id, track in active_tracks.items():
            if track.bbox is None:
                continue
            
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
                
                # 중심점 거리 계산 (로컬용과 동일한 방식)
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
                dynamic_iou_threshold = max(0.05, dynamic_iou_threshold)  # 최소 0.05
                dynamic_center_dist_threshold = self.center_dist_threshold * (1.0 + track.missed_frames * 0.1)
                dynamic_center_dist_threshold = min(2.0, dynamic_center_dist_threshold)  # 최대 2.0
                
                # 매칭 조건: IoU 또는 중심점 거리 또는 텍스트 매칭 또는 예측 위치 기반 거리
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
                    # 복합 점수 계산 (로컬용과 동일)
                    if text_match:
                        score = 1000.0 + iou * 100  # 텍스트 매칭 우선
                    elif pred_center_dist is not None and self.linear_motion_boost:
                        # 일직선 움직임 가정 시 예측 위치 기반 점수 (가중치 증가)
                        score = iou * 100 + (1.0 - center_dist) * 50 + (1.0 - pred_center_dist) * 100
                    else:
                        # IoU와 중심점 거리를 조합한 점수
                        score = iou * 100 + (1.0 - center_dist) * 50
                    
                    match_scores.append((track_id, idx, score, iou, center_dist, text_match, pred_center_dist))
        
        match_scores.sort(key=lambda x: x[2], reverse=True)
        
        for match_data in match_scores:
            if len(match_data) == 7:
                track_id, detection_idx, score, iou, center_dist, text_match, pred_center_dist = match_data
            else:
                track_id, detection_idx, score, iou, center_dist, text_match = match_data[:6]
                pred_center_dist = None
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            
            track = active_tracks[track_id]
            det = detected_bboxes[detection_idx]
            track.update(det['qr'], frame_number)
            matched_detections.add(detection_idx)
            matched_tracks.add(track_id)
        
        for idx, det in enumerate(detected_bboxes):
            if idx not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                new_track = QRTrack(track_id, det['qr'], frame_number)
                self.tracks[track_id] = new_track
        
        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks:
                track.missed_frames += 1
                track.frame_number = frame_number
        
        tracked_qrs = []
        
        # 탐지된 QR (매칭된 것) - 로컬용과 동일한 방식
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
        
        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks and track.missed_frames > 0:
                predicted_bbox = track.predict_position()
                if predicted_bbox is not None:
                    tracked_qr = track.qr_data.copy()
                    tracked_qr['track_id'] = track_id
                    tracked_qr['tracked'] = True
                    tracked_qr['predicted'] = True
                    tracked_qr['missed_frames'] = track.missed_frames
                    if 'detection' in tracked_qr:
                        tracked_qr['detection'] = tracked_qr['detection'].copy()
                        tracked_qr['detection']['bbox_xyxy'] = list(predicted_bbox)
                    tracked_qrs.append(tracked_qr)
        
        tracks_to_remove = [
            tid for tid, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for tid in tracks_to_remove:
            del self.tracks[tid]
        
        return tracked_qrs
    
    def get_active_track_count(self):
        return len([t for t in self.tracks.values() if t.missed_frames <= self.max_missed_frames])

def video_player_with_qr(video_path, output_dir="video_player_results", 
                         show_preview=True, preview_interval=30, verbose_log=False):
    """영상 플레이어 + 실시간 QR 탐지 (코랩용)"""
    
    total_start_time = time.time()
    
    import datetime
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_run_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_run_dir, f"qr_detection_log_{run_id}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    log_file_closed = False
    
    log_buffer = []  # 로그 버퍼링 (성능 개선)
    log_flush_count = 0
    
    def log_print(message, force_flush=False):
        nonlocal log_buffer, log_flush_count  # 외부 변수 수정을 위해 nonlocal 선언
        print(message)
        try:
            if not log_file_closed and not log_file.closed:
                log_buffer.append(message + '\n')
                log_flush_count += 1
                # 10개마다 또는 강제 플러시 시 파일에 쓰기
                if force_flush or log_flush_count >= 10:
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()
                    log_flush_count = 0
        except (ValueError, AttributeError):
            # 파일이 닫혔거나 오류 발생 시 콘솔에만 출력
            pass
    
    log_print(f"🖥️  구글 코랩 환경에서 실행 중")
    log_print(f"📁 결과 폴더: {output_run_dir}")
    
    # YOLO 모델 초기화
    yolo_model = None
    use_yolo_mode = True
    
    if YOLO_AVAILABLE and use_yolo_mode:
        try:
            model_path = 'model1.pt'
            if os.path.exists(model_path):
                # GPU 사용 여부 먼저 확인
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                if device == 'cuda':
                    gpu_name = torch.cuda.get_device_name(0)
                    log_print(f"🖥️  GPU 감지: {gpu_name}")
                else:
                    log_print("🖥️  GPU 미감지 - CPU 모드로 실행됩니다")
                    log_print("⚠️ GPU를 사용하려면: 런타임 > 런타임 유형 변경 > GPU 선택")
                
                # YOLO 모델 초기화
                yolo_model = YOLO(model_path)
                
                # GPU 사용 여부 재확인 (YOLO가 자동으로 GPU 사용)
                if device == 'cuda':
                    log_print(f"✅ YOLO 모델 초기화 완료 (GPU 모드: {gpu_name})")
                else:
                    log_print("✅ YOLO 모델 초기화 완료 (CPU 모드)")
            else:
                log_print(f"⚠️ YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
                use_yolo_mode = False
        except Exception as e:
            log_print(f"❌ YOLO 모델 초기화 실패: {e}")
            use_yolo_mode = False
    else:
        use_yolo_mode = False
    
    # Dynamsoft 초기화
    dbr_reader = None
    if DBR_AVAILABLE:
        try:
            license_key = os.environ.get('DYNAMSOFT_LICENSE_KEY', '')
            if not license_key:
                license_key = 't0085YQEAADYdcL2llMa8vH1Rtnun+43saE/kdAE7ZbIxMQGRMtSzVSZRI8vfOK4Ids52rjekwzh87yABFLraXw5Va1BV7NnBjI8m7qbw3kxOprI75ExJpw=='
            
            if license_key:
                if DBR_VERSION == "bundle_v11":
                    error = license.LicenseManager.init_license(license_key)
                    if error[0] != 0:
                        log_print(f"⚠️ Dynamsoft 라이선스 초기화 실패: {error[1]}")
                    else:
                        dbr_reader = cvr.CaptureVisionRouter()
                        from dynamsoft_barcode_reader_bundle import EnumPresetTemplate
                        error_code, error_msg, settings = dbr_reader.get_simplified_settings(EnumPresetTemplate.PT_DEFAULT)
                        if error_code == 0 and settings:
                            barcode_settings = settings.barcode_settings
                            if barcode_settings:
                                barcode_settings.barcode_format_ids = dbr.EnumBarcodeFormat.BF_QR_CODE
                                if hasattr(barcode_settings, 'expected_barcodes_count'):
                                    barcode_settings.expected_barcodes_count = 10
                                if hasattr(barcode_settings, 'deblur_level'):
                                    barcode_settings.deblur_level = 9
                            dbr_reader.update_settings(EnumPresetTemplate.PT_DEFAULT, settings)
                        log_print("✅ Dynamsoft Barcode Reader 초기화 완료")
        except Exception as e:
            log_print(f"❌ Dynamsoft 초기화 실패: {e}")
            dbr_reader = None
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log_print(f"\n📹 비디오 정보:")
    log_print(f"  파일: {video_path}")
    log_print(f"  해상도: {width}x{height}")
    log_print(f"  FPS: {fps:.2f}")
    log_print(f"  총 프레임: {total_frames}")
    
    # 출력 비디오 설정
    output_video_path = os.path.join(output_run_dir, f"output_{run_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 추적 초기화
    qr_tracker = QRTracker(max_missed_frames=10, iou_threshold=0.15, center_dist_threshold=1.2, 
                           linear_motion_boost=True)
    
    # 비동기 해독 워커
    decode_queue = None
    decode_results = {}
    decode_worker_thread = None
    stop_decode_worker = None
    decode_lock = threading.Lock()
    
    if dbr_reader is not None:
        decode_queue = Queue(maxsize=10)
        stop_decode_worker = threading.Event()
        
        def decode_worker():
            while not stop_decode_worker.is_set():
                try:
                    item = decode_queue.get(timeout=0.1)
                    if item is None:
                        return
                    
                    if len(item) == 5:
                        track_id, roi, bbox, roi_offset, frame_num = item
                    else:
                        track_id, roi, bbox, roi_offset = item
                        frame_num = None
                    
                    decoded_text = None
                    quad_xy = None
                    decode_method_detail = None
                    
                    try:
                        if dbr_reader is not None:
                            if len(roi.shape) == 3:
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            else:
                                roi_gray = roi.copy()
                            
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            roi_enhanced = clahe.apply(roi_gray)
                            roi_norm = cv2.normalize(roi_enhanced, None, 0, 255, cv2.NORM_MINMAX)
                            
                            rh, rw = roi_norm.shape
                            border_size = 20
                            white_canvas = np.full((rh + border_size*2, rw + border_size*2), 255, dtype=np.uint8)
                            white_canvas[border_size:border_size+rh, border_size:border_size+rw] = roi_norm
                            roi_rgb = cv2.cvtColor(white_canvas, cv2.COLOR_GRAY2RGB)
                            
                            if DBR_VERSION == "bundle_v11":
                                from dynamsoft_barcode_reader_bundle import dbr as dbr_module
                                
                                items = None
                                captured_result = dbr_reader.capture(roi_rgb, dbr_module.EnumImagePixelFormat.IPF_RGB_888)
                                barcode_result = captured_result.get_decoded_barcodes_result()
                                if barcode_result:
                                    items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
                                    if items and len(items) > 0:
                                        decode_method_detail = "원본(흰테두리)"
                                
                                if not items or len(items) == 0:
                                    roi_inverted_gray = cv2.bitwise_not(roi_norm)
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
                                    barcode_item = items[0]
                                    text = None
                                    if hasattr(barcode_item, 'get_text'):
                                        text = barcode_item.get_text()
                                    elif hasattr(barcode_item, 'text'):
                                        text = barcode_item.text
                                    
                                    if text:
                                        decoded_text = text
                                        decoded_text = _process_decoded_text(decoded_text)
                                        
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
                                                
                                                if result_points:
                                                    roi_x1, roi_y1 = roi_offset
                                                    quad_xy = []
                                                    for point in result_points:
                                                        abs_x = roi_x1 + int(point.x)
                                                        abs_y = roi_y1 + int(point.y)
                                                        quad_xy.append([abs_x, abs_y])
                                        except:
                                            pass
                    except Exception as e:
                        pass
                    
                    if decoded_text:
                        if quad_xy is None:
                            x1, y1, x2, y2 = bbox
                            quad_xy = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        
                        with decode_lock:
                            decode_results[track_id] = {
                                'text': decoded_text,
                                'quad_xy': quad_xy,
                                'decode_bbox': list(bbox),
                                'decode_method': "Dynamsoft",
                                'decode_method_detail': decode_method_detail,
                                'frame': frame_num if frame_num is not None else 0
                            }
                    
                    decode_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    if 'item' in locals() and item:
                        decode_queue.task_done()
        
        decode_worker_thread = threading.Thread(target=decode_worker, daemon=True)
        decode_worker_thread.start()
        log_print("✅ 비동기 해독 워커 스레드 시작")
    
    # 통계 변수
    frame_count = 0
    detected_count = 0
    success_count = 0
    failed_count = 0
    
    log_print(f"\n🎬 영상 처리 시작!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_print("\n📺 영상 재생 완료!")
                break
            
            frame_count += 1
            
            # QR 코드 탐지
            if use_yolo_mode and yolo_model is not None:
                filtered_locations = process_frame_with_yolo(frame, yolo_model, conf_threshold=0.25)
                
                detected_qrs = []
                for location in filtered_locations:
                    qr_data = {
                        'bbox': location['bbox'],
                        'confidence': location['confidence'],
                        'text': '',
                        'detection': {
                            'bbox_xyxy': location['bbox'],
                            'quad_xy': None
                        },
                        'method': 'YOLO',
                        'success': False
                    }
                    detected_qrs.append(qr_data)
                
                tracked_qrs = qr_tracker.update(detected_qrs, frame_count)
                
                # 비동기 해독 큐에 추가
                if decode_queue is not None and dbr_reader is not None:
                    for tracked_qr in tracked_qrs:
                        track_id = tracked_qr.get('track_id')
                        if track_id is not None:
                            with decode_lock:
                                if track_id in decode_results:
                                    # 해독 결과 업데이트 (로컬용과 동일한 로직)
                                    decode_result = decode_results[track_id]
                                    tracked_qr['text'] = decode_result['text']
                                    tracked_qr['success'] = True
                                    # 실제 사용된 방법으로 업데이트 (YOLO + 해독 방법)
                                    decode_method = decode_result.get('decode_method', 'Unknown')
                                    tracked_qr['method'] = f"YOLO+{decode_method}"
                                    if 'detection' in tracked_qr and decode_result.get('quad_xy'):
                                        # quad_xy를 현재 추적 위치에 맞춰서 변환 (로컬용과 동일)
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
                            
                            bbox = tracked_qr.get('bbox', tracked_qr.get('detection', {}).get('bbox_xyxy'))
                            if bbox is not None and len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                roi = frame[y1:y2, x1:x2]
                                if roi.size > 0:
                                    try:
                                        decode_queue.put_nowait((track_id, roi, bbox, (x1, y1), frame_count))
                                    except:
                                        pass
                
                # 결과 시각화
                result_frame = frame.copy()
                
                for qr in tracked_qrs:
                    detection = qr.get('detection')
                    if detection is None:
                        continue
                    
                    # 해독 결과 확인 및 업데이트 (시각화 전에 다시 확인 - 로컬용과 동일)
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
                    
                    qr_points = None
                    if 'quad_xy' in detection and detection['quad_xy'] is not None:
                        quad = detection['quad_xy']
                        if len(quad) == 4:
                            quad_array = np.array(quad)
                            qr_points = quad_array.astype(np.int32)
                    elif 'bbox_xyxy' in detection:
                        bbox = detection['bbox_xyxy']
                        x1, y1, x2, y2 = bbox
                        qr_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    
                    if qr_points is not None:
                        text = qr.get('text', '')
                        success = qr.get('success', False)
                        track_id = qr.get('track_id', None)
                        
                        color = (0, 255, 0) if success else (0, 0, 255)
                        cv2.polylines(result_frame, [qr_points], True, color, 2)
                        
                        if text:
                            display_text = text[:30] + "..." if len(text) > 30 else text
                            if track_id is not None:
                                display_text = f"#{track_id} {display_text}"
                            text_pos = (int(qr_points[0][0]), int(qr_points[0][1]) - 15)
                            result_frame = put_korean_text(result_frame, display_text, text_pos, font_size=14, color=color)
                        
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1
                
                if tracked_qrs:
                    detected_count += 1
                
                # ★★★★★ 프레임별 상세 로그 출력 (verbose_log가 True일 때만) ★★★★★
                if verbose_log:
                    log_print(f"\n📹 프레임 {frame_count}/{total_frames}")
                    if tracked_qrs:
                        log_print(f"  🔍 탐지: {len(tracked_qrs)}개 QR 코드 발견")
                        for idx, qr in enumerate(tracked_qrs):
                            track_id = qr.get('track_id', 'N/A')
                            bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy', []))
                            conf = qr.get('confidence', 0.0)
                            text = qr.get('text', '')
                            method = qr.get('method', 'Unknown')
                            success = qr.get('success', False)
                            
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
                
                # 비디오 저장
                out_video.write(result_frame)
                
                # 코랩에서 프리뷰 표시 (일정 간격마다) - 최적화: 메모리 정리 추가
                if show_preview and frame_count % preview_interval == 0:
                    try:
                        clear_output(wait=True)
                        display_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        plt.figure(figsize=(12, 8))
                        plt.imshow(display_frame)
                        plt.axis('off')
                        plt.title(f'Frame {frame_count}/{total_frames} - Detected: {len(tracked_qrs)} QR codes', 
                                 fontsize=14)
                        plt.tight_layout()
                        plt.show()
                        plt.close()  # 메모리 정리 (성능 개선)
                        print(f"프레임 {frame_count}/{total_frames} 처리 중... (탐지: {len(tracked_qrs)}개)")
                    except Exception as e:
                        # 프리뷰 표시 실패해도 계속 진행
                        pass
            
            # 진행 상황 출력
            if frame_count % 100 == 0:
                log_print(f"   처리 중... {frame_count}/{total_frames} 프레임 ({frame_count/total_frames*100:.1f}%)")
    
    except KeyboardInterrupt:
        log_print("\n⏹️ 사용자가 중단했습니다.")
    finally:
        # 정리
        if stop_decode_worker is not None:
            stop_decode_worker.set()
            if decode_queue is not None:
                try:
                    decode_queue.put(None, timeout=0.1)
                except:
                    pass
            if decode_worker_thread is not None:
                decode_worker_thread.join(timeout=1.0)
        
        out_video.release()
        cap.release()
    
    # 최종 통계 (파일 닫기 전에 출력)
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    log_print(f"\n📊 결과 통계!", force_flush=True)
    log_print(f"  총 프레임: {total_frames}", force_flush=True)
    log_print(f"  처리된 프레임: {frame_count}", force_flush=True)
    log_print(f"  총 실행 시간: {total_execution_time:.1f}초", force_flush=True)
    log_print(f"  탐지된 프레임: {detected_count}개", force_flush=True)
    log_print(f"  ✅ 성공: {success_count}개", force_flush=True)
    log_print(f"  ❌ 실패: {failed_count}개", force_flush=True)
    log_print(f"  결과 저장: {output_run_dir}/", force_flush=True)
    log_print(f"  📹 출력 영상: {output_video_path}", force_flush=True)
    
    # 버퍼에 남은 로그 모두 쓰기
    try:
        if not log_file_closed and not log_file.closed and log_buffer:
            log_file.writelines(log_buffer)
            log_file.flush()
            log_buffer.clear()
    except:
        pass
    
    # 이제 파일 닫기
    try:
        if not log_file.closed:
            log_file.close()
        log_file_closed = True
    except:
        pass
    
    print(f"\n✅ 처리 완료!")
    print(f"📹 출력 영상: {output_video_path}")
    print(f"📁 결과 폴더: {output_run_dir}")
    
    # 코랩에서 최종 결과 영상 표시
    if show_preview and os.path.exists(output_video_path):
        from IPython.display import Video
        display(Video(output_video_path, width=800))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python yolo_dynamsoft_colab.py <비디오_파일_경로>")
        print("\n코랩에서 사용 예시:")
        print("  video_player_with_qr('video.mp4', output_dir='results', show_preview=True)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_player_with_qr(video_path, show_preview=True)

