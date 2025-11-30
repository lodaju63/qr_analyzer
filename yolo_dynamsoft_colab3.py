"""
구글 코랩용: 영상 플레이어 + 실시간 QR 탐지 (yolo_dynamsoft.py와 동일한 기능)
[코랩 최적화]: GUI 제거, matplotlib/IPython.display 사용
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
from IPython.display import display, Image, clear_output, Video
import matplotlib.pyplot as plt
import warnings
import datetime

warnings.filterwarnings('ignore')

# 코랩 환경 확인
IN_COLAB = 'google.colab' in sys.modules

# Dynamsoft Barcode Reader import (dynamsoft-barcode-reader-bundle v11)
try:
    from dynamsoft_barcode_reader_bundle import dbr, license, cvr
    DBR_AVAILABLE = True
    DBR_VERSION = "bundle_v11"
except ImportError:
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
    
    decoded_text = decoded_text.replace('–', '-').replace('—', '-')
    
    try:
        if isinstance(decoded_text, bytes):
            decoded_text = decoded_text.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_text = decoded_text.decode('cp949')
        except:
            decoded_text = str(decoded_text)
    
    return decoded_text

def preprocess_frame_for_detection(frame, use_clahe=True, use_normalize=True, clahe_clip_limit=2.0):
    """탐지 성능 향상을 위한 프레임 전처리"""
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        else:
            enhanced = gray
        
        if use_normalize:
            normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = enhanced
        
        processed = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        return processed
    except:
        return frame

def calculate_iou(bbox1, bbox2):
    """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def filter_overlapping_yolo_rois(locations, iou_threshold=0.5):
    """YOLO가 반환한 ROI 리스트에서 겹치는 ROI를 제거 (NMS와 유사)"""
    if not locations:
        return []
    
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

def yolo_detect_qr_locations(model, frame, conf_threshold=0.25, use_preprocessing=False, 
                             use_clahe=True, use_normalize=True, clahe_clip_limit=2.0, 
                             detect_both_frames=True, iou_threshold=0.5):
    """YOLO 모델로 QR 코드 위치 빠르게 탐지"""
    try:
        all_locations = []
        
        frames_to_detect = []
        if use_preprocessing:
            processed_frame = preprocess_frame_for_detection(frame, use_clahe=use_clahe, 
                                                           use_normalize=use_normalize, 
                                                           clahe_clip_limit=clahe_clip_limit)
            if detect_both_frames:
                frames_to_detect = [processed_frame, frame]
            else:
                frames_to_detect = [processed_frame]
        else:
            frames_to_detect = [frame]
        
        for detect_frame in frames_to_detect:
            results = model(detect_frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                h_orig, w_orig = frame.shape[:2]
                h_detect, w_detect = detect_frame.shape[:2]
                
                scale_x = w_orig / w_detect if w_detect > 0 else 1.0
                scale_y = h_orig / h_detect if h_detect > 0 else 1.0
                
                for box in result.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    pad = 20
                    h, w = frame.shape[:2]
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    
                    all_locations.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        if len(all_locations) > 1:
            filtered_locations = filter_overlapping_yolo_rois(all_locations, iou_threshold=iou_threshold)
            return filtered_locations
        
        return all_locations
    except Exception as e:
        return []

def process_frame_with_yolo(frame, yolo_model, conf_threshold=0.25, use_preprocessing=False,
                            use_clahe=True, use_normalize=True, clahe_clip_limit=2.0, 
                            detect_both_frames=True, iou_threshold=0.5):
    """YOLO로 빠르게 위치만 탐지 (해독 제거, 비동기 해독으로 분리)"""
    if yolo_model is not None:
        qr_locations = yolo_detect_qr_locations(yolo_model, frame, conf_threshold, 
                                               use_preprocessing=use_preprocessing,
                                               use_clahe=use_clahe,
                                               use_normalize=use_normalize,
                                               clahe_clip_limit=clahe_clip_limit,
                                               detect_both_frames=detect_both_frames,
                                               iou_threshold=iou_threshold)
        return qr_locations
    
    return []

def create_single_frame(frame):
    """원본 프레임만 사용"""
    return frame, [1.0]

def get_platform_font_paths():
    """플랫폼별 한글 폰트 경로 반환 (코랩용)"""
    font_paths = []
    
    if IN_COLAB:
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        system = platform.system()
        if system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/gulim.ttc",
            ]
        elif system == "Darwin":
            font_paths = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ]
        else:
            font_paths = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            ]
    
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
    
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    diag1 = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
    diag2 = np.sqrt((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)
    max_diag = max(diag1, diag2)
    
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
        """이전 위치 기반으로 다음 위치 예측 (일직선 움직임 가정)"""
        if self.bbox is None:
            return None
        
        if len(self.history) < 2:
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
    """QR 코드 프레임 간 추적 관리자 (일직선 움직임 최적화)"""
    def __init__(self, max_missed_frames=10, iou_threshold=0.15, center_dist_threshold=1.2, 
                 linear_motion_boost=True):
        self.tracks = {}
        self.next_track_id = 0
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.center_dist_threshold = center_dist_threshold
        self.linear_motion_boost = linear_motion_boost
    
    def update(self, detected_qrs, frame_number):
        """탐지된 QR 코드들과 추적 중인 QR 코드들을 매칭하여 업데이트"""
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
                iou = calculate_iou(track_bbox, det['bbox'])
                center_dist = calculate_center_distance(track_bbox, det['bbox'])
                
                det_text = det['qr'].get('text', '')
                text_match = (track_text != '' and det_text != '' and track_text == det_text)
                
                predicted_bbox = track.predict_position()
                pred_center_dist = None
                if predicted_bbox is not None and self.linear_motion_boost:
                    pred_center_dist = calculate_center_distance(predicted_bbox, det['bbox'])
                
                dynamic_iou_threshold = self.iou_threshold * (1.0 - track.missed_frames * 0.05)
                dynamic_iou_threshold = max(0.05, dynamic_iou_threshold)
                dynamic_center_dist_threshold = self.center_dist_threshold * (1.0 + track.missed_frames * 0.1)
                dynamic_center_dist_threshold = min(2.0, dynamic_center_dist_threshold)
                
                matches = False
                if text_match:
                    matches = True
                elif iou >= dynamic_iou_threshold:
                    matches = True
                elif center_dist <= dynamic_center_dist_threshold:
                    matches = True
                elif pred_center_dist is not None and pred_center_dist <= dynamic_center_dist_threshold * 1.2:
                    matches = True
                
                if matches:
                    if text_match:
                        score = 1000.0 + iou * 100
                    elif pred_center_dist is not None and self.linear_motion_boost:
                        score = iou * 100 + (1.0 - center_dist) * 50 + (1.0 - pred_center_dist) * 100
                    else:
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
        
        detection_to_track = {}
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
                         use_preprocessing=False, use_clahe=True, use_normalize=True,
                         clahe_clip_limit=2.0, detect_both_frames=True, conf_threshold=0.25,
                         iou_threshold=0.5, show_preview=True, preview_interval=30):
    """영상 플레이어 + 실시간 QR 탐지 (코랩용)"""
    
    total_start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_run_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_run_dir, f"qr_detection_log_{run_id}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    log_file_closed = False
    
    log_buffer = []
    log_flush_count = 0
    
    def log_print(message, force_flush=False):
        nonlocal log_buffer, log_flush_count
        print(message)
        try:
            if not log_file_closed and not log_file.closed:
                log_buffer.append(message + '\n')
                log_flush_count += 1
                if force_flush or log_flush_count >= 10:
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()
                    log_flush_count = 0
        except:
            pass
    
    log_print(f"🖥️  구글 코랩 환경에서 실행 중")
    log_print(f"📁 결과 폴더: {output_run_dir}")
    
    # YOLO 모델 초기화
    yolo_model = None
    use_yolo_mode = True
    
    if YOLO_AVAILABLE and use_yolo_mode:
        try:
            model_path = YOLO_MODEL_PATH
            if os.path.exists(model_path):
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                if device == 'cuda':
                    gpu_name = torch.cuda.get_device_name(0)
                    log_print(f"🖥️  GPU 감지: {gpu_name}")
                else:
                    log_print("🖥️  GPU 미감지 - CPU 모드로 실행됩니다")
                
                yolo_model = YOLO(model_path)
                
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
                        log_print(f"   라이선스 오류 코드: {error[0]}")
                        log_print(f"   다른 PC에서 실행 시 라이선스 제한이 있을 수 있습니다.")
                        log_print(f"   환경 변수 DYNAMSOFT_LICENSE_KEY에 유효한 라이선스 키를 설정하세요.")
                    else:
                        log_print(f"✅ Dynamsoft 라이선스 초기화 성공")
                        dbr_reader = cvr.CaptureVisionRouter()
                        from dynamsoft_barcode_reader_bundle import EnumPresetTemplate
                        error_code, error_msg, settings = dbr_reader.get_simplified_settings(EnumPresetTemplate.PT_DEFAULT)
                        if error_code == 0 and settings:
                            barcode_settings = settings.barcode_settings
                            if barcode_settings:
                                barcode_settings.barcode_format_ids = dbr.EnumBarcodeFormat.BF_QR_CODE
                                # 한 번에 많이 찾도록 설정 (colab2와 동일)
                                if hasattr(barcode_settings, 'expected_barcodes_count'):
                                    barcode_settings.expected_barcodes_count = 50  # 10 -> 50으로 증가
                                if hasattr(barcode_settings, 'deblur_level'):
                                    barcode_settings.deblur_level = 9  # 최대 디블러 레벨
                                # 추가 최적화 설정
                                if hasattr(barcode_settings, 'min_barcode_text_length'):
                                    barcode_settings.min_barcode_text_length = 1
                            # 타임아웃 설정도 추가
                            if hasattr(settings, 'timeout'):
                                settings.timeout = 500  # 500ms 타임아웃
                            # 설정 업데이트 및 확인
                            update_error = dbr_reader.update_settings(EnumPresetTemplate.PT_DEFAULT, settings)
                            if update_error[0] != 0:
                                log_print(f"⚠️ Dynamsoft 설정 업데이트 경고: {update_error[1]}")
                            else:
                                log_print(f"✅ Dynamsoft Barcode Reader 초기화 완료 (expected_barcodes_count=50, deblur_level=9)")
                        else:
                            log_print(f"⚠️ Dynamsoft 설정 가져오기 실패: error_code={error_code}, msg={error_msg}")
                            log_print("✅ Dynamsoft Barcode Reader 초기화 완료 (기본 설정 사용)")
        except Exception as e:
            log_print(f"❌ Dynamsoft 초기화 실패: {e}")
            dbr_reader = None
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 첫 프레임을 읽어서 실제 해상도 확인 (메타데이터가 부정확할 수 있음)
    ret, first_frame = cap.read()
    if not ret:
        log_print(f"❌ 비디오 파일에서 프레임을 읽을 수 없습니다: {video_path}")
        cap.release()
        return
    
    # 실제 프레임 크기 확인
    actual_height, actual_width = first_frame.shape[:2]
    
    # 메타데이터에서 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_meta = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_meta = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 실제 프레임 크기와 메타데이터가 다르면 실제 크기 사용
    if actual_width != width_meta or actual_height != height_meta:
        log_print(f"⚠️ 메타데이터와 실제 프레임 크기가 다릅니다:", force_flush=True)
        log_print(f"   메타데이터: {width_meta}x{height_meta}", force_flush=True)
        log_print(f"   실제 프레임: {actual_width}x{actual_height}", force_flush=True)
        log_print(f"   실제 프레임 크기를 사용합니다.", force_flush=True)
        width = actual_width
        height = actual_height
    else:
        width = width_meta
        height = height_meta
        log_print(f"✅ 메타데이터와 실제 프레임 크기 일치: {width}x{height}", force_flush=True)
    
    # 프레임 카운트가 0이거나 부정확할 수 있으므로 실제로 카운트
    if total_frames_meta <= 0:
        log_print(f"⚠️ 메타데이터에서 프레임 수를 가져올 수 없습니다. 실제로 카운트합니다...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        total_frames = frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음으로 되돌리기
    else:
        total_frames = total_frames_meta
        # 첫 프레임을 읽었으므로 다시 처음으로 돌아가기
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # OpenCV 버전 및 백엔드 정보 (디버깅용)
    opencv_version = cv2.__version__
    try:
        backend = cap.getBackendName()
    except:
        backend = "Unknown"
    
    # 파일 크기 확인
    file_size_mb = "확인 불가"
    file_size_bytes = 0
    if os.path.exists(video_path):
        try:
            file_size_bytes = os.path.getsize(video_path)
            file_size_mb = f"{file_size_bytes / (1024*1024):.2f} MB"
        except:
            file_size_mb = "확인 불가"
    
    log_print(f"\n📹 비디오 정보:", force_flush=True)
    log_print(f"  파일: {video_path}", force_flush=True)
    log_print(f"  파일명: {os.path.basename(video_path)}", force_flush=True)
    log_print(f"  파일 크기: {file_size_mb} ({file_size_bytes:,} bytes)", force_flush=True)
    log_print(f"  해상도 (사용): {width}x{height}", force_flush=True)
    log_print(f"  해상도 (메타데이터): {width_meta}x{height_meta}", force_flush=True)
    log_print(f"  해상도 (실제 프레임): {actual_width}x{actual_height}", force_flush=True)
    log_print(f"  FPS: {fps:.2f}", force_flush=True)
    log_print(f"  총 프레임 (사용): {total_frames}", force_flush=True)
    log_print(f"  총 프레임 (메타데이터): {total_frames_meta}", force_flush=True)
    log_print(f"  OpenCV 버전: {opencv_version}", force_flush=True)
    log_print(f"  비디오 백엔드: {backend}", force_flush=True)
    
    # 출력 비디오 설정
    output_video_path = os.path.join(output_run_dir, f"output_{run_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 추적 초기화
    qr_tracker = QRTracker(max_missed_frames=10, iou_threshold=0.15, center_dist_threshold=1.2, 
                           linear_motion_boost=True)
    use_tracking = True
    base_detection_interval = 1
    max_detection_interval = 1
    detection_interval = base_detection_interval
    last_detection_frame = 0
    
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
            log_count = 0
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
    tracking_stats = {
        'total_tracks': 0,
        'active_tracks': 0,
        'predicted_frames': 0
    }
    
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
    dynamsoft_method_stats = {
        "원본(흰테두리)": 0,
        "반전(정규화후,검은테두리)": 0
    }
    
    log_print(f"\n🎬 영상 처리 시작!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_print("\n📺 영상 재생 완료!")
                break
            
            frame_count += 1
            
            # QR 코드 탐지
            detected = False
            unique_qrs = []
            
            should_detect = (frame_count - last_detection_frame) >= detection_interval
            
            if use_tracking and not should_detect:
                tracked_qrs = []
                for track_id, track in qr_tracker.tracks.items():
                    if track.missed_frames <= qr_tracker.max_missed_frames:
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
                
                if tracked_qrs:
                    for qr in tracked_qrs:
                        track_id = qr.get('track_id')
                        if track_id is not None and decode_results is not None:
                            with decode_lock:
                                if track_id in decode_results:
                                    decode_result = decode_results[track_id]
                                    if decode_result.get('text'):
                                        qr['text'] = decode_result['text']
                                        qr['success'] = True
                                        decode_method = decode_result.get('decode_method', 'Unknown')
                                        qr['method'] = f"YOLO+{decode_method}"
                    
                    unique_qrs = tracked_qrs
                    detected = True
            
            if should_detect:
                current_success = 0
                current_failed = 0
                
                try:
                    start_time = time.time()
                    
                    single_frame, scales = create_single_frame(frame)
                    
                    if use_yolo_mode and yolo_model is not None:
                        filtered_locations = process_frame_with_yolo(single_frame, yolo_model, 
                                                                      conf_threshold=conf_threshold,
                                                                      use_preprocessing=use_preprocessing,
                                                                      use_clahe=use_clahe,
                                                                      use_normalize=use_normalize,
                                                                      clahe_clip_limit=clahe_clip_limit,
                                                                      detect_both_frames=detect_both_frames,
                                                                      iou_threshold=iou_threshold)
                        
                        detected_qrs = []
                        for i, location in enumerate(filtered_locations):
                            x1, y1, x2, y2 = location['bbox']
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
                        
                        unique_qrs = detected_qrs
                    else:
                        unique_qrs = []
                    
                    if use_tracking:
                        tracked_qrs = qr_tracker.update(unique_qrs, frame_count)
                        unique_qrs = tracked_qrs
                        
                        active_count = qr_tracker.get_active_track_count()
                        tracking_stats['active_tracks'] = max(tracking_stats['active_tracks'], active_count)
                        tracking_stats['total_tracks'] = max(tracking_stats['total_tracks'], qr_tracker.next_track_id)
                        
                        predicted_count = sum(1 for qr in tracked_qrs if qr.get('predicted', False))
                        if predicted_count > 0:
                            tracking_stats['predicted_frames'] += predicted_count
                        
                        if decode_queue is not None and dbr_reader is not None:
                            for tracked_qr in tracked_qrs:
                                track_id = tracked_qr.get('track_id')
                                if track_id is not None:
                                    with decode_lock:
                                        if track_id in decode_results:
                                            decode_result = decode_results[track_id]
                                            tracked_qr['text'] = decode_result['text']
                                            tracked_qr['success'] = True
                                            decode_method = decode_result.get('decode_method', 'Unknown')
                                            tracked_qr['method'] = f"YOLO+{decode_method}"
                                            if 'detection' in tracked_qr and decode_result.get('quad_xy'):
                                                current_bbox = tracked_qr.get('bbox', tracked_qr.get('detection', {}).get('bbox_xyxy'))
                                                decode_bbox = decode_result.get('decode_bbox')
                                                
                                                if current_bbox is not None and len(current_bbox) == 4 and \
                                                   decode_bbox is not None and len(decode_bbox) == 4:
                                                    decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                                    curr_x1, curr_y1, curr_x2, curr_y2 = map(int, current_bbox)
                                                    
                                                    decode_cx = (decode_x1 + decode_x2) / 2
                                                    decode_cy = (decode_y1 + decode_y2) / 2
                                                    curr_cx = (curr_x1 + curr_x2) / 2
                                                    curr_cy = (curr_y1 + curr_y2) / 2
                                                    
                                                    dx = curr_cx - decode_cx
                                                    dy = curr_cy - decode_cy
                                                    
                                                    quad_xy_original = decode_result['quad_xy']
                                                    quad_xy_transformed = []
                                                    for qx, qy in quad_xy_original:
                                                        quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                                    tracked_qr['detection']['quad_xy'] = quad_xy_transformed
                                                else:
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
                    
                    parallel_time = time.time() - start_time
                    
                    log_print(f"\n📹 프레임 {frame_count}/{total_frames} (처리 시간: {parallel_time*1000:.1f}ms)")
                    
                    if unique_qrs:
                        log_print(f"  🔍 탐지: {len(unique_qrs)}개 QR 코드 발견")
                        
                        for idx, qr in enumerate(unique_qrs):
                            if isinstance(qr, dict) and 'meta' in qr:
                                continue
                            
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
                            
                            if success:
                                current_success += 1
                            else:
                                current_failed += 1
                            
                            method_name = qr['method']
                            original_method = method_name
                            
                            if original_method in method_stats:
                                method_stats[original_method] += 1
                            
                            if original_method in method_detection_count:
                                method_detection_count[original_method] += 1
                            if original_method in method_unique_detection_count:
                                method_unique_detection_count[original_method] += 1
                            
                            track_id = qr.get('track_id')
                            if track_id is not None and decode_results is not None:
                                with decode_lock:
                                    if track_id in decode_results:
                                        decode_result = decode_results[track_id]
                                        if decode_result.get('text'):
                                            qr['text'] = decode_result['text']
                                            qr['success'] = True
                                            decode_method = decode_result.get('decode_method', 'Unknown')
                                            qr['method'] = f"YOLO+{decode_method}"
                                            
                                            if decode_method == 'Dynamsoft' and track_id in decode_results:
                                                method_detail = decode_result.get('decode_method_detail')
                                                if method_detail and method_detail in dynamsoft_method_stats:
                                                    dynamsoft_method_stats[method_detail] += 1
                                        
                                        if 'detection' in qr and decode_result.get('quad_xy'):
                                            current_bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                                            decode_bbox = decode_result.get('decode_bbox')
                                            
                                            if current_bbox is not None and len(current_bbox) == 4 and \
                                               decode_bbox is not None and len(decode_bbox) == 4:
                                                decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                                curr_x1, curr_y1, curr_x2, curr_y2 = map(int, current_bbox)
                                                
                                                decode_cx = (decode_x1 + decode_x2) / 2
                                                decode_cy = (decode_y1 + decode_y2) / 2
                                                curr_cx = (curr_x1 + curr_x2) / 2
                                                curr_cy = (curr_y1 + curr_y2) / 2
                                                
                                                dx = curr_cx - decode_cx
                                                dy = curr_cy - decode_cy
                                                
                                                quad_xy_original = decode_result['quad_xy']
                                                quad_xy_transformed = []
                                                for qx, qy in quad_xy_original:
                                                    quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                                qr['detection']['quad_xy'] = quad_xy_transformed
                                            else:
                                                qr['detection']['quad_xy'] = decode_result['quad_xy']
                    
                    success_count += current_success
                    failed_count += current_failed
                    
                    if unique_qrs:
                        detected_count += 1
                    
                    last_detection_frame = frame_count
                except Exception as e:
                    log_print(f"⚠️ 프레임 처리 오류: {e}")
            
            # 결과 시각화
            result_frame = frame.copy()
            
            for qr in unique_qrs:
                detection = qr.get('detection')
                if detection is None:
                    continue
                
                track_id = qr.get('track_id')
                if track_id is not None and decode_results is not None:
                    with decode_lock:
                        if track_id in decode_results:
                            decode_result = decode_results[track_id]
                            if decode_result.get('text'):
                                qr['text'] = decode_result['text']
                                qr['success'] = True
                                decode_method = decode_result.get('decode_method', 'Unknown')
                                qr['method'] = f"YOLO+{decode_method}"
                            
                            if 'detection' in qr and decode_result.get('quad_xy'):
                                current_bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                                decode_bbox = decode_result.get('decode_bbox')
                                
                                if current_bbox is not None and len(current_bbox) == 4 and \
                                   decode_bbox is not None and len(decode_bbox) == 4:
                                    decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                    curr_x1, curr_y1, curr_x2, curr_y2 = map(int, current_bbox)
                                    
                                    decode_cx = (decode_x1 + decode_x2) / 2
                                    decode_cy = (decode_y1 + decode_y2) / 2
                                    curr_cx = (curr_x1 + curr_x2) / 2
                                    curr_cy = (curr_y1 + curr_y2) / 2
                                    
                                    dx = curr_cx - decode_cx
                                    dy = curr_cy - decode_cy
                                    
                                    quad_xy_original = decode_result['quad_xy']
                                    quad_xy_transformed = []
                                    for qx, qy in quad_xy_original:
                                        quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                    qr['detection']['quad_xy'] = quad_xy_transformed
                                else:
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
            
            # 비디오 저장
            out_video.write(result_frame)
            
            # 코랩에서 프리뷰 표시 (일정 간격마다)
            if show_preview and frame_count % preview_interval == 0:
                try:
                    clear_output(wait=True)
                    display_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(display_frame)
                    plt.axis('off')
                    plt.title(f'Frame {frame_count}/{total_frames} - Detected: {len(unique_qrs)} QR codes', 
                             fontsize=14)
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    print(f"프레임 {frame_count}/{total_frames} 처리 중... (탐지: {len(unique_qrs)}개)")
                except Exception as e:
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
    
    # 최종 통계
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
    
    # 파일 닫기
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
        display(Video(output_video_path, width=800))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python yolo_dynamsoft_colab3.py <비디오_파일_경로>")
        print("\n코랩에서 사용 예시:")
        print("  video_player_with_qr('video.mp4', output_dir='results', show_preview=True)")
        print("\n옵션:")
        print("  --preprocessing: 전처리 사용")
        print("  --no-clahe: CLAHE 사용 안 함")
        print("  --no-normalize: 정규화 사용 안 함")
        print("  --clahe-clip-limit 2.0: CLAHE clipLimit 값")
        print("  --conf 0.25: YOLO 신뢰도 임계값")
        print("  --iou 0.5: 겹침 임계값")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 간단한 옵션 파싱 (argparse 대신)
    use_preprocessing = '--preprocessing' in sys.argv
    use_clahe = '--no-clahe' not in sys.argv if use_preprocessing else False
    use_normalize = '--no-normalize' not in sys.argv if use_preprocessing else False
    
    clahe_clip_limit = 2.0
    conf_threshold = 0.25
    iou_threshold = 0.5
    
    for i, arg in enumerate(sys.argv):
        if arg == '--clahe-clip-limit' and i + 1 < len(sys.argv):
            clahe_clip_limit = float(sys.argv[i + 1])
        elif arg == '--conf' and i + 1 < len(sys.argv):
            conf_threshold = float(sys.argv[i + 1])
        elif arg == '--iou' and i + 1 < len(sys.argv):
            iou_threshold = float(sys.argv[i + 1])
    
    detect_both_frames = True
    
    video_player_with_qr(
        video_path=video_path,
        output_dir="video_player_results",
        use_preprocessing=use_preprocessing,
        use_clahe=use_clahe,
        use_normalize=use_normalize,
        clahe_clip_limit=clahe_clip_limit,
        detect_both_frames=detect_both_frames,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        show_preview=True
    )

