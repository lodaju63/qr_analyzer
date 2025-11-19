"""
l.pt 모델 파일 동작 확인 스크립트
영상 및 이미지 테스트 지원
"""

import cv2
import numpy as np
import os
from pathlib import Path
import torch
import time
import datetime
import threading
from queue import Queue, Empty

# Ultralytics YOLO 모델 로드 시도
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics를 사용할 수 없습니다.")

# QReader import (정확한 QR 위치 탐지용)
try:
    from qreader import QReader
    QREADER_AVAILABLE = True
except ImportError:
    QREADER_AVAILABLE = False
    print("⚠️ QReader를 사용할 수 없습니다. pip install qreader로 설치하세요.")

# 표시용 설정
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------
# ★★★★★ IoU 기반 중복 제거 함수 ★★★★★
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


def calculate_center_distance(bbox1, bbox2):
    """두 바운딩 박스의 중심점 간 거리 계산 (정규화)"""
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
    # bbox_xyxy 사용 (YOLO 탐지 결과)
    if 'bbox' in detection:
        bbox = detection['bbox']
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return center_x, center_y, x1, y1, x2, y2
    
    # quad_xy가 있으면 사용
    if 'quad_xy' in detection:
        quad = detection['quad_xy']
        if len(quad) == 4:
            quad_array = np.array(quad)
            center = np.mean(quad_array, axis=0)
            x_coords = quad_array[:, 0]
            y_coords = quad_array[:, 1]
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)
            return center[0], center[1], x1, y1, x2, y2
    
    # bbox_xyxy 사용 (fallback)
    if 'bbox_xyxy' in detection:
        bbox = detection['bbox_xyxy']
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y, x1, y1, x2, y2
    
    return None, None, None, None, None, None


def filter_overlapping_detections(detections, iou_threshold=0.5):
    """
    겹치는 탐지 결과 제거 (NMS와 유사)
    위치 기반 중복 제거 (텍스트 기반 아님)
    """
    if not detections:
        return []
    
    # 신뢰도(confidence) 기준으로 정렬 (높은 것이 우선)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    for detection in detections:
        is_overlapping = False
        bbox1 = detection['bbox']
        
        # bbox가 리스트인 경우 튜플로 변환
        if isinstance(bbox1, (list, np.ndarray)):
            if len(bbox1) == 4:
                bbox1 = (bbox1[0], bbox1[1], bbox1[2], bbox1[3])
            else:
                continue
        
        for filtered in filtered_detections:
            bbox2 = filtered['bbox']
            if isinstance(bbox2, (list, np.ndarray)):
                if len(bbox2) == 4:
                    bbox2 = (bbox2[0], bbox2[1], bbox2[2], bbox2[3])
                else:
                    continue
            
            iou = calculate_iou(bbox1, bbox2)
            
            if iou > iou_threshold:
                is_overlapping = True
                break
        
        if not is_overlapping:
            filtered_detections.append(detection)
    
    return filtered_detections


# -----------------------------------------------------------------
# ★★★★★ 프레임 간 추적 기능 ★★★★★
# -----------------------------------------------------------------
class QRTrack:
    """단일 QR 코드 추적 정보"""
    def __init__(self, track_id, qr_data, frame_number):
        self.track_id = track_id
        self.qr_data = qr_data  # {'text': str, 'detection': dict, 'bbox': list, ...}
        self.frame_number = frame_number
        self.last_seen_frame = frame_number
        self.missed_frames = 0
        self.history = []  # 위치 이력 [(x1, y1, x2, y2), ...]
        
        # 위치 정보 추출
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr_data)
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
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr_data)
        if center_x is not None:
            self.bbox = (x1, y1, x2, y2)
            self.center = (center_x, center_y)
            self.history.append(self.bbox)
            # 최근 10개만 유지
            if len(self.history) > 10:
                self.history.pop(0)
    
    def predict_position(self):
        """이전 위치 기반으로 다음 위치 예측"""
        if self.bbox is None:
            return None
        
        if len(self.history) < 2:
            return self.bbox
        
        # 최근 2개 위치로 속도 계산
        prev_bbox = self.history[-2]
        curr_bbox = self.history[-1]
        
        # 속도 계산 (중심점 기준)
        prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
        curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
        curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
        
        vx = curr_center_x - prev_center_x
        vy = curr_center_y - prev_center_y
        
        # missed_frames를 고려하여 예측
        frames_to_predict = self.missed_frames + 1
        predicted_center_x = curr_center_x + vx * frames_to_predict
        predicted_center_y = curr_center_y + vy * frames_to_predict
        
        # 박스 크기 유지
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
    def __init__(self, max_missed_frames=5, iou_threshold=0.2, center_dist_threshold=0.8):
        self.tracks = {}  # {track_id: QRTrack}
        self.next_track_id = 0
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.center_dist_threshold = center_dist_threshold
    
    def update(self, detected_qrs, frame_number):
        """탐지된 QR 코드들과 추적 중인 QR 코드들을 매칭하여 업데이트"""
        # 1. 탐지된 QR 코드들의 bbox 추출
        detected_bboxes = []
        for qr in detected_qrs:
            center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(qr)
            if center_x is not None:
                detected_bboxes.append({
                    'qr': qr,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
        
        # 2. 활성 추적 목록
        active_tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.missed_frames <= self.max_missed_frames
        }
        
        # 3. 매칭 점수 계산
        matched_detections = set()
        matched_tracks = set()
        match_scores = []
        
        for track_id, track in active_tracks.items():
            if track.bbox is None:
                continue
            
            # 예측 위치 계산
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
                
                # 동적 임계값
                dynamic_iou_threshold = self.iou_threshold * (1.0 - track.missed_frames * 0.1)
                dynamic_iou_threshold = max(0.1, dynamic_iou_threshold)
                
                # 매칭 조건
                if (iou >= dynamic_iou_threshold or 
                    center_dist <= self.center_dist_threshold or 
                    text_match):
                    
                    # 복합 점수 계산
                    if text_match:
                        score = 1000.0 + iou * 100
                    else:
                        score = iou * 100 + (1.0 - center_dist) * 50
                    
                    match_scores.append((track_id, idx, score, iou, center_dist, text_match))
        
        # 점수 순으로 정렬
        match_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 최적 매칭 수행
        for track_id, detection_idx, score, iou, center_dist, text_match in match_scores:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            
            # 매칭 성공: 추적 업데이트
            track = active_tracks[track_id]
            det = detected_bboxes[detection_idx]
            
            # 탐지된 QR 정보가 더 정확하면 업데이트
            if not track.qr_data.get('text') or det['qr'].get('text'):
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
        
        # 6. 추적 결과 반환
        tracked_qrs = []
        
        # 매칭된 탐지 및 새로 생성된 추적
        for idx, det in enumerate(detected_bboxes):
            if idx in matched_detections:
                # 매칭된 track_id 찾기
                track_id = None
                for tid, didx, _, _, _, _ in match_scores:
                    if didx == idx and tid in matched_tracks:
                        track_id = tid
                        break
                
                if track_id is not None:
                    qr = det['qr'].copy()
                    qr['track_id'] = track_id
                    qr['tracked'] = True
                    qr['predicted'] = False
                    tracked_qrs.append(qr)
            elif idx not in matched_detections:
                # 새로 생성된 추적 (여기서는 처리하지 않음. 4단계에서 이미 tracks에 추가됨)
                pass
        
        # 매칭되지 않은 추적 (예측 위치 사용)
        for track_id, track in active_tracks.items():
            if track_id not in matched_tracks and track.missed_frames <= self.max_missed_frames:
                predicted_bbox = track.predict_position()
                if predicted_bbox is not None:
                    tracked_qr = track.qr_data.copy()
                    tracked_qr['track_id'] = track_id
                    tracked_qr['tracked'] = True
                    tracked_qr['predicted'] = True
                    tracked_qr['missed_frames'] = track.missed_frames
                    
                    # detection에 예측 위치 추가
                    if 'bbox' not in tracked_qr:
                        tracked_qr['bbox'] = list(predicted_bbox)
                    
                    tracked_qrs.append(tracked_qr)
        
        # 새로 생성된 트래킹도 최종 목록에 포함
        newly_created_track_ids = [
            tid for tid in self.tracks 
            if self.tracks[tid].frame_number == frame_number and tid not in matched_tracks
        ]
        for tid in newly_created_track_ids:
            track = self.tracks[tid]
            tracked_qr = track.qr_data.copy()
            tracked_qr['track_id'] = tid
            tracked_qr['tracked'] = True
            tracked_qr['predicted'] = False
            tracked_qrs.append(tracked_qr)
        
        return tracked_qrs
    
    def get_active_track_count(self):
        """활성 추적 개수 반환"""
        return len([t for t in self.tracks.values() if t.missed_frames <= self.max_missed_frames])


def test_model_info(model_path='l.pt'):
    """모델 정보 확인"""
    print("=" * 60)
    print(f"📦 모델 파일: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📊 파일 크기: {file_size:.2f} MB")
    
    # Ultralytics YOLO 모델로 로드 시도
    if ULTRALYTICS_AVAILABLE:
        try:
            print("\n🔍 Ultralytics YOLO 모델로 로드 시도...")
            model = YOLO(model_path)
            
            print(f"✅ 모델 타입: YOLO (Ultralytics)")
            print(f"📋 모델 정보:")
            print(f"   - Task: {model.task if hasattr(model, 'task') else 'Unknown'}")
            print(f"   - Classes: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
            
            if hasattr(model, 'names'):
                print(f"   - 클래스 목록:")
                for idx, name in model.names.items():
                    print(f"     [{idx}] {name}")
            
            if hasattr(model, 'model'):
                model_info = model.model
                print(f"   - 모델 구조: {type(model_info).__name__}")
            
            return model, 'yolo'
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
    
    # 일반 PyTorch 모델로 로드 시도
    try:
        print("\n🔍 PyTorch 모델로 로드 시도...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✅ 모델 타입: PyTorch")
        print(f"📋 체크포인트 키:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"   - {key}: (dict with {len(checkpoint[key])} keys)")
                if len(checkpoint[key].keys()) < 10:
                    for subkey in checkpoint[key].keys():
                        print(f"     - {subkey}: {type(checkpoint[subkey]).__name__}")
            elif isinstance(checkpoint[key], (list, tuple)):
                print(f"   - {key}: ({type(checkpoint[key]).__name__} with {len(checkpoint[key])} items)")
            else:
                print(f"   - {key}: {type(checkpoint[key]).__name__}")
        
        return checkpoint, 'pytorch'
    except Exception as e:
        print(f"❌ PyTorch 모델 로드 실패: {e}")
        return None, None

def test_yolo_detection(model, image_path, conf_threshold=0.25):
    """YOLO 모델로 이미지 탐지 테스트"""
    print(f"\n🖼️  이미지 테스트: {os.path.basename(image_path)}")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    print(f"   이미지 크기: {image.shape[1]}x{image.shape[0]}")
    
    # 탐지 실행
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        result = results[0]
        
        print(f"   탐지된 객체 수: {len(result.boxes) if result.boxes is not None else 0}")
        
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = result.names[cls] if hasattr(result, 'names') else f"Class_{cls}"
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': xyxy,
                    'class_id': cls
                })
                
                print(f"   [{i+1}] {class_name}: {conf:.2%} at [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
        
        return {
            'image': image,
            'detections': detections,
            'result': result
        }
    except Exception as e:
        print(f"❌ 탐지 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_results(image, detections, save_path=None):
    """탐지 결과 시각화"""
    if detections is None or len(detections) == 0:
        print("   ⚠️ 탐지된 객체가 없습니다.")
        return
    
    # 이미지 복사
    vis_image = image.copy()
    
    # BGR to RGB 변환
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    # Matplotlib으로 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(vis_image_rgb)
    
    # 바운딩 박스 그리기
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # 박스 그리기
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 라벨 추가
        label = f"{det['class']}: {det['confidence']:.2%}"
        ax.text(x1, y1-5, label, color='red', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 결과 저장: {save_path}")
    
    plt.show()

def test_video_detection(model, video_path, conf_threshold=0.25, 
                        frame_interval=1, show_video=True, save_output=True,
                         process_scale=1.0, enable_decode=True, qreader=None,
                         use_qreader_detect=False, qreader_detect_interval=5,
                         use_tracking=True):
    """YOLO 모델로 영상 탐지 테스트 (추적 + 비동기 해독 최적화 버전)
    
    Args:
        model: YOLO 모델
        video_path: 비디오 파일 경로
        conf_threshold: 신뢰도 임계값
        frame_interval: 탐지 간격 (1=모든 프레임, 2=2프레임마다, 5=5프레임마다)
        show_video: 화면 표시 여부
        save_output: 결과 영상 저장 여부
        process_scale: 처리 해상도 스케일 (1.0=원본, 0.5=50%, 0.25=25%)
        enable_decode: 해독 기능 사용 여부 (기본: True)
        qreader: QReader 인스턴스 (해독 사용 시 필요, None이면 자동 생성)
        use_qreader_detect: QReader의 detect()로 정확한 QR 위치 탐지 여부 (기본: False, 느림)
        qreader_detect_interval: QReader detect() 실행 간격 (N프레임마다, 기본: 5)
        use_tracking: 프레임 간 추적 기능 사용 여부 (기본: True, 끊김 없는 시각화)
    
    Note:
        - 탐지: 동기 처리 (빠름, 원본 속도 유지)
        - 해독: 비동기 처리 (느림, 백그라운드에서 처리)
        - 추적: 프레임 간 추적으로 끊김 없는 시각화
        - 중복 제거: IoU 기반 (위치 기반)
        - 시각화: 해독 성공=초록색, 해독 실패=빨간색
    """
    print(f"\n🎬 영상 테스트: {os.path.basename(video_path)}")
    print("=" * 60)
    
    # 비디오 파일 확인
    if not os.path.exists(video_path):
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
        return None
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {video_path}")
        return None
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # 처리 해상도 설정
    process_width = int(width * process_scale)
    process_height = int(height * process_scale)
    scale_x = width / process_width if process_width > 0 else 1.0
    scale_y = height / process_height if process_height > 0 else 1.0
    
    print(f"📹 영상 정보:")
    print(f"   원본 해상도: {width}x{height}")
    print(f"   처리 해상도: {process_width}x{process_height} (스케일: {process_scale*100:.0f}%)")
    print(f"   FPS: {fps:.2f}")
    print(f"   총 프레임: {total_frames}")
    print(f"   길이: {duration:.2f}초")
    print(f"   탐지 간격: {frame_interval}프레임마다")
    
    # 출력 설정
    output_dir = Path('l_pt_test_results')
    output_dir.mkdir(exist_ok=True)
    
    video_output_path = None
    out_video = None
    if save_output:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_output_path = output_dir / f"video_result_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        print(f"   출력 파일: {video_output_path}")
    
    # 통계
    frame_count = 0
    detection_count = 0
    total_detections = 0
    detection_times = []
    detections_per_frame = []
    frame_processing_times = []  # 각 프레임 처리 시간 (탐지 + 시각화 등)
    
    # 마지막 탐지 결과 저장 (다음 탐지 전까지 표시)
    last_detections = []
    last_qreader_detect_frame = 0  # QReader detect() 마지막 실행 프레임
    
    # 로그 파일
    log_file_path = output_dir / f"video_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    # QReader 인스턴스 생성 (해독용)
    if enable_decode and qreader is None and QREADER_AVAILABLE:
        try:
            qreader = QReader()
            log_print("✅ QReader 인스턴스 생성 (해독용)")
        except Exception as e:
            log_print(f"⚠️ QReader 초기화 실패: {e}")
            enable_decode = False
    
    # QReader 인스턴스 생성 (정확한 QR 위치 탐지용)
    qreader_detector = None
    if use_qreader_detect and QREADER_AVAILABLE:
        try:
            qreader_detector = QReader()
            log_print(f"✅ QReader detect() 활성화 (정확한 QR 위치 탐지, {qreader_detect_interval}프레임마다)")
            log_print(f"   ⚠️ 주의: detect()는 느리므로 간격을 조정하여 사용하세요.")
        except Exception as e:
            log_print(f"⚠️ QReader 초기화 실패: {e}")
            qreader_detector = None
    elif use_qreader_detect and not QREADER_AVAILABLE:
        log_print("⚠️ QReader를 사용할 수 없습니다. 바운딩 박스만 표시됩니다.")
    
    # 추적 기능 초기화
    qr_tracker = None
    if use_tracking:
        qr_tracker = QRTracker(max_missed_frames=5, iou_threshold=0.2, center_dist_threshold=0.8)
        log_print("✅ 추적 기능 활성화 (끊김 없는 시각화)")
    
    log_print(f"영상 테스트 시작: {video_path}")
    log_print(f"설정: conf_threshold={conf_threshold}, frame_interval={frame_interval}, process_scale={process_scale}")
    log_print(f"해상도: 원본 {width}x{height} → 처리 {process_width}x{process_height}")
    log_print(f"모드: 동기 탐지 처리 (최적화)")
    if enable_decode:
        log_print(f"해독: 비동기 처리 활성화")
    log_print("-" * 60)
    
    # ★★★★★ 해독용 비동기 처리 설정 ★★★★★
    decode_queue = None
    decode_results = {}  # {track_id: {'text': str, 'frame': int, 'quad_xy': list, 'decode_bbox': list}} - 해독 결과 저장
    decode_worker_thread = None
    stop_decode_worker = None
    
    if enable_decode and qreader is not None:
        decode_queue = Queue(maxsize=10)
        stop_decode_worker = threading.Event()
        decode_lock = threading.Lock()
        
        def decode_worker():
            """백그라운드에서 해독 수행하는 워커 스레드"""
            # 외부 변수 frame_count를 참조할 수 없으므로, 초기 로그를 위한 임시 track_id 변수 사용
            log_count = 0 
            
            while not stop_decode_worker.is_set():
                try:
                    item = decode_queue.get(timeout=0.1)
                    if item is None:
                        # 큐에 None이 들어오면 스레드 종료
                        return 
                
                    track_id, roi, bbox, roi_offset = item  # roi_offset 추가: (roi_x1, roi_y1)
                    try:
                        # QReader로 해독 시도 (detect() 먼저 호출하여 성공률 향상)
                        decoded_text = None
                        quad_xy = None
                        detections = qreader.detect(roi)
                        
                        if detections and len(detections) > 0:
                            # detect()로 찾은 힌트를 사용하여 decode()
                            detection = detections[0]
                            decoded_text = qreader.decode(roi, detection)
                            
                            # quad_xy 추출 (ROI 내 상대 좌표)
                            if 'quad_xy' in detection:
                                quad_xy_roi = detection['quad_xy']
                                if len(quad_xy_roi) == 4:
                                    # ROI 내 상대 좌표를 원본 이미지 절대 좌표로 변환
                                    roi_x1, roi_y1 = roi_offset
                                    quad_xy = []
                                    for qx, qy in quad_xy_roi:
                                        abs_x = roi_x1 + int(qx)
                                        abs_y = roi_y1 + int(qy)
                                        quad_xy.append([abs_x, abs_y])
                            else:
                                # detect() 실패 시 직접 decode() 시도
                                decoded_text = qreader.decode(roi)
                            
                            if decoded_text:
                                with decode_lock:
                                    decode_results[track_id] = {
                                        'text': decoded_text,
                                        # 'frame': frame_count, # 외부 변수 참조 불가
                                        'quad_xy': quad_xy,  # 정확한 QR 위치 (4개 꼭짓점)
                                        'decode_bbox': list(bbox)  # 해독 시점의 bbox (위치 변환용)
                                    }
                                # 디버깅: 해독 성공 로그 (처음 몇 개만)
                                if log_count < 10:
                                    log_print(f"✅ 해독 성공 [T{track_id}]: {decoded_text[:50]}")
                                    log_count += 1
                        
                    except Exception as e:
                        # 해독 실패 시 무시 (너무 많은 로그 방지)
                        # 처음 몇 개만 로그 출력
                        if log_count < 3 and track_id <= 3:
                            log_print(f"⚠️ 해독 실패 [T{track_id}]: {str(e)[:50]}")
                            log_count += 1
                        pass
                    
                    decode_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    log_print(f"해독 워커 오류: {e}")
                    # item이 정의되었고 None이 아니며, 큐에 들어있는 작업이라면 task_done 호출
                    if 'item' in locals() and item:
                        decode_queue.task_done()
        
        decode_worker_thread = threading.Thread(target=decode_worker, daemon=True)
        decode_worker_thread.start()
        log_print("✅ 해독 워커 스레드 시작")
    
    print(f"\n▶️  영상 처리 시작... (동기 탐지 + {'비동기 해독' if enable_decode else '해독 없음'})")
    start_time = time.time()
    last_detection_frame = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps if fps > 0 else 0
            
            # 프레임 처리 시작 시간
            frame_start_time = time.time()
            
            # ★★★★★ 동기 탐지 처리 (빠름) ★★★★★
            should_detect = (frame_count - last_detection_frame) >= frame_interval or frame_count == 1
            
            if should_detect:
                # 처리용 해상도로 축소
                if process_scale < 1.0:
                    process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
                else:
                    process_frame = frame
                
                # YOLO 탐지 수행 (동기)
                detect_start = time.time()
                results = model(process_frame, conf=conf_threshold, verbose=False)
                detect_time = time.time() - detect_start
                detection_times.append(detect_time)
                
                result = results[0]
                detections = []
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # 원본 해상도 좌표로 변환
                        if process_scale < 1.0:
                            xyxy = [
                                xyxy[0] * scale_x,
                                xyxy[1] * scale_y,
                                xyxy[2] * scale_x,
                                xyxy[3] * scale_y
                            ]
                        
                        # ★★★★★ 패딩 추가 (video_synch.py와 동일하게) ★★★★★
                        # 패딩을 처음부터 추가하여 더 정확한 ROI 확보
                        pad = 20  # video_synch.py와 동일한 패딩 크기
                        x1, y1, x2, y2 = xyxy
                        h, w = frame.shape[:2]
                        x1 = max(0, int(x1 - pad))
                        y1 = max(0, int(y1 - pad))
                        x2 = min(w, int(x2 + pad))
                        y2 = min(h, int(y2 + pad))
                        xyxy = [x1, y1, x2, y2]
                        
                        class_name = result.names[cls] if hasattr(result, 'names') else f"Class_{cls}"
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': xyxy,
                            'class_id': cls
                        })
                
                    # ★★★★★ IoU 기반 중복 제거 ★★★★★
                    filtered_detections = filter_overlapping_detections(detections, iou_threshold=0.5)
                    
                    if len(detections) > len(filtered_detections):
                        log_print(f"    ⚡ 중복 제거: {len(detections)}개 → {len(filtered_detections)}개")
                    
                    # ★★★★★ QReader detect()로 정확한 QR 위치 탐지 (quad_xy) ★★★★★
                    # 성능 최적화: 일정 간격으로만 실행 (기본: 5프레임마다)
                    should_run_qreader_detect = (
                        qreader_detector is not None and 
                        filtered_detections and
                        (frame_count - last_qreader_detect_frame) >= qreader_detect_interval
                    )
                    
                    if should_run_qreader_detect:
                        for det in filtered_detections:
                            x1, y1, x2, y2 = map(int, det['bbox'])
                            # ROI 추출 (이미 패딩이 포함된 bbox이므로 그대로 사용)
                            roi = frame[y1:y2, x1:x2]
                            
                            if roi.size > 0:
                                try:
                                    # QReader의 detect()로 정확한 위치 찾기
                                    qr_detections = qreader_detector.detect(roi)
                                    if qr_detections and len(qr_detections) > 0:
                                        detection = qr_detections[0]
                                        if 'quad_xy' in detection:
                                            # ROI 내 상대 좌표를 원본 이미지 절대 좌표로 변환
                                            # video_synch.py와 동일: 패딩이 포함된 bbox 좌표 사용
                                            quad_xy = []
                                            for qx, qy in detection['quad_xy']:
                                                abs_x = x1 + int(qx)
                                                abs_y = y1 + int(qy)
                                                quad_xy.append([abs_x, abs_y])
                                            det['quad_xy'] = quad_xy
                                except Exception as e:
                                    # detect 실패 시 무시 (바운딩 박스만 사용)
                                    pass
                        last_qreader_detect_frame = frame_count
                    
                    # 통계 업데이트
                    if filtered_detections:
                        frame_detections_count = len(filtered_detections)
                        detections_per_frame.append(frame_detections_count)
                        detection_count += 1
                    total_detections += len(filtered_detections)
                    
                    # 로그 출력
                    if filtered_detections:
                        log_print(f"프레임 {frame_count} ({current_time:.2f}초): {frame_detections_count}개 탐지")
                        for i, det in enumerate(filtered_detections):
                            log_print(f"  [{i+1}] {det['class']}: {det['confidence']:.2%} "
                                        f"at [{int(det['bbox'][0])}, {int(det['bbox'][1])}, "
                                        f"{int(det['bbox'][2])}, {int(det['bbox'][3])}]")
                    
                    # ★★★★★ 추적 기능 적용 ★★★★★
                    if use_tracking and qr_tracker is not None:
                        # 탐지 결과를 추적 형식으로 변환
                        tracked_qr_list = []
                        for det in filtered_detections:
                            qr_data = {
                                'bbox': det['bbox'],
                                'quad_xy': det.get('quad_xy'), # QReader detect 결과 전달
                                'text': '',  # 아직 해독 안됨
                                'detection': {
                                    'bbox_xyxy': det['bbox'],
                                    'quad_xy': det.get('quad_xy')  # detection에도 quad_xy 포함 (추적에 유지)
                                }
                            }
                            tracked_qr_list.append(qr_data)
                        
                        # 추적 업데이트
                        tracked_qrs = qr_tracker.update(tracked_qr_list, frame_count)
                        
                        # 해독 큐에 추가 (비동기 해독)
                        if enable_decode and decode_queue is not None:
                            for tracked_qr in tracked_qrs:
                                track_id = tracked_qr.get('track_id')
                                if track_id is not None:
                                    # 이미 해독된 것은 스킵 (하지만 quad_xy는 업데이트)
                                    with decode_lock:
                                        if track_id in decode_results:
                                            decode_result = decode_results[track_id]
                                            tracked_qr['text'] = decode_result['text']
                                            
                                            # quad_xy가 있으면 추적 위치에 맞춰서 변환
                                            if 'quad_xy' in decode_result and decode_result['quad_xy'] is not None:
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
                                                    tracked_qr['quad_xy'] = quad_xy_transformed
                                                else:
                                                    # bbox 정보가 없으면 원본 quad_xy 사용
                                                    tracked_qr['quad_xy'] = decode_result['quad_xy']
                                            continue # 이미 해독된 것은 큐에 다시 넣지 않음
                                    
                                    # ROI 추출하여 해독 큐에 추가
                                    bbox = tracked_qr.get('bbox', tracked_qr.get('detection', {}).get('bbox_xyxy'))
                                    if bbox is not None and len(bbox) == 4:
                                        x1, y1, x2, y2 = map(int, bbox)
                                        # 이미 패딩이 포함된 bbox이므로 그대로 사용 (video_synch.py와 동일)
                                        roi = frame[y1:y2, x1:x2]
                                        if roi.size > 0:
                                            try:
                                                # ROI 오프셋 정보도 함께 전달 (quad_xy 좌표 변환용)
                                                # video_synch.py와 동일: 패딩이 포함된 bbox 좌표 사용
                                                decode_queue.put_nowait((track_id, roi, bbox, (x1, y1)))
                                                # 디버깅: 큐 추가 로그 (처음 몇 개만)
                                                if track_id <= 3 and len(decode_results) < 5:
                                                    log_print(f"📤 해독 큐 추가 [T{track_id}] (ROI 크기: {roi.shape})")
                                            except:
                                                # 큐가 가득 차면 스킵
                                                if track_id <= 3 and len(decode_results) < 5:
                                                    log_print(f"⚠️ 해독 큐 가득참 [T{track_id}]")
                                                pass
                        
                        last_detections = tracked_qrs
                    else:
                        # 추적 없이 탐지 결과만 사용
                        last_detections = filtered_detections.copy()
                
                last_detection_frame = frame_count
            else:
                # 탐지하지 않는 프레임: 추적 결과 사용
                if use_tracking and qr_tracker is not None:
                    # 추적만 사용 (탐지 없이)
                    tracked_qrs = []
                    for track_id, track in qr_tracker.tracks.items():
                        if track.missed_frames <= qr_tracker.max_missed_frames:
                            predicted_bbox = track.predict_position()
                            if predicted_bbox is not None:
                                tracked_qr = track.qr_data.copy()
                                tracked_qr['track_id'] = track_id
                                tracked_qr['tracked'] = True
                                tracked_qr['predicted'] = True
                                tracked_qr['bbox'] = list(predicted_bbox)
                                
                                # 해독 결과 확인
                                if enable_decode:
                                    with decode_lock:
                                        if track_id in decode_results:
                                            decode_result = decode_results[track_id]
                                            tracked_qr['text'] = decode_result['text']
                                            
                                            # quad_xy 우선순위: 탐지 프레임의 quad_xy > 해독 결과의 quad_xy
                                            # 탐지 프레임에서 얻은 정확한 quad_xy가 있으면 우선 사용
                                            if 'quad_xy' not in tracked_qr or tracked_qr.get('quad_xy') is None:
                                                # 탐지 프레임의 quad_xy가 없을 때만 해독 결과의 quad_xy 사용
                                                if 'quad_xy' in decode_result and decode_result['quad_xy'] is not None:
                                                    current_bbox = tracked_qr.get('bbox')
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
                                                        tracked_qr['quad_xy'] = quad_xy_transformed
                                                    else:
                                                        # bbox 정보가 없으면 원본 quad_xy 사용
                                                        tracked_qr['quad_xy'] = decode_result['quad_xy']
                                            continue # 이미 해독된 것은 큐에 다시 넣지 않음
                                    
                                    # ROI 추출하여 해독 큐에 추가
                                    bbox = tracked_qr.get('bbox', tracked_qr.get('detection', {}).get('bbox_xyxy'))
                                    if bbox is not None and len(bbox) == 4:
                                        x1, y1, x2, y2 = map(int, bbox)
                                        # 이미 패딩이 포함된 bbox이므로 그대로 사용 (video_synch.py와 동일)
                                        roi = frame[y1:y2, x1:x2]
                                        if roi.size > 0:
                                            try:
                                                # ROI 오프셋 정보도 함께 전달 (quad_xy 좌표 변환용)
                                                # video_synch.py와 동일: 패딩이 포함된 bbox 좌표 사용
                                                decode_queue.put_nowait((track_id, roi, bbox, (x1, y1)))
                                                # 디버깅: 큐 추가 로그 (처음 몇 개만)
                                                if track_id <= 3 and len(decode_results) < 5:
                                                    log_print(f"📤 해독 큐 추가 [T{track_id}] (ROI 크기: {roi.shape})")
                                            except:
                                                # 큐가 가득 차면 스킵
                                                if track_id <= 3 and len(decode_results) < 5:
                                                    log_print(f"⚠️ 해독 큐 가득참 [T{track_id}]")
                                                pass
                                
                                tracked_qrs.append(tracked_qr)
                    last_detections = tracked_qrs
            
            # 결과 시각화 (최신 탐지 결과 사용)
            vis_frame = frame.copy()
            
            # 바운딩 박스 그리기 (해독 성공=초록색, 실패=빨간색)
            display_detections = last_detections
            for det in display_detections:
                # 해독 상태 확인 (최신 해독 결과 확인)
                track_id = det.get('track_id', None)
                has_text = det.get('text', '') != ''
                
                # 해독이 활성화되어 있고 track_id가 있으면 최신 해독 결과 확인
                if enable_decode and track_id is not None and decode_results is not None:
                    with decode_lock:
                        if track_id in decode_results:
                            decode_result = decode_results[track_id]
                            det['text'] = decode_result['text']
                            has_text = True
                            
                            # quad_xy 우선순위: 탐지 프레임의 quad_xy > 해독 결과의 quad_xy
                            # 탐지 프레임에서 얻은 정확한 quad_xy가 있으면 우선 사용
                            if 'quad_xy' not in det or det.get('quad_xy') is None:
                                # 탐지 프레임의 quad_xy가 없을 때만 해독 결과의 quad_xy 사용
                                if 'quad_xy' in decode_result and decode_result['quad_xy'] is not None:
                                    current_bbox = det.get('bbox', det.get('detection', {}).get('bbox_xyxy'))
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
                                        det['quad_xy'] = quad_xy_transformed
                                    else:
                                        # bbox 정보가 없으면 원본 quad_xy 사용
                                        det['quad_xy'] = decode_result['quad_xy']
                
                is_predicted = det.get('predicted', False)
                
                # 색상 결정: 해독 성공=초록색, 실패=빨간색
                if has_text:
                    box_color = (0, 255, 0)  # 초록색 (BGR)
                    text_color = (0, 0, 0)  # 검은색
                else:
                    box_color = (0, 0, 255)  # 빨간색 (BGR)
                    text_color = (255, 255, 255)  # 흰색
                
                # 예측 위치는 점선으로 표시
                line_type = cv2.LINE_AA
                thickness = 2
                
                # quad_xy가 있으면 정확한 4개 꼭짓점으로 그리기
                if 'quad_xy' in det and det['quad_xy'] is not None:
                    quad_xy = det['quad_xy']
                    if len(quad_xy) == 4:
                        points = np.array(quad_xy, dtype=np.int32)
                        if is_predicted:
                            # 점선 효과 (간격을 두고 선 그리기)
                            for i in range(4):
                                p1 = tuple(points[i])
                                p2 = tuple(points[(i+1)%4])
                                # 간단한 점선은 구현이 복잡하므로, 일단은 실선으로 표시 (두께를 얇게)
                                cv2.line(vis_frame, p1, p2, box_color, 1, line_type) 
                        else:
                            cv2.polylines(vis_frame, [points], True, box_color, thickness, line_type)
                        
                        # 라벨 위치는 좌측 상단 꼭짓점
                        label_x, label_y = int(quad_xy[0][0]), int(quad_xy[0][1])
                    else:
                        # quad_xy가 잘못되었으면 bbox 사용
                        bbox = det.get('bbox', det.get('detection', {}).get('bbox_xyxy', []))
                        if bbox is not None and len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, thickness, line_type)
                            label_x, label_y = x1, y1
                        else:
                            continue
                else:
                    # 바운딩 박스 사용
                    bbox = det.get('bbox', det.get('detection', {}).get('bbox_xyxy', []))
                    if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, thickness, line_type)
                        label_x, label_y = x1, y1
                    else:
                        continue
                
                # 라벨 구성
                label_parts = []
                if track_id is not None:
                    label_parts.append(f"T{track_id}")
                if is_predicted:
                    label_parts.append("P")
                if has_text:
                    text = det['text']
                    # 텍스트가 너무 길면 자르기
                    if len(text) > 30:
                        text = text[:27] + "..."
                    # OpenCV putText에서 문제가 되는 특수 문자들을 표준 하이픈으로 변경
                    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
                    text = text.replace('？', '?').replace('！', '!').replace('，', ',')
                    label_parts.append(text)
                else:
                    conf = det.get('confidence', 0)
                    class_name = det.get('class', 'QR')
                    if conf > 0:
                        label_parts.append(f"{class_name} {conf:.1%}")
                    else:
                        label_parts.append(f"{class_name} (미해독)")
                
                label = " | ".join(label_parts)
                font_scale = 0.6
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                
                # 라벨 배경
                # 라벨이 위로 튀어나가지 않도록 y 좌표 조정
                label_rect_y1 = max(0, label_y - label_size[1] - 10)
                label_rect_y2 = label_y
                label_text_y = label_y - 5
                
                cv2.rectangle(vis_frame, (label_x, label_rect_y1), 
                              (label_x + label_size[0] + 5, label_rect_y2), box_color, -1)
                cv2.putText(vis_frame, label, (label_x, label_text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
            
            # 프레임 정보 표시
            info_text = f"Frame: {frame_count}/{total_frames} | Time: {current_time:.1f}s"
            if display_detections:
                info_text += f" | Active Tracks: {qr_tracker.get_active_track_count() if qr_tracker else len(display_detections)}"
            cv2.putText(vis_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 출력 비디오에 저장
            if out_video is not None:
                out_video.write(vis_frame)
            
            # 화면에 표시
            if show_video:
                # 화면 크기에 맞게 리사이즈
                display_width = 1280
                if width > display_width:
                    scale = display_width / width
                    display_height = int(height * scale)
                    display_frame = cv2.resize(vis_frame, (display_width, display_height))
                else:
                    display_frame = vis_frame
                
                cv2.imshow('QR Detection Test (Optimized)', display_frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ 사용자가 중단했습니다.")
                    break
            
            # 프레임 처리 시간 측정
            frame_processing_time = time.time() - frame_start_time
            frame_processing_times.append(frame_processing_time)
            
            # 진행 상황 출력 (실시간 처리 속도 포함)
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                # 현재까지의 평균 처리 FPS 계산
                elapsed_time = time.time() - start_time
                current_processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                speed_ratio = (current_processing_fps / fps * 100) if fps > 0 else 0
                print(f"   진행: {progress:.1f}% ({frame_count}/{total_frames} 프레임) | "
                      f"처리 속도: {current_processing_fps:.2f} FPS (원본 {fps:.2f} FPS의 {speed_ratio:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 중단했습니다.")
    finally:
        # ★★★★★ 해독 워커 스레드 종료 ★★★★★
        if enable_decode and stop_decode_worker is not None and decode_queue is not None:
            stop_decode_worker.set()
            # 큐에 None을 넣어 워커 스레드가 종료되도록 신호 전송
            try:
                decode_queue.put_nowait(None)
            except:
                pass # 큐가 가득차도 종료 신호는 보내야 함
            if decode_worker_thread is not None:
                decode_worker_thread.join(timeout=2.0)
            log_print("✅ 해독 워커 스레드 종료")
        
        # 정리
        total_time = time.time() - start_time
        cap.release()
        if out_video is not None:
            out_video.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # 통계 출력
        print(f"\n📊 처리 완료!")
        print(f"   총 프레임: {frame_count}")
        print(f"   탐지된 프레임: {detection_count}")
        print(f"   총 탐지 수: {total_detections}")
        print(f"   처리 시간: {total_time:.2f}초")
        
        # 처리 속도 통계
        if frame_count > 0 and total_time > 0:
            actual_fps = frame_count / total_time
            print(f"\n⚡ 처리 속도 분석:")
            print(f"   원본 영상 FPS: {fps:.2f}")
            print(f"   실제 처리 FPS: {actual_fps:.2f}")
            speed_ratio = (actual_fps / fps * 100) if fps > 0 else 0
            print(f"   속도 비율: {speed_ratio:.1f}% (원본 대비)")
            if actual_fps >= fps:
                print(f"   ✅ 실시간 처리 가능! (원본보다 {actual_fps/fps:.2f}x 빠름)")
            else:
                print(f"   ⚠️ 실시간 처리 불가 (원본의 {actual_fps/fps:.2f}x 느림)")
        
        if frame_processing_times:
            avg_frame_time = np.mean(frame_processing_times)
            min_frame_time = np.min(frame_processing_times)
            max_frame_time = np.max(frame_processing_times)
            print(f"\n📈 프레임 처리 시간:")
            print(f"   평균: {avg_frame_time*1000:.2f}ms")
            print(f"   최소: {min_frame_time*1000:.2f}ms")
            print(f"   최대: {max_frame_time*1000:.2f}ms")
            if fps > 0:
                target_frame_time = 1.0 / fps
                print(f"   목표 (원본 FPS 기준): {target_frame_time*1000:.2f}ms")
        
        if detection_times:
            avg_detect_time = np.mean(detection_times)
            print(f"\n🔍 탐지 시간:")
            print(f"   평균 탐지 시간: {avg_detect_time*1000:.2f}ms")
        if detections_per_frame:
            avg_detections = np.mean(detections_per_frame)
            print(f"   프레임당 평균 탐지: {avg_detections:.2f}개")
        
        log_print("-" * 60)
        log_print(f"처리 완료")
        log_print(f"총 프레임: {frame_count}, 탐지 프레임: {detection_count}, 총 탐지: {total_detections}")
        log_print(f"처리 시간: {total_time:.2f}초")
        
        # 처리 속도 로그
        if frame_count > 0 and total_time > 0:
            actual_fps = frame_count / total_time
            log_print(f"\n⚡ 처리 속도 분석:")
            log_print(f"   원본 영상 FPS: {fps:.2f}")
            log_print(f"   실제 처리 FPS: {actual_fps:.2f}")
            speed_ratio = (actual_fps / fps * 100) if fps > 0 else 0
            log_print(f"   속도 비율: {speed_ratio:.1f}% (원본 대비)")
            if actual_fps >= fps:
                log_print(f"   ✅ 실시간 처리 가능! (원본보다 {actual_fps/fps:.2f}x 빠름)")
            else:
                log_print(f"   ⚠️ 실시간 처리 불가 (원본의 {actual_fps/fps:.2f}x 느림)")
        
        if frame_processing_times:
            avg_frame_time = np.mean(frame_processing_times)
            min_frame_time = np.min(frame_processing_times)
            max_frame_time = np.max(frame_processing_times)
            log_print(f"\n📈 프레임 처리 시간:")
            log_print(f"   평균: {avg_frame_time*1000:.2f}ms")
            log_print(f"   최소: {min_frame_time*1000:.2f}ms")
            log_print(f"   최대: {max_frame_time*1000:.2f}ms")
            if fps > 0:
                target_frame_time = 1.0 / fps
                log_print(f"   목표 (원본 FPS 기준): {target_frame_time*1000:.2f}ms")
        
        log_file.close()
        
        if video_output_path:
            print(f"   💾 결과 영상 저장: {video_output_path}")
            print(f"   📝 로그 파일: {log_file_path}")
    
    return {
        'total_frames': frame_count,
        'detection_frames': detection_count,
        'total_detections': total_detections,
        'output_video': video_output_path,
        'log_file': log_file_path
    }

def main():
    """메인 함수"""
    model_path = 'l.pt'
    video_path = r'C:\Users\Administrator\qr_sh\data\video\sample_video3-1.mp4'
    
    # 1. 모델 정보 확인
    model, model_type = test_model_info(model_path)
    
    if model is None:
        print("\n❌ 모델을 로드할 수 없습니다.")
        return
    
    # 2. YOLO 모델인 경우 테스트
    if model_type == 'yolo':
        print("\n" + "=" * 60)
        print("🎬 영상 테스트 모드")
        print("=" * 60)
        
        # 영상 테스트
        if os.path.exists(video_path):
            print(f"\n📹 영상 파일: {video_path}")
            
            # 사용자 설정
            conf_threshold = 0.25  # 신뢰도 임계값
            frame_interval = 2     # 2프레임마다 탐지 (속도 향상)
            process_scale = 1.0    # 처리 해상도 스케일 (1.0=원본, 0.5=50%, 0.25=25%)
            show_video = True      # 화면에 표시 여부
            save_output = True     # 결과 영상 저장 여부
            
            result = test_video_detection(
                model, video_path,
                conf_threshold=conf_threshold,
                frame_interval=frame_interval,
                show_video=show_video,
                save_output=save_output,
                process_scale=process_scale,
                enable_decode=True,  # 해독 활성화
                qreader=None,  # 자동 생성
                use_qreader_detect=False,  # 기본값: False (느림, 필요시 True로 변경)
                qreader_detect_interval=5,  # QReader detect() 실행 간격 (5프레임마다)
                use_tracking=True  # 추적 기능 활성화 (끊김 없는 시각화)
            )
            
            if result:
                print(f"\n✅ 영상 테스트 완료!")
        else:
            print(f"\n❌ 영상 파일을 찾을 수 없습니다: {video_path}")
            print(f"\n📁 이미지 테스트로 전환...")
            
            # 테스트 이미지 경로
            test_images_dir = Path('data/250723_test')
            
            if test_images_dir.exists():
                test_images = list(test_images_dir.glob('*.jpg'))[:5]
                
                if len(test_images) > 0:
                    print(f"\n📁 테스트 이미지: {len(test_images)}개")
                    
                    output_dir = Path('l_pt_test_results')
                    output_dir.mkdir(exist_ok=True)
                    
                    for i, img_path in enumerate(test_images):
                        print(f"\n{'='*60}")
                        result_data = test_yolo_detection(model, str(img_path))
                        
                        if result_data:
                            save_path = output_dir / f"result_{i+1}_{img_path.stem}.png"
                            visualize_results(result_data['image'], result_data['detections'], 
                                            str(save_path))
                    
                    print(f"\n✅ 테스트 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
                else:
                    print("❌ 테스트 이미지를 찾을 수 없습니다.")
            else:
                print(f"❌ 테스트 이미지 디렉토리를 찾을 수 없습니다: {test_images_dir}")
    
    elif model_type == 'pytorch':
        print("\n⚠️ 일반 PyTorch 모델입니다. YOLO 전용 테스트는 사용할 수 없습니다.")
        print("모델 구조를 확인하려면 추가 분석이 필요합니다.")

if __name__ == '__main__':
    main()