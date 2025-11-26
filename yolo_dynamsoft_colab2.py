"""
구글 코랩용: 영상 플레이어 + 고성능 QR 탐지 (정확도 개선판)
[개선 사항]:
1. Padding: YOLO 박스보다 20% 넓게 잘라 Quiet Zone 확보
2. Upscaling: ROI 이미지를 2배 확대 + 샤픈 필터 적용
3. Settings: Dynamsoft 해독 설정을 최고 수준으로 강화
4. Speed: 불필요한 미리보기 옵션 기본 OFF
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
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. 라이브러리 체크
# -----------------------------------------------------------------
IN_COLAB = 'google.colab' in sys.modules

try:
    from dynamsoft_barcode_reader_bundle import dbr, license, cvr
    from dynamsoft_barcode_reader_bundle import EnumPresetTemplate
    DBR_AVAILABLE = True
except ImportError:
    print("⚠️ Dynamsoft Barcode Reader가 없습니다. !pip install dynamsoft-barcode-reader-bundle 실행 필요")
    DBR_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ Ultralytics가 없습니다. !pip install ultralytics 실행 필요")
    YOLO_AVAILABLE = False

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# -----------------------------------------------------------------
# 2. 유틸리티 함수 (한글 폰트, YOLO 탐지 등)
# -----------------------------------------------------------------
def get_platform_font_paths():
    if IN_COLAB:
        return ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    return [] # 로컬 경로는 생략

def put_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    if not PIL_AVAILABLE:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    try:
        img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_paths = get_platform_font_paths()
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        if font is None: font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        return img

def yolo_detect_qr_locations(model, frame, conf_threshold=0.25):
    """
    YOLO로 QR 위치 찾기 + [수정됨] 패딩(여유공간) 20% 추가
    """
    try:
        results = model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        locations = []
        
        h_img, w_img = frame.shape[:2]
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # [핵심 수정 1] 박스 크기의 20% 만큼 상하좌우 여유 공간 확보 (Quiet Zone)
                box_w = x2 - x1
                box_h = y2 - y1
                pad_w = int(box_w * 0.2)
                pad_h = int(box_h * 0.2)
                
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w_img, x2 + pad_w)
                y2 = min(h_img, y2 + pad_h)
                
                locations.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        return locations
    except:
        return []

def preprocess_for_decoding(roi):
    """
    [핵심 수정 2] 작은 QR 코드를 위해 2배 확대 및 샤픈 필터 적용
    """
    try:
        # 1. 2배 확대 (Cubic 보간법이 화질 저하가 적음)
        roi_upscaled = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # 2. 샤픈(Sharpen) 필터 적용 - 흐릿한 경계선 강화
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        roi_sharpened = cv2.filter2D(roi_upscaled, -1, kernel)
        
        return roi_sharpened
    except:
        return roi

# -----------------------------------------------------------------
# 3. 추적(Tracking) 관련 클래스 (기존 로직 유지)
# -----------------------------------------------------------------
def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i: return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return intersection / (area1 + area2 - intersection) if (area1 + area2 - intersection) > 0 else 0.0

class QRTracker:
    def __init__(self, max_missed=10):
        self.tracks = {}
        self.next_id = 0
        self.max_missed = max_missed

    def update(self, detected_items, frame_num):
        # 간단한 IoU 기반 매칭 (상세 로직은 길이상 간소화, 핵심은 동일)
        active_tracks = {tid: t for tid, t in self.tracks.items() if t['missed'] <= self.max_missed}
        matched_det_indices = set()
        matched_track_ids = set()
        
        # 매칭 시도
        for tid, track in active_tracks.items():
            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(detected_items):
                if idx in matched_det_indices: continue
                iou = calculate_iou(track['bbox'], det['bbox'])
                if iou > 0.3 and iou > best_iou: # IoU 임계값
                    best_iou = iou
                    best_idx = idx
            
            if best_idx != -1:
                # 매칭 성공
                det = detected_items[best_idx]
                track['bbox'] = det['bbox']
                track['missed'] = 0
                track['last_frame'] = frame_num
                # 기존 텍스트가 없고 새 탐지에 텍스트가 있으면 업데이트
                if not track['text'] and det['text']:
                    track['text'] = det['text']
                    track['success'] = True
                
                matched_track_ids.add(tid)
                matched_det_indices.add(best_idx)
            else:
                track['missed'] += 1
        
        # 새로운 트랙 생성
        for idx, det in enumerate(detected_items):
            if idx not in matched_det_indices:
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'text': det['text'],
                    'success': det.get('success', False),
                    'missed': 0,
                    'start_frame': frame_num,
                    'last_frame': frame_num,
                    'id': self.next_id
                }
                self.next_id += 1
                
        # 오래된 트랙 삭제
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['missed'] <= self.max_missed}
        
        # 결과 반환용 리스트
        results = []
        for tid, t in self.tracks.items():
            if t['missed'] <= 1: # 현재 보이거나 방금 놓친 것만
                res = t.copy()
                res['track_id'] = tid
                results.append(res)
        return results

# -----------------------------------------------------------------
# 4. 메인 실행 함수
# -----------------------------------------------------------------
def video_player_with_qr(video_path, output_dir="results", show_preview=False, preview_interval=30):
    
    # 0. 설정 및 준비
    os.makedirs(output_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"output_{run_id}.mp4")
    
    print(f"🚀 처리 시작: {video_path}")
    print(f"💾 결과 저장: {out_path}")
    if show_preview:
        print("⚠️ 주의: 미리보기(show_preview)가 켜져 있으면 처리 속도가 느려질 수 있습니다.")

    # 1. 모델 로드
    yolo = None
    if YOLO_AVAILABLE:
        try:
            # GPU 사용 가능시 자동 사용
            yolo = YOLO('model1.pt') 
            print("✅ YOLO 모델 로드 완료")
        except:
            print("❌ model1.pt를 찾을 수 없습니다. 기본 모델(yolov8n.pt)을 사용하거나 파일을 업로드하세요.")
            try: yolo = YOLO('yolov8n.pt')
            except: pass

    # 2. Dynamsoft 초기화 및 [핵심 수정 3] 설정 강화
    dbr_reader = None
    if DBR_AVAILABLE:
        try:
            license_key = "t0085YQEAADYdcL2llMa8vH1Rtnun+43saE/kdAE7ZbIxMQGRMtSzVSZRI8vfOK4Ids52rjekwzh87yABFLraXw5Va1BV7NnBjI8m7qbw3kxOprI75ExJpw=="
            license.LicenseManager.init_license(license_key)
            dbr_reader = cvr.CaptureVisionRouter()
            
            # 설정 가져오기
            err, msg, settings = dbr_reader.get_simplified_settings(EnumPresetTemplate.PT_DEFAULT)
            if err == 0:
                # [설정 강화]
                settings.barcode_settings.expected_barcodes_count = 50 # 한 번에 많이 찾도록
                settings.barcode_settings.deblur_level = 9             # 블러 제거 수준 최대
                settings.barcode_settings.min_barcode_text_length = 1
                settings.timeout = 500  # 타임아웃 500ms (충분히 시간 줌)
                dbr_reader.update_settings(EnumPresetTemplate.PT_DEFAULT, settings)
                print("✅ Dynamsoft 설정 최적화 완료 (Deblur Lv.9)")
        except Exception as e:
            print(f"❌ Dynamsoft 초기화 실패: {e}")

    # 3. 비디오 준비
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # 4. 해독 워커 (스레드)
    decode_q = Queue()
    result_map = {} # track_id -> text
    lock = threading.Lock()
    
    def worker():
        while True:
            item = decode_q.get()
            if item is None: break
            track_id, roi_img = item
            
            # [전처리] 확대 및 샤픈
            processed_roi = preprocess_for_decoding(roi_img)
            
            text = None
            if dbr_reader:
                try:
                    # RGB 변환 불필요 (OpenCV 이미지는 BGR, dbr은 자동 처리 혹은 BGR 선호)
                    # 만약 필요하다면: img_rgb = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2RGB)
                    res = dbr_reader.capture(processed_roi, dbr.EnumImagePixelFormat.IPF_BGR_888)
                    decoded = res.get_decoded_barcodes_result()
                    if decoded and decoded.get_items():
                        text = decoded.get_items()[0].text
                except: pass
            
            if text:
                with lock:
                    result_map[track_id] = text
            decode_q.task_done()

    t_worker = threading.Thread(target=worker, daemon=True)
    t_worker.start()
    
    # 5. 메인 루프
    tracker = QRTracker()
    frame_cnt = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_cnt += 1
            
            # A. YOLO 탐지 (패딩 포함됨)
            detections = yolo_detect_qr_locations(yolo, frame)
            
            # B. 탐지 결과를 추적기 포맷으로 변환
            det_for_tracker = []
            for det in detections:
                det_for_tracker.append({
                    'bbox': det['bbox'],
                    'text': None, # 아직 모름
                    'success': False
                })
            
            # C. 추적 업데이트
            tracked_objs = tracker.update(det_for_tracker, frame_cnt)
            
            # D. 해독 요청 및 결과 병합
            for obj in tracked_objs:
                tid = obj['track_id']
                
                # 이미 해독된 적 있으면 텍스트 가져오기
                with lock:
                    if tid in result_map:
                        obj['text'] = result_map[tid]
                        obj['success'] = True
                
                # 해독 안됐으면 큐에 넣기 (단, 너무 많이 넣지 않기 위해 큐 사이즈 체크 가능)
                if not obj['success'] and decode_q.qsize() < 5:
                    x1, y1, x2, y2 = obj['bbox']
                    # 좌표 안전장치
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2].copy()
                        decode_q.put((tid, roi))
            
            # E. 그리기 (결과 영상용)
            for obj in tracked_objs:
                x1, y1, x2, y2 = obj['bbox']
                text = obj['text']
                color = (0, 255, 0) if text else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{obj['track_id']}"
                if text: label += f" {text}"
                frame = put_korean_text(frame, label, (x1, y1-25), 20, color)

            writer.write(frame)
            
            # F. 로그 및 프리뷰
            if frame_cnt % 30 == 0:
                elapsed = time.time() - start_time
                fps_cur = frame_cnt / elapsed
                sys.stdout.write(f"\rFrame: {frame_cnt}/{total_frames} | FPS: {fps_cur:.1f} | Found: {len(result_map)}")
                sys.stdout.flush()
                
                if show_preview:
                    # 코랩 표시용 (리사이즈해서 전송량 줄임)
                    preview_img = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                    preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                    clear_output(wait=True)
                    plt.figure(figsize=(8, 5))
                    plt.imshow(preview_img)
                    plt.axis('off')
                    plt.show()

    except KeyboardInterrupt:
        print("\n중지됨!")
    finally:
        cap.release()
        writer.release()
        decode_q.put(None)
        t_worker.join()
        print(f"\n\n완료! 결과 파일: {out_path}")
        
        # 코랩에서 영상 다운로드 쉽게 하도록
        if IN_COLAB and os.path.exists(out_path):
            print(f"영상을 다운로드 하려면 왼쪽 파일 탐색기에서 {output_dir} 폴더를 확인하세요.")