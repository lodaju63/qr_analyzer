"""
영상 플레이어 + 실시간 QR 탐지 (병렬 처리)
[최종 최적화]: YOLO ROI 리스트를 먼저 필터링하여 중복 스레드 생성을 방지
"""

import cv2
import time
import os
import numpy as np
import threading
import queue

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# PyZbar 경고 메시지 완전히 숨기기
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['ZBAR_WARNINGS'] = '0'

# 표준 출력 리다이렉션으로 경고 숨기기
import sys
from contextlib import redirect_stderr
import io

# QReader import
try:
    from qreader import QReader
    QREADER_AVAILABLE = True
    # QReader 경고 메시지 숨기기
    warnings.filterwarnings('ignore', category=UserWarning, module='qreader')
except ImportError:
    QREADER_AVAILABLE = False
    print("⚠️ QReader를 사용할 수 없습니다. pip install qreader로 설치하세요.")

# YOLO 모델 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics를 사용할 수 없습니다. pip install ultralytics로 설치하세요.")

# PyZbar 관련 코드 제거됨
PYZBAR_AVAILABLE = False

# PIL import (한글 폰트 지원용)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL을 사용할 수 없습니다. pip install Pillow로 설치하세요.")

# 병렬 처리용 QR 탐지 함수들
def qreader_detect_parallel(frame, qreader, results_queue):
    """QReader 탐지 (병렬 처리용) - [비-YOLO 모드용]"""
    try:
        detections = qreader.detect(frame)
        if detections and len(detections) > 0:
            results = []
            for i, detection in enumerate(detections):
                try:
                    decoded_text = qreader.decode(frame, detection)
                    if decoded_text:
                        # 특수 문자 처리
                        decoded_text = decoded_text.replace('–', '-')
                        decoded_text = decoded_text.replace('—', '-')
                        
                        # 한글 인코딩 처리
                        try:
                            if isinstance(decoded_text, bytes):
                                decoded_text = decoded_text.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                decoded_text = decoded_text.decode('cp949')
                            except:
                                decoded_text = str(decoded_text)
                        
                        
                        results.append({
                            'text': decoded_text,
                            'detection': detection,
                            'method': f'QReader-{i+1}',
                            'success': True
                        })
                    else:
                        results.append({
                            'text': '',  # 실패한 경우 텍스트 없음
                            'detection': detection,
                            'method': f'QReader-{i+1}-실패',
                            'success': False
                        })
                except Exception as e:
                    continue
            
            if results:
                results_queue.put(('QReader', results))
    except Exception as e:
        pass

# PyZbar 함수 제거됨

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

def brightness_qreader_detect_parallel(frame, qreader, results_queue):
    """밝기향상+QReader 탐지 (병렬 처리용, 파라미터 스윕) - [비-YOLO 모드용]"""
    try:
        # 성능 최적 조합: 밝기향상 파라미터 (속도·성공률 균형)
        params = [
            (1.1, 5),
            (1.2, 10),
            (1.3, 12),
            (1.3, 15),
            (1.3, 18),
            (1.4, 20)
        ]
        aggregate = []
        for alpha, beta in params:
            bright = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            bright = cv2.medianBlur(bright, 3)
            detections = qreader.detect(bright)
            if detections and len(detections) > 0:
                for i, detection in enumerate(detections):
                    try:
                        decoded_text = qreader.decode(bright, detection)
                        decoded_text_processed = _process_decoded_text(decoded_text)
                        if decoded_text_processed:
                            aggregate.append({'text': decoded_text_processed,'detection': detection,'method': f'밝기향상+QReader-{i+1}','success': True,'params': f'α={alpha},β={beta}'})
                        else:
                            aggregate.append({'text': '','detection': detection,'method': f'밝기향상+QReader-{i+1}-실패','success': False,'params': f'α={alpha},β={beta}'})
                            print(f"    ⚠️ 밝기향상 실패: 원본={decoded_text}, 처리후={decoded_text_processed}")
                    except Exception as e:
                        print(f"    ❌ 밝기향상 예외: {e}")
                        continue
        if aggregate:
            results_queue.put(('밝기향상+QReader', aggregate))
    except Exception:
        pass

def clahe_qreader_detect_parallel(frame, qreader, results_queue):
    """CLAHE+QReader 탐지 (병렬 처리용, 파라미터 스윕) - [비-YOLO 모드용]"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 성능 최적 조합: CLAHE 파라미터 (tile=(3,3) 고정)
        clip_limits = [1.0, 3.0, 3.5, 4.0, 5.0, 6.0]
        tiles = [(3, 3), (2, 2)]
        aggregate = []
        for cl in clip_limits:
            for ts in tiles:
                clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=ts)
                enhanced = clahe.apply(gray)
                enhanced = cv2.medianBlur(enhanced, 3)
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                detections = qreader.detect(enhanced_bgr)
                if detections and len(detections) > 0:
                    for i, detection in enumerate(detections):
                        try:
                            decoded_text = qreader.decode(enhanced_bgr, detection)
                            decoded_text = _process_decoded_text(decoded_text)
                            if decoded_text:
                                # CLAHE+QReader-1 제거, 2, 3번만 유지
                                method_num = i + 2 if i == 0 else i + 1
                                aggregate.append({'text': decoded_text,'detection': detection,'method': f'CLAHE+QReader-{method_num}','success': True,'params': f'clip={cl},tile={ts}'})
                            else:
                                method_num = i + 2 if i == 0 else i + 1
                                aggregate.append({'text': '','detection': detection,'method': f'CLAHE+QReader-{method_num}-실패','success': False,'params': f'clip={cl},tile={ts}'})
                        except Exception:
                            continue
        if aggregate:
            results_queue.put(('CLAHE+QReader', aggregate))
    except Exception as e:
        pass

# 반전+QReader (흰색 QR용)
def inverted_qreader_detect_parallel(frame, qreader, results_queue):
    """[비-YOLO 모드용]"""
    try:
        inverted = cv2.bitwise_not(frame)
        detections = qreader.detect(inverted)
        if detections and len(detections) > 0:
            results = []
            for i, detection in enumerate(detections):
                try:
                    decoded_text = qreader.decode(inverted, detection)
                    if decoded_text:
                        decoded_text = decoded_text.replace('–', '-').replace('—', '-')
                        try:
                            if isinstance(decoded_text, bytes):
                                decoded_text = decoded_text.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                decoded_text = decoded_text.decode('cp949')
                            except:
                                decoded_text = str(decoded_text)
                        results.append({
                            'text': decoded_text,
                            'detection': detection,
                            'method': f'Inverted+QReader-{i+1}',
                            'success': True
                        })
                    else:
                        results.append({
                            'text': '',  # 실패한 경우 텍스트 없음
                            'detection': detection,
                            'method': f'Inverted+QReader-{i+1}-실패',
                            'success': False
                        })
                except Exception:
                    continue
            if results:
                results_queue.put(('Inverted+QReader', results))
    except Exception:
        pass

# Binary+QReader 방법 제거됨 (성능상 이점 없음)

# 반전+CLAHE+QReader
def inverted_clahe_qreader_detect_parallel(frame, qreader, results_queue):
    """[비-YOLO 모드용]"""
    try:
        inverted = cv2.bitwise_not(frame)
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        # 성능 최적 조합: Inverted+CLAHE 파라미터 (tile=(3,3) 고정)
        clip_limits = [3.0, 5.0, 5.5, 6.0, 8.0]
        tiles = [(3, 3), (2, 2)]
        aggregate = []
        for cl in clip_limits:
            for ts in tiles:
                clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=ts)
                enhanced = clahe.apply(gray)
                enhanced = cv2.medianBlur(enhanced, 3)
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                detections = qreader.detect(enhanced_bgr)
                if detections and len(detections) > 0:
                    for i, detection in enumerate(detections):
                        try:
                            decoded_text = qreader.decode(enhanced_bgr, detection)
                            if decoded_text:
                                aggregate.append({'text': decoded_text,'detection': detection,'method': f'Inverted+CLAHE+QReader-{i+1}','success': True,'params': f'clip={cl},tile={ts}'})
                            else:
                                aggregate.append({'text': '','detection': detection,'method': f'Inverted+CLAHE+QReader-{i+1}-실패','success': False,'params': f'clip={cl},tile={ts}'})
                        except Exception:
                            continue
        if aggregate:
            results_queue.put(('Inverted+CLAHE+QReader', aggregate))
    except Exception:
        pass

# Inverted+Binary+QReader 방법 제거됨 (성능상 이점 없음)
# 밝기향상+PyZbar 함수 제거됨

def apply_clahe(img, clip_limit=3.0, tile_grid_size=(3, 3)):
    """CLAHE 전처리 적용"""
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)
        enhanced = cv2.medianBlur(enhanced, 3)
        if len(img.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    except Exception:
        return img

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
# ★★★★★ 원본 복원 ★★★★★
#
# `qreader.detect(roi)`를 다시 호출하여 정확한 시각화 좌표(`quad_xy`)를
# 확보하고 해독 성공률을 높이는 원본 로직으로 복원합니다.
# -----------------------------------------------------------------
def decode_roi_parallel(roi, qreader, bbox, results_queue, method_name="YOLO+QReader"):
    """ROI 영역에서 QR 코드 해독 (병렬 처리용) - [원본 버전]"""
    try:
        # 1단계: detect()로 위치 찾기 (더 정확한 위치, quad_xy 확보)
        detections = qreader.detect(roi)
        
        if detections and len(detections) > 0:
            # 첫 번째 detection 사용
            detection = detections[0]
            # 2단계: 찾은 힌트(detection)로 decode() 실행
            decoded_text = qreader.decode(roi, detection)
        else:
            # detect 실패 시 ROI 전체에서 직접 decode 시도
            decoded_text = qreader.decode(roi)
            detection = None # 힌트 없음
        
        if decoded_text:
            decoded_text = _process_decoded_text(decoded_text)
            if decoded_text:
                # 원본 이미지 좌표로 변환
                x1, y1, x2, y2 = bbox
                
                if detection and 'quad_xy' in detection:
                    # ROI 내 좌표를 원본 이미지 좌표로 변환
                    quad_xy = []
                    for qx, qy in detection['quad_xy']:
                        # ROI 내 상대 좌표를 원본 이미지 절대 좌표로 변환
                        abs_x = x1 + int(qx)
                        abs_y = y1 + int(qy)
                        quad_xy.append([abs_x, abs_y])
                    
                    detection_result = {
                        'bbox_xyxy': [x1, y1, x2, y2], # YOLO의 넓은 bbox
                        'quad_xy': quad_xy # QReader의 정밀한 quad
                    }
                else:
                    # detection 정보가 없으면 YOLO의 bbox 기반으로 생성
                    detection_result = {
                        'bbox_xyxy': [x1, y1, x2, y2],
                        'quad_xy': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }
                
                results_queue.put((method_name, [{
                    'text': decoded_text,
                    'detection': detection_result,
                    'method': method_name,
                    'success': True
                }]))
                return
    except Exception:
        pass

def decode_roi_with_preprocessing_parallel(roi, qreader, bbox, results_queue, method_name, preprocessing_func):
    """전처리된 ROI에서 QR 코드 해독 (병렬 처리용)"""
    try:
        processed_roi = preprocessing_func(roi)
        if processed_roi is not None:
            # 원본 decode_roi_parallel (정확도 우선)을 호출합니다.
            decode_roi_parallel(processed_roi, qreader, bbox, results_queue, method_name)
    except Exception:
        pass

# -----------------------------------------------------------------
# ★★★★★ IoU 계산 함수들을 위로 이동 ★★★★★
# `process_frame_with_yolo` 보다 먼저 정의되어야 합니다.
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
    qreader 스레드를 생성하기 전에 호출하여 중복 스레드 생성을 방지합니다.
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
def process_frame_with_yolo(frame, yolo_model, qreader, conf_threshold=0.25, use_preprocessing=False):
    """YOLO로 빠르게 위치 찾고, ROI에서만 해독하는 최적화된 처리
    
    Args:
        frame: 입력 프레임
        yolo_model: YOLO 모델
        qreader: QReader 인스턴스
        conf_threshold: YOLO 신뢰도 임계값
        use_preprocessing: 전처리 방법 사용 여부 (False면 원본만 사용, 속도 향상)
    """
    results_queue = queue.Queue()
    threads = []
    
    # 1단계: YOLO로 빠르게 QR 코드 위치 탐지
    if yolo_model is not None:
        qr_locations = yolo_detect_qr_locations(yolo_model, frame, conf_threshold)
        
        # ★★★★★ 새로운 최적화 단계 ★★★★★
        # qreader 스레드를 생성하기 *전에* 겹치는 ROI를 먼저 제거
        filtered_locations = filter_overlapping_yolo_rois(qr_locations, iou_threshold=0.5)
        
        # (디버깅용)
        if len(qr_locations) > len(filtered_locations):
            print(f"    ⚡ ROI 필터링: {len(qr_locations)}개 -> {len(filtered_locations)}개 (중복 스레드 방지)")
        
        if filtered_locations: # ★ 수정: filtered_locations 사용
            # 2단계: 각 ROI에서 병렬로 해독 시도
            for i, location in enumerate(filtered_locations): # ★ 수정: filtered_locations 사용
                x1, y1, x2, y2 = location['bbox']
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # 원본 ROI 해독 (항상 실행) - ★ 원본 함수(정확도 우선) 호출
                if qreader:
                    threads.append(threading.Thread(
                        target=decode_roi_parallel,
                        args=(roi, qreader, location['bbox'], results_queue, f"YOLO+QReader-{i+1}")
                    ))
                    
                    # 전처리된 ROI 해독 (옵션, 원본 실패 시에만 유용)
                    if use_preprocessing:
                        threads.append(threading.Thread(
                            target=decode_roi_with_preprocessing_parallel,
                            args=(roi, qreader, location['bbox'], results_queue, 
                                  f"YOLO+밝기향상+QReader-{i+1}",
                                  lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=15))
                        ))
                        
                        threads.append(threading.Thread(
                            target=decode_roi_with_preprocessing_parallel,
                            args=(roi, qreader, location['bbox'], results_queue,
                                  f"YOLO+CLAHE+QReader-{i+1}",
                                  lambda img: apply_clahe(img))
                        ))
        
        # 모든 스레드 시작
        for thread in threads:
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        # 결과 수집
        all_results = {}
        while not results_queue.empty():
            method, results = results_queue.get()
            all_results[method] = results
        
        return all_results
    
    # YOLO 모델이 없으면 기존 방식 사용
    return process_frame_parallel(frame, qreader)

def process_frame_parallel(frame, qreader):
    """프레임을 병렬로 처리하여 모든 QR 탐지 방법 실행 (기존 방식 - 비-YOLO 모드용)"""
    results_queue = queue.Queue()
    threads = []
    
    # 여러 방법을 동시에 실행 (Binary 방법들 제거됨)
    if qreader:
        threads.append(threading.Thread(target=qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=brightness_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=clahe_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=inverted_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=inverted_clahe_qreader_detect_parallel, args=(frame, qreader, results_queue)))
    
    # 모든 스레드 시작
    for thread in threads:
        thread.start()
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()
    
    # 결과 수집
    all_results = {}
    while not results_queue.empty():
        method, results = results_queue.get()
        all_results[method] = results
    
    return all_results

def create_single_frame(frame):
    """원본 프레임만 사용"""
    return frame, [1.0]

def get_scale_color(scale):
    """스케일별 색상 반환 (BGR 형식)"""
    if scale == 1.0:
        return (0, 255, 0)    # 초록색
    elif scale == 1.5:
        return (255, 0, 0)    # 파란색
    elif scale == 2.0:
        return (0, 255, 255)  # 노란색
    else:
        return (255, 255, 255)  # 기본 흰색

def put_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """OpenCV 이미지에 한글 텍스트를 그리는 함수"""
    if not PIL_AVAILABLE:
        # PIL이 없으면 OpenCV 기본 폰트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    
    try:
        # OpenCV 이미지를 PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 한글 폰트 로드 (Windows 기본 폰트들 시도)
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",   # 굴림
            "C:/Windows/Fonts/batang.ttc",  # 바탕
            "C:/Windows/Fonts/arial.ttf"    # Arial (fallback)
        ]
        
        font = None
        for font_path in font_paths:
            try:
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

def get_english_method_name(method_name):
    """한글 방법명을 영문으로 변환 (OpenCV putText 호환)"""
    method_map = {
        "QReader": "QReader",
        "밝기향상+QReader": "Bright+QReader",
        "CLAHE+QReader": "CLAHE+QReader",
        "Inverted+QReader": "Inverted+QReader",
        "Inverted+CLAHE+QReader": "Inverted+CLAHE+QReader"
    }
    return method_map.get(method_name, method_name)

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
        if len(quad) == 4:
            quad_array = np.array(quad)
            center = np.mean(quad_array, axis=0)
            # quad_xy의 바운딩 박스 계산
            x_coords = quad_array[:, 0]
            y_coords = quad_array[:, 1]
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)
            return center[0], center[1], x1, y1, x2, y2
    
    # polygon_xy가 있으면 사용
    elif 'polygon_xy' in detection:
        polygon = detection['polygon_xy']
        if len(polygon) >= 4:
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

def process_single_results(results):
    """원본 스케일 결과 처리 - 중심점 기반 중복 제거
    
    로직:
    1. 성공한 QR들을 먼저 수집
    2. 같은 위치에 성공/실패가 모두 있으면 성공만 유지
    3. 같은 위치에 실패만 있으면 실패도 유지
    """
    unique_qrs = []
    
    # 디버깅: 탐지 결과 출력
    total_detected = sum(len(qr_list) for qr_list in results.values())
    successful = sum(len([qr for qr in qr_list if isinstance(qr, dict) and 'success' in qr and qr['success']]) for qr_list in results.values())
    print(f"    🔍 탐지 결과: {successful}/{total_detected} 성공")
    
    # 성공한 QR들과 실패한 QR들을 분리
    successful_qrs = []
    failed_qrs = []
    for method, qr_list in results.items():
        for qr in qr_list:
            # 메타데이터는 건너뛰기
            if isinstance(qr, dict) and 'meta' in qr:
                continue
            
            # 디버깅: 각 QR의 원본 상태 출력
            if qr.get('success'):
                successful_qrs.append(qr)
            else:
                print(f"    🔴 실패 QR 발견: method={qr['method']}, text='{qr['text']}', success={qr['success']}")
                failed_qrs.append(qr)
    
    # 성공한 QR들 중에서 중복 제거
    successful_unique = []
    for qr in successful_qrs:
        detection = qr['detection']
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(detection)
        
        if center_x is not None:
            # 기존 QR들과 중복 체크
            is_duplicate = False
            for existing_qr in successful_unique:
                existing_detection = existing_qr['detection']
                existing_center_x, existing_center_y, existing_x1, existing_y1, existing_x2, existing_y2 = get_qr_center_and_bbox(existing_detection)
                
                if existing_center_x is not None:
                    # IoU 기반 중복 체크 (임계값 0.5)
                    iou = calculate_iou((x1, y1, x2, y2), (existing_x1, existing_y1, existing_x2, existing_y2))
                    if iou > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                qr['scale'] = 1.0
                successful_unique.append(qr)
    
    # 실패한 QR들 중에서 중복 제거 (단, 성공한 QR과 겹치는 것은 제외)
    failed_unique = []
    for qr in failed_qrs:
        detection = qr['detection']
        center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(detection)
        
        if center_x is not None:
            # 성공한 QR과 겹치는지 체크
            overlaps_with_success = False
            for success_qr in successful_unique:
                success_detection = success_qr['detection']
                success_center_x, success_center_y, success_x1, success_y1, success_x2, success_y2 = get_qr_center_and_bbox(success_detection)
                
                if success_center_x is not None:
                    # IoU 기반 겹침 체크
                    iou = calculate_iou((x1, y1, x2, y2), (success_x1, success_y1, success_x2, success_y2))
                    # 중심점 거리 기반 체크 (추가)
                    center_dist = calculate_center_distance((x1, y1, x2, y2), (success_x1, success_y1, success_x2, success_y2))
                    # IoU > 0.3 또는 정규화된 중심점 거리 < 0.5면 겹침
                    if iou > 0.2 or center_dist < 0.5:
                        overlaps_with_success = True
                        # 디버깅: 겹침 확인
                        print(f"    🔴 실패 QR과 성공 QR 겹침 감지: IoU={iou:.2f}, 중심거리={center_dist:.2f}")
                        break
            
            # 성공한 QR과 겹치지 않는 경우에만 실패한 QR 추가
            if not overlaps_with_success:
                # 실패한 QR들 간 중복 체크
                is_duplicate = False
                for existing_qr in failed_unique:
                    existing_detection = existing_qr['detection']
                    existing_center_x, existing_center_y, existing_x1, existing_y1, existing_x2, existing_y2 = get_qr_center_and_bbox(existing_detection)
                    
                    if existing_center_x is not None:
                        # IoU 기반 중복 체크 (임계값 0.5)
                        iou = calculate_iou((x1, y1, x2, y2), (existing_x1, existing_y1, existing_x2, existing_y2))
                        if iou > 0.5:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    qr['scale'] = 1.0
                    failed_unique.append(qr)
    
    # 성공한 QR들과 실패한 QR들을 합침
    unique_qrs = successful_unique + failed_unique
    
    return unique_qrs

def extract_bounding_box(detection, image_width=None, image_height=None):
    """
    QReader detection 결과에서 우선순위에 따라 Bounding Box 추출
    
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
    os.makedirs(os.path.join(output_run_dir, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_run_dir, "failed"), exist_ok=True)
    
    # 로그 파일 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_run_dir, f"qr_detection_log_{timestamp}.txt")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log_print(message):
        """콘솔 출력과 파일 저장을 동시에"""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print(f"📁 결과 폴더 생성: {output_run_dir}")
    log_print(f"📁 로그 파일: {log_file_path}")
    
    # QR 탐지기 초기화
    detector = cv2.QRCodeDetector()
    
    # YOLO 모델 초기화
    yolo_model = None
    use_yolo_mode = True  # YOLO 모드 사용 여부
    
    if YOLO_AVAILABLE and use_yolo_mode:
        try:
            model_path = 'l.pt'
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
    
    # QReader 초기화
    qreader = None
    if QREADER_AVAILABLE:
        try:
            qreader = QReader()
            log_print("✅ QReader 초기화 완료")
        except Exception as e:
            log_print(f"❌ QReader 초기화 실패: {e}")
            qreader = None
    
    log_print(f"📊 사용 가능한 탐지기:")
    log_print(f"  - YOLO 모델: {'✅ (빠른 탐지 모드)' if yolo_model else '❌'}")
    log_print(f"  - OpenCV: ❌")
    log_print(f"  - QReader: {'✅' if qreader else '❌'}")
    log_print(f"  - PyZbar: ❌ (제거됨)")
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
    
    # 탐지 간격 설정 (성능 향상)
    detection_interval = 2  # 5프레임마다 탐지
    last_detection_frame = 0
    
    # 통계 변수
    success_count = 0
    failed_count = 0
    
    # 방법별 성공률 추적 (테스트용 확장)
    method_stats = {
        "YOLO+QReader": 0,
        "YOLO+밝기향상+QReader": 0,
        "YOLO+CLAHE+QReader": 0,
        "QReader": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "Inverted+QReader": 0,
        "Inverted+CLAHE+QReader": 0
    }
    
    # 테스트용: 방법별 탐지 개수 및 고유 탐지 추적
    method_detection_count = {
        "YOLO+QReader": 0,
        "YOLO+밝기향상+QReader": 0,
        "YOLO+CLAHE+QReader": 0,
        "QReader": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "Inverted+QReader": 0,
        "Inverted+CLAHE+QReader": 0
    }
    
    method_unique_detection_count = {
        "YOLO+QReader": 0,
        "YOLO+밝기향상+QReader": 0,
        "YOLO+CLAHE+QReader": 0,
        "QReader": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "Inverted+QReader": 0,
        "Inverted+CLAHE+QReader": 0
    }
    
    # 모든 방법에서 찾은 QR 코드들을 저장 (중복 제거용)
    all_detected_qrs = []
    
    # 현재 프레임용 변수
    current_success = 0
    current_failed = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n📺 영상 재생 완료!")
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
            
            # 탐지 간격 체크 (성능 향상)
            should_detect = (frame_count - last_detection_frame) >= detection_interval
            
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
                        # 🚀 YOLO 기반 빠른 탐지 + ROI 해독
                        # [최적화] use_preprocessing=False로 설정하여
                        # 원본 ROI에 대해서만 decode_roi_parallel (원본 버전)을 호출합니다.
                        use_preprocessing_mode = False 
                        results = process_frame_with_yolo(single_frame, yolo_model, qreader, conf_threshold=0.25, use_preprocessing=use_preprocessing_mode)
                    else:
                        # 기존 병렬 처리 방식 (비-YOLO 모드)
                        results = process_frame_parallel(single_frame, qreader)
                    
                    # Binary 방법 제거로 파라미터별 결과 출력 로직 제거됨
                    
                    # 결과 통합 및 중복 제거
                    unique_qrs = process_single_results(results)
                
                    # 전체 탐지 개수 업데이트 (테스트용 상세 통계)
                    for method, qr_list in results.items():
                        if method in method_detection_count:
                            # 메타데이터는 카운트에서 제외
                            actual_list = [qr for qr in qr_list if not (isinstance(qr, dict) and 'meta' in qr)]
                            method_detection_count[method] += len(actual_list)
                    
                    parallel_time = time.time() - start_time
                    
                    # 결과 처리 및 통계 업데이트
                    if unique_qrs:
                        log_print(f"\n🔍 프레임 {frame_count}: {len(unique_qrs)}개의 고유 QR 코드 발견")
                        
                        for qr in unique_qrs:
                            # 메타데이터는 건너뛰기
                            if isinstance(qr, dict) and 'meta' in qr:
                                continue
                            
                            # 성공/실패 통계
                            if qr.get('success'):
                                # 파라미터 정보가 있으면 표시
                                params_info = f" [{qr.get('params', '')}]" if qr.get('params') else ""
                                log_print(f"    ✅ QR 코드: {qr['text']} ({qr['method']}{params_info})")
                                current_success += 1
                            else:
                                current_failed += 1
                            
                            # 원본 방법명으로 통계 업데이트
                            # "YOLO+QReader-1" → "YOLO+QReader"
                            # "YOLO+밝기향상+QReader-1" → "YOLO+밝기향상+QReader"
                            method_name = qr['method']
                            
                            # YOLO 방식인 경우 처리
                            if method_name.startswith('YOLO+'):
                                # 마지막 숫자 제거 (예: "YOLO+QReader-1" → "YOLO+QReader")
                                if '-' in method_name and method_name[-1].isdigit():
                                    # 마지막 하이픈과 숫자 제거
                                    parts = method_name.rsplit('-', 1)
                                    if parts[1].isdigit():
                                        original_method = parts[0]
                                    else:
                                        original_method = method_name
                                else:
                                    original_method = method_name
                            # 기존 방식 (스케일 정보 제거)
                            elif '-0.5x' in method_name or '-0.75x' in method_name or '-1.0x' in method_name or '-1.25x' in method_name or '-1.5x' in method_name:
                                # 스케일 정보와 인덱스 모두 제거
                                temp_method = method_name.rsplit('-', 1)[0]  # 스케일 제거
                                if temp_method.endswith('-1'):
                                    original_method = temp_method[:-2]  # "-1" 제거
                                else:
                                    original_method = temp_method.split('-')[0]  # 첫 번째 부분만
                            else:
                                # 일반적인 경우: 마지막 숫자 제거
                                parts = method_name.rsplit('-', 1)
                                if parts[1].isdigit():
                                    original_method = parts[0]
                                else:
                                    original_method = method_name.split('-')[0]  # 첫 번째 부분만
                            
                            # method_stats에 존재하는 키인지 확인
                            if original_method in method_stats:
                                method_stats[original_method] += 1
                            else:
                                print(f"    ⚠️ 알 수 없는 방법: {original_method}")
                            
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
                            
                            # QReader 결과 처리 - quad_xy로 정확한 기울어진 형태 사용
                            if 'quad_xy' in detection:
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
                                    "scale": qr.get('scale', 1.0)
                                })
                    
                    # 첫 번째 성공한 결과를 메인 시각화용으로 설정
                    if unique_qrs:
                        detected = True
                        detected_text = unique_qrs[0]['text']
                        detection_method = unique_qrs[0]['method']
                    
                    print(f"    ⚡ 병렬 처리 시간: {parallel_time:.3f}초")
                    
                   
                    
                    # 처리한 프레임은 항상 업데이트 (QR 발견 여부와 관계없이)
                    last_detection_frame = frame_count
                    
                    if detected:
                        detected_count += 1
                        
                        # 현재 프레임의 성공/실패 통계 출력 (중복 제거)
                        total_found = current_success + current_failed
                        if total_found > 0:
                            print(f"    📊 결과: {total_found}개 중 {current_success}개 성공, {current_failed}개 실패")
                        
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
                                                # 디버깅: 실패로 분류된 이유 출력
                                                log_print(f"    🔴 실패 분류: success={qr_success}, text='{qr_text}', method='{qr_method}'")
                                            else:
                                                # 스케일별 색상 적용
                                                scale = qr_viz.get('scale', 1.0)
                                                box_color = get_scale_color(scale)
                                                text_color = box_color
                                                # 디버깅: 성공으로 분류된 경우 출력
                                                log_print(f"    🟢 성공 분류: success={qr_success}, text='{qr_text}', method='{qr_method}'")
                                            
                                            # QR 코드 영역 그리기 (선 두께 줄임)
                                            cv2.polylines(display_frame, [display_points], True, box_color, 2)
                                            
                                            # 텍스트 표시 (하이픈 문자 정리)
                                            display_text = qr_text[:30] + "..." if len(qr_text) > 30 else qr_text
                                            # OpenCV putText에서 문제가 되는 특수 문자들을 표준 하이픈으로 변경
                                            display_text = display_text.replace('–', '-').replace('—', '-').replace('−', '-')
                                            display_text = display_text.replace('？', '?').replace('！', '!').replace('，', ',')
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
                        # 시각화 데이터가 없을 때 기본 시각화 (폰트 크기 줄임, 하이픈 문자 정리)
                        print(f"    ⚠️ 시각화 데이터 없음")
                        text = detected_text[:30] + "..." if len(detected_text) > 30 else detected_text
                        text = text.replace('–', '-').replace('—', '-').replace('−', '-')
                        text = text.replace('？', '?').replace('！', '!').replace('，', ',')
                        cv2.putText(display_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        method_text = f"Method: {detection_method}"
                        display_frame = put_korean_text(display_frame, method_text, (10, 25), font_size=16, color=(0, 255, 0))
                    
                    # 결과 저장 (해독 실패 시 failed 폴더에 저장)
                    if "실패" in detected_text or "실패" in detection_method:
                        result_path = os.path.join(output_run_dir, "failed", f"frame_{frame_count:06d}.jpg")
                        failed_count += 1
                    else:
                        result_path = os.path.join(output_run_dir, "enhanced", f"frame_{frame_count:06d}.jpg")
                        success_count += 1
                    
                        # 시각화된 프레임 저장
                        cv2.imwrite(result_path, display_frame)
                
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
            
            # 화면에 표시
            cv2.imshow("Video Player + QR Detection", display_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
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
    cap.release()
    cv2.destroyAllWindows()
    
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
    
    log_print(f"\n🎯 방법별 성공률:")
    total_method_success = sum(method_stats.values())
    for method, count in method_stats.items():
        if total_method_success > 0:
            percentage = (count / total_method_success) * 100
            log_print(f"  {method}: {count}개 ({percentage:.1f}%)")
    
    log_print(f"\n📊 테스트용 상세 통계:")
    log_print(f"  방법별 탐지 개수:")
    for method, count in method_detection_count.items():
        log_print(f"    {method}: {count}개")
    
    log_print(f"  방법별 성공률 (탐지 대비):")
    for method in method_stats.keys():
        detected = method_detection_count[method]
        success = method_stats[method]
        if detected > 0:
            success_rate = (success / detected) * 100
            log_print(f"    {method}: {success}/{detected} ({success_rate:.1f}%)")
        else:
            log_print(f"    {method}: 0/0 (0.0%)")
    
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
    
    print(f"\n🎯 방법별 성공률:")
    total_method_success = sum(method_stats.values())
    for method, count in method_stats.items():
        if total_method_success > 0:
            percentage = (count / total_method_success) * 100
            print(f"  {method}: {count}개 ({percentage:.1f}%)")
    
    print(f"\n📊 테스트용 상세 통계:")
    print(f"  방법별 탐지 개수:")
    for method, count in method_detection_count.items():
        print(f"    {method}: {count}개")
    
    print(f"  방법별 성공률 (탐지 대비):")
    for method in method_stats.keys():
        detected = method_detection_count[method]
        success = method_stats[method]
        if detected > 0:
            success_rate = (success / detected) * 100
            print(f"    {method}: {success}/{detected} ({success_rate:.1f}%)")
        else:
            print(f"    {method}: 0/0 (0.0%)")
    
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
        # ★★★★★ 수정된 부분 ★★★★★
        # 오류 메시지의 파일 이름을 현재 파일(video_player_qr_parallel.py)로 수정
        print("사용법: python video_player_qr_parallel.py <비디오_파일_경로>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_player_with_qr(video_path)