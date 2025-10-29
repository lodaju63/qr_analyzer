"""
영상 플레이어 + 실시간 QR 탐지
영상을 화면에 보여주면서 QR 코드 탐지 시 시각화
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
    """QReader 탐지 (병렬 처리용)"""
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
                            'text': '해독 실패',
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

def brightness_qreader_detect_parallel(frame, qreader, results_queue):
    """밝기향상+QReader 탐지 (병렬 처리용, 파라미터 스윕)"""
    try:
        params = [(1.3, 20), (1.5, 30), (1.7, 40)]
        aggregate = []
        for alpha, beta in params:
            bright = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            bright = cv2.medianBlur(bright, 3)
            detections = qreader.detect(bright)
            if detections and len(detections) > 0:
                for i, detection in enumerate(detections):
                    try:
                        decoded_text = qreader.decode(bright, detection)
                        decoded_text = _process_decoded_text(decoded_text)
                        if decoded_text:
                            aggregate.append({'text': decoded_text,'detection': detection,'method': f'밝기향상+QReader-{i+1}','success': True})
                        else:
                            aggregate.append({'text': '해독 실패','detection': detection,'method': f'밝기향상+QReader-{i+1}-실패','success': False})
                    except Exception:
                        continue
        if aggregate:
            results_queue.put(('밝기향상+QReader', aggregate))
    except Exception:
        pass

def clahe_qreader_detect_parallel(frame, qreader, results_queue):
    """CLAHE+QReader 탐지 (병렬 처리용, 파라미터 스윕)"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clip_limits = [2.0, 3.0]
        tiles = [(8, 8), (12, 12)]
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
                                aggregate.append({'text': decoded_text,'detection': detection,'method': f'CLAHE+QReader-{i+1}','success': True})
                            else:
                                aggregate.append({'text': '해독 실패','detection': detection,'method': f'CLAHE+QReader-{i+1}-실패','success': False})
                        except Exception:
                            continue
        if aggregate:
            results_queue.put(('CLAHE+QReader', aggregate))
    except Exception as e:
        pass

# 반전+QReader (흰색 QR용)
def inverted_qreader_detect_parallel(frame, qreader, results_queue):
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
                            'text': '해독 실패',
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

# 원본 이진화+QReader
def binary_qreader_detect_parallel(frame, qreader, results_queue):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aggregate = []
        blurs = [0, 3]
        blocks = [11, 15, 21]
        consts = [2, 5, 10]
        for k in blurs:
            src = cv2.medianBlur(gray, k) if k else gray
            for b in blocks:
                for c in consts:
                    try:
                        binary = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c)
                        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                        detections = qreader.detect(binary_bgr)
                        if detections and len(detections) > 0:
                            for i, detection in enumerate(detections):
                                try:
                                    decoded_text = qreader.decode(binary_bgr, detection)
                                    if decoded_text:
                                        aggregate.append({'text': decoded_text,'detection': detection,'method': f'Binary+QReader-{i+1}','success': True})
                                    else:
                                        aggregate.append({'text': '해독 실패','detection': detection,'method': f'Binary+QReader-{i+1}-실패','success': False})
                                except Exception:
                                    continue
                    except Exception:
                        continue
        if aggregate:
            results_queue.put(('Binary+QReader', aggregate))
    except Exception:
        pass

# 반전+CLAHE+QReader
def inverted_clahe_qreader_detect_parallel(frame, qreader, results_queue):
    try:
        inverted = cv2.bitwise_not(frame)
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        clip_limits = [2.0, 3.0]
        tiles = [(8, 8), (12, 12)]
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
                                aggregate.append({'text': decoded_text,'detection': detection,'method': f'Inverted+CLAHE+QReader-{i+1}','success': True})
                            else:
                                aggregate.append({'text': '해독 실패','detection': detection,'method': f'Inverted+CLAHE+QReader-{i+1}-실패','success': False})
                        except Exception:
                            continue
        if aggregate:
            results_queue.put(('Inverted+CLAHE+QReader', aggregate))
    except Exception:
        pass

# 반전+이진화+QReader
def inverted_binary_qreader_detect_parallel(frame, qreader, results_queue):
    try:
        inverted = cv2.bitwise_not(frame)
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        aggregate = []
        blurs = [0, 3]
        blocks = [11, 15, 21]
        consts = [2, 5, 10]
        for k in blurs:
            src = cv2.medianBlur(gray, k) if k else gray
            for b in blocks:
                for c in consts:
                    try:
                        binary = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c)
                        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                        detections = qreader.detect(binary_bgr)
                        if detections and len(detections) > 0:
                            for i, detection in enumerate(detections):
                                try:
                                    decoded_text = qreader.decode(binary_bgr, detection)
                                    if decoded_text:
                                        aggregate.append({'text': decoded_text,'detection': detection,'method': f'Inverted+Binary+QReader-{i+1}','success': True})
                                    else:
                                        aggregate.append({'text': '해독 실패','detection': detection,'method': f'Inverted+Binary+QReader-{i+1}-실패','success': False})
                                except Exception:
                                    continue
                    except Exception:
                        continue
        if aggregate:
            results_queue.put(('Inverted+Binary+QReader', aggregate))
    except Exception:
        pass
# 밝기향상+PyZbar 함수 제거됨

def process_frame_parallel(frame, qreader):
    """프레임을 병렬로 처리하여 모든 QR 탐지 방법 실행"""
    results_queue = queue.Queue()
    threads = []
    
    # 여러 방법을 동시에 실행 (반전/이진화 계열 포함)
    if qreader:
        threads.append(threading.Thread(target=qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=brightness_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=clahe_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=inverted_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=binary_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=inverted_clahe_qreader_detect_parallel, args=(frame, qreader, results_queue)))
        threads.append(threading.Thread(target=inverted_binary_qreader_detect_parallel, args=(frame, qreader, results_queue)))
    
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
        "Binary+QReader": "Binary+QReader",
        "Inverted+CLAHE+QReader": "Inverted+CLAHE+QReader",
        "Inverted+Binary+QReader": "Inverted+Binary+QReader"
    }
    return method_map.get(method_name, method_name)

def is_center_in_bbox(center_x, center_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
    """중심점이 사각형 안에 있는지 확인"""
    return bbox_x1 <= center_x <= bbox_x2 and bbox_y1 <= center_y <= bbox_y2

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
    """원본 스케일 결과 처리 - 중심점 기반 중복 제거"""
    unique_qrs = []
    
    # 디버깅: 탐지 결과 출력
    total_detected = sum(len(qr_list) for qr_list in results.values())
    successful = sum(len([qr for qr in qr_list if qr['success']]) for qr_list in results.values())
    print(f"    🔍 탐지 결과: {successful}/{total_detected} 성공")
    
    for method, qr_list in results.items():
        for qr in qr_list:
            if qr['success']:
                detection = qr['detection']
                center_x, center_y, x1, y1, x2, y2 = get_qr_center_and_bbox(detection)
                
                if center_x is not None:
                    # 기존 QR들과 중복 체크
                    is_duplicate = False
                    for existing_qr in unique_qrs:
                        existing_detection = existing_qr['detection']
                        existing_center_x, existing_center_y, existing_x1, existing_y1, existing_x2, existing_y2 = get_qr_center_and_bbox(existing_detection)
                        
                        if existing_center_x is not None:
                            # 현재 QR의 중심이 기존 QR의 사각형 안에 있거나
                            # 기존 QR의 중심이 현재 QR의 사각형 안에 있으면 중복
                            if (is_center_in_bbox(center_x, center_y, existing_x1, existing_y1, existing_x2, existing_y2) or
                                is_center_in_bbox(existing_center_x, existing_center_y, x1, y1, x2, y2)):
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        qr['scale'] = 1.0
                        unique_qrs.append(qr)
    
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
    
    # 기존 결과 폴더 삭제 후 재생성
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"🗑️ 기존 결과 폴더 삭제: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "failed"), exist_ok=True)
    print(f"📁 결과 폴더 생성: {output_dir}")
    
    # QR 탐지기 초기화
    detector = cv2.QRCodeDetector()
    
    # QReader 초기화
    qreader = None
    if QREADER_AVAILABLE:
        try:
            qreader = QReader()
            print("✅ QReader 초기화 완료")
        except Exception as e:
            print(f"❌ QReader 초기화 실패: {e}")
            qreader = None
    
    print(f"📊 사용 가능한 탐지기:")
    print(f"  - OpenCV: ❌")
    print(f"  - QReader: {'✅' if qreader else '❌'}")
    print(f"  - PyZbar: ❌ (제거됨)")
    print(f"  - PIL (한글폰트): {'✅' if PIL_AVAILABLE else '❌'}")
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 비디오 정보:")
    print(f"  파일: {video_path}")
    print(f"  해상도: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  총 프레임: {total_frames}")
    print(f"  길이: {total_frames/fps:.2f}초")
    
    # 해상도 조정 (화면에 맞게)
    display_width = 1280
    display_height = 720
    
    if width > display_width:
        scale = display_width / width
        display_width = int(width * scale)
        display_height = int(height * scale)
    
    print(f"  화면 해상도: {display_width}x{display_height}")
    print(f"\n🎬 영상 재생 시작!")
    print(f"  - ESC 키: 종료")
    print(f"  - SPACE 키: 일시정지/재생")
    print(f"  - S 키: 현재 프레임 저장")
    
    # 재생 제어 변수
    paused = False
    frame_count = 0
    detected_count = 0
    start_time = time.time()
    
    # FPS 계산용
    fps_counter = 0
    fps_start_time = time.time()
    
    # 탐지 간격 설정 (성능 향상)
    detection_interval = 3  # 3프레임마다 탐지 (0.3초 간격)
    last_detection_frame = 0
    
    # 통계 변수
    success_count = 0
    failed_count = 0
    
    # 방법별 성공률 추적 (테스트용 확장)
    method_stats = {
        "QReader": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "Inverted+QReader": 0,
        "Binary+QReader": 0,
        "Inverted+CLAHE+QReader": 0,
        "Inverted+Binary+QReader": 0
    }
    
    # 테스트용: 방법별 탐지 개수 및 고유 탐지 추적
    method_detection_count = {
        "QReader": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "Inverted+QReader": 0,
        "Binary+QReader": 0,
        "Inverted+CLAHE+QReader": 0,
        "Inverted+Binary+QReader": 0
    }
    
    method_unique_detection_count = {
        "QReader": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "Inverted+QReader": 0,
        "Binary+QReader": 0,
        "Inverted+CLAHE+QReader": 0,
        "Inverted+Binary+QReader": 0
    }
    
    # 모든 방법에서 찾은 QR 코드들을 저장 (중복 제거용)
    all_detected_qrs = []
    
    # 현재 프레임용 변수
    current_success = 0
    current_failed = 0
    
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
                
                # 병렬 처리
                results = process_frame_parallel(single_frame, qreader)
                
                # 결과 통합 및 중복 제거
                unique_qrs = process_single_results(results)
                
                # 전체 탐지 개수 업데이트 (테스트용 상세 통계)
                for method, qr_list in results.items():
                    if method in method_detection_count:
                        method_detection_count[method] += len(qr_list)
                
                parallel_time = time.time() - start_time
                
                # 결과 처리 및 통계 업데이트
                if unique_qrs:
                    print(f"\n🔍 프레임 {frame_count}: {len(unique_qrs)}개의 고유 QR 코드 발견")
                    
                    for qr in unique_qrs:
                        if qr['success']:
                            print(f"    ✅ QR 코드: {qr['text']} ({qr['method']})")
                            current_success += 1
                            # 원본 방법명으로 통계 업데이트 (스케일 정보 제거)
                            # "PyZbar-1-0.5x" → "PyZbar"
                            if '-0.5x' in qr['method'] or '-0.75x' in qr['method'] or '-1.0x' in qr['method'] or '-1.25x' in qr['method'] or '-1.5x' in qr['method']:
                                # 스케일 정보와 인덱스 모두 제거
                                temp_method = qr['method'].rsplit('-', 1)[0]  # 스케일 제거
                                if temp_method.endswith('-1'):
                                    original_method = temp_method[:-2]  # "-1" 제거
                                else:
                                    original_method = temp_method.split('-')[0]  # 첫 번째 부분만
                            else:
                                original_method = qr['method'].split('-')[0]  # 첫 번째 부분만
                            
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
                            
                            # 시각화 데이터 추가 - 실제 QR 형태 반영
                            qr_points = None
                            detection = qr['detection']
                            
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
                                    "success": True,
                                    "scale": qr.get('scale', 1.0)
                                })
                        else:
                            current_failed += 1
                            
                            # 해독 실패한 QR도 unique_qrs에 추가하여 콘솔 출력
                            qr['scale'] = 1.0
                            unique_qrs.append(qr)
                            
                            # 해독 실패한 QR도 시각화 (빨간 박스)
                            qr_points = None
                            detection = qr['detection']
                            
                            # QReader 결과 처리 - quad_xy로 정확한 기울어진 형태 사용
                            if 'quad_xy' in detection:
                                quad = detection['quad_xy']
                                if len(quad) == 4:
                                    quad_array = np.array(quad)
                                    center = np.mean(quad_array, axis=0)
                                    angles = np.arctan2(quad_array[:, 1] - center[1], quad_array[:, 0] - center[0])
                                    sorted_indices = np.argsort(angles)
                                    sorted_quad = quad_array[sorted_indices]
                                    qr_points = np.array([sorted_quad], dtype=np.float32)
                            
                            elif 'bbox_xyxy' in detection:
                                bbox = detection['bbox_xyxy']
                                x1, y1, x2, y2 = bbox
                                qr_points = np.array([[
                                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                                ]], dtype=np.float32)
                            
                            elif 'cxcy' in detection and 'wh' in detection:
                                cx, cy = detection['cxcy']
                                w, h = detection['wh']
                                x1, y1 = cx - w/2, cy - h/2
                                x2, y2 = cx + w/2, cy + h/2
                                qr_points = np.array([[
                                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                                ]], dtype=np.float32)
                            
                            else:
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
                                    "success": False,  # 실패 표시
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
                                        else:
                                            # 스케일별 색상 적용
                                            scale = qr_viz.get('scale', 1.0)
                                            box_color = get_scale_color(scale)
                                            text_color = box_color
                                        
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
                                    pass  # 개별 QR 시각화 오류 (콘솔 출력 제거)
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
                        result_path = os.path.join(output_dir, "failed", f"frame_{frame_count:06d}.jpg")
                        failed_count += 1
                    else:
                        result_path = os.path.join(output_dir, "enhanced", f"frame_{frame_count:06d}.jpg")
                        success_count += 1
                    
                    # 시각화된 프레임 저장
                    cv2.imwrite(result_path, display_frame)
                
            except Exception as e:
                print(f"  ❌ 프레임 {frame_count} 처리 오류: {e}")
        
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
            save_path = os.path.join(output_dir, f"screenshot_{frame_count:06d}.jpg")
            cv2.imwrite(save_path, display_frame)
            print(f"📷 스크린샷 저장: {save_path}")
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    
    # 결과 요약
    elapsed = time.time() - start_time
    # 🕐 전체 실행 시간 계산
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    print(f"\n📊 결과 통계!")
    print(f"  총 프레임: {total_frames}")
    print(f"  재생 시간: {elapsed:.1f}초")
    print(f"  🚀 총 실행 시간: {total_execution_time:.1f}초 (병렬 처리)")
    print(f"  탐지된 QR 코드: {detected_count}개")
    print(f"  탐지율: {detected_count/frame_count*100:.1f}%" if frame_count > 0 else "  탐지율: 0.0%")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {failed_count}개")
    print(f"  결과 저장: {output_dir}/")
    
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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python video_player_qr.py <비디오_파일_경로>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_player_with_qr(video_path)
