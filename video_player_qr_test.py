"""
영상 플레이어 + 실시간 QR 탐지
영상을 화면에 보여주면서 QR 코드 탐지 시 시각화
"""

import cv2
import time
import os
import numpy as np

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

# QReader와 PyZbar import (선택적)
try:
    from qreader import QReader
    QREADER_AVAILABLE = True
    # QReader 경고 메시지 숨기기
    warnings.filterwarnings('ignore', category=UserWarning, module='qreader')
except ImportError:
    QREADER_AVAILABLE = False
    print("⚠️ QReader를 사용할 수 없습니다. pip install qreader로 설치하세요.")

try:
    from pyzbar import pyzbar
    from PIL import Image
    PYZBAR_AVAILABLE = True
    # PyZbar 경고 메시지 숨기기
    warnings.filterwarnings('ignore', category=UserWarning, module='pyzbar')
except ImportError:
    PYZBAR_AVAILABLE = False
    print("⚠️ PyZbar를 사용할 수 없습니다. pip install pyzbar로 설치하세요.")

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
    print(f"  - OpenCV: ✅")
    print(f"  - QReader: {'✅' if qreader else '❌'}")
    print(f"  - PyZbar: {'✅' if PYZBAR_AVAILABLE else '❌'}")
    
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
        "PyZbar": 0, 
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "밝기향상+PyZbar": 0
    }
    
    # 테스트용: 방법별 탐지 개수 및 고유 탐지 추적
    method_detection_count = {
        "QReader": 0,
        "PyZbar": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "밝기향상+PyZbar": 0
    }
    
    method_unique_detection_count = {
        "QReader": 0,
        "PyZbar": 0,
        "밝기향상+QReader": 0,
        "CLAHE+QReader": 0,
        "밝기향상+PyZbar": 0
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
        
        if should_detect:
            # 현재 프레임용 변수 초기화
            current_success = 0
            current_failed = 0
            
            try:
                # 방법 1: QReader 탐지 (AI 기반 - 다중 QR 코드 지원) - 테스트용: 항상 실행
                if qreader:
                    try:
                        # 1단계: detect()로 위치 찾기
                        detections = qreader.detect(frame)
                        if detections and len(detections) > 0:
                            print(f"\n🔍 프레임 {frame_count}: {len(detections)}개의 QR 코드 발견 (QReader)")
                            method_detection_count["QReader"] += len(detections)
                            
                            # 모든 QR 코드 처리
                            for i, detection in enumerate(detections):
                                try:
                                    # 2단계: decode()로 텍스트 추출
                                    decoded_text = qreader.decode(frame, detection)
                                    if decoded_text:
                                        # 특수 문자 처리 (엔 대시 → 일반 하이픈)
                                        decoded_text = decoded_text.replace('–', '-')  # 엔 대시
                                        decoded_text = decoded_text.replace('—', '-')  # 엠 대시
                                        
                                        # 한글 인코딩 처리 (안전장치)
                                        try:
                                            if isinstance(decoded_text, bytes):
                                                decoded_text = decoded_text.decode('utf-8')
                                        except UnicodeDecodeError:
                                            try:
                                                decoded_text = decoded_text.decode('cp949')  # 한글 인코딩
                                            except:
                                                decoded_text = str(decoded_text)  # 최후 수단
                                        
                                        detected = True
                                        detected_text = decoded_text
                                        detection_method = f"QReader-{i+1}"
                                        print(f"    ✅ QR 코드 {i+1}: {decoded_text} (QReader)")
                                        current_success += 1
                                        method_stats["QReader"] += 1
                                        
                                        # 모든 QR 코드 시각화
                                        qr_points = None
                                        # 1순위: 정확한 bbox_xyxy 사용
                                        if 'bbox_xyxy' in detection:
                                                bbox = detection['bbox_xyxy']
                                                x1, y1, x2, y2 = bbox
                                                qr_points = np.array([[
                                                    [x1, y1],  # 좌상단
                                                    [x2, y1],  # 우상단
                                                    [x2, y2],  # 우하단
                                                    [x1, y2]   # 좌하단
                                                ]], dtype=np.float32)
                                                # QReader bbox (콘솔 출력 제거)
                                        else:
                                                # 2순위: bbox_xyxy가 없을 때만 추정 시각화
                                                bbox, method_info = extract_bounding_box(detection, frame.shape[1], frame.shape[0])
                                                
                                                if bbox is not None:
                                                    x1, y1, x2, y2 = bbox
                                                    qr_points = np.array([[
                                                        [x1, y1],  # 좌상단
                                                        [x2, y1],  # 우상단
                                                        [x2, y2],  # 우하단
                                                        [x1, y2]   # 좌하단
                                                    ]], dtype=np.float32)
                                                    # 추정 시각화 (콘솔 출력 제거)
                                                else:
                                                    print(f"    {method_info}")
                                                    qr_points = None
                                        
                                        # 시각화 데이터 추가
                                        if qr_points is not None:
                                            all_qr_visualizations.append({
                                                "points": qr_points,
                                                "text": decoded_text,
                                                "method": f"QReader-{i+1}",
                                                "success": True
                                            })
                                        
                                        # 모든 QR 코드 처리 (조선소 T-bar 공정용 - 완전한 정보 수집)
                                    else:
                                        # QR 코드 해독 실패
                                        print(f"    ❌ QR 코드 {i+1} 해독 실패 (QReader)")
                                        current_failed += 1
                                        
                                        # 해독 실패해도 위치 정보가 있으면 시각화 시도
                                        # 모든 QR 코드 시각화 (다중 QR 지원)
                                        qr_points = None
                                        if 'bbox_xyxy' in detection:
                                            bbox = detection['bbox_xyxy']
                                            x1, y1, x2, y2 = bbox
                                            qr_points = np.array([[
                                                [x1, y1],  # 좌상단
                                                [x2, y1],  # 우상단
                                                [x2, y2],  # 우하단
                                                [x1, y2]   # 좌하단
                                            ]], dtype=np.float32)
                                            # QReader bbox (해독실패) (콘솔 출력 제거)
                                            # 위치 정보가 있으면 시각화를 위해 detected = True 설정
                                            detected = True
                                            detected_text = "해독 실패"
                                            detection_method = f"QReader-{i+1}-실패"
                                        else:
                                                # 추정 시각화
                                                bbox, method_info = extract_bounding_box(detection, frame.shape[1], frame.shape[0])
                                                if bbox is not None:
                                                    x1, y1, x2, y2 = bbox
                                                    points = np.array([[
                                                        [x1, y1],  # 좌상단
                                                        [x2, y1],  # 우상단
                                                        [x2, y2],  # 우하단
                                                        [x1, y2]   # 좌하단
                                                    ]], dtype=np.float32)
                                                    # 추정 시각화 (해독실패) (콘솔 출력 제거)
                                                    # 위치 정보가 있으면 시각화를 위해 detected = True 설정
                                                    detected = True
                                                    detected_text = "해독 실패"
                                                    detection_method = f"QReader-{i+1}-실패"
                                                else:
                                                    print(f"    {method_info} (해독실패)")
                                                    qr_points = None
                                        
                                        # 실패한 QR 코드도 시각화 데이터에 추가
                                        if qr_points is not None:
                                            all_qr_visualizations.append({
                                                "points": qr_points,
                                                "text": "해독 실패",
                                                "method": f"QReader-{i+1}-실패",
                                                "success": False
                                            })
                                        
                                except Exception as e:
                                    print(f"    ❌ QR 코드 {i+1} 처리 오류: {e}")
                                    continue
                                else:
                                    # 해독 실패 시에도 위치 정보 확인 (bbox_xyxy가 없을 때만)
                                    if 'bbox_xyxy' not in detection:
                                        bbox, method_info = extract_bounding_box(detection, frame.shape[1], frame.shape[0])
                                        if bbox is not None:
                                            print(f"    {method_info}: 위치 정보 있음")
                                        else:
                                            print(f"    {method_info}")
                                    else:
                                        pass  # bbox_xyxy 있지만 해독 실패 (콘솔 출력 제거)
                    except Exception as e:
                        print(f"    ❌ QReader 오류: {e}")
                        pass
                
                # 방법 2: PyZbar 탐지 (QR 코드만 - 보조용) - 테스트용: 항상 실행
                if PYZBAR_AVAILABLE:
                    try:
                        # OpenCV를 PIL로 변환
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        # QR 코드만 탐지 (다른 바코드 제외)
                        pyzbar_results = pyzbar.decode(pil_image, symbols=[pyzbar.ZBarSymbol.QRCODE])
                        
                        if pyzbar_results:
                            print(f"\n🔍 프레임 {frame_count}: {len(pyzbar_results)}개의 QR 코드 발견 (PyZbar)")
                            method_detection_count["PyZbar"] += len(pyzbar_results)
                            # PyZbar 다중 QR 코드 처리
                            for i, result in enumerate(pyzbar_results):
                                try:
                                    qr_data = result.data.decode('utf-8')
                                    # 모든 QR 코드 시각화
                                    qr_points = None
                                    rect = result.rect
                                    qr_points = np.array([[
                                        [rect.left, rect.top],
                                        [rect.left + rect.width, rect.top],
                                        [rect.left + rect.width, rect.top + rect.height],
                                        [rect.left, rect.top + rect.height]
                                    ]], dtype=np.float32)
                                    
                                    if not detected:  # 첫 번째 QR 코드만 detected 설정
                                        detected = True
                                        detected_text = qr_data
                                        detection_method = f"PyZbar-{i+1}"
                                        method_stats["PyZbar"] += 1
                                        current_success += 1
                                        print(f"    ✅ QR 코드 {i+1}: {qr_data} (PyZbar)")
                                    else:
                                        # 추가 QR 코드는 출력만
                                        print(f"    ✅ QR 코드 {i+1}: {qr_data} (PyZbar)")
                                        current_success += 1
                                        method_stats["PyZbar"] += 1
                                    
                                    # 모든 QR 코드를 시각화 리스트에 추가
                                    all_qr_visualizations.append({
                                        "points": qr_points,
                                        "text": qr_data,
                                        "method": f"PyZbar-{i+1}",
                                        "success": True
                                    })
                                except Exception as e:
                                    print(f"    ❌ QR 코드 {i+1} 해독 실패 (PyZbar)")
                                    current_failed += 1
                    except Exception as e:
                        pass
                
                # 방법 3: 밝기 향상 + QReader - 테스트용: 항상 실행
                if qreader:
                    try:
                        # 밝기 향상 전처리
                        brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
                        # 1단계: detect()로 위치 찾기
                        detections = qreader.detect(brightened)
                        if detections and len(detections) > 0:
                            print(f"\n🔍 프레임 {frame_count}: {len(detections)}개의 QR 코드 발견 (밝기향상+QReader)")
                            method_detection_count["밝기향상+QReader"] += len(detections)
                            
                            # 모든 QR 코드 처리
                            for i, detection in enumerate(detections):
                                try:
                                    # 2단계: decode()로 텍스트 추출
                                    decoded_text = qreader.decode(brightened, detection)
                                    if decoded_text:
                                        # 특수 문자 처리 (엔 대시 → 일반 하이픈)
                                        decoded_text = decoded_text.replace('–', '-')  # 엔 대시
                                        decoded_text = decoded_text.replace('—', '-')  # 엠 대시
                                        
                                        # 한글 인코딩 처리 (안전장치)
                                        try:
                                            if isinstance(decoded_text, bytes):
                                                decoded_text = decoded_text.decode('utf-8')
                                        except UnicodeDecodeError:
                                            try:
                                                decoded_text = decoded_text.decode('cp949')
                                            except:
                                                decoded_text = str(decoded_text)
                                        
                                        detected = True
                                        detected_text = decoded_text
                                        detection_method = f"밝기향상+QReader-{i+1}"
                                        method_stats["밝기향상+QReader"] += 1
                                        current_success += 1
                                        print(f"    ✅ QR 코드 {i+1}: {decoded_text} (밝기향상+QReader)")
                                        
                                        # QReader bbox를 points로 변환하고 시각화 데이터에 추가
                                        if 'bbox_xyxy' in detection:
                                            bbox = detection['bbox_xyxy']
                                            x1, y1, x2, y2 = bbox
                                            points = np.array([[
                                                [x1, y1],  # 좌상단
                                                [x2, y1],  # 우상단
                                                [x2, y2],  # 우하단
                                                [x1, y2]   # 좌하단
                                            ]], dtype=np.float32)
                                            
                                            # 시각화 데이터 추가
                                            all_qr_visualizations.append({
                                                "points": points,
                                                "text": decoded_text,
                                                "method": f"밝기향상+QReader-{i+1}",
                                                "success": True
                                            })
                                        # 모든 QR 코드 처리 (조선소 T-bar 공정용 - 완전한 정보 수집)
                                    else:
                                        # QR 코드 해독 실패
                                        print(f"    ❌ QR 코드 {i+1} 해독 실패 (밝기향상+QReader)")
                                        current_failed += 1
                                        
                                        # 해독 실패해도 위치 정보가 있으면 시각화 시도
                                        qr_points = None
                                        if 'bbox_xyxy' in detection:
                                            bbox = detection['bbox_xyxy']
                                            x1, y1, x2, y2 = bbox
                                            qr_points = np.array([[
                                                [x1, y1],  # 좌상단
                                                [x2, y1],  # 우상단
                                                [x2, y2],  # 우하단
                                                [x1, y2]   # 좌하단
                                            ]], dtype=np.float32)
                                            detected = True
                                            detected_text = "해독 실패"
                                            detection_method = f"밝기향상+QReader-{i+1}-실패"
                                        
                                        # 실패한 QR 코드도 시각화 데이터에 추가
                                        if qr_points is not None:
                                            all_qr_visualizations.append({
                                                "points": qr_points,
                                                "text": "해독 실패",
                                                "method": f"밝기향상+QReader-{i+1}-실패",
                                                "success": False
                                            })
                                        # 모든 QR 코드 처리 (조선소 T-bar 공정용 - 완전한 정보 수집)
                                except Exception as e:
                                    print(f"    ❌ QR 코드 {i+1} 처리 오류: {e}")
                                    continue
                    except Exception as e:
                        print(f"    ❌ 밝기향상+QReader 오류: {e}")
                        pass
                
                # 방법 4: CLAHE + QReader - 테스트용: 항상 실행
                if qreader:
                    try:
                        # CLAHE 전처리
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        # 그레이스케일을 BGR로 변환
                        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                        
                        # 1단계: detect()로 위치 찾기
                        detections = qreader.detect(enhanced_bgr)
                        if detections and len(detections) > 0:
                            print(f"\n🔍 프레임 {frame_count}: {len(detections)}개의 QR 코드 발견 (CLAHE+QReader)")
                            method_detection_count["CLAHE+QReader"] += len(detections)
                            
                            # 모든 QR 코드 처리
                            for i, detection in enumerate(detections):
                                try:
                                    # 2단계: decode()로 텍스트 추출
                                    decoded_text = qreader.decode(enhanced_bgr, detection)
                                    if decoded_text:
                                        # 특수 문자 처리 (엔 대시 → 일반 하이픈)
                                        decoded_text = decoded_text.replace('–', '-')  # 엔 대시
                                        decoded_text = decoded_text.replace('—', '-')  # 엠 대시
                                        
                                        # 한글 인코딩 처리 (안전장치)
                                        try:
                                            if isinstance(decoded_text, bytes):
                                                decoded_text = decoded_text.decode('utf-8')
                                        except UnicodeDecodeError:
                                            try:
                                                decoded_text = decoded_text.decode('cp949')
                                            except:
                                                decoded_text = str(decoded_text)
                                        
                                        detected = True
                                        detected_text = decoded_text
                                        detection_method = f"CLAHE+QReader-{i+1}"
                                        method_stats["CLAHE+QReader"] += 1
                                        current_success += 1
                                        print(f"    ✅ QR 코드 {i+1}: {decoded_text} (CLAHE+QReader)")
                                        
                                        # QReader bbox를 points로 변환하고 시각화 데이터에 추가
                                        if 'bbox_xyxy' in detection:
                                            bbox = detection['bbox_xyxy']
                                            x1, y1, x2, y2 = bbox
                                            points = np.array([[
                                                [x1, y1],  # 좌상단
                                                [x2, y1],  # 우상단
                                                [x2, y2],  # 우하단
                                                [x1, y2]   # 좌하단
                                            ]], dtype=np.float32)
                                            
                                            # 시각화 데이터 추가
                                            all_qr_visualizations.append({
                                                "points": points,
                                                "text": decoded_text,
                                                "method": f"CLAHE+QReader-{i+1}",
                                                "success": True
                                            })
                                        # 모든 QR 코드 처리 (조선소 T-bar 공정용 - 완전한 정보 수집)
                                    else:
                                        # QR 코드 해독 실패
                                        print(f"    ❌ QR 코드 {i+1} 해독 실패 (CLAHE+QReader)")
                                        current_failed += 1
                                        
                                        # 해독 실패해도 위치 정보가 있으면 시각화 시도
                                        qr_points = None
                                        if 'bbox_xyxy' in detection:
                                            bbox = detection['bbox_xyxy']
                                            x1, y1, x2, y2 = bbox
                                            qr_points = np.array([[
                                                [x1, y1],  # 좌상단
                                                [x2, y1],  # 우상단
                                                [x2, y2],  # 우하단
                                                [x1, y2]   # 좌하단
                                            ]], dtype=np.float32)
                                            detected = True
                                            detected_text = "해독 실패"
                                            detection_method = f"CLAHE+QReader-{i+1}-실패"
                                        
                                        # 실패한 QR 코드도 시각화 데이터에 추가
                                        if qr_points is not None:
                                            all_qr_visualizations.append({
                                                "points": qr_points,
                                                "text": "해독 실패",
                                                "method": f"CLAHE+QReader-{i+1}-실패",
                                                "success": False
                                            })
                                        # 모든 QR 코드 처리 (조선소 T-bar 공정용 - 완전한 정보 수집)
                                except Exception as e:
                                    print(f"    ❌ QR 코드 {i+1} 처리 오류: {e}")
                                    continue
                    except Exception as e:
                        print(f"    ❌ CLAHE+QReader 오류: {e}")
                        pass
                
                # 방법 5: 밝기 향상 + PyZbar - 테스트용: 항상 실행
                if PYZBAR_AVAILABLE:
                    try:
                        # 밝기 향상 전처리
                        brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
                        pil_image = Image.fromarray(cv2.cvtColor(brightened, cv2.COLOR_BGR2RGB))
                        
                        # QR 코드만 탐지
                        pyzbar_results = pyzbar.decode(pil_image, symbols=[pyzbar.ZBarSymbol.QRCODE])
                        
                        if pyzbar_results:
                            print(f"\n🔍 프레임 {frame_count}: {len(pyzbar_results)}개의 QR 코드 발견 (밝기향상+PyZbar)")
                            method_detection_count["밝기향상+PyZbar"] += len(pyzbar_results)
                            # 밝기향상+PyZbar 다중 QR 코드 처리
                            for i, result in enumerate(pyzbar_results):
                                try:
                                    qr_data = result.data.decode('utf-8')
                                    if not detected:  # 첫 번째 QR 코드만 시각화
                                        detected = True
                                        detected_text = qr_data
                                        detection_method = f"밝기향상+PyZbar-{i+1}"
                                        method_stats["밝기향상+PyZbar"] += 1
                                        current_success += 1
                                        print(f"    ✅ QR 코드 {i+1}: {qr_data} (밝기향상+PyZbar)")
                                        
                                        # PyZbar rect를 points로 변환
                                        rect = result.rect
                                        points = np.array([[
                                            [rect.left, rect.top],
                                            [rect.left + rect.width, rect.top],
                                            [rect.left + rect.width, rect.top + rect.height],
                                            [rect.left, rect.top + rect.height]
                                        ]], dtype=np.float32)
                                    else:
                                        # 추가 QR 코드는 출력만
                                        print(f"    ✅ QR 코드 {i+1}: {qr_data} (밝기향상+PyZbar)")
                                        current_success += 1
                                        method_stats["밝기향상+PyZbar"] += 1
                                except Exception as e:
                                    print(f"    ❌ QR 코드 {i+1} 해독 실패 (밝기향상+PyZbar)")
                                    current_failed += 1
                    except Exception as e:
                        pass
                
                
                if detected:
                    detected_count += 1
                    last_detection_frame = frame_count  # 탐지 성공 시 마지막 탐지 프레임 업데이트
                    
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
                                        
                                        # 해독 실패 시 빨간 박스, 성공 시 초록 박스
                                        if not qr_success or "실패" in qr_text or "실패" in qr_method:
                                            box_color = (0, 0, 255)  # 빨간색 (BGR)
                                            text_color = (0, 0, 255)  # 빨간색
                                        else:
                                            box_color = (0, 255, 0)  # 초록색 (BGR)
                                            text_color = (0, 255, 0)  # 초록색
                                        
                                        # QR 코드 영역 그리기 (선 두께 줄임)
                                        cv2.polylines(display_frame, [display_points], True, box_color, 2)
                                        
                                        # 텍스트 표시 (폰트 크기 줄임)
                                        text = qr_text[:30] + "..." if len(qr_text) > 30 else qr_text
                                        text_pos = (int(display_points[0][0]), int(display_points[0][1]) - 15 - (j * 20))
                                        cv2.putText(display_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                                        
                                        # 탐지 방법 표시 (첫 번째 QR만)
                                        if j == 0:
                                            method_text = f"Method: {qr_method}"
                                            cv2.putText(display_frame, method_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                                    else:
                                        pass  # points_2d 변환 실패 (콘솔 출력 제거)
                                except Exception as e:
                                    pass  # 개별 QR 시각화 오류 (콘솔 출력 제거)
                        except Exception as e:
                            print(f"    ❌ 시각화 오류: {e}")
                            # 기본 시각화 (폰트 크기 줄임)
                            text = detected_text[:30] + "..." if len(detected_text) > 30 else detected_text
                            cv2.putText(display_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            method_text = f"Method: {detection_method}"
                            cv2.putText(display_frame, method_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    else:
                        # 시각화 데이터가 없을 때 기본 시각화 (폰트 크기 줄임)
                        print(f"    ⚠️ 시각화 데이터 없음")
                        text = detected_text[:30] + "..." if len(detected_text) > 30 else detected_text
                        cv2.putText(display_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        method_text = f"Method: {detection_method}"
                        cv2.putText(display_frame, method_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
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
    print(f"  🐌 총 실행 시간: {total_execution_time:.1f}초 (순차 처리)")
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
