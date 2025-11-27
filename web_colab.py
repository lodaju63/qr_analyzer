"""
Streamlit 기반 QR 탐지 웹 애플리케이션 (코랩용)
yolo_dynamsoft.py와 동일한 성능의 웹 버전
코랩 환경에서 바로 실행 가능하도록 최적화
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
import platform
import threading
from queue import Queue, Empty
import datetime
import zipfile
import io

from streamlit.runtime.scriptrunner import add_script_run_ctx
import shutil

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# Streamlit ScriptRunContext 경고 억제
import logging

# Streamlit 관련 로거 레벨 조정
streamlit_loggers = [
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.state',
    'streamlit.runtime.session_state',
    'streamlit.runtime.media_file_storage',  # MediaFileStorageError 억제
    'streamlit.web.server.media_file_handler',  # MediaFileHandler 억제
]
for logger_name in streamlit_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)  # CRITICAL로 설정하여 완전히 억제
    logger.propagate = False

# 모든 Streamlit 경고 메시지 억제
logging.getLogger('streamlit').setLevel(logging.CRITICAL)

# 코랩 환경 확인
IN_COLAB = 'google.colab' in sys.modules

# 코랩 환경에서 작업 디렉토리 설정
if IN_COLAB:
    # 코랩에서는 /content 디렉토리 사용
    if os.getcwd() != '/content':
        os.chdir('/content')
    # 모델 파일 경로 확인
    MODEL_PATH = 'model1.pt'
    if not os.path.exists(MODEL_PATH):
        # 여러 경로에서 모델 파일 찾기
        possible_paths = [
            '/content/model1.pt',
            '/content/drive/MyDrive/model1.pt',
            './model1.pt',
            'model1.pt'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                MODEL_PATH = path
                break
else:
    MODEL_PATH = 'model1.pt'

# yolo_dynamsoft.py 함수들 import
try:
    # 동일한 디렉토리에 있는 경우 직접 import
    from yolo_dynamsoft import (
        _process_decoded_text,
        preprocess_frame_for_detection,
        yolo_detect_qr_locations,
        process_frame_with_yolo,
        create_single_frame,
        put_korean_text,
        get_qr_center_and_bbox,
        extract_bounding_box,
        calculate_iou,
        filter_overlapping_yolo_rois,
        QRTracker,
        QRTrack
    )
    YOLO_DYNASOFT_IMPORTED = True
except ImportError:
    # import 실패 시 경로 추가 시도
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from yolo_dynamsoft import (
            _process_decoded_text,
            preprocess_frame_for_detection,
            yolo_detect_qr_locations,
            process_frame_with_yolo,
            create_single_frame,
            put_korean_text,
            get_qr_center_and_bbox,
            extract_bounding_box,
            calculate_iou,
            filter_overlapping_yolo_rois,
            QRTracker,
            QRTrack
        )
        YOLO_DYNASOFT_IMPORTED = True
    except Exception as e:
        st.error(f"yolo_dynamsoft.py 모듈을 import할 수 없습니다: {e}")
        YOLO_DYNASOFT_IMPORTED = False

# Dynamsoft Barcode Reader import
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

# YOLO 모델 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# PIL import
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="QR 탐지 시스템",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'should_stop' not in st.session_state:
    st.session_state.should_stop = False
if 'processing_thread' not in st.session_state:
    st.session_state.processing_thread = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'current_results' not in st.session_state:
    st.session_state.current_results = {
        'detected_qrs': [],
        'frame_num': 0,
        'total_frames': 0,
        'fps': 0.0
    }
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None
if 'accumulated_qr_records' not in st.session_state:
    st.session_state.accumulated_qr_records = []
if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None
if 'temp_video_path' not in st.session_state:
    st.session_state.temp_video_path = None
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None
if 'processing_completed' not in st.session_state:
    st.session_state.processing_completed = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'dbr_reader' not in st.session_state:
    st.session_state.dbr_reader = None
if 'decode_queue' not in st.session_state:
    st.session_state.decode_queue = None
if 'decode_results' not in st.session_state:
    st.session_state.decode_results = {}
if 'decode_lock' not in st.session_state:
    st.session_state.decode_lock = threading.Lock()
if 'decode_worker_thread' not in st.session_state:
    st.session_state.decode_worker_thread = None
if 'stop_decode_worker' not in st.session_state:
    st.session_state.stop_decode_worker = None
if 'qr_tracker' not in st.session_state:
    st.session_state.qr_tracker = None
if 'batch_files' not in st.session_state:
    st.session_state.batch_files = []
if 'batch_processing' not in st.session_state:
    st.session_state.batch_processing = False
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = {}
if 'current_batch_file_index' not in st.session_state:
    st.session_state.current_batch_file_index = 0

# 결과 디렉토리 설정 (코랩에서는 /content 사용)
if IN_COLAB:
    OUTPUT_BASE_DIR = "/content/output_results"
else:
    OUTPUT_BASE_DIR = "output_results"

def initialize_models():
    """모델 초기화"""
    if not YOLO_DYNASOFT_IMPORTED:
        return None, None, "yolo_dynamsoft.py 모듈을 import할 수 없습니다."
    
    yolo_model = None
    dbr_reader = None
    
    # YOLO 모델 초기화
    if YOLO_AVAILABLE:
        try:
            # 코랩 환경에서 모델 경로 확인
            model_path = MODEL_PATH
            if not os.path.exists(model_path):
                # 여러 경로에서 모델 파일 찾기
                possible_paths = [
                    '/content/model1.pt',
                    '/content/drive/MyDrive/model1.pt',
                    './model1.pt',
                    'model1.pt'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
            
            if os.path.exists(model_path):
                yolo_model = YOLO(model_path)
                st.success(f"✅ YOLO 모델 로드 완료: {model_path}")
            else:
                if IN_COLAB:
                    st.warning(f"⚠️ YOLO 모델 파일을 찾을 수 없습니다.")
                    st.info("💡 코랩에서 모델 파일을 업로드하거나 Google Drive에서 복사하세요.")
                    st.code("""
# Google Drive 사용 시:
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/model1.pt /content/

# 또는 직접 업로드:
from google.colab import files
uploaded = files.upload()  # model1.pt 선택
                    """)
                else:
                    st.warning(f"⚠️ YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
        except Exception as e:
            st.error(f"❌ YOLO 모델 초기화 실패: {e}")
    
    # Dynamsoft 초기화
    if DBR_AVAILABLE:
        try:
            license_key = os.environ.get('DYNAMSOFT_LICENSE_KEY', '')
            if not license_key:
                license_key = 't0085YQEAADYdcL2llMa8vH1Rtnun+43saE/kdAE7ZbIxMQGRMtSzVSZRI8vfOK4Ids52rjekwzh87yABFLraXw5Va1BV7NnBjI8m7qbw3kxOprI75ExJpw=='
            
            if license_key:
                if DBR_VERSION == "bundle_v11":
                    error = license.LicenseManager.init_license(license_key)
                    if error[0] != 0:
                        st.warning(f"⚠️ Dynamsoft 라이선스 초기화 실패: {error[1]}")
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
                        st.success("✅ Dynamsoft Barcode Reader 초기화 완료")
        except Exception as e:
            st.error(f"❌ Dynamsoft 초기화 실패: {e}")
    
    return yolo_model, dbr_reader, None

def process_image_file(image_path, conf_threshold, iou_threshold, use_preprocessing,
                      use_clahe, use_normalize, clahe_clip_limit, detect_both_frames):
    """이미지 파일 처리"""
    if not YOLO_DYNASOFT_IMPORTED:
        return None, None, "yolo_dynamsoft.py 모듈을 import할 수 없습니다."
    
    try:
        # 이미지 읽기
        frame = cv2.imread(image_path)
        
        if frame is None:
            return None, None, "이미지를 읽을 수 없습니다."
        
        # YOLO 탐지
        yolo_model = st.session_state.yolo_model
        if yolo_model is None:
            return None, None, "YOLO 모델이 초기화되지 않았습니다."
        
        filtered_locations = process_frame_with_yolo(
            frame, yolo_model, 
            conf_threshold=conf_threshold,
            use_preprocessing=use_preprocessing,
            use_clahe=use_clahe,
            use_normalize=use_normalize,
            clahe_clip_limit=clahe_clip_limit,
            detect_both_frames=detect_both_frames,
            iou_threshold=iou_threshold
        )
        
        # 결과 표시용 프레임 생성
        display_frame = frame.copy()
        detected_qrs = []
        
        # Dynamsoft 해독 시도
        dbr_reader = st.session_state.dbr_reader
        
        for i, location in enumerate(filtered_locations):
            x1, y1, x2, y2 = location['bbox']
            roi = frame[y1:y2, x1:x2]
            
            # 해독 시도
            decoded_text = None
            quad_xy = None
            
            if dbr_reader and roi.size > 0:
                try:
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
                        captured_result = dbr_reader.capture(roi_rgb, dbr_module.EnumImagePixelFormat.IPF_RGB_888)
                        barcode_result = captured_result.get_decoded_barcodes_result()
                        if barcode_result:
                            items = barcode_result.get_items() if hasattr(barcode_result, 'get_items') else None
                            if items and len(items) > 0:
                                barcode_item = items[0]
                                text = None
                                if hasattr(barcode_item, 'get_text'):
                                    text = barcode_item.get_text()
                                elif hasattr(barcode_item, 'text'):
                                    text = barcode_item.text
                                elif hasattr(barcode_item, 'barcode_text'):
                                    text = barcode_item.barcode_text
                                if text:
                                    decoded_text = _process_decoded_text(text)
                except:
                    pass
            
            # 시각화
            if decoded_text:
                color = (0, 255, 0)
                success = True
                method = "YOLO+Dynamsoft"
            else:
                color = (0, 0, 255)
                success = False
                method = "YOLO"
            
            if quad_xy and len(quad_xy) == 4:
                quad_array = np.array(quad_xy, dtype=np.int32)
                cv2.polylines(display_frame, [quad_array], True, color, 2)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # QR 번호만 표시 (해독정보는 표에 표시됨)
            track_id_text = f"#{i}"
            cv2.putText(display_frame, track_id_text, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            detected_qrs.append({
                'track_id': i,
                'bbox': location['bbox'],
                'confidence': location['confidence'],
                'text': decoded_text or '',
                'method': method,
                'success': success,
                'frame': 0,
                'detection': {
                    'bbox_xyxy': location['bbox'],
                    'quad_xy': quad_xy
                }
            })
        
        return display_frame, detected_qrs, None
    except Exception as e:
        import traceback
        return None, None, f"{str(e)}\n{traceback.format_exc()}"

def decode_worker_func_with_ref(dbr_reader, decode_queue, stop_event, session_state_ref):
    """해독 워커 스레드 - 스레드 안전 버전"""
    if not dbr_reader or not decode_queue:
        return
    
    decode_lock = session_state_ref.get('decode_lock')
    
    while not stop_event.is_set():
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
                            elif hasattr(barcode_item, 'barcode_text'):
                                text = barcode_item.barcode_text
                            
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
                                        elif hasattr(location, 'get_result_points'):
                                            result_points = location.get_result_points()
                                        
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
                # 에러는 무시 (로그 파일 제거됨)
                pass
            
            # 해독 실패는 무시
            
            if decoded_text:
                if quad_xy is None:
                    x1, y1, x2, y2 = bbox
                    quad_xy = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                
                # decode_results 업데이트 (항상 session_state_ref에서 최신 상태 가져오기)
                with decode_lock:
                    decode_results = session_state_ref.get('decode_results', {})
                    if decode_results is None:
                        decode_results = {}
                    decode_results[track_id] = {
                        'text': decoded_text,
                        'quad_xy': quad_xy,
                        'decode_bbox': list(bbox),
                        'decode_method': 'Dynamsoft',
                        'decode_method_detail': decode_method_detail,
                        'frame': frame_num if frame_num is not None else 0
                    }
                    session_state_ref['decode_results'] = decode_results
                    
                    # 해독 결과만 저장 (누적은 process_video_thread에서 프레임마다 수행)
            
            decode_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            if 'item' in locals() and item:
                decode_queue.task_done()

def process_video_thread(video_path, output_dir, conf_threshold, iou_threshold,
                        use_preprocessing, use_clahe, use_normalize, clahe_clip_limit,
                        detect_both_frames, session_state_ref):
    """비디오 처리 스레드 - 스레드 안전 버전"""
    # Streamlit 경고 억제 (스레드 내에서)
    import logging
    import warnings
    import os
    
    # 모든 Streamlit 로거 레벨을 ERROR로 설정
    for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner', 
                        'streamlit.runtime.scriptrunner.script_runner']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # CRITICAL로 설정하여 더 확실하게 억제
        logger.propagate = False
    
    # 모든 경고 억제
    warnings.filterwarnings('ignore')
    
    # 환경 변수로 Streamlit 로깅 레벨 설정
    os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'
    
    try:
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"비디오 파일을 열 수 없습니다: {video_path}"
            session_state_ref['processing'] = False
            session_state_ref['error'] = error_msg
            return
        
        session_state_ref['cap'] = cap
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 비디오 설정
        display_width = 1280
        display_height = 720
        if width > display_width:
            scale = display_width / width
            display_width = int(width * scale)
            display_height = int(height * scale)
        
        # 출력 비디오 파일
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(output_dir, f"output_{run_id}.mp4")
        
        system = platform.system()
        if system == "Windows":
            codec = 'mp4v'
        elif system == "Darwin":
            codec = 'avc1'
        else:
            codec = 'mp4v'
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (display_width, display_height))
        session_state_ref['video_writer'] = out_video
        
        # 로그 파일 제거됨 - 해독된 QR 기록은 누적 저장됨
        # 해독된 QR 기록 초기화 (새 처리 시작 시)
        if 'accumulated_qr_records' not in session_state_ref:
            session_state_ref['accumulated_qr_records'] = []
        else:
            session_state_ref['accumulated_qr_records'] = []  # 새 처리 시작 시 초기화
        
        # processing 상태를 명시적으로 확인 및 설정
        if not session_state_ref.get('processing', False):
            session_state_ref['processing'] = True
        
        # QR 추적기 초기화
        qr_tracker = QRTracker(max_missed_frames=10, iou_threshold=0.15, 
                              center_dist_threshold=1.2, linear_motion_boost=True)
        session_state_ref['qr_tracker'] = qr_tracker
        
        # 해독 워커 시작
        dbr_reader = session_state_ref.get('dbr_reader')
        yolo_model = session_state_ref.get('yolo_model')
        
        decode_queue = None
        stop_decode_worker = None
        decode_worker_thread = None
        
        if dbr_reader:
            decode_queue = Queue(maxsize=10)
            stop_decode_worker = threading.Event()
            session_state_ref['decode_queue'] = decode_queue
            session_state_ref['stop_decode_worker'] = stop_decode_worker
            
            # 해독 워커 함수에 필요한 객체 전달
            def decode_worker_with_ref():
                decode_worker_func_with_ref(dbr_reader, decode_queue, stop_decode_worker, session_state_ref)
            
            decode_worker_thread = threading.Thread(target=decode_worker_with_ref, daemon=True)
            # ★★★ [핵심 수정] 해독 워커 스레드에도 Streamlit 컨텍스트 주입
            add_script_run_ctx(decode_worker_thread)
            decode_worker_thread.start()
            session_state_ref['decode_worker_thread'] = decode_worker_thread
        frame_count = 0
        start_time = time.time()
        fps_counter = 0
        fps_start_time = time.time()
        
        # 메인 처리 루프
        while True:
            # 상태 확인
            processing = session_state_ref.get('processing', False)
            should_stop = session_state_ref.get('should_stop', False)
            
            if not processing:
                break
            
            if should_stop:
                break
            
            if session_state_ref.get('paused', False):
                # 일시정지 중
                time.sleep(0.1)
                continue
            
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 해상도 조정
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # 첫 프레임은 즉시 표시 (QR 탐지 전에)
            if frame_count == 1:
                # RGB로 변환하여 저장 (app.py 방식)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                session_state_ref['current_frame'] = frame_rgb
                session_state_ref['current_results'] = {
                    'detected_qrs': [],
                    'frame_num': frame_count,
                    'total_frames': total_frames,
                    'fps': fps
                }
            
            # QR 탐지
            detected_qrs = []
            
            if yolo_model:
                filtered_locations = process_frame_with_yolo(
                    frame, yolo_model,
                    conf_threshold=conf_threshold,
                    use_preprocessing=use_preprocessing,
                    use_clahe=use_clahe,
                    use_normalize=use_normalize,
                    clahe_clip_limit=clahe_clip_limit,
                    detect_both_frames=detect_both_frames,
                    iou_threshold=iou_threshold
                )
                
                for i, location in enumerate(filtered_locations):
                    detected_qrs.append({
                        'bbox': location['bbox'],
                        'confidence': location['confidence'],
                        'text': '',
                        'detection': {
                            'bbox_xyxy': location['bbox'],
                            'quad_xy': None
                        },
                        'method': 'YOLO',
                        'success': False
                    })
            
            # 추적 업데이트
            if qr_tracker:
                tracked_qrs = qr_tracker.update(detected_qrs, frame_count)
                detected_qrs = tracked_qrs
                
                # 해독 결과 확인 (항상 최신 상태에서 읽기)
                decode_lock = session_state_ref.get('decode_lock')
                
                for qr in detected_qrs:
                    track_id = qr.get('track_id')
                    if track_id is not None:
                        # decode_results를 매번 최신 상태에서 읽기
                        with decode_lock:
                            decode_results = session_state_ref.get('decode_results', {})
                            if decode_results is None:
                                decode_results = {}
                            
                            if track_id in decode_results:
                                decode_result = decode_results[track_id]
                                if decode_result.get('text'):
                                    qr['text'] = decode_result['text']
                                    qr['success'] = True
                                    qr['method'] = f"YOLO+{decode_result.get('decode_method', 'Unknown')}"
                                    if 'detection' in qr and decode_result.get('quad_xy'):
                                        qr['detection']['quad_xy'] = decode_result['quad_xy']
                        
                        # 해독 큐에 추가 (해독되지 않은 경우에만)
                        if not qr.get('success') and decode_queue:
                            bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                roi = frame[y1:y2, x1:x2]
                                if roi.size > 0:
                                    try:
                                        decode_queue.put_nowait(
                                            (track_id, roi, bbox, (x1, y1), frame_count)
                                        )
                                    except:
                                        pass
            
            # 시각화 직전에 해독 결과 다시 확인 및 업데이트 (yolo_dynamsoft.py 방식)
            decode_lock = session_state_ref.get('decode_lock')
            decode_results = session_state_ref.get('decode_results', {})
            if decode_results is None:
                decode_results = {}
            
            for qr in detected_qrs:
                track_id = qr.get('track_id')
                if track_id is not None:
                    with decode_lock:
                        if track_id in decode_results:
                            decode_result = decode_results[track_id]
                            if decode_result.get('text'):
                                qr['text'] = decode_result['text']
                                qr['success'] = True
                                qr['method'] = f"YOLO+{decode_result.get('decode_method', 'Unknown')}"
                                # quad_xy를 현재 bbox 위치에 맞춰 변환
                                if 'detection' in qr and decode_result.get('quad_xy'):
                                    # 원본 quad_xy는 원본 프레임 좌표일 수 있음
                                    # 현재 bbox 위치를 기준으로 변환 필요
                                    current_bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                                    if current_bbox:
                                        # decode_result의 bbox와 비교하여 offset 계산
                                        decode_bbox = decode_result.get('decode_bbox')
                                        if decode_bbox:
                                            # offset 계산: 현재 bbox - 해독 시점 bbox
                                            dx = current_bbox[0] - decode_bbox[0]
                                            dy = current_bbox[1] - decode_bbox[1]
                                            
                                            quad_xy_original = decode_result['quad_xy']
                                            quad_xy_transformed = []
                                            for qx, qy in quad_xy_original:
                                                quad_xy_transformed.append([int(qx + dx), int(qy + dy)])
                                            qr['detection']['quad_xy'] = quad_xy_transformed
                                        else:
                                            # decode_bbox가 없으면 원본 quad_xy 사용
                                            qr['detection']['quad_xy'] = decode_result['quad_xy']
                                    else:
                                        # bbox가 없으면 원본 quad_xy 사용
                                        qr['detection']['quad_xy'] = decode_result['quad_xy']
            
            # 시각화 직전에 해독 결과 다시 확인 및 업데이트 (yolo_dynamsoft.py 방식)
            decode_lock = session_state_ref.get('decode_lock')
            decode_results = session_state_ref.get('decode_results', {})
            if decode_results is None:
                decode_results = {}
            
            for qr in detected_qrs:
                track_id = qr.get('track_id')
                if track_id is not None:
                    with decode_lock:
                        if track_id in decode_results:
                            decode_result = decode_results[track_id]
                            if decode_result.get('text') and decode_result.get('quad_xy'):
                                # 해독 결과의 quad_xy를 현재 프레임의 bbox 위치에 맞춰 변환
                                current_bbox = qr.get('bbox', qr.get('detection', {}).get('bbox_xyxy'))
                                decode_bbox = decode_result.get('decode_bbox')
                                
                                if current_bbox and decode_bbox and len(current_bbox) == 4 and len(decode_bbox) == 4:
                                    # 중심점 이동량 계산 (yolo_dynamsoft.py 방식)
                                    decode_x1, decode_y1, decode_x2, decode_y2 = decode_bbox
                                    curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
                                    
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
                                elif decode_result.get('quad_xy'):
                                    # bbox 정보가 없으면 원본 quad_xy 사용
                                    qr['detection']['quad_xy'] = decode_result['quad_xy']
            
            # 시각화
            scale_x = display_width / width
            scale_y = display_height / height
            
            for qr in detected_qrs:
                detection = qr.get('detection', {})
                if 'quad_xy' in detection and detection['quad_xy']:
                    quad = np.array(detection['quad_xy'])
                    if len(quad) == 4:
                        quad_array = np.array(quad)
                        center = np.mean(quad_array, axis=0)
                        angles = np.arctan2(quad_array[:, 1] - center[1], 
                                          quad_array[:, 0] - center[0])
                        sorted_indices = np.argsort(angles)
                        sorted_quad = quad_array[sorted_indices]
                        # float로 변환 후 스케일링 (타입 오류 방지)
                        points = sorted_quad.astype(np.float32)
                        points[:, 0] *= scale_x
                        points[:, 1] *= scale_y
                        points = points.astype(np.int32)
                        
                        color = (0, 255, 0) if qr.get('success') else (0, 0, 255)
                        cv2.polylines(display_frame, [points], True, color, 2)
                        
                        # QR 번호만 표시 (해독정보는 표에 표시됨)
                        track_id = qr.get('track_id')
                        if track_id is not None:
                            track_id_text = f"#{track_id}"
                            cv2.putText(display_frame, track_id_text, 
                                      (int(points[0][0]), int(points[0][1]) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                elif 'bbox_xyxy' in detection:
                    x1, y1, x2, y2 = detection['bbox_xyxy']
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    color = (0, 255, 0) if qr.get('success') else (0, 0, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # QR 번호만 표시 (해독정보는 표에 표시됨)
                    track_id = qr.get('track_id')
                    if track_id is not None:
                        track_id_text = f"#{track_id}"
                        cv2.putText(display_frame, track_id_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # FPS 계산
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                info_text = f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames}"
                cv2.putText(display_frame, info_text, (10, display_height - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 일시정지 표시
            if session_state_ref.get('paused', False):
                cv2.putText(display_frame, "PAUSED", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 현재 프레임 및 결과 저장 (app.py 방식: RGB로 변환하여 저장)
            # RGB로 변환하여 저장 (Streamlit image는 RGB를 기대함)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            session_state_ref['current_frame'] = frame_rgb
            session_state_ref['current_results'] = {
                'detected_qrs': detected_qrs,
                'frame_num': frame_count,
                'total_frames': total_frames,
                'fps': fps
            }
            
            # 현재 프레임의 해독된 QR을 누적 리스트에 추가 (웹에서 표시되는 것과 동일)
            decoded_qrs = [qr for qr in detected_qrs if qr.get('success')]
            if decoded_qrs:
                if 'accumulated_qr_records' not in session_state_ref:
                    session_state_ref['accumulated_qr_records'] = []
                
                accumulated_records = session_state_ref['accumulated_qr_records']
                for qr in decoded_qrs:
                    track_id = qr.get('track_id')
                    qr_text = qr.get('text', '')
                    confidence = qr.get('confidence', None)
                    method = qr.get('method', 'Unknown')
                    
                    # 같은 프레임과 track_id가 이미 있으면 업데이트, 없으면 추가
                    found = False
                    for record in accumulated_records:
                        if (record.get('track_id') == track_id and 
                            record.get('frame') == frame_count):
                            # 기존 기록 업데이트
                            record['text'] = qr_text
                            record['confidence'] = confidence if confidence is not None else record.get('confidence')
                            record['method'] = method
                            found = True
                            break
                    
                    if not found:
                        # 새 기록 추가
                        accumulated_records.append({
                            'frame': frame_count,
                            'track_id': track_id,
                            'text': qr_text,
                            'confidence': confidence,
                            'method': method
                        })
                
                session_state_ref['accumulated_qr_records'] = accumulated_records
            
            # 비디오에 쓰기
            if out_video.isOpened():
                out_video.write(display_frame)
            
            
            # 첫 프레임은 빠르게 표시되도록 딜레이 없음
            # 이후 프레임은 적절한 딜레이
            if frame_count == 1:
                time.sleep(0.01)  # 첫 프레임은 빠르게
            else:
                time.sleep(0.05)  # 이후 프레임은 적절한 속도
        
        # 정리
        if stop_decode_worker:
            stop_decode_worker.set()
            if decode_queue:
                try:
                    decode_queue.put(None, timeout=0.1)
                except:
                    pass
        
        if decode_worker_thread:
            decode_worker_thread.join(timeout=1.0)
        
        # 비디오 릴리스 및 파일 경로 저장 (종료 시에도 처리된 결과 저장)
        if out_video.isOpened():
            out_video.release()
        
        # 비디오 파일 경로를 session_state에 저장 (종료 시에도 저장)
        # 파일이 존재하는 경우에만 경로 저장
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            session_state_ref['temp_video_path'] = output_video_path
        
        # 누적된 QR 기록을 CSV로 저장
        accumulated_records = session_state_ref.get('accumulated_qr_records', [])
        if accumulated_records:
            import csv
            import io
            csv_path = os.path.join(output_dir, f"qr_records_{run_id}.csv")
            
            # CSV 파일 작성 (웹 표시와 동일: 프레임, QR번호, 해독정보, 신뢰도만 저장)
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['프레임', 'QR번호', '해독정보', '신뢰도']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in accumulated_records:
                    confidence = record.get('confidence')
                    if confidence is None:
                        confidence_str = 'N/A'
                    elif isinstance(confidence, (int, float)):
                        confidence_str = f"{confidence:.3f}"
                    else:
                        confidence_str = str(confidence)
                    
                    writer.writerow({
                        '프레임': record.get('frame', 0),
                        'QR번호': record.get('track_id', 'N/A'),
                        '해독정보': record.get('text', ''),
                        '신뢰도': confidence_str
                    })
            
            session_state_ref['temp_qr_records_path'] = csv_path
        
        cap.release()
        
        session_state_ref['processing'] = False
        session_state_ref['processing_completed'] = True  # 처리 완료 플래그 (종료 시에도 설정)
        
        # 프레임이 하나도 처리되지 않은 경우 경고
        if frame_count == 0:
            error_msg = "경고: 프레임이 하나도 처리되지 않았습니다. processing 상태를 확인하세요."
            print(f"[ERROR] {error_msg}")
            session_state_ref['error'] = error_msg
        
    except Exception as e:
        import traceback
        error_msg = f"처리 중 오류 발생: {e}\n{traceback.format_exc()}"
        session_state_ref['processing'] = False
        session_state_ref['error'] = error_msg
        print(f"ERROR in video thread: {error_msg}")  # 콘솔에도 출력

def process_batch_files_thread(files_info, output_dir, conf_threshold, iou_threshold,
                              use_preprocessing, use_clahe, use_normalize, clahe_clip_limit,
                              detect_both_frames, session_state_ref):
    """여러 파일을 순차적으로 처리하는 스레드"""
    try:
        total_files = len(files_info)
        session_state_ref['batch_processing'] = True
        session_state_ref['current_batch_file_index'] = 0
        session_state_ref['batch_results'] = {}
        session_state_ref['batch_files'] = [f['name'] for f in files_info]
        
        for idx, file_info in enumerate(files_info):
            if session_state_ref.get('should_stop', False):
                break
            
            file_path = file_info['path']
            file_name = file_info['name']
            file_ext = file_info['ext']
            is_image = file_info['is_image']
            
            session_state_ref['current_batch_file_index'] = idx
            session_state_ref['batch_results'][file_name] = {
                'status': 'processing',
                'error': None,
                'video_path': None,
                'csv_path': None
            }
            
            try:
                if is_image:
                    # 이미지 처리
                    display_frame, detected_qrs, error = process_image_file(
                        file_path, conf_threshold, iou_threshold,
                        use_preprocessing, use_clahe, use_normalize,
                        clahe_clip_limit, detect_both_frames
                    )
                    
                    if error:
                        session_state_ref['batch_results'][file_name]['status'] = 'error'
                        session_state_ref['batch_results'][file_name]['error'] = error
                    else:
                        # 이미지 결과 저장 및 프레임 저장
                        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_image_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{run_id}.jpg")
                        if display_frame is not None:
                            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(output_image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                            session_state_ref['batch_results'][file_name]['image_path'] = output_image_path
                        
                        # CSV 저장
                        if detected_qrs:
                            decoded_qrs = [qr for qr in detected_qrs if qr.get('success')]
                            if decoded_qrs:
                                import csv
                                csv_path = os.path.join(output_dir, f"qr_records_{os.path.splitext(file_name)[0]}_{run_id}.csv")
                                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                                    fieldnames = ['프레임', 'QR번호', '해독정보', '신뢰도']
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    for qr in decoded_qrs:
                                        confidence = qr.get('confidence')
                                        confidence_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else 'N/A'
                                        writer.writerow({
                                            '프레임': 1,
                                            'QR번호': qr.get('track_id', 'N/A'),
                                            '해독정보': qr.get('text', ''),
                                            '신뢰도': confidence_str
                                        })
                                session_state_ref['batch_results'][file_name]['csv_path'] = csv_path
                        
                        session_state_ref['batch_results'][file_name]['status'] = 'completed'
                else:
                    # 비디오 처리 - 별도 스레드로 실행하고 실시간 프레임 업데이트
                    video_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
                    os.makedirs(video_output_dir, exist_ok=True)
                    
                    # 비디오 처리용 임시 session_state 생성
                    video_session_state = {
                        'processing': True,
                        'paused': False,
                        'should_stop': False,
                        'yolo_model': session_state_ref.get('yolo_model'),
                        'dbr_reader': session_state_ref.get('dbr_reader'),
                        'decode_queue': None,
                        'decode_results': {},
                        'decode_lock': threading.Lock(),
                        'accumulated_qr_records': [],
                        'qr_tracker': None
                    }
                    
                    # 비디오 처리 스레드 시작
                    video_thread = threading.Thread(
                        target=process_video_thread,
                        args=(file_path, video_output_dir, conf_threshold, iou_threshold,
                             use_preprocessing, use_clahe, use_normalize,
                             clahe_clip_limit, detect_both_frames, video_session_state),
                        daemon=True
                    )
                    add_script_run_ctx(video_thread)
                    video_thread.start()
                    
                    # 비디오 처리 완료 대기
                    
                    # 비디오 처리 완료 대기
                    video_thread.join()
                    
                    # 결과 확인
                    if video_session_state.get('temp_video_path'):
                        session_state_ref['batch_results'][file_name]['video_path'] = video_session_state['temp_video_path']
                    if video_session_state.get('temp_qr_records_path'):
                        session_state_ref['batch_results'][file_name]['csv_path'] = video_session_state['temp_qr_records_path']
                    
                    if video_session_state.get('error'):
                        session_state_ref['batch_results'][file_name]['status'] = 'error'
                        session_state_ref['batch_results'][file_name]['error'] = video_session_state['error']
                    else:
                        session_state_ref['batch_results'][file_name]['status'] = 'completed'
                        
            except Exception as e:
                import traceback
                error_msg = f"파일 처리 중 오류: {str(e)}\n{traceback.format_exc()}"
                session_state_ref['batch_results'][file_name]['status'] = 'error'
                session_state_ref['batch_results'][file_name]['error'] = error_msg
        
        session_state_ref['batch_processing'] = False
        session_state_ref['processing'] = False
        session_state_ref['processing_completed'] = True
        
    except Exception as e:
        import traceback
        error_msg = f"배치 처리 중 오류 발생: {e}\n{traceback.format_exc()}"
        session_state_ref['batch_processing'] = False
        session_state_ref['processing'] = False
        session_state_ref['error'] = error_msg

def main():
    st.title("📱 QR 코드 탐지 시스템")
    st.markdown("---")
    
    # 사이드바 - 탐지 옵션
    with st.sidebar:
        st.header("⚙️ 탐지 옵션")
        
        conf_threshold = st.slider("신뢰도 임계값 (Confidence)", 
                                  min_value=0.0, max_value=1.0, 
                                  value=0.25, step=0.01)
        
        iou_threshold = st.slider("IoU 임계값", 
                                 min_value=0.0, max_value=1.0, 
                                 value=0.5, step=0.01)
        
        st.markdown("---")
        st.subheader("전처리 옵션")
        
        use_preprocessing = st.checkbox("전처리 사용", value=False,
                                       help="전처리를 사용하면 CLAHE와 정규화가 적용됩니다.")
        
        use_clahe = st.checkbox("CLAHE 적용", value=True, 
                               help="CLAHE (Contrast Limited Adaptive Histogram Equalization)를 사용하여 대비를 개선합니다. 전처리 사용 시 적용됩니다.")
        
        clahe_clip_limit = st.slider("CLAHE Clip Limit", 
                                    min_value=0.5, max_value=5.0, 
                                    value=2.0, step=0.1,
                                    help="CLAHE의 clip limit 값. 낮을수록 대비 개선이 약하고 오탐지가 감소합니다. (기본값: 2.0)")
        
        if not use_preprocessing:
            st.info("💡 전처리를 활성화하면 CLAHE 설정이 적용됩니다.")
        
        if use_preprocessing:
            use_normalize = st.checkbox("정규화 적용", value=True,
                                       help="이미지 정규화를 적용하여 대비를 끌어올립니다.")
            detect_both_frames = st.checkbox("원본과 전처리 프레임 모두 탐지", value=True,
                                            help="원본 프레임과 전처리된 프레임 모두에서 QR 코드를 탐지합니다.")
        else:
            use_normalize = False
            detect_both_frames = True
        
        st.markdown("---")
        st.header("📁 파일 업로드")
        
        # 업로드 모드 선택
        upload_mode = st.radio(
            "업로드 모드",
            ["단일 파일", "여러 파일 (배치 처리)"],
            horizontal=True,
            help="단일 파일 또는 여러 파일을 한번에 처리할 수 있습니다. (비디오는 단일 파일만 가능, 이미지는 여러 파일 가능)"
        )
        
        if upload_mode == "단일 파일":
            uploaded_files = st.file_uploader(
                "비디오 또는 이미지 파일을 선택하세요",
                type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
                help="비디오 파일 또는 이미지 파일을 업로드하세요",
                accept_multiple_files=False
            )
            # 단일 파일 모드에서는 리스트로 변환
            if uploaded_files:
                uploaded_files = [uploaded_files]
            else:
                uploaded_files = []
        else:
            uploaded_files = st.file_uploader(
                "이미지 파일을 선택하세요 (여러 개 선택 가능)",
                type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
                help="여러 이미지 파일을 선택하여 배치 처리할 수 있습니다. 비디오 파일은 단일 파일만 처리 가능합니다. Ctrl+클릭 또는 Shift+클릭으로 여러 파일 선택",
                accept_multiple_files=True
            )
            if uploaded_files is None:
                uploaded_files = []
            
            # 비디오 파일이 여러 개 선택되었는지 확인
            if uploaded_files:
                video_files = [f for f in uploaded_files if os.path.splitext(f.name)[1].lower() in ['.mp4', '.avi', '.mov']]
                if len(video_files) > 1:
                    st.error(f"⚠️ 비디오 파일은 단일 파일만 처리할 수 있습니다. 현재 {len(video_files)}개의 비디오 파일이 선택되었습니다.")
                    st.info("💡 비디오 파일은 하나만 선택하거나, 단일 파일 모드를 사용하세요.")
                elif len(video_files) == 1 and len(uploaded_files) > 1:
                    st.warning(f"⚠️ 비디오 파일과 이미지 파일을 함께 선택했습니다. 비디오 파일은 단일 파일 모드에서만 처리됩니다.")
        
        # 업로드된 파일 목록 표시
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)}개 파일이 선택되었습니다:")
            for i, file in enumerate(uploaded_files, 1):
                file_ext = os.path.splitext(file.name)[1].lower()
                file_type = "🖼️ 이미지" if file_ext in ['.jpg', '.jpeg', '.png'] else "🎬 비디오"
                st.text(f"  {i}. {file_type} - {file.name}")
        
        # 단일 파일 모드 호환성을 위해 uploaded_file 변수 유지
        uploaded_file = uploaded_files[0] if len(uploaded_files) == 1 else None
        
        st.markdown("---")
        
        # 모델 초기화 버튼
        if st.button("🔧 모델 초기화", width='stretch'):
            with st.spinner("모델 초기화 중..."):
                yolo_model, dbr_reader, error = initialize_models()
                if error:
                    st.error(error)
                else:
                    st.session_state.yolo_model = yolo_model
                    st.session_state.dbr_reader = dbr_reader
                    if yolo_model:
                        st.success("✅ YOLO 모델 초기화 완료")
                    if dbr_reader:
                        st.success("✅ Dynamsoft 초기화 완료")
    
    # 메인 영역 - 처리 화면과 QR 정보를 나란히 배치
    col_video, col_qr = st.columns([1.5, 1])
    
    with col_video:
        # 헤더와 프레임 정보를 같은 줄에 표시
        col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
        with col_header1:
            st.header("📺 처리 화면")
        with col_header2:
            # 단일 파일 처리 모드일 때만 프레임 정보 표시
            if not (st.session_state.get('batch_processing', False) or 
                   (st.session_state.get('processing_completed', False) and 
                    len(st.session_state.get('batch_results', {})) > 0)):
                current_results = st.session_state.current_results
                if current_results:
                    frame_num = current_results.get('frame_num', 0)
                    total_frames = current_results.get('total_frames', 0)
                    st.metric("현재 프레임", f"{frame_num}/{total_frames}", delta=None)
                else:
                    st.metric("현재 프레임", "0/0", delta=None)
            else:
                st.metric("현재 프레임", "-", delta=None)
        with col_header3:
            # 단일 파일 처리 모드일 때만 FPS 표시
            if not (st.session_state.get('batch_processing', False) or 
                   (st.session_state.get('processing_completed', False) and 
                    len(st.session_state.get('batch_results', {})) > 0)):
                current_results = st.session_state.current_results
                if current_results:
                    fps = current_results.get('fps', 0.0)
                    st.metric("FPS", f"{fps:.2f}", delta=None)
                else:
                    st.metric("FPS", "0.00", delta=None)
            else:
                st.metric("FPS", "-", delta=None)
        
        video_placeholder = st.empty()
        
        # 배치 처리 모드 확인
        is_batch_mode = (st.session_state.get('batch_processing', False) or 
                        (st.session_state.get('processing_completed', False) and 
                         len(st.session_state.get('batch_results', {})) > 0))
        
        # current_frame 초기화
        current_frame = None
        
        if is_batch_mode:
            # 배치 처리 중이거나 완료된 경우 화면 결과 표시 안 함
            if st.session_state.get('batch_processing', False):
                video_placeholder.info("🔄 배치 처리 중... 처리 완료 후 결과를 다운로드할 수 있습니다.")
            else:
                video_placeholder.info("✅ 배치 처리 완료! 아래에서 결과를 다운로드할 수 있습니다.")
        else:
            # 단일 파일 처리 모드
            current_frame = st.session_state.get('current_frame')
        
        # 단일 파일 처리 모드일 때만 프레임 표시
        if current_frame is not None and not is_batch_mode:
            try:
                if isinstance(current_frame, np.ndarray):
                    # 이미지를 최대 높이로 제한하여 한 화면에 들어오도록 조정
                    h, w = current_frame.shape[:2]
                    max_height = 500  # 최대 높이 설정
                    if h > max_height:
                        scale = max_height / h
                        new_width = int(w * scale)
                        new_height = int(h * scale)
                        current_frame_resized = cv2.resize(current_frame, (new_width, new_height))
                    else:
                        current_frame_resized = current_frame
                    
                    # 이미 RGB로 저장되어 있으므로 바로 표시
                    # MediaFileStorageError 억제를 위해 예외 처리
                    try:
                        video_placeholder.image(current_frame_resized, channels="RGB", width='stretch')
                    except Exception as img_error:
                        # MediaFileStorageError는 내부 캐시 관련 경고로 무시 가능
                        error_str = str(img_error)
                        if 'MediaFileStorageError' not in error_str and 'MediaFileHandler' not in error_str:
                            # 다른 실제 에러는 무시 (이미 내부적으로 처리됨)
                            pass
                else:
                    video_placeholder.error(f"프레임 형식 오류: {type(current_frame)}")
            except Exception as e:
                # MediaFileStorageError 관련 예외는 무시
                error_msg = str(e)
                if 'MediaFileStorageError' not in error_msg and 'MediaFileHandler' not in error_msg:
                    # 실제 에러만 표시
                    pass
        else:
            if st.session_state.get('processing', False):
                video_placeholder.info("🔄 처리 중... 첫 프레임을 준비하고 있습니다. 잠시만 기다려주세요.")
            else:
                video_placeholder.info("처리할 파일을 업로드하고 시작 버튼을 클릭하세요.")
        
        # 제어 버튼
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            is_batch_mode = len(uploaded_files) > 1
            button_label = "▶️ 배치 처리 시작" if is_batch_mode else "▶️ 처리 시작"
            if st.button(button_label, disabled=st.session_state.processing or st.session_state.batch_processing, width='stretch'):
                if not uploaded_files or len(uploaded_files) == 0:
                    st.warning("⚠️ 먼저 파일을 업로드하세요.")
                elif st.session_state.yolo_model is None:
                    st.warning("⚠️ 먼저 모델을 초기화하세요.")
                else:
                    # 처리 시작
                    st.session_state.processing = True
                    st.session_state.paused = False
                    st.session_state.should_stop = False
                    
                    # 단일 파일 처리 모드인 경우 배치 처리 상태 초기화
                    if len(uploaded_files) == 1:
                        st.session_state.batch_processing = False
                        st.session_state.processing_completed = False
                        st.session_state.batch_results = {}
                        st.session_state.batch_files = []
                        st.session_state.current_batch_file_index = 0
                    
                    # 임시 디렉토리 생성
                    import tempfile
                    temp_dir = tempfile.mkdtemp(prefix="qr_temp_")
                    st.session_state.output_dir = temp_dir
                    st.session_state.temp_dir = temp_dir
                    
                    if is_batch_mode:
                        # 배치 처리 모드
                        # 비디오 파일이 여러 개인지 확인
                        video_files = [f for f in uploaded_files if os.path.splitext(f.name)[1].lower() in ['.mp4', '.avi', '.mov']]
                        if len(video_files) > 1:
                            st.error(f"⚠️ 비디오 파일은 단일 파일만 처리할 수 있습니다. 현재 {len(video_files)}개의 비디오 파일이 선택되었습니다.")
                            st.info("💡 비디오 파일은 하나만 선택하거나, 단일 파일 모드를 사용하세요.")
                            st.session_state.processing = False
                        else:
                            files_info = []
                            for uploaded_file in uploaded_files:
                                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(temp_file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                                is_image = file_ext in ['.jpg', '.jpeg', '.png']
                                
                                files_info.append({
                                    'name': uploaded_file.name,
                                    'path': temp_file_path,
                                    'ext': file_ext,
                                    'is_image': is_image
                                })
                            
                            # 배치 처리 스레드 시작
                            batch_thread = threading.Thread(
                                target=process_batch_files_thread,
                                args=(files_info, temp_dir, conf_threshold, iou_threshold,
                                     use_preprocessing, use_clahe, use_normalize,
                                     clahe_clip_limit, detect_both_frames, st.session_state),
                                daemon=True
                            )
                            add_script_run_ctx(batch_thread)
                            batch_thread.start()
                            st.session_state.processing_thread = batch_thread
                            st.success(f"✅ {len(uploaded_files)}개 파일 배치 처리 시작!")
                            st.rerun()
                    else:
                        # 단일 파일 처리 모드
                        uploaded_file = uploaded_files[0]
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # 이미지 파일인지 비디오 파일인지 확인
                        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                        is_image = file_ext in ['.jpg', '.jpeg', '.png']
                        
                        if is_image:
                            # 이미지 처리
                            display_frame, detected_qrs, error = process_image_file(
                                temp_file_path, conf_threshold, iou_threshold,
                                use_preprocessing, use_clahe, use_normalize,
                                clahe_clip_limit, detect_both_frames
                            )
                            if error:
                                st.error(f"이미지 처리 오류: {error}")
                                st.session_state.processing = False
                            else:
                                # BGR을 RGB로 변환 (Streamlit image는 RGB를 기대함)
                                if display_frame is not None:
                                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                                    st.session_state.current_frame = frame_rgb.copy()
                                    
                                    # 이미지 결과 저장
                                    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_image_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}_{run_id}.jpg")
                                    cv2.imwrite(output_image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                                    st.session_state.temp_image_path = output_image_path
                                    
                                    # CSV 저장 (QR 기록이 있는 경우)
                                    if detected_qrs:
                                        decoded_qrs = [qr for qr in detected_qrs if qr.get('success')]
                                        if decoded_qrs:
                                            import csv
                                            csv_path = os.path.join(temp_dir, f"qr_records_{os.path.splitext(uploaded_file.name)[0]}_{run_id}.csv")
                                            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                                                fieldnames = ['프레임', 'QR번호', '해독정보', '신뢰도']
                                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                                writer.writeheader()
                                                for qr in decoded_qrs:
                                                    confidence = qr.get('confidence')
                                                    confidence_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else 'N/A'
                                                    writer.writerow({
                                                        '프레임': 1,
                                                        'QR번호': qr.get('track_id', 'N/A'),
                                                        '해독정보': qr.get('text', ''),
                                                        '신뢰도': confidence_str
                                                    })
                                            st.session_state.temp_qr_records_path = csv_path
                                else:
                                    st.session_state.current_frame = None
                                st.session_state.current_results = {
                                    'detected_qrs': detected_qrs,
                                    'frame_num': 1,
                                    'total_frames': 1,
                                    'fps': 0.0
                                }
                                st.session_state.processing = False
                                st.session_state.processing_completed = True
                                st.success("✅ 이미지 처리 완료!")
                                st.rerun()  # 화면 갱신
                        else:
                            # 비디오 처리 스레드 시작 - session_state 참조 전달
                            processing_thread = threading.Thread(
                                target=process_video_thread,
                                args=(temp_file_path, temp_dir, conf_threshold, iou_threshold,
                                     use_preprocessing, use_clahe, use_normalize, 
                                     clahe_clip_limit, detect_both_frames, st.session_state),
                                daemon=True
                            )
                            # ★★★ [핵심 수정] 스레드에 Streamlit 컨텍스트 주입
                            add_script_run_ctx(processing_thread)
                            processing_thread.start()
                            st.session_state.processing_thread = processing_thread
                            st.success("✅ 처리 시작!")
                            time.sleep(0.1)  # 스레드 시작 대기
                            st.rerun()  # 즉시 화면 갱신하여 처리 화면 표시
        
        with col_btn2:
            pause_button_label = "⏸️ 일시정지" if not st.session_state.paused else "▶️ 재개"
            if st.button(pause_button_label, disabled=not st.session_state.processing, width='stretch'):
                st.session_state.paused = not st.session_state.paused
        
        with col_btn3:
            if st.button("⏹️ 종료", disabled=not st.session_state.processing, width='stretch'):
                st.session_state.should_stop = True
                st.session_state.processing = False
                st.warning("⏹️ 처리 종료 중... (종료 시점까지의 결과를 다운로드할 수 있습니다)")
        
        # 처리 상태
        batch_results = st.session_state.get('batch_results', {})
        has_batch_results = len(batch_results) > 0
        
        if st.session_state.batch_processing or (has_batch_results and st.session_state.get('processing_completed', False)):
            # 배치 처리 중이거나 완료된 경우
            current_idx = st.session_state.get('current_batch_file_index', 0)
            total_files = len(st.session_state.get('batch_files', []))
            if total_files == 0:
                total_files = len([f for f in batch_results.keys()])
            
            completed = len([r for r in batch_results.values() if r.get('status') == 'completed'])
            errors = len([r for r in batch_results.values() if r.get('status') == 'error'])
            
            if st.session_state.batch_processing:
                # 배치 처리 중
                st.progress((completed + errors) / total_files if total_files > 0 else 0)
                st.info(f"🔄 배치 처리 중... ({completed + errors}/{total_files} 완료, {errors}개 오류)")
            else:
                # 배치 처리 완료
                st.progress(1.0)  # 100% 완료
                st.success(f"✅ 배치 처리 완료! ({completed}/{total_files} 성공, {errors}개 오류)")
            
            # 배치 처리 결과 미리보기
            if batch_results:
                with st.expander("📋 배치 처리 진행 상황", expanded=True):
                    for file_name, result in batch_results.items():
                        status = result.get('status', 'pending')
                        if status == 'completed':
                            st.success(f"✅ {file_name}")
                        elif status == 'error':
                            st.error(f"❌ {file_name}: {result.get('error', '알 수 없는 오류')[:50]}")
                        elif status == 'processing':
                            st.info(f"🔄 {file_name} (처리 중...)")
                        else:
                            st.text(f"⏳ {file_name} (대기 중...)")
        elif st.session_state.processing:
            if st.session_state.paused:
                st.info("⏸️ 일시정지 중...")
            else:
                st.info("▶️ 처리 중...")
        
    
    with col_qr:
        st.header("📊 해독된 QR 정보")
        
        # 배치 처리 중일 때는 QR 정보 표시 안 함
        if st.session_state.get('batch_processing', False) or \
           (st.session_state.get('processing_completed', False) and 
            len(st.session_state.get('batch_results', {})) > 0):
            current_results = None
        else:
            current_results = st.session_state.current_results
        
        if current_results and current_results.get('detected_qrs'):
            detected_qrs = current_results['detected_qrs']
            decoded_qrs = [qr for qr in detected_qrs if qr.get('success')]
            
            if decoded_qrs:
                # 표 형식으로 표시
                try:
                    import pandas as pd
                    
                    table_data = []
                    frame_num = current_results.get('frame_num', 0)
                    
                    for qr in decoded_qrs:
                        track_id = qr.get('track_id', 'N/A')
                        qr_text = qr.get('text', '')
                        confidence = qr.get('confidence', 0.0)
                        
                        table_data.append({
                            '프레임': frame_num,
                            'QR번호': track_id,
                            '해독정보': qr_text[:40] + ('...' if len(qr_text) > 40 else ''),  # 좀 더 짧게
                            '신뢰도': f"{confidence:.2f}" if isinstance(confidence, (int, float)) else 'N/A'
                        })
                    
                    if table_data:
                        df = pd.DataFrame(table_data)
                        # 표 형식으로 표시 (인덱스 제거)
                        st.dataframe(
                            df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                "프레임": st.column_config.NumberColumn("프레임", format="%d", width="small"),
                                "QR번호": st.column_config.NumberColumn("QR번호", format="%d", width="small"),
                                "해독정보": st.column_config.TextColumn("해독정보", width="medium"),
                                "신뢰도": st.column_config.TextColumn("신뢰도", width="small")
                            }
                        )
                except ImportError:
                    # pandas가 없으면 기존 방식으로 표시
                    for qr in decoded_qrs:
                        track_id = qr.get('track_id', 'N/A')
                        qr_text = qr.get('text', '')
                        frame_num = current_results.get('frame_num', 0)
                        confidence = qr.get('confidence', 0.0)
                        st.text(f"프레임 {frame_num} | QR #{track_id} | {qr_text[:30]} | 신뢰도: {confidence:.2f}")
            else:
                st.info("현재 프레임에서 해독된 QR이 없습니다.")
        else:
            st.info("처리 결과가 없습니다.")
        
        # 처리 완료 후 현재 결과 다운로드 섹션
        batch_results = st.session_state.get('batch_results', {})
        is_batch_completed = (st.session_state.get('processing_completed', False) and 
                             not st.session_state.processing and 
                             not st.session_state.batch_processing and 
                             len(batch_results) > 0)
        
        if is_batch_completed:
            # 배치 처리 완료
            st.markdown("---")
            completed_count = len([r for r in batch_results.values() if r.get('status') == 'completed'])
            error_count = len([r for r in batch_results.values() if r.get('status') == 'error'])
            st.success(f"✅ 배치 처리 완료! ({completed_count}개 성공, {error_count}개 오류)")
            
            # 배치 처리 결과 다운로드
            st.subheader("📦 배치 처리 결과 다운로드")
            
            # 모든 결과 파일 수집
            all_files = []
            for file_name, result in batch_results.items():
                if result.get('status') == 'completed':
                    if result.get('video_path') and os.path.exists(result['video_path']):
                        all_files.append(('video', result['video_path'], file_name))
                    if result.get('image_path') and os.path.exists(result['image_path']):
                        all_files.append(('image', result['image_path'], file_name))
                    if result.get('csv_path') and os.path.exists(result['csv_path']):
                        all_files.append(('csv', result['csv_path'], file_name))
            
            if all_files:
                # ZIP으로 전체 다운로드
                def create_batch_zip():
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file_type, file_path, original_name in all_files:
                            if os.path.exists(file_path):
                                base_name = os.path.splitext(original_name)[0]
                                ext = os.path.splitext(file_path)[1]
                                zip_name = f"{base_name}{ext}"
                                zip_file.write(file_path, zip_name)
                    zip_buffer.seek(0)
                    return zip_buffer.getvalue()
                
                zip_data = create_batch_zip()
                st.download_button(
                    label=f"📦 전체 결과 다운로드 ({len(all_files)}개 파일)",
                    data=zip_data,
                    file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width='stretch'
                )
                
                # 개별 파일 다운로드
                st.subheader("📥 개별 파일 다운로드")
                for file_name, result in batch_results.items():
                    if result.get('status') == 'completed':
                        with st.expander(f"📁 {file_name}", expanded=False):
                            if result.get('video_path') and os.path.exists(result['video_path']):
                                with open(result['video_path'], "rb") as f:
                                    st.download_button(
                                        label="⬇️ 영상 다운로드",
                                        data=f.read(),
                                        file_name=os.path.basename(result['video_path']),
                                        mime="video/mp4",
                                        key=f"batch_video_{file_name}"
                                    )
                            if result.get('image_path') and os.path.exists(result['image_path']):
                                with open(result['image_path'], "rb") as f:
                                    st.download_button(
                                        label="⬇️ 이미지 다운로드",
                                        data=f.read(),
                                        file_name=os.path.basename(result['image_path']),
                                        mime="image/jpeg",
                                        key=f"batch_image_{file_name}"
                                    )
                            if result.get('csv_path') and os.path.exists(result['csv_path']):
                                with open(result['csv_path'], "rb") as f:
                                    st.download_button(
                                        label="⬇️ QR 기록 다운로드 (CSV)",
                                        data=f.read(),
                                        file_name=os.path.basename(result['csv_path']),
                                        mime="text/csv",
                                        key=f"batch_csv_{file_name}"
                                    )
                
                # output_results에 저장하기 버튼
                st.markdown("---")
                if st.button("💾 output_results에 저장하기", width='stretch', type="primary"):
                    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(OUTPUT_BASE_DIR, f"batch_{run_id}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    saved_files = []
                    for file_type, file_path, original_name in all_files:
                        if os.path.exists(file_path):
                            dest_path = os.path.join(output_dir, os.path.basename(file_path))
                            shutil.copy2(file_path, dest_path)
                            saved_files.append(dest_path)
                    
                    if saved_files:
                        st.success(f"✅ {len(saved_files)}개 파일이 저장되었습니다: {output_dir}")
                        # 상태 초기화는 사용자가 "새 작업 시작" 버튼을 눌러야 함
                        # st.session_state.processing_completed = False
                        # st.session_state.batch_results = {}
        
        elif st.session_state.get('processing_completed', False) and not st.session_state.processing:
            # 단일 파일 처리 완료
            st.markdown("---")
            st.success("✅ 처리 완료! 결과를 다운로드하거나 저장할 수 있습니다.")
            
            temp_video_path = st.session_state.get('temp_video_path')
            temp_image_path = st.session_state.get('temp_image_path')
            temp_qr_records_path = st.session_state.get('temp_qr_records_path')
            
            # 이미지 파일인지 비디오 파일인지 확인
            is_image_file = temp_image_path is not None and os.path.exists(temp_image_path)
            is_video_file = temp_video_path is not None and os.path.exists(temp_video_path)
            
            if is_image_file:
                # 이미지 파일 다운로드
                col_save1, col_save2 = st.columns(2)
                
                with col_save1:
                    st.subheader("🖼️ 이미지 다운로드")
                    with open(temp_image_path, "rb") as f:
                        st.download_button(
                            label="⬇️ 이미지 다운로드",
                            data=f.read(),
                            file_name=os.path.basename(temp_image_path),
                            mime="image/jpeg",
                            width='stretch'
                        )
                
                with col_save2:
                    if temp_qr_records_path and os.path.exists(temp_qr_records_path):
                        st.subheader("📊 해독된 QR 기록")
                        with open(temp_qr_records_path, "rb") as f:
                            st.download_button(
                                label="⬇️ QR 기록 다운로드 (CSV)",
                                data=f.read(),
                                file_name=os.path.basename(temp_qr_records_path),
                                mime="text/csv",
                                width='stretch'
                            )
                
                # output_results에 저장하기 버튼
                if temp_image_path or temp_qr_records_path:
                    st.markdown("---")
                    if st.button("💾 output_results에 저장하기", width='stretch', type="primary"):
                        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = os.path.join(OUTPUT_BASE_DIR, f"single_{run_id}")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        saved_files = []
                        if temp_image_path and os.path.exists(temp_image_path):
                            dest_path = os.path.join(output_dir, os.path.basename(temp_image_path))
                            shutil.copy2(temp_image_path, dest_path)
                            saved_files.append(dest_path)
                        if temp_qr_records_path and os.path.exists(temp_qr_records_path):
                            dest_path = os.path.join(output_dir, os.path.basename(temp_qr_records_path))
                            shutil.copy2(temp_qr_records_path, dest_path)
                            saved_files.append(dest_path)
                        
                        if saved_files:
                            st.success(f"✅ {len(saved_files)}개 파일이 저장되었습니다: {output_dir}")
                            st.session_state.processing_completed = False
                            st.session_state.temp_image_path = None
                            st.session_state.temp_qr_records_path = None
                            st.rerun()
            
            elif is_video_file:
                # 비디오 파일 다운로드
                col_save1, col_save2 = st.columns(2)
                
                with col_save1:
                    st.subheader("📹 영상 다운로드")
                    with open(temp_video_path, "rb") as f:
                        st.download_button(
                            label="⬇️ 영상 다운로드",
                            data=f.read(),
                            file_name=os.path.basename(temp_video_path),
                            mime="video/mp4",
                            width='stretch'
                        )
                
                with col_save2:
                    if temp_qr_records_path and os.path.exists(temp_qr_records_path):
                        st.subheader("📊 해독된 QR 기록")
                        with open(temp_qr_records_path, "rb") as f:
                            st.download_button(
                                label="⬇️ QR 기록 다운로드 (CSV)",
                                data=f.read(),
                                file_name=os.path.basename(temp_qr_records_path),
                                mime="text/csv",
                                width='stretch'
                            )
                
                # output_results에 저장하기 버튼
                if temp_video_path or temp_qr_records_path:
                    st.markdown("---")
                    if st.button("💾 output_results에 저장하기", width='stretch', type="primary"):
                        # output_results 디렉토리 생성
                        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = os.path.join(OUTPUT_BASE_DIR, run_id)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 파일 복사
                        saved_files = []
                        if temp_video_path and os.path.exists(temp_video_path):
                            dest_video = os.path.join(output_dir, os.path.basename(temp_video_path))
                            shutil.copy2(temp_video_path, dest_video)
                            saved_files.append(dest_video)
                        
                        if temp_qr_records_path and os.path.exists(temp_qr_records_path):
                            dest_csv = os.path.join(output_dir, os.path.basename(temp_qr_records_path))
                            shutil.copy2(temp_qr_records_path, dest_csv)
                            saved_files.append(dest_csv)
                        
                        if saved_files:
                            st.success(f"✅ {len(saved_files)}개 파일이 저장되었습니다: {output_dir}")
                            # 저장 완료 후 플래그 리셋
                            st.session_state.processing_completed = False
    
    # 결과 다운로드 섹션
    st.markdown("---")
    st.header("💾 저장된 결과")
    
    # output_results 디렉토리가 없으면 생성
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_BASE_DIR):
        result_dirs = sorted([d for d in os.listdir(OUTPUT_BASE_DIR) 
                             if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))], 
                           reverse=True)
        
        if result_dirs:
            selected_dir = st.selectbox("결과 디렉토리 선택", result_dirs)
            
            if selected_dir:
                result_dir_path = os.path.join(OUTPUT_BASE_DIR, selected_dir)
                files = os.listdir(result_dir_path)
                
                video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]
                csv_files = [f for f in files if f.endswith('.csv')]
                
                # 세션 상태에 선택된 파일 목록 초기화
                session_key = f"selected_files_{selected_dir}"
                if session_key not in st.session_state:
                    st.session_state[session_key] = []
                
                st.subheader("📁 파일 선택")
                
                # 모든 파일 목록 (영상 + CSV)
                all_files = {}
                for video_file in video_files:
                    all_files[video_file] = {'type': 'video', 'path': os.path.join(result_dir_path, video_file)}
                for csv_file in csv_files:
                    all_files[csv_file] = {'type': 'csv', 'path': os.path.join(result_dir_path, csv_file)}
                
                if all_files:
                    # 체크박스로 파일 선택
                    selected_files = []
                    col_check1, col_check2 = st.columns([3, 1])
                    
                    with col_check1:
                        st.write("**선택할 파일:**")
                        for filename in sorted(all_files.keys()):
                            file_type = "📹" if all_files[filename]['type'] == 'video' else "📝"
                            is_checked = filename in st.session_state.get(session_key, [])
                            if st.checkbox(f"{file_type} {filename}", value=is_checked, key=f"file_{selected_dir}_{filename}"):
                                if filename not in selected_files:
                                    selected_files.append(filename)
                            else:
                                if filename in st.session_state.get(session_key, []):
                                    st.session_state[session_key].remove(filename)
                    
                    # 선택된 파일 목록 업데이트
                    st.session_state[session_key] = selected_files
                    
                    # 선택된 파일 정보 표시
                    if selected_files:
                        st.success(f"✅ {len(selected_files)}개 파일 선택됨")
                        
                        # 선택한 파일 다운로드 (ZIP으로)
                        if len(selected_files) > 1:
                            def create_zip():
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for filename in selected_files:
                                        file_path = all_files[filename]['path']
                                        if os.path.exists(file_path):
                                            zip_file.write(file_path, filename)
                                zip_buffer.seek(0)
                                return zip_buffer.getvalue()
                            
                            zip_data = create_zip()
                            st.download_button(
                                label=f"📦 선택한 파일 {len(selected_files)}개 ZIP으로 다운로드",
                                data=zip_data,
                                file_name=f"selected_files_{selected_dir}.zip",
                                mime="application/zip",
                                width='stretch'
                            )
                        
                        # 개별 다운로드 버튼들
                        st.subheader("📥 개별 다운로드")
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            selected_videos = [f for f in selected_files if all_files[f]['type'] == 'video']
                            if selected_videos:
                                st.write("**📹 영상 파일:**")
                                for video_file in selected_videos:
                                    video_path = all_files[video_file]['path']
                                    if os.path.exists(video_path):
                                        with open(video_path, "rb") as f:
                                            st.download_button(
                                                label=f"⬇️ {video_file}",
                                                data=f.read(),
                                                file_name=video_file,
                                                mime="video/mp4",
                                                key=f"dl_video_{selected_dir}_{video_file}",
                                                width='stretch'
                                            )
                        
                        with col_dl2:
                            selected_csvs = [f for f in selected_files if all_files[f]['type'] == 'csv']
                            if selected_csvs:
                                st.write("**📊 QR 기록 파일 (CSV):**")
                                for csv_file in selected_csvs:
                                    csv_path = all_files[csv_file]['path']
                                    if os.path.exists(csv_path):
                                        with open(csv_path, "rb") as f:
                                            csv_content = f.read()
                                            st.download_button(
                                                label=f"⬇️ {csv_file}",
                                                data=csv_content,
                                                file_name=csv_file,
                                                mime="text/csv",
                                                key=f"dl_csv_{selected_dir}_{csv_file}",
                                                width='stretch'
                                            )
                        
                        # CSV 미리보기
                        if selected_csvs:
                            st.subheader("📖 QR 기록 미리보기")
                            selected_csv = st.selectbox("미리볼 CSV 파일 선택", selected_csvs, key=f"preview_{selected_dir}")
                            if selected_csv:
                                csv_path = all_files[selected_csv]['path']
                                try:
                                    import pandas as pd
                                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                                    st.dataframe(df, width='stretch', hide_index=True)
                                except:
                                    with open(csv_path, "r", encoding='utf-8') as f:
                                        csv_content = f.read()
                                        st.text_area("CSV 내용", csv_content, height=300, key=f"csv_area_{selected_dir}")
                    else:
                        st.info("💡 위에서 다운로드할 파일을 선택하세요.")
                else:
                    st.info("선택한 디렉토리에 파일이 없습니다.")
        else:
            st.info("저장된 결과가 없습니다.")
    else:
        st.info("결과 디렉토리가 없습니다.")
    
    # 실시간 업데이트 (프로세싱 중일 때만)
    # 배치 처리 완료 후에는 갱신하지 않음 (완료 메시지 표시를 위해)
    is_batch_completed = (st.session_state.get('processing_completed', False) and 
                         not st.session_state.processing and 
                         not st.session_state.batch_processing and
                         len(st.session_state.get('batch_results', {})) > 0)
    
    if st.session_state.processing or st.session_state.batch_processing:
        # 처리 중일 때만 주기적으로 화면 업데이트
        if st.session_state.batch_processing:
            # 배치 처리 중일 때는 주기적으로 화면 갱신
            time.sleep(0.2)
        else:
            # 단일 파일 처리 중일 때
            current_results = st.session_state.get('current_results', {})
            frame_num = current_results.get('frame_num', 0)
            current_frame = st.session_state.get('current_frame')
            
            if frame_num == 0:
                # 아직 프레임이 생성되지 않음 - 빠르게 갱신
                time.sleep(0.05)  # 더 빠르게
            else:
                # 프레임이 생성됨 - 적절한 속도로 갱신
                time.sleep(0.15)  # 프레임 속도에 맞춤
        
        st.rerun()
    elif is_batch_completed:
        # 배치 처리 완료 직후 한 번만 갱신하여 완료 메시지 표시
        st.rerun()

if __name__ == "__main__":
    main()

