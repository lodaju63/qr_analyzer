"""
QR 코드 탐지 능력 테스트
YOLO 모델의 QR 코드 탐지 성능을 테스트합니다.
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime

# YOLO 모델 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics를 사용할 수 없습니다. pip install ultralytics로 설치하세요.")
    sys.exit(1)

# PIL import (한글 폰트 지원용)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def put_korean_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """한글 텍스트를 이미지에 그리기"""
    if not PIL_AVAILABLE:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    
    try:
        font_path = 'data/font/NanumGothic.ttf'
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


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


def filter_overlapping_detections(detections, iou_threshold=0.5):
    """
    겹치는 탐지 결과 제거 (NMS - Non-Maximum Suppression)
    
    Args:
        detections: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        iou_threshold: 겹침 임계값 (0.5 = 50% 이상 겹치면 중복으로 간주)
    
    Returns:
        filtered_detections: 필터링된 탐지 결과
    """
    if not detections:
        return []
    
    # 신뢰도(confidence) 기준으로 정렬 (높은 것이 우선)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered = []
    for det in detections:
        is_overlapping = False
        bbox1 = det['bbox']
        
        for filtered_det in filtered:
            bbox2 = filtered_det['bbox']
            iou = calculate_iou(bbox1, bbox2)
            
            if iou > iou_threshold:
                is_overlapping = True
                break
        
        if not is_overlapping:
            filtered.append(det)
    
    return filtered


def detect_qr_with_yolo(model, frame, conf_threshold=0.25, iou_threshold=0.5):
    """
    YOLO 모델로 QR 코드 위치 탐지
    
    Args:
        model: YOLO 모델
        frame: 입력 프레임 (BGR)
        conf_threshold: 신뢰도 임계값 (기본: 0.25)
        iou_threshold: 겹침 임계값 (기본: 0.5)
    
    Returns:
        detections: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
    """
    detections = []
    
    try:
        # YOLO 탐지
        results = model(frame, conf=conf_threshold, verbose=False, imgsz=640)
        result = results[0]
        
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
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        # Overlap threshold 적용 (NMS)
        detections = filter_overlapping_detections(detections, iou_threshold=iou_threshold)
    
    except Exception as e:
        print(f"⚠️ 탐지 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return detections


def test_single_image(model, image_path, output_dir="test_results", conf_threshold=0.25, iou_threshold=0.5, save_result=True):
    """단일 이미지에서 QR 코드 탐지 테스트"""
    
    # 이미지 읽기
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    h, w = frame.shape[:2]
    
    # QR 코드 탐지
    start_time = time.time()
    detections = detect_qr_with_yolo(model, frame, conf_threshold, iou_threshold)
    detect_time = time.time() - start_time
    
    # 결과 시각화
    result_frame = frame.copy()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # 바운딩 박스 그리기 (초록색)
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 신뢰도 표시
        text = f"QR{i+1}: {conf:.2f}"
        text_pos = (x1, y1 - 10) if y1 > 20 else (x1, y2 + 20)
        result_frame = put_korean_text(result_frame, text, text_pos, font_size=16, color=(0, 255, 0))
    
    # 정보 표시
    info_text = f"Detections: {len(detections)} | Time: {detect_time*1000:.1f}ms | Conf: {conf_threshold} | IoU: {iou_threshold}"
    result_frame = put_korean_text(result_frame, info_text, (10, 30), font_size=16, color=(255, 255, 255))
    
    # 결과 저장
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
        cv2.imwrite(output_path, result_frame)
    
    # 결과 정보
    result_info = {
        'image_path': image_path,
        'image_size': f"{w}x{h}",
        'detections': len(detections),
        'detect_time_ms': detect_time * 1000,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'detection_details': [
            {
                'id': i+1,
                'bbox': det['bbox'],
                'confidence': float(det['confidence'])
            }
            for i, det in enumerate(detections)
        ]
    }
    
    return result_info


def test_image_batch(model, image_dir, output_dir="test_results", conf_threshold=0.25, iou_threshold=0.5):
    """여러 이미지를 일괄 테스트"""
    
    if not os.path.exists(image_dir):
        print(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        return
    
    # 이미지 파일 찾기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {image_dir}")
        return
    
    image_files = sorted(image_files)
    total_images = len(image_files)
    
    print(f"\n{'='*60}")
    print(f"📷 일괄 이미지 테스트 시작")
    print(f"{'='*60}")
    print(f"   이미지 디렉토리: {image_dir}")
    print(f"   총 이미지 개수: {total_images}개")
    print(f"   Confidence Threshold: {conf_threshold}")
    print(f"   IoU Threshold: {iou_threshold}")
    print(f"{'='*60}\n")
    
    # 결과 저장용
    all_results = []
    total_detections = 0
    total_time = 0
    
    # 각 이미지 테스트
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{total_images}] 처리 중: {image_path.name}...", end=' ')
        
        result = test_single_image(model, str(image_path), output_dir, conf_threshold, iou_threshold, save_result=True)
        
        if result:
            all_results.append(result)
            total_detections += result['detections']
            total_time += result['detect_time_ms']
            print(f"✅ {result['detections']}개 탐지 ({result['detect_time_ms']:.1f}ms)")
        else:
            print("❌ 실패")
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"📊 테스트 결과 통계")
    print(f"{'='*60}")
    print(f"   총 이미지: {total_images}개")
    print(f"   총 탐지 개수: {total_detections}개")
    print(f"   평균 탐지 개수: {total_detections/total_images:.2f}개/이미지")
    print(f"   총 처리 시간: {total_time/1000:.2f}초")
    print(f"   평균 처리 시간: {total_time/total_images:.1f}ms/이미지")
    
    # 탐지 성공률
    detected_images = sum(1 for r in all_results if r['detections'] > 0)
    detection_rate = (detected_images / total_images * 100) if total_images > 0 else 0
    print(f"   탐지 성공 이미지: {detected_images}개 ({detection_rate:.1f}%)")
    
    # Confidence 통계
    all_confidences = []
    for result in all_results:
        for det in result['detection_details']:
            all_confidences.append(det['confidence'])
    
    if all_confidences:
        print(f"   평균 Confidence: {np.mean(all_confidences):.3f}")
        print(f"   최소 Confidence: {np.min(all_confidences):.3f}")
        print(f"   최대 Confidence: {np.max(all_confidences):.3f}")
    
    print(f"{'='*60}")
    
    # JSON 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'model_path': str(model.ckpt_path) if hasattr(model, 'ckpt_path') else 'unknown',
                'image_dir': image_dir,
                'total_images': total_images,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'timestamp': timestamp
            },
            'summary': {
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections/total_images if total_images > 0 else 0,
                'total_time_sec': total_time/1000,
                'avg_time_ms': total_time/total_images if total_images > 0 else 0,
                'detected_images': detected_images,
                'detection_rate_percent': detection_rate,
                'avg_confidence': float(np.mean(all_confidences)) if all_confidences else 0,
                'min_confidence': float(np.min(all_confidences)) if all_confidences else 0,
                'max_confidence': float(np.max(all_confidences)) if all_confidences else 0
            },
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 상세 결과 저장: {json_path}")
    print(f"💾 이미지 결과 저장: {output_dir}/")


def test_video(model, video_path, output_dir="test_results", conf_threshold=0.25, iou_threshold=0.5, max_frames=None, show_display=True):
    """비디오에서 QR 코드 탐지 테스트 (화면 표시 포함)"""
    
    print(f"\n{'='*60}")
    print(f"🎬 비디오 탐지 테스트: {video_path}")
    print(f"{'='*60}")
    
    # 비디오 열기
    if not os.path.exists(video_path):
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # FPS가 0이거나 이상한 경우 기본값 사용
    if fps <= 0 or fps > 120:
        fps = 30.0
        print(f"   ⚠️ FPS 정보가 없거나 이상합니다. 기본값 30 FPS 사용")
    
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   총 프레임: {total_frames}")
    print(f"   Confidence Threshold: {conf_threshold}")
    print(f"   IoU Threshold: {iou_threshold}")
    print(f"   화면 표시: {'ON' if show_display else 'OFF'}")
    if max_frames:
        print(f"   최대 처리 프레임: {max_frames}")
    print(f"{'='*60}\n")
    
    # 출력 비디오 설정
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{video_name}_detected_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"💾 출력 비디오: {output_path}")
    
    # 화면 표시용 해상도 조정
    display_width = 1280
    display_height = 720
    if width > display_width:
        scale = display_width / width
        display_height = int(height * scale)
    
    # 통계
    frame_count = 0
    total_detections = 0
    detected_frames = 0
    total_detect_time = 0
    max_detections_in_frame = 0
    all_confidences = []
    
    # FPS 제어를 위한 변수
    frame_interval = 1.0 / fps
    paused = False
    start_time = time.time()  # 전체 시작 시간
    result_frame = None
    
    print("▶️ 탐지 시작... (원본 속도로 재생)")
    if show_display:
        print("   💡 ESC 키: 종료, SPACE 키: 일시정지/재생")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n📺 영상 재생 완료!")
                    break
                
                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break
                
                # QR 코드 탐지
                detect_start = time.time()
                detections = detect_qr_with_yolo(model, frame, conf_threshold, iou_threshold)
                detect_time = time.time() - detect_start
                total_detect_time += detect_time
                
                # 통계 업데이트
                num_detections = len(detections)
                if num_detections > 0:
                    detected_frames += 1
                    total_detections += num_detections
                    max_detections_in_frame = max(max_detections_in_frame, num_detections)
                    for det in detections:
                        all_confidences.append(det['confidence'])
                
                # 결과 시각화
                result_frame = frame.copy()
                
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    
                    # 바운딩 박스 그리기 (초록색)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 신뢰도 표시
                    text = f"QR{i+1}: {conf:.2f}"
                    text_pos = (x1, y1 - 10) if y1 > 20 else (x1, y2 + 20)
                    result_frame = put_korean_text(result_frame, text, text_pos, font_size=14, color=(0, 255, 0))
                
                # 정보 표시
                current_fps = 1.0 / detect_time if detect_time > 0 else 0
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {num_detections} | FPS: {current_fps:.1f}"
                result_frame = put_korean_text(result_frame, info_text, (10, 30), font_size=16, color=(255, 255, 255))
                
                # 비디오 저장 (원본 해상도로)
                out.write(result_frame)
            
            # 화면 표시
            if show_display and result_frame is not None:
                display_frame = cv2.resize(result_frame, (display_width, display_height))
                
                if paused:
                    pause_text = "PAUSED - Press SPACE to resume"
                    cv2.putText(display_frame, pause_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.imshow('QR Detection Test - Video', display_frame)
                
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
            
            # FPS 제어 (원본 속도로 재생)
            if not paused and frame_count > 0:
                elapsed = time.time() - start_time
                expected_time = frame_count * frame_interval
                sleep_time = expected_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 진행 상황 출력 (10프레임마다, 화면 표시 안 할 때만)
            if not show_display and frame_count % 10 == 0:
                print(f"   처리 중... {frame_count}/{total_frames} 프레임 ({frame_count/total_frames*100:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 중단했습니다.")
    finally:
        end_time = time.time()  # 전체 종료 시간
        total_elapsed_time = end_time - start_time  # 전체 경과 시간
        cap.release()
        out.release()
        if show_display:
            cv2.destroyAllWindows()
    
    # 최종 통계
    avg_detect_time = total_detect_time / frame_count if frame_count > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    detection_rate = (detected_frames / frame_count * 100) if frame_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 최종 통계:")
    print(f"{'='*60}")
    print(f"   처리된 프레임: {frame_count}개")
    print(f"   탐지된 프레임: {detected_frames}개 ({detection_rate:.1f}%)")
    print(f"   총 탐지 개수: {total_detections}개")
    print(f"   프레임당 평균 탐지: {avg_detections:.2f}개")
    print(f"   프레임당 최대 탐지: {max_detections_in_frame}개")
    print(f"   평균 탐지 시간: {avg_detect_time*1000:.1f}ms/프레임")
    print(f"   순수 탐지 시간: {total_detect_time:.2f}초 (탐지만)")
    print(f"   총 처리 시간: {total_elapsed_time:.2f}초 (전체 경과 시간)")
    print(f"   원본 비디오 길이: {total_frames/fps:.2f}초")
    
    if all_confidences:
        print(f"   평균 Confidence: {np.mean(all_confidences):.3f}")
        print(f"   최소 Confidence: {np.min(all_confidences):.3f}")
        print(f"   최대 Confidence: {np.max(all_confidences):.3f}")
    
    print(f"\n💾 비디오 저장 완료: {output_path}")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='QR 코드 탐지 능력 테스트')
    parser.add_argument('input_path', type=str, help='입력 이미지/비디오 파일 또는 이미지 디렉토리 경로')
    parser.add_argument('--model', type=str, default='model1.pt', help='YOLO 모델 파일 경로 (기본: model1.pt)')
    parser.add_argument('--output', type=str, default='test_results', help='출력 디렉토리 (기본: test_results)')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값 (기본: 0.25)')
    parser.add_argument('--iou', type=float, default=0.5, help='겹침 임계값 (기본: 0.5)')
    parser.add_argument('--max-frames', type=int, default=None, help='비디오 최대 처리 프레임 수 (기본: 전체)')
    parser.add_argument('--no-display', action='store_true', help='비디오 화면 표시 안 함 (진행 상황만 출력)')
    
    args = parser.parse_args()
    
    # YOLO 모델 로드
    if not os.path.exists(args.model):
        print(f"❌ YOLO 모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)
    
    print(f"🔍 YOLO 모델 로드 중: {args.model}")
    try:
        model = YOLO(args.model)
        print("✅ YOLO 모델 로드 완료")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        sys.exit(1)
    
    # 입력 경로 확인
    if not os.path.exists(args.input_path):
        print(f"❌ 입력 경로를 찾을 수 없습니다: {args.input_path}")
        sys.exit(1)
    
    # 파일 타입 확인
    if os.path.isdir(args.input_path):
        # 디렉토리: 일괄 이미지 테스트
        test_image_batch(model, args.input_path, args.output, args.conf, args.iou)
    else:
        # 파일: 단일 파일 테스트
        file_ext = Path(args.input_path).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 이미지 테스트
            result = test_single_image(model, args.input_path, args.output, args.conf, args.iou, save_result=True)
            if result:
                print(f"\n✅ 탐지 완료: {result['detections']}개 QR 코드 발견")
                print(f"   처리 시간: {result['detect_time_ms']:.1f}ms")
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 비디오 테스트
            show_display = not args.no_display
            test_video(model, args.input_path, args.output, args.conf, args.iou, args.max_frames, show_display=show_display)
        else:
            print(f"❌ 지원하지 않는 파일 형식입니다: {file_ext}")
            print("   지원 형식: .jpg, .jpeg, .png, .bmp, .mp4, .avi, .mov, .mkv")
            sys.exit(1)


if __name__ == "__main__":
    main()

