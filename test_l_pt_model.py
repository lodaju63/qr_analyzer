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

# Ultralytics YOLO 모델 로드 시도
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics를 사용할 수 없습니다.")

# 표시용 설정
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
            print(f"   - Task: {model.task if hasattr(model, 'task') else 'Unknown'}")
            print(f"   - Classes: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
            
            if hasattr(model, 'names'):
                print(f"   - 클래스 목록:")
                for idx, name in model.names.items():
                    print(f"     [{idx}] {name}")
            
            if hasattr(model, 'model'):
                model_info = model.model
                print(f"   - 모델 구조: {type(model_info).__name__}")
            
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
                print(f"   - {key}: (dict with {len(checkpoint[key])} keys)")
                if len(checkpoint[key].keys()) < 10:
                    for subkey in checkpoint[key].keys():
                        print(f"     - {subkey}: {type(checkpoint[subkey]).__name__}")
            elif isinstance(checkpoint[key], (list, tuple)):
                print(f"   - {key}: ({type(checkpoint[key]).__name__} with {len(checkpoint[key])} items)")
            else:
                print(f"   - {key}: {type(checkpoint[key]).__name__}")
        
        return checkpoint, 'pytorch'
    except Exception as e:
        print(f"❌ PyTorch 모델 로드 실패: {e}")
        return None, None

def test_yolo_detection(model, image_path, conf_threshold=0.25):
    """YOLO 모델로 이미지 탐지 테스트"""
    print(f"\n🖼️  이미지 테스트: {os.path.basename(image_path)}")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    print(f"   이미지 크기: {image.shape[1]}x{image.shape[0]}")
    
    # 탐지 실행
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        result = results[0]
        
        print(f"   탐지된 객체 수: {len(result.boxes) if result.boxes is not None else 0}")
        
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
                
                print(f"   [{i+1}] {class_name}: {conf:.2%} at [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
        
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
        print("   ⚠️ 탐지된 객체가 없습니다.")
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
        print(f"   💾 결과 저장: {save_path}")
    
    plt.show()

def test_video_detection(model, video_path, conf_threshold=0.25, 
                        frame_interval=30, show_video=True, save_output=True):
    """YOLO 모델로 영상 탐지 테스트"""
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
    
    print(f"📹 영상 정보:")
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   총 프레임: {total_frames}")
    print(f"   길이: {duration:.2f}초")
    print(f"   탐지 간격: {frame_interval}프레임마다")
    
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
        print(f"   출력 파일: {video_output_path}")
    
    # 통계
    frame_count = 0
    detection_count = 0
    total_detections = 0
    detection_times = []
    detections_per_frame = []
    
    # 마지막 탐지 결과 저장 (다음 탐지 전까지 표시)
    last_detections = []
    
    # 로그 파일
    log_file_path = output_dir / f"video_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print(f"영상 테스트 시작: {video_path}")
    log_print(f"설정: conf_threshold={conf_threshold}, frame_interval={frame_interval}")
    log_print("-" * 60)
    
    print(f"\n▶️  영상 처리 시작...")
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps if fps > 0 else 0
            
            # 프레임 간격에 따라 탐지
            should_detect = (frame_count % frame_interval == 0) or (frame_count == 1)
            
            detections = []
            if should_detect:
                # 탐지 수행
                detect_start = time.time()
                results = model(frame, conf=conf_threshold, verbose=False)
                detect_time = time.time() - detect_start
                detection_times.append(detect_time)
                
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    frame_detections = 0
                    for box in result.boxes:
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
                        
                        frame_detections += 1
                        total_detections += 1
                    
                    detection_count += 1
                    detections_per_frame.append(frame_detections)
                    
                    # 마지막 탐지 결과 업데이트
                    last_detections = detections.copy()
                    
                    log_print(f"프레임 {frame_count} ({current_time:.2f}초): {frame_detections}개 탐지")
                    for i, det in enumerate(detections):
                        log_print(f"  [{i+1}] {det['class']}: {det['confidence']:.2%} "
                                 f"at [{int(det['bbox'][0])}, {int(det['bbox'][1])}, "
                                 f"{int(det['bbox'][2])}, {int(det['bbox'][3])}]")
            
            # 결과 시각화 (마지막 탐지 결과 사용)
            vis_frame = frame.copy()
            
            # 바운딩 박스 그리기
            display_detections = last_detections if not should_detect or len(detections) == 0 else detections
            for det in display_detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                
                # 박스 그리기
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 라벨 추가
                label = f"{det['class']}: {det['confidence']:.1%}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(vis_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 프레임 정보 표시
            info_text = f"Frame: {frame_count}/{total_frames} | Time: {current_time:.1f}s"
            if display_detections:
                info_text += f" | Detections: {len(display_detections)}"
            if not should_detect and display_detections:
                info_text += " (cached)"
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
                
                cv2.imshow('QR Detection Test', display_frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ 사용자가 중단했습니다.")
                    break
            
            # 진행 상황 출력
            if frame_count % (frame_interval * 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   진행: {progress:.1f}% ({frame_count}/{total_frames} 프레임)")
    
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 중단했습니다.")
    finally:
        # 정리
        total_time = time.time() - start_time
        cap.release()
        if out_video is not None:
            out_video.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # 통계 출력
        print(f"\n📊 처리 완료!")
        print(f"   총 프레임: {frame_count}")
        print(f"   탐지된 프레임: {detection_count}")
        print(f"   총 탐지 수: {total_detections}")
        print(f"   처리 시간: {total_time:.2f}초")
        if detection_times:
            avg_detect_time = np.mean(detection_times)
            print(f"   평균 탐지 시간: {avg_detect_time*1000:.2f}ms")
        if detections_per_frame:
            avg_detections = np.mean(detections_per_frame)
            print(f"   프레임당 평균 탐지: {avg_detections:.2f}개")
        
        log_print("-" * 60)
        log_print(f"처리 완료")
        log_print(f"총 프레임: {frame_count}, 탐지 프레임: {detection_count}, 총 탐지: {total_detections}")
        log_print(f"처리 시간: {total_time:.2f}초")
        log_file.close()
        
        if video_output_path:
            print(f"   💾 결과 영상 저장: {video_output_path}")
            print(f"   📝 로그 파일: {log_file_path}")
    
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
            frame_interval = 30    # N프레임마다 탐지 (성능 향상을 위해)
            show_video = True      # 화면에 표시 여부
            save_output = True     # 결과 영상 저장 여부
            
            result = test_video_detection(
                model, 
                video_path, 
                conf_threshold=conf_threshold,
                frame_interval=frame_interval,
                show_video=show_video,
                save_output=save_output
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

