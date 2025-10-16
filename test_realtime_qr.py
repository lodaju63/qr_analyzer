"""
실시간 QR 탐지 모듈 테스트 스크립트
조선소 T-Bar 제작 공정을 위한 QR 코드 인식 시스템 테스트
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict
import argparse
import sys
import os

# 모듈 import
from realtime_qr_detector import RealtimeQRDetector
from qr_utils import config, PerformanceStats
from qr_detection import get_detection_pipeline
from qr_preprocessing import get_preprocessing_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QRDetectionTester:
    """QR 탐지 테스터"""
    
    def __init__(self):
        """테스터 초기화"""
        self.test_results = []
        self.performance_stats = PerformanceStats()
    
    def test_static_images(self, image_folder: str) -> Dict:
        """정적 이미지 테스트"""
        logger.info(f"정적 이미지 테스트 시작: {image_folder}")
        
        if not os.path.exists(image_folder):
            logger.error(f"이미지 폴더를 찾을 수 없습니다: {image_folder}")
            return {}
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_folder}")
            return {}
        
        logger.info(f"총 {len(image_files)}개 이미지 테스트")
        
        # 탐지 파이프라인 가져오기
        detection_pipeline = get_detection_pipeline("optimized")
        
        results = {
            'total_images': len(image_files),
            'successful_detections': 0,
            'total_detection_time': 0.0,
            'detection_results': []
        }
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            logger.info(f"처리 중 ({i+1}/{len(image_files)}): {image_file}")
            
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"이미지를 읽을 수 없습니다: {image_file}")
                continue
            
            # QR 코드 탐지
            start_time = time.time()
            detection_results, detection_time = detection_pipeline.detect_with_preprocessing(image)
            total_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'image_file': image_file,
                'detection_count': len(detection_results),
                'detection_time': detection_time,
                'total_time': total_time,
                'success': len(detection_results) > 0,
                'results': detection_results
            }
            
            results['detection_results'].append(result)
            
            if result['success']:
                results['successful_detections'] += 1
                logger.info(f"  ✅ 성공: {len(detection_results)}개 QR 코드 탐지 ({detection_time:.3f}s)")
            else:
                logger.warning(f"  ❌ 실패: QR 코드 탐지 실패 ({detection_time:.3f}s)")
            
            results['total_detection_time'] += detection_time
        
        # 통계 계산
        results['success_rate'] = results['successful_detections'] / results['total_images']
        results['avg_detection_time'] = results['total_detection_time'] / results['total_images']
        
        logger.info(f"정적 이미지 테스트 완료:")
        logger.info(f"  성공률: {results['success_rate']:.1%}")
        logger.info(f"  평균 탐지 시간: {results['avg_detection_time']:.3f}s")
        
        return results
    
    def test_realtime_detection(self, duration: int = 30) -> Dict:
        """실시간 탐지 테스트"""
        logger.info(f"실시간 탐지 테스트 시작 (지속 시간: {duration}초)")
        
        # 실시간 탐지기 초기화
        detector = RealtimeQRDetector(
            camera_id=0,
            frame_width=640,
            frame_height=480,
            fps=30,
            detection_interval=5
        )
        
        if not detector.start_detection():
            logger.error("실시간 탐지 시작 실패")
            return {}
        
        logger.info("실시간 탐지 시작됨. ESC 키로 조기 종료 가능")
        
        start_time = time.time()
        test_results = {
            'start_time': start_time,
            'duration': duration,
            'frames_processed': 0,
            'detections_found': 0,
            'detection_history': []
        }
        
        try:
            while time.time() - start_time < duration:
                # 현재 프레임 가져오기
                frame = detector.get_current_frame()
                if frame is None:
                    continue
                
                # 탐지 결과 가져오기
                results = detector.get_detection_results()
                
                # 통계 업데이트
                test_results['frames_processed'] += 1
                if results:
                    test_results['detections_found'] += len(results)
                    test_results['detection_history'].append({
                        'timestamp': time.time(),
                        'detection_count': len(results),
                        'results': results
                    })
                
                # 결과 시각화
                if results:
                    frame = detector.visualize_results(frame, results)
                
                # 성능 정보 표시
                from qr_utils import VisualizationHelper
                stats = detector.get_performance_stats()
                frame = VisualizationHelper.draw_performance_info(frame, stats)
                
                # 프레임 표시
                cv2.imshow('Realtime QR Detection Test', frame)
                
                # ESC 키로 조기 종료
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    logger.info("사용자에 의해 조기 종료")
                    break
        
        except KeyboardInterrupt:
            logger.info("키보드 인터럽트로 종료")
        
        finally:
            detector.stop_detection()
            cv2.destroyAllWindows()
        
        # 최종 통계
        end_time = time.time()
        actual_duration = end_time - start_time
        
        test_results['end_time'] = end_time
        test_results['actual_duration'] = actual_duration
        test_results['fps'] = test_results['frames_processed'] / actual_duration if actual_duration > 0 else 0
        test_results['detection_rate'] = test_results['detections_found'] / test_results['frames_processed'] if test_results['frames_processed'] > 0 else 0
        
        logger.info(f"실시간 탐지 테스트 완료:")
        logger.info(f"  처리된 프레임: {test_results['frames_processed']}")
        logger.info(f"  탐지된 QR 코드: {test_results['detections_found']}개")
        logger.info(f"  평균 FPS: {test_results['fps']:.1f}")
        logger.info(f"  탐지율: {test_results['detection_rate']:.1%}")
        
        return test_results
    
    def test_performance_benchmark(self, image_folder: str) -> Dict:
        """성능 벤치마크 테스트"""
        logger.info("성능 벤치마크 테스트 시작")
        
        if not os.path.exists(image_folder):
            logger.error(f"이미지 폴더를 찾을 수 없습니다: {image_folder}")
            return {}
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_folder}")
            return {}
        
        # 테스트할 파이프라인들
        pipelines = {
            'realtime': get_detection_pipeline("realtime"),
            'optimized': get_detection_pipeline("optimized"),
            'full': get_detection_pipeline("full")
        }
        
        benchmark_results = {}
        
        for pipeline_name, pipeline in pipelines.items():
            logger.info(f"파이프라인 테스트: {pipeline_name}")
            
            results = {
                'pipeline': pipeline_name,
                'total_images': len(image_files),
                'successful_detections': 0,
                'total_detection_time': 0.0,
                'detection_times': []
            }
            
            for image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # 탐지 실행
                start_time = time.time()
                detection_results, detection_time = pipeline.detect(image)
                total_time = time.time() - start_time
                
                # 결과 저장
                results['detection_times'].append(detection_time)
                results['total_detection_time'] += detection_time
                
                if detection_results:
                    results['successful_detections'] += 1
            
            # 통계 계산
            results['success_rate'] = results['successful_detections'] / results['total_images']
            results['avg_detection_time'] = results['total_detection_time'] / results['total_images']
            results['min_detection_time'] = min(results['detection_times']) if results['detection_times'] else 0
            results['max_detection_time'] = max(results['detection_times']) if results['detection_times'] else 0
            
            benchmark_results[pipeline_name] = results
            
            logger.info(f"  {pipeline_name}: 성공률 {results['success_rate']:.1%}, "
                       f"평균 시간 {results['avg_detection_time']:.3f}s")
        
        return benchmark_results

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='QR 탐지 모듈 테스트')
    parser.add_argument('--mode', choices=['static', 'realtime', 'benchmark'], 
                       default='static', help='테스트 모드')
    parser.add_argument('--image_folder', type=str, default='data/first_goal',
                       help='이미지 폴더 경로')
    parser.add_argument('--duration', type=int, default=30,
                       help='실시간 테스트 지속 시간 (초)')
    
    args = parser.parse_args()
    
    tester = QRDetectionTester()
    
    if args.mode == 'static':
        results = tester.test_static_images(args.image_folder)
        print(f"\n=== 정적 이미지 테스트 결과 ===")
        print(f"성공률: {results.get('success_rate', 0):.1%}")
        print(f"평균 탐지 시간: {results.get('avg_detection_time', 0):.3f}s")
        
    elif args.mode == 'realtime':
        results = tester.test_realtime_detection(args.duration)
        print(f"\n=== 실시간 탐지 테스트 결과 ===")
        print(f"처리된 프레임: {results.get('frames_processed', 0)}")
        print(f"탐지된 QR 코드: {results.get('detections_found', 0)}개")
        print(f"평균 FPS: {results.get('fps', 0):.1f}")
        
    elif args.mode == 'benchmark':
        results = tester.test_performance_benchmark(args.image_folder)
        print(f"\n=== 성능 벤치마크 결과 ===")
        for pipeline_name, result in results.items():
            print(f"{pipeline_name}: 성공률 {result['success_rate']:.1%}, "
                  f"평균 시간 {result['avg_detection_time']:.3f}s")

if __name__ == "__main__":
    main()

