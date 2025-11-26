"""
YOLO 모델(.pt)을 ONNX 형식으로 변환하는 스크립트
"""
import os
import sys
from ultralytics import YOLO

def convert_pt_to_onnx(model_path, output_path=None, imgsz=640, opset=12, simplify=True):
    """
    YOLO .pt 모델을 ONNX 형식으로 변환
    
    Args:
        model_path: 입력 .pt 모델 파일 경로
        output_path: 출력 ONNX 파일 경로 (None이면 자동 생성)
        imgsz: 입력 이미지 크기 (기본: 640)
        opset: ONNX opset 버전 (기본: 12)
        simplify: 모델 단순화 여부 (기본: True)
    
    Returns:
        변환된 ONNX 파일 경로
    """
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    if output_path is None:
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}.onnx"
    
    print(f"🔄 모델 변환 시작:")
    print(f"   입력: {model_path}")
    print(f"   출력: {output_path}")
    print(f"   이미지 크기: {imgsz}")
    print(f"   ONNX opset: {opset}")
    print(f"   모델 단순화: {simplify}")
    
    try:
        # YOLO 모델 로드
        model = YOLO(model_path)
        
        # ONNX로 변환
        model.export(
            format='onnx',
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,  # 고정 크기 입력 (더 빠름)
            half=False,     # FP32 사용 (호환성)
        )
        
        # 출력 파일 경로 확인 (YOLO가 자동으로 경로 생성)
        exported_path = os.path.splitext(model_path)[0] + '.onnx'
        if os.path.exists(exported_path):
            if exported_path != output_path and os.path.exists(output_path):
                # 원하는 경로로 이동
                import shutil
                shutil.move(exported_path, output_path)
                print(f"✅ 모델을 {output_path}로 이동했습니다.")
            else:
                output_path = exported_path
            
            # 파일 크기 비교
            pt_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            onnx_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            print(f"\n✅ 변환 완료!")
            print(f"   ONNX 파일: {output_path}")
            print(f"   원본 크기: {pt_size:.2f} MB")
            print(f"   ONNX 크기: {onnx_size:.2f} MB")
            print(f"   크기 변화: {((onnx_size - pt_size) / pt_size * 100):+.1f}%")
            
            return output_path
        else:
            print(f"⚠️ ONNX 파일을 찾을 수 없습니다. 변환에 실패했을 수 있습니다.")
            return None
            
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO .pt 모델을 ONNX로 변환')
    parser.add_argument('--model', type=str, default='l.pt', help='입력 .pt 모델 파일 (기본: l.pt)')
    parser.add_argument('--output', type=str, default=None, help='출력 ONNX 파일 경로 (기본: 자동)')
    parser.add_argument('--imgsz', type=int, default=640, help='입력 이미지 크기 (기본: 640)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset 버전 (기본: 12)')
    parser.add_argument('--no-simplify', action='store_true', help='모델 단순화 비활성화')
    
    args = parser.parse_args()
    
    convert_pt_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify
    )


