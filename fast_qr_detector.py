import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import pyzbar.pyzbar as pyzbar
import imutils
from qreader import QReader
import time

class FastQRCodeDetector:
    def __init__(self, output_dir="results", clear_previous=True):
        self.detector = cv2.QRCodeDetector()
        # QReader를 UTF-8만 사용하도록 설정
        try:
            # QReader 인코딩 설정 (UTF-8만 사용)
            self.qreader = QReader()
            # 경고 메시지 숨기기
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='qreader')
        except:
            self.qreader = QReader()
        self.output_dir = output_dir

        # 이전 결과 초기화
        if clear_previous and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            print(f"🗑️  이전 결과 폴더 삭제됨: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "failed"), exist_ok=True)
        print(f"📁 새로운 결과 폴더 생성됨: {output_dir}")

    def draw_korean_text(self, image, text, position, color=(0, 255, 0), font_size=20):
        """한글 텍스트 그리기"""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("malgun.ttf", font_size)  # Windows
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)  # 대체 폰트
                except:
                    font = ImageFont.load_default()
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"한글 텍스트 그리기 오류: {e}")
            return image

    def enhance_image(self, image):
        """이미지 품질 향상"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced

    def apply_binary_threshold(self, image):
        """적응적 이진화 적용"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 적응적 이진화
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 컬러 이미지로 변환
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary

    def apply_pil_enhancement(self, image):
        """PIL을 사용한 이미지 향상"""
        try:
            # OpenCV 이미지를 PIL로 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.5)
            
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            # PIL 이미지를 OpenCV로 변환
            if len(image.shape) == 3:
                return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            else:
                return np.array(enhanced)
        except Exception as e:
            print(f"PIL 향상 오류: {e}")
            return image

    def center_crop(self, image, crop_ratio=0.8):
        """중심 크롭"""
        h, w = image.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h + crop_h, start_w:start_w + crop_w]

    def apply_gaussian_blur(self, image, kernel_size=5):
        """가우시안 블러 적용"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def detect_with_opencv(self, image):
        """OpenCV로 QR 코드 탐지"""
        try:
            retval, decoded_info, points = self.detector.detectAndDecode(image)
            if retval and decoded_info:
                return decoded_info, points
        except Exception as e:
            print(f"OpenCV 탐지 오류: {e}")
        return None, None

    def detect_with_pyzbar(self, image):
        """PyZbar로 QR 코드 탐지"""
        try:
            # OpenCV 이미지를 PIL로 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # PyZbar로 탐지
            qr_codes = pyzbar.decode(pil_image)
            if qr_codes:
                return qr_codes[0].data.decode('utf-8'), qr_codes[0].rect
        except Exception as e:
            print(f"PyZbar 탐지 오류: {e}")
        return None, None

    def detect_with_qreader(self, image):
        """QReader로 QR 코드 탐지"""
        try:
            # QReader로 탐지
            detections = self.qreader.detect(image)
            if detections and len(detections) > 0:
                # 첫 번째 탐지 결과에서 텍스트 추출
                decoded_text = self.qreader.decode(image, detections[0])
                if decoded_text:
                    return decoded_text, detections[0]
        except Exception as e:
            print(f"QReader 탐지 오류: {e}")
        return None, None

    def detect_qr_comprehensive(self, image, filename="unknown"):
        """종합적인 QR 코드 탐지"""
        print(f"\n🔍 QR 코드 탐지 시작: {filename}")
        
        # 원본 이미지 정보
        h, w = image.shape[:2]
        print(f"  이미지 크기: {w}x{h}")
        
        # 1단계: 원본 이미지로 탐지
        print("  1단계: 원본 이미지 탐지...")
        for method_name, method_func in [
            ("OpenCV", lambda img: self.detect_with_opencv(img)),
            ("PyZbar", lambda img: self.detect_with_pyzbar(img)),
            ("QReader", lambda img: self.detect_with_qreader(img))
        ]:
            try:
                result, info = method_func(image)
                if result:
                    print(f"    ✅ {method_name}: {result}")
                    return result, info, "original", method_name
            except Exception as e:
                print(f"    ❌ {method_name}: {e}")
        
        # 2단계: 이미지 향상 후 탐지
        print("  2단계: 이미지 향상 후 탐지...")
        enhanced = self.enhance_image(image)
        for method_name, method_func in [
            ("OpenCV", lambda img: self.detect_with_opencv(img)),
            ("PyZbar", lambda img: self.detect_with_pyzbar(img)),
            ("QReader", lambda img: self.detect_with_qreader(img))
        ]:
            try:
                result, info = method_func(enhanced)
                if result:
                    print(f"    ✅ {method_name} (향상): {result}")
                    return result, info, "enhanced", method_name
            except Exception as e:
                print(f"    ❌ {method_name} (향상): {e}")
        
        # 3단계: 이진화 후 탐지
        print("  3단계: 이진화 후 탐지...")
        binary = self.apply_binary_threshold(image)
        for method_name, method_func in [
            ("OpenCV", lambda img: self.detect_with_opencv(img)),
            ("PyZbar", lambda img: self.detect_with_pyzbar(img)),
            ("QReader", lambda img: self.detect_with_qreader(img))
        ]:
            try:
                result, info = method_func(binary)
                if result:
                    print(f"    ✅ {method_name} (이진화): {result}")
                    return result, info, "binary", method_name
            except Exception as e:
                print(f"    ❌ {method_name} (이진화): {e}")
        
        # 4단계: PIL 향상 후 탐지
        print("  4단계: PIL 향상 후 탐지...")
        pil_enhanced = self.apply_pil_enhancement(image)
        for method_name, method_func in [
            ("OpenCV", lambda img: self.detect_with_opencv(img)),
            ("PyZbar", lambda img: self.detect_with_pyzbar(img)),
            ("QReader", lambda img: self.detect_with_qreader(img))
        ]:
            try:
                result, info = method_func(pil_enhanced)
                if result:
                    print(f"    ✅ {method_name} (PIL): {result}")
                    return result, info, "pil_enhanced", method_name
            except Exception as e:
                print(f"    ❌ {method_name} (PIL): {e}")
        
        # 5단계: 중심 크롭 후 탐지
        print("  5단계: 중심 크롭 후 탐지...")
        cropped = self.center_crop(image)
        for method_name, method_func in [
            ("OpenCV", lambda img: self.detect_with_opencv(img)),
            ("PyZbar", lambda img: self.detect_with_pyzbar(img)),
            ("QReader", lambda img: self.detect_with_qreader(img))
        ]:
            try:
                result, info = method_func(cropped)
                if result:
                    print(f"    ✅ {method_name} (크롭): {result}")
                    return result, info, "cropped", method_name
            except Exception as e:
                print(f"    ❌ {method_name} (크롭): {e}")
        
        # 6단계: 가우시안 블러 후 탐지
        print("  6단계: 가우시안 블러 후 탐지...")
        blurred = self.apply_gaussian_blur(image)
        for method_name, method_func in [
            ("OpenCV", lambda img: self.detect_with_opencv(img)),
            ("PyZbar", lambda img: self.detect_with_pyzbar(img)),
            ("QReader", lambda img: self.detect_with_qreader(img))
        ]:
            try:
                result, info = method_func(blurred)
                if result:
                    print(f"    ✅ {method_name} (블러): {result}")
                    return result, info, "blurred", method_name
            except Exception as e:
                print(f"    ❌ {method_name} (블러): {e}")
        
        print("  ❌ 모든 방법 실패")
        return None, None, None, None

    def process_image(self, image_path, save_result=True):
        """이미지 처리 및 결과 저장"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return False
        
        filename = os.path.basename(image_path)
        
        # QR 코드 탐지
        result, info, preprocessing, method = self.detect_qr_comprehensive(image, filename)
        
        if result:
            print(f"✅ 탐지 성공: {result}")
            
            if save_result:
                # 결과 이미지 생성
                result_image = image.copy()
                
                # QR 코드 영역 표시
                if info is not None:
                    if isinstance(info, np.ndarray) and len(info) >= 4:
                        # OpenCV points
                        points = info.astype(np.int32)
                        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
                        
                        # 텍스트 표시
                        text = f"{result[:20]}..." if len(result) > 20 else result
                        cv2.putText(result_image, text, (int(points[0][0]), int(points[0][1]) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    elif hasattr(info, 'left'):
                        # PyZbar rect
                        cv2.rectangle(result_image, (info.left, info.top), 
                                    (info.left + info.width, info.top + info.height), (0, 255, 0), 2)
                        
                        # 텍스트 표시
                        text = f"{result[:20]}..." if len(result) > 20 else result
                        cv2.putText(result_image, text, (info.left, info.top - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    elif isinstance(info, dict) and 'bbox_xyxy' in info:
                        # QReader bbox
                        bbox = info['bbox_xyxy']
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 텍스트 표시
                        text = f"{result[:20]}..." if len(result) > 20 else result
                        cv2.putText(result_image, text, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 결과 정보 추가
                info_text = f"Method: {method} | Preprocessing: {preprocessing}"
                cv2.putText(result_image, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 결과 저장
                result_path = os.path.join(self.output_dir, "enhanced", filename)
                cv2.imwrite(result_path, result_image)
                print(f"💾 결과 저장: {result_path}")
            
            return True
        else:
            print(f"❌ 탐지 실패: {filename}")
            
            if save_result:
                # 실패한 이미지 저장
                failed_path = os.path.join(self.output_dir, "failed", filename)
                cv2.imwrite(failed_path, image)
                print(f"💾 실패 이미지 저장: {failed_path}")
            
            return False

    def process_folder(self, folder_path, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """폴더 내 모든 이미지 처리"""
        if not os.path.exists(folder_path):
            print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
            return
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {folder_path}")
            return
        
        print(f"📁 폴더 처리 시작: {folder_path}")
        print(f"  총 {len(image_files)}개 이미지")
        
        # 처리 통계
        total_images = len(image_files)
        successful = 0
        failed = 0
        
        # 각 이미지 처리
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, filename)
            print(f"\n[{i}/{total_images}] 처리 중: {filename}")
            
            if self.process_image(image_path):
                successful += 1
            else:
                failed += 1
        
        # 결과 요약
        print(f"\n📊 처리 완료!")
        print(f"  총 이미지: {total_images}")
        print(f"  성공: {successful}")
        print(f"  실패: {failed}")
        print(f"  성공률: {successful/total_images*100:.1f}%")
        
        return {
            'total': total_images,
            'successful': successful,
            'failed': failed,
            'success_rate': successful/total_images*100
        }

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QR 코드 탐지기')
    parser.add_argument('--input', '-i', required=True, help='입력 이미지 또는 폴더 경로')
    parser.add_argument('--output', '-o', default='results', help='출력 폴더 경로')
    parser.add_argument('--clear', action='store_true', help='이전 결과 삭제')
    
    args = parser.parse_args()
    
    # 탐지기 초기화
    detector = FastQRCodeDetector(output_dir=args.output, clear_previous=args.clear)
    
    # 입력 경로 확인
    if os.path.isfile(args.input):
        # 단일 이미지 처리
        print(f"🖼️  단일 이미지 처리: {args.input}")
        detector.process_image(args.input)
    elif os.path.isdir(args.input):
        # 폴더 처리
        print(f"📁 폴더 처리: {args.input}")
        detector.process_folder(args.input)
    else:
        print(f"❌ 입력 경로를 찾을 수 없습니다: {args.input}")

if __name__ == "__main__":
    main()
