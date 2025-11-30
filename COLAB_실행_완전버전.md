# 🚀 코랩 실행 - 완전한 버전 (ngrok 사용)

Google Drive 공유 폴더 링크에서 파일을 다운로드하고 ngrok으로 실행하는 **완전한 코드**입니다.

## 📋 전체 실행 코드 (복사해서 바로 사용!)

```python
# ==========================================
# 전체 실행 코드 - Google Drive 공유 링크에서 다운로드 + ngrok 실행
# ==========================================

# 1. 패키지 설치
!pip install -q streamlit ultralytics opencv-python numpy pillow dynamsoft-barcode-reader-bundle pyngrok gdown
!apt-get -qq install -y fonts-nanum

# 2. 파일 준비
import os
import shutil
import glob

os.chdir('/content')
os.makedirs('pages', exist_ok=True)

# 3. Google Drive 공유 폴더에서 파일 다운로드
FOLDER_ID = '1lT2kc6h4gOJ6IMoFh0W6TeqUahbQxA7X'
print("📥 Google Drive 공유 폴더에서 파일 다운로드 중...")
print(f"🔗 폴더 ID: {FOLDER_ID}\n")

# gdown으로 폴더 다운로드
!gdown --folder "https://drive.google.com/drive/folders/{FOLDER_ID}?usp=sharing" -O /tmp/qr_files --remaining-ok

# 4. 파일 찾기 및 이동
print("\n📋 파일 정리 중...")

file_map = {
    'Home_colab.py': 'Home_colab.py',
    'yolo_dynamsoft.py': 'yolo_dynamsoft.py',
    'model1.pt': 'model1.pt',
    '1__비디오_QR_탐지.py': 'pages/1__비디오_QR_탐지.py',
    '2__이미지_QR_탐지.py': 'pages/2__이미지_QR_탐지.py',
    '3__프레임_추출.py': 'pages/3__프레임_추출.py',
}

# 다운로드된 파일 찾기
all_files = glob.glob('/tmp/qr_files/**/*', recursive=True)

for target, dest in file_map.items():
    found = None
    for f in all_files:
        if os.path.isfile(f) and target in os.path.basename(f):
            found = f
            break
    
    if found:
        shutil.copy(found, dest)
        size = os.path.getsize(dest) / 1024
        print(f"  ✅ {dest} ({size:.1f} KB)")
    else:
        print(f"  ⚠️ {target} - 찾을 수 없음")

# 5. 파일 확인
print("\n📋 최종 파일 확인:")
required = ['Home_colab.py', 'model1.pt', 'yolo_dynamsoft.py',
            'pages/1__비디오_QR_탐지.py', 'pages/2__이미지_QR_탐지.py', 'pages/3__프레임_추출.py']

all_ok = all(os.path.exists(f) for f in required)
for f in required:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"  ✅ {f} ({size:.2f} MB)")
    else:
        print(f"  ❌ {f} - 없음!")

# 6. ngrok으로 Streamlit 실행 (별도 브라우저 창!)
if all_ok:
    from pyngrok import ngrok
    import subprocess
    import threading
    import time
    
    # ⚠️ ngrok 토큰 설정 (선택사항 - 더 안정적)
    # 1. https://dashboard.ngrok.com/get-started/your-authtoken 에서 토큰 복사
    # 2. 아래 주석을 해제하고 토큰 입력
    # ngrok.set_auth_token("여기에_토큰_입력")  # 예: "2abc123def456ghi789jkl012mno345pq"
    
    # 또는 환경 변수로 설정
    # import os
    # os.environ['NGROK_AUTHTOKEN'] = '여기에_토큰_입력'
    
    def run_streamlit():
        subprocess.run([
            'streamlit', 'run', 'Home_colab.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    
    print("\n🚀 Streamlit 서버 시작 중...")
    thread = threading.Thread(target=run_streamlit, daemon=True)
    thread.start()
    
    # 서버 시작 대기
    print("⏳ 서버 시작 대기 중... (5초)")
    time.sleep(5)
    
    # ngrok 터널 생성
    print("🌐 ngrok 터널 생성 중...")
    try:
        public_url = ngrok.connect(8501)
        
        print("\n" + "="*70)
        print("✅ Streamlit이 성공적으로 실행 중입니다!")
        print("="*70)
        print(f"\n🔗 공개 URL: {public_url}")
        print(f"\n📱 별도 브라우저 창에서 위 링크를 클릭하세요!")
        print(f"💡 이 링크는 코랩 런타임이 실행 중일 때만 유효합니다.")
        print("="*70)
    except Exception as e:
        print(f"\n❌ ngrok 터널 생성 실패: {e}")
        print("💡 대체 방법을 시도합니다...")
        
        # 대체 방법: iframe 사용
        from google.colab import output
        try:
            output.serve_kernel_port_as_iframe(8501)
            print("✅ iframe으로 실행 중입니다.")
        except:
            print("❌ 실행 실패. 터미널 출력을 확인하세요.")
else:
    print("\n❌ 일부 필수 파일이 없습니다. 다운로드를 확인하세요.")
```

## 🎯 더 간단한 버전 (한 줄로 실행)

```python
# 한 번에 실행 (복사해서 붙여넣기)
exec("""# 설치
import subprocess
subprocess.run(['pip', 'install', '-q', 'streamlit', 'ultralytics', 'opencv-python', 'numpy', 'pillow', 'dynamsoft-barcode-reader-bundle', 'pyngrok', 'gdown'], check=False)
subprocess.run(['apt-get', 'update', '-qq'], check=False)
subprocess.run(['apt-get', 'install', '-qq', '-y', 'fonts-nanum'], check=False)

# 파일 준비
import os, shutil, glob
os.chdir('/content')
os.makedirs('pages', exist_ok=True)

# 다운로드
FOLDER_ID = '1lT2kc6h4gOJ6IMoFh0W6TeqUahbQxA7X'
os.system(f'gdown --folder "https://drive.google.com/drive/folders/{FOLDER_ID}?usp=sharing" -O /tmp/qr_files --remaining-ok')

# 파일 정리
file_map = {
    'Home_colab.py': 'Home_colab.py',
    'yolo_dynamsoft.py': 'yolo_dynamsoft.py',
    'model1.pt': 'model1.pt',
    '1__비디오_QR_탐지.py': 'pages/1__비디오_QR_탐지.py',
    '2__이미지_QR_탐지.py': 'pages/2__이미지_QR_탐지.py',
    '3__프레임_추출.py': 'pages/3__프레임_추출.py',
}
all_files = glob.glob('/tmp/qr_files/**/*', recursive=True)
for target, dest in file_map.items():
    for f in all_files:
        if os.path.isfile(f) and target in os.path.basename(f):
            shutil.copy(f, dest)
            break

# 실행
from pyngrok import ngrok
import threading, time
def run(): subprocess.run(['streamlit', 'run', 'Home_colab.py', '--server.port', '8501', '--server.address', 'localhost'])
threading.Thread(target=run, daemon=True).start()
time.sleep(5)
public_url = ngrok.connect(8501)
print(f"\\n✅ Streamlit 실행 중: {public_url}")
""")
```

## 📝 실행 순서

1. 위 코드를 코랩 노트북에 복사
2. 실행 (Runtime > Run all 또는 Shift+Enter)
3. 파일 다운로드 완료 대기
4. ngrok URL 확인
5. URL 클릭하여 별도 브라우저 창에서 열기

## ⚡ ngrok 대신 iframe 사용하려면

ngrok이 필요 없다면, 마지막 부분만 이렇게 변경:

```python
# ngrok 대신 iframe 사용
from google.colab import output
output.serve_kernel_port_as_iframe(8501)
```

## 💡 주의사항

1. **첫 실행**: 파일 다운로드에 시간이 걸릴 수 있습니다 (약 1-2분)
2. **ngrok 무료 버전**: 세션 시간 제한 (약 2시간)
3. **런타임 재시작**: 모든 파일이 사라지므로 코드를 다시 실행해야 합니다
4. **필수 파일**: 6개 파일이 모두 다운로드되어야 합니다

## 🔧 문제 해결

### 파일을 찾을 수 없을 때

```python
# 다운로드된 파일 확인
import glob
files = glob.glob('/tmp/qr_files/**/*', recursive=True)
print("다운로드된 파일들:")
for f in files[:20]:  # 처음 20개만 표시
    print(f"  - {f}")
```

### ngrok 연결 실패 시

```python
# iframe 방식으로 대체
from google.colab import output
output.serve_kernel_port_as_iframe(8501)
```

---

**🎯 사용법**: 위의 전체 실행 코드를 복사해서 코랩 노트북에 붙여넣고 실행하면 됩니다! 🚀

