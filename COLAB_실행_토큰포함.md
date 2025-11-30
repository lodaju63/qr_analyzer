# 🚀 코랩 실행 - ngrok 토큰 포함 버전

ngrok 토큰을 설정하여 더 안정적으로 실행하는 버전입니다.

## 🔑 ngrok 토큰 받기

1. **ngrok 대시보드 접속**: https://dashboard.ngrok.com/
2. **무료 계정 생성** (이메일로 가입)
3. **토큰 복사**: https://dashboard.ngrok.com/get-started/your-authtoken
   - 토큰 예시: `2abc123def456ghi789jkl012mno345pqrs678tuv901wxyz234`

## 📋 전체 실행 코드 (토큰 설정 포함)

```python
# ==========================================
# 전체 실행 코드 - Google Drive 공유 링크 + ngrok 토큰
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
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  ✅ {f} ({size:.2f} MB)")
    else:
        print(f"  ❌ {f} - 없음!")

# 6. ngrok 토큰 설정 및 Streamlit 실행
if all_ok:
    from pyngrok import ngrok
    import subprocess
    import threading
    import time
    
    # ⚠️⚠️⚠️ 여기에 ngrok 토큰 입력! ⚠️⚠️⚠️
    # https://dashboard.ngrok.com/get-started/your-authtoken 에서 토큰 복사
    NGROK_TOKEN = "여기에_토큰_입력"  # 예: "2abc123def456ghi789jkl012mno345pqrs678tuv901wxyz234"
    
    # 토큰 설정 (토큰이 입력된 경우만)
    if NGROK_TOKEN != "여기에_토큰_입력" and NGROK_TOKEN:
        try:
            ngrok.set_auth_token(NGROK_TOKEN)
            print("✅ ngrok 토큰 설정 완료!")
        except Exception as e:
            print(f"⚠️ 토큰 설정 실패: {e}")
            print("💡 토큰 없이 계속 진행합니다...")
    else:
        print("⚠️ ngrok 토큰이 설정되지 않았습니다. 무료 버전으로 실행됩니다.")
    
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
        print("\n💡 해결 방법:")
        print("   1. 위의 NGROK_TOKEN 변수에 토큰을 입력하세요")
        print("   2. 또는 iframe 방식으로 대체합니다...")
        
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

## 🎯 토큰 입력 위치 (간단히)

코드에서 이 부분만 찾아서 수정하세요:

```python
# ⚠️⚠️⚠️ 이 부분!
NGROK_TOKEN = "여기에_토큰_입력"  # ← 여기에 토큰 붙여넣기
```

예시:
```python
NGROK_TOKEN = "2abc123def456ghi789jkl012mno345pqrs678tuv901wxyz234"
```

## 💡 토큰 없이 사용하기

토큰 없이도 사용 가능하지만 제한이 있습니다:
- 세션 시간 제한 (약 2시간)
- 랜덤 URL
- 연결 수 제한

토큰 없이 사용하려면 `NGROK_TOKEN = ""` 또는 그대로 두면 됩니다. 실패 시 자동으로 iframe 방식으로 전환됩니다.

## 🔧 토큰 찾는 방법 (요약)

1. https://dashboard.ngrok.com/ 접속
2. 무료 계정 생성/로그인  
3. https://dashboard.ngrok.com/get-started/your-authtoken 접속
4. 토큰 복사 (긴 문자열)
5. 코드의 `NGROK_TOKEN = "여기에_토큰_입력"` 부분에 붙여넣기

---

**💡 팁**: 토큰을 설정하면 더 안정적이고 긴 세션을 사용할 수 있습니다! 🚀

