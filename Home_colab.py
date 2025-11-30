"""
QR 탐지 시스템 - 메인 홈페이지 (코랩용)
Streamlit 멀티페이지 앱의 진입점
코랩 환경에서 바로 실행 가능하도록 최적화
"""

import streamlit as st
import os
import sys

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
    'streamlit.runtime.media_file_storage',
    'streamlit.web.server.media_file_handler',
]
for logger_name in streamlit_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
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
    
    # pages 폴더가 없으면 생성
    pages_dir = '/content/pages'
    if not os.path.exists(pages_dir):
        os.makedirs(pages_dir)
        st.warning("⚠️ `pages` 폴더를 생성했습니다. 필요한 페이지 파일들을 업로드하거나 복사하세요.")
    
    # ============================================================================
    # 🎯 YOLO 모델 경로 설정 (여기서만 수정하면 됩니다!)
    # ============================================================================
    YOLO_MODEL_PATH = 'best.pt'  # 다른 모델 테스트 시 여기만 변경하세요 (예: 'model1.pt', 'yolov8n.pt' 등) - best.pt는 OBB 모델입니다
    
    # 여러 경로에서 모델 파일 찾기
    if not os.path.exists(YOLO_MODEL_PATH):
        possible_paths = [
            f'/content/{YOLO_MODEL_PATH}',
            f'/content/drive/MyDrive/{YOLO_MODEL_PATH}',
            f'./{YOLO_MODEL_PATH}',
            YOLO_MODEL_PATH
        ]
        for path in possible_paths:
            if os.path.exists(path):
                YOLO_MODEL_PATH = path
                break
    
    # 환경 변수로 설정하여 다른 파일들이 참조할 수 있도록 함
    os.environ['YOLO_MODEL_PATH'] = YOLO_MODEL_PATH

# 페이지 설정
st.set_page_config(
    page_title="QR 탐지 시스템",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📱 QR 탐지 시스템")
    st.markdown("---")
    
    # 코랩 환경 정보 표시
    if IN_COLAB:
        st.info("🌐 **코랩 환경에서 실행 중** - 모든 기능을 사용할 수 있습니다.")
    
    st.markdown("""
    ## 🎯 시스템 개요
    
    이 시스템은 다양한 환경에서 QR 코드를 탐지하고 해독하는 도구들을 제공합니다.
    
    ### 📑 사용 가능한 페이지
    
    왼쪽 사이드바에서 원하는 도구를 선택하세요:
    
    #### 1️⃣ 🎬 비디오 QR 탐지
    - 동영상 파일에서 QR 코드를 실시간으로 탐지하고 해독
    - 추적 기능으로 QR 코드 추적
    - 결과 비디오 저장 및 분석
    
    #### 2️⃣ 🖼️ 이미지 QR 탐지
    - 단일 이미지에서 QR 코드 탐지 및 해독
    - 다양한 전처리 옵션 제공 (노이즈 제거, 대비 조정, 이진화 등)
    - 딥러닝 기반 이미지 향상 (U-Net/SegFormer)
    - 원본 및 전처리 결과 이미지 다운로드
    
    #### 3️⃣ 🎥 프레임 추출
    - 비디오 파일에서 원하는 프레임 추출
    - 프레임 탐색기로 비디오 탐색
    - 썸네일 그리드 보기
    - 여러 프레임 일괄 추출
    
    ---
    
    ### 🔧 시작하기
    
    1. 왼쪽 사이드바에서 원하는 도구 선택
    2. 각 도구의 가이드에 따라 사용
    3. 필요한 모델 파일이 준비되어 있는지 확인하세요
    
    ### 📚 더 알아보기
    
    - 각 페이지의 사이드바에 상세한 설명과 가이드가 있습니다.
    - 문제가 발생하면 각 페이지의 도움말을 참고하세요.
    """)
    
    # 코랩 환경 안내
    if IN_COLAB:
        st.markdown("---")
        st.subheader("💡 코랩 사용 팁")
        with st.expander("📋 필요한 파일 및 설정"):
            st.markdown("""
            ### 필수 파일:
            1. **모델 파일**: `best.pt` (YOLO OBB 모델)
            2. **pages 폴더의 파일들**:
               - `pages/1__비디오_QR_탐지.py`
               - `pages/2__이미지_QR_탐지.py`
               - `pages/3__프레임_추출.py`
            
            ### 권장 위치:
            - `/content/` 또는 `/content/drive/MyDrive/`
            
            ### 필요한 패키지:
            ```python
            !pip install streamlit ultralytics opencv-python numpy pillow
            !pip install dynamsoft-barcode-reader-bundle
            ```
            
            ### 실행 방법:
            ```python
            !streamlit run Home_colab.py --server.port 8501 --server.address 0.0.0.0
            ```
            
            ⚠️ **주의**: 런타임 재시작 시 `/content`의 파일이 사라지므로, Google Drive에 파일을 저장하세요!
            """)
    
    # 빠른 링크
    st.markdown("---")
    st.subheader("🚀 빠른 시작")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎬 비디오 QR 탐지**
        
        동영상 처리에 최적화
        """)
        if st.button("비디오 QR 탐지 시작", width='stretch', use_container_width=True):
            st.switch_page("pages/1__비디오_QR_탐지.py")
    
    with col2:
        st.markdown("""
        **🖼️ 이미지 QR 탐지**
        
        단일 이미지 처리
        """)
        if st.button("이미지 QR 탐지 시작", width='stretch', use_container_width=True):
            st.switch_page("pages/2__이미지_QR_탐지.py")
    
    with col3:
        st.markdown("""
        **🎥 프레임 추출**
        
        비디오 프레임 추출
        """)
        if st.button("프레임 추출 시작", width='stretch', use_container_width=True):
            st.switch_page("pages/3__프레임_추출.py")

if __name__ == "__main__":
    main()

