"""
비디오 프레임 추출 도구 (Streamlit)
영상 파일에서 원하는 프레임을 추출하고 저장할 수 있는 간단한 도구
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import zipfile
import io
from typing import List, Tuple

# 페이지 설정
st.set_page_config(
    page_title="프레임 추출",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_video_info(video_path: str) -> Tuple[int, float, int, int, int]:
    """비디오 정보 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return total_frames, fps, width, height, duration

def extract_frame(video_path: str, frame_number: int) -> np.ndarray:
    """특정 프레임 번호의 이미지 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

def extract_frame_by_time(video_path: str, time_seconds: float, fps: float) -> np.ndarray:
    """특정 시간(초)의 프레임 추출"""
    frame_number = int(time_seconds * fps)
    return extract_frame(video_path, frame_number)

def create_zip_from_frames(frames_data: List[Tuple[np.ndarray, str]]) -> bytes:
    """여러 프레임을 ZIP 파일로 생성"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for frame, filename in frames_data:
            # 이미지를 메모리에 저장
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                zip_file.writestr(filename, buffer.tobytes())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def get_thumbnail(frame: np.ndarray, max_size: Tuple[int, int] = (200, 150)) -> np.ndarray:
    """프레임을 썸네일 크기로 리사이즈"""
    h, w = frame.shape[:2]
    max_w, max_h = max_size
    
    # 비율 유지하면서 리사이즈
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(frame, (new_w, new_h))

def generate_thumbnails(video_path: str, num_thumbnails: int = 20, 
                       start_frame: int = 0, end_frame: int = None) -> List[Tuple[int, np.ndarray]]:
    """비디오에서 썸네일 생성"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames - 1
    
    end_frame = min(end_frame, total_frames - 1)
    
    # 프레임 간격 계산
    frame_range = end_frame - start_frame + 1
    if num_thumbnails > frame_range:
        num_thumbnails = frame_range
    
    interval = frame_range // num_thumbnails if num_thumbnails > 0 else 1
    
    thumbnails = []
    for i in range(num_thumbnails):
        frame_num = start_frame + (i * interval)
        frame_num = min(frame_num, end_frame)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            thumbnail = get_thumbnail(frame)
            thumbnails.append((frame_num, thumbnail))
    
    cap.release()
    return thumbnails

def main():
    st.title("🎬 비디오 프레임 추출 도구")
    st.markdown("영상 파일에서 원하는 프레임을 추출하고 저장할 수 있습니다.")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    if 'temp_video_path' not in st.session_state:
        st.session_state.temp_video_path = None
    if 'extracted_frames' not in st.session_state:
        st.session_state.extracted_frames = []
    if 'current_preview_frame' not in st.session_state:
        st.session_state.current_preview_frame = 0
    if 'thumbnails' not in st.session_state:
        st.session_state.thumbnails = []
    
    # 사이드바
    with st.sidebar:
        st.header("📁 비디오 업로드")
        
        uploaded_file = st.file_uploader(
            "비디오 파일 선택",
            type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
            help="MP4, AVI, MOV 등 다양한 비디오 형식 지원"
        )
        
        if uploaded_file is not None:
            # 임시 파일로 저장
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_video_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.temp_video_path = temp_video_path
            
            # 비디오 정보 가져오기
            with st.spinner("비디오 정보 분석 중..."):
                total_frames, fps, width, height, duration = get_video_info(temp_video_path)
                
                if total_frames:
                    st.session_state.video_info = {
                        'total_frames': total_frames,
                        'fps': fps,
                        'width': width,
                        'height': height,
                        'duration': duration,
                        'filename': uploaded_file.name
                    }
                    st.success("✅ 비디오 로드 완료!")
                else:
                    st.error("❌ 비디오 파일을 읽을 수 없습니다.")
        
        st.markdown("---")
        
        # 비디오 정보 표시
        if st.session_state.video_info:
            info = st.session_state.video_info
            st.header("📊 비디오 정보")
            st.text(f"파일명: {info['filename']}")
            st.text(f"해상도: {info['width']} x {info['height']}")
            st.text(f"FPS: {info['fps']:.2f}")
            st.text(f"총 프레임: {info['total_frames']:,}개")
            st.text(f"재생 시간: {info['duration']:.2f}초")
            
            st.markdown("---")
    
    # 메인 영역
    if st.session_state.video_info:
        info = st.session_state.video_info
        
        # 탭으로 기능 분리
        tab1, tab2 = st.tabs(["🔍 장면 탐색", "📸 프레임 추출"])
        
        with tab1:
            st.header("🔍 장면 탐색 - 원하는 프레임 찾기")
            
            # 탐색 모드 선택
            search_mode = st.radio(
                "탐색 방법",
                ["프레임 탐색기", "썸네일 그리드"],
                horizontal=True,
                help="프레임 탐색기: 슬라이더로 프레임 탐색\n썸네일 그리드: 여러 프레임을 한눈에 보기"
            )
            
            st.markdown("---")
            
            if search_mode == "프레임 탐색기":
                # 프레임 탐색기 모드
                st.subheader("프레임 탐색기")
                
                # 현재 프레임 표시
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    current_frame = st.slider(
                        "프레임 탐색",
                        min_value=0,
                        max_value=info['total_frames'] - 1,
                        value=st.session_state.current_preview_frame,
                        key="frame_explorer_slider",
                        help="슬라이더를 움직여 원하는 프레임을 찾으세요"
                    )
                    st.session_state.current_preview_frame = current_frame
                
                with col2:
                    time_sec = current_frame / info['fps'] if info['fps'] > 0 else 0
                    st.metric("시간", f"{time_sec:.2f}초")
                
                with col3:
                    progress = (current_frame / info['total_frames']) * 100 if info['total_frames'] > 0 else 0
                    st.metric("진행률", f"{progress:.1f}%")
                
                with col4:
                    # 프레임 이동 버튼
                    col_prev, col_next = st.columns(2)
                    with col_prev:
                        if st.button("◀️", key="prev_frame"):
                            st.session_state.current_preview_frame = max(0, current_frame - 1)
                            st.rerun()
                    with col_next:
                        if st.button("▶️", key="next_frame"):
                            st.session_state.current_preview_frame = min(info['total_frames'] - 1, current_frame + 1)
                            st.rerun()
                
                # 빠른 이동 버튼
                st.caption("빠른 이동:")
                col_fast = st.columns(5)
                jump_values = [10, 30, 100, info['total_frames'] // 4, info['total_frames'] // 2]
                jump_labels = ["+10프레임", "+30프레임", "+100프레임", "1/4지점", "중간지점"]
                for i, (jump, label) in enumerate(zip(jump_values, jump_labels)):
                    with col_fast[i]:
                        if st.button(label, key=f"jump_{jump}"):
                            new_frame = min(info['total_frames'] - 1, current_frame + jump)
                            st.session_state.current_preview_frame = new_frame
                            st.rerun()
                
                st.markdown("---")
                
                # 현재 프레임 미리보기
                with st.spinner("프레임 로딩 중..."):
                    frame = extract_frame(st.session_state.temp_video_path, current_frame)
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, width='stretch', caption=f"프레임 #{current_frame} | {time_sec:.2f}초")
                        
                        # 프레임 정보
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.info(f"**프레임 번호**: {current_frame:,} / {info['total_frames']:,}")
                        with col_info2:
                            st.info(f"**시간**: {time_sec:.2f}초 / {info['duration']:.2f}초")
                        
                        # 이 프레임 추출 버튼
                        if st.button("📸 이 프레임 추출하기", width='stretch', type="primary"):
                            frame_rgb_copy = frame_rgb.copy()
                            filename = f"frame_{current_frame:06d}_{time_sec:.2f}s.jpg"
                            st.session_state.extracted_frames = [(frame_rgb_copy, filename)]
                            st.success(f"✅ 프레임 #{current_frame}이 추출 목록에 추가되었습니다!")
                            st.rerun()
                    else:
                        st.error("❌ 프레임을 읽을 수 없습니다.")
            
            else:
                # 썸네일 그리드 모드
                st.subheader("썸네일 그리드 - 여러 프레임 한눈에 보기")
                
                col_grid1, col_grid2 = st.columns([3, 1])
                
                with col_grid1:
                    # 탐색 범위 설정
                    search_start = st.number_input(
                        "시작 프레임",
                        min_value=0,
                        max_value=info['total_frames'] - 1,
                        value=st.session_state.current_preview_frame,
                        key="search_start"
                    )
                    search_end = st.number_input(
                        "끝 프레임",
                        min_value=0,
                        max_value=info['total_frames'] - 1,
                        value=min(search_start + 500, info['total_frames'] - 1),
                        key="search_end"
                    )
                
                with col_grid2:
                    num_thumbnails = st.number_input(
                        "썸네일 개수",
                        min_value=4,
                        max_value=50,
                        value=20,
                        help="생성할 썸네일 개수"
                    )
                
                if st.button("🔍 썸네일 생성", width='stretch', type="primary"):
                    with st.spinner(f"{num_thumbnails}개 썸네일 생성 중..."):
                        thumbnails = generate_thumbnails(
                            st.session_state.temp_video_path,
                            num_thumbnails,
                            search_start,
                            search_end
                        )
                        st.session_state.thumbnails = thumbnails
                
                # 썸네일 표시
                if st.session_state.thumbnails:
                    st.markdown("---")
                    st.subheader(f"썸네일 그리드 ({len(st.session_state.thumbnails)}개)")
                    st.caption("썸네일을 클릭하면 해당 프레임으로 이동합니다.")
                    
                    # 그리드 레이아웃 (5열)
                    num_cols = 5
                    for i in range(0, len(st.session_state.thumbnails), num_cols):
                        cols = st.columns(num_cols)
                        for j, (frame_num, thumbnail) in enumerate(st.session_state.thumbnails[i:i+num_cols]):
                            with cols[j]:
                                thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                                time_sec = frame_num / info['fps'] if info['fps'] > 0 else 0
                                
                                st.image(thumbnail_rgb, width='stretch', 
                                        caption=f"#{frame_num}\n{time_sec:.1f}초")
                                
                                # 썸네일 클릭 시 해당 프레임으로 이동
                                if st.button(f"선택", key=f"select_thumb_{i+j}", width='stretch'):
                                    st.session_state.current_preview_frame = frame_num
                                    st.info(f"✅ 프레임 #{frame_num} 선택됨. 프레임 탐색기로 이동하세요.")
                                    st.rerun()
                                
                                # 바로 추출 버튼
                                if st.button(f"추출", key=f"extract_thumb_{i+j}", width='stretch'):
                                    full_frame = extract_frame(st.session_state.temp_video_path, frame_num)
                                    if full_frame is not None:
                                        frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
                                        filename = f"frame_{frame_num:06d}_{time_sec:.2f}s.jpg"
                                        st.session_state.extracted_frames = [(frame_rgb, filename)]
                                        st.success(f"✅ 프레임 #{frame_num} 추출 완료!")
                                        st.rerun()
        
        with tab2:
            st.header("🎯 프레임 추출")
        
        # 추출 방법 선택
        extraction_method = st.radio(
            "추출 방법 선택",
            ["프레임 번호로 추출", "시간(초)으로 추출", "여러 프레임 일괄 추출"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if extraction_method == "프레임 번호로 추출":
            # 단일 프레임 추출 (프레임 번호)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                frame_number = st.slider(
                    "프레임 번호",
                    min_value=0,
                    max_value=info['total_frames'] - 1,
                    value=0,
                    help=f"0부터 {info['total_frames'] - 1}까지 선택 가능"
                )
            
            with col2:
                # 시간으로 프레임 번호 계산
                time_seconds = frame_number / info['fps'] if info['fps'] > 0 else 0
                st.metric("시간", f"{time_seconds:.2f}초")
            
            if st.button("📸 프레임 추출", width='stretch', type="primary"):
                with st.spinner("프레임 추출 중..."):
                    frame = extract_frame(st.session_state.temp_video_path, frame_number)
                    
                    if frame is not None:
                        # BGR을 RGB로 변환
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.session_state.extracted_frames = [(frame_rgb, f"frame_{frame_number:06d}.jpg")]
                        
                        st.success(f"✅ 프레임 #{frame_number} 추출 완료!")
                    else:
                        st.error("❌ 프레임 추출 실패")
        
        elif extraction_method == "시간(초)으로 추출":
            # 단일 프레임 추출 (시간)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                time_seconds = st.slider(
                    "시간 (초)",
                    min_value=0.0,
                    max_value=info['duration'],
                    value=0.0,
                    step=0.1,
                    help=f"0부터 {info['duration']:.2f}초까지 선택 가능"
                )
            
            with col2:
                # 프레임 번호 계산
                frame_number = int(time_seconds * info['fps']) if info['fps'] > 0 else 0
                frame_number = min(frame_number, info['total_frames'] - 1)
                st.metric("프레임 번호", f"#{frame_number}")
            
            if st.button("📸 프레임 추출", width='stretch', type="primary"):
                with st.spinner("프레임 추출 중..."):
                    frame = extract_frame_by_time(st.session_state.temp_video_path, 
                                                  time_seconds, info['fps'])
                    
                    if frame is not None:
                        # BGR을 RGB로 변환
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.session_state.extracted_frames = [(frame_rgb, f"frame_{frame_number:06d}_{time_seconds:.2f}s.jpg")]
                        
                        st.success(f"✅ {time_seconds:.2f}초 프레임 추출 완료!")
                    else:
                        st.error("❌ 프레임 추출 실패")
        
        else:
            # 여러 프레임 일괄 추출
            st.subheader("여러 프레임 일괄 추출")
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_frame = st.number_input(
                    "시작 프레임",
                    min_value=0,
                    max_value=info['total_frames'] - 1,
                    value=0,
                    help="추출 시작 프레임 번호"
                )
                
                interval = st.number_input(
                    "간격 (프레임)",
                    min_value=1,
                    max_value=info['total_frames'],
                    value=30,
                    help="몇 프레임마다 추출할지 설정"
                )
            
            with col2:
                end_frame = st.number_input(
                    "끝 프레임",
                    min_value=0,
                    max_value=info['total_frames'] - 1,
                    value=min(300, info['total_frames'] - 1),
                    help="추출 종료 프레임 번호"
                )
                
                max_frames = st.number_input(
                    "최대 추출 개수",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="최대 추출할 프레임 개수 제한"
                )
            
            # 추출할 프레임 목록 미리보기
            frames_to_extract = []
            current_frame = start_frame
            while current_frame <= end_frame and len(frames_to_extract) < max_frames:
                frames_to_extract.append(current_frame)
                current_frame += interval
            
            st.info(f"💡 총 {len(frames_to_extract)}개 프레임이 추출됩니다: {frames_to_extract[:10]}{'...' if len(frames_to_extract) > 10 else ''}")
            
            if st.button("📸 여러 프레임 추출", width='stretch', type="primary"):
                extracted = []
                progress_bar = st.progress(0)
                
                for i, frame_num in enumerate(frames_to_extract):
                    frame = extract_frame(st.session_state.temp_video_path, frame_num)
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        time_sec = frame_num / info['fps'] if info['fps'] > 0 else 0
                        filename = f"frame_{frame_num:06d}_{time_sec:.2f}s.jpg"
                        extracted.append((frame_rgb, filename))
                    
                    progress_bar.progress((i + 1) / len(frames_to_extract))
                
                if extracted:
                    st.session_state.extracted_frames = extracted
                    st.success(f"✅ {len(extracted)}개 프레임 추출 완료!")
                else:
                    st.error("❌ 프레임 추출 실패")
                
                progress_bar.empty()
        
        # 추출된 프레임 표시 및 다운로드
        if st.session_state.extracted_frames:
            st.markdown("---")
            st.header("📷 추출된 프레임")
            
            # 다운로드 옵션
            col1, col2 = st.columns([1, 4])
            with col1:
                if len(st.session_state.extracted_frames) > 1:
                    download_as_zip = st.checkbox("ZIP으로 다운로드", value=True)
                else:
                    download_as_zip = False
            
            # 프레임 표시
            num_cols = 3
            for i in range(0, len(st.session_state.extracted_frames), num_cols):
                cols = st.columns(num_cols)
                for j, (frame_rgb, filename) in enumerate(st.session_state.extracted_frames[i:i+num_cols]):
                    with cols[j]:
                        st.image(frame_rgb, width='stretch', caption=filename)
                        
                        # 개별 다운로드 버튼
                        if not download_as_zip:
                            # 이미지를 바이트로 변환
                            pil_image = Image.fromarray(frame_rgb)
                            buf = io.BytesIO()
                            pil_image.save(buf, format='JPEG')
                            buf.seek(0)
                            
                            st.download_button(
                                label="⬇️ 다운로드",
                                data=buf.getvalue(),
                                file_name=filename,
                                mime="image/jpeg",
                                key=f"download_{i+j}"
                            )
            
            # ZIP 다운로드 (여러 프레임인 경우)
            if download_as_zip and len(st.session_state.extracted_frames) > 1:
                st.markdown("---")
                st.subheader("📦 일괄 다운로드")
                
                if st.button("📥 모든 프레임 ZIP으로 다운로드", width='stretch', type="primary"):
                    # BGR로 변환하여 ZIP 생성
                    frames_bgr = []
                    for frame_rgb, filename in st.session_state.extracted_frames:
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        frames_bgr.append((frame_bgr, filename))
                    
                    zip_data = create_zip_from_frames(frames_bgr)
                    
                    zip_filename = f"{os.path.splitext(info['filename'])[0]}_frames.zip"
                    st.download_button(
                        label=f"⬇️ {len(st.session_state.extracted_frames)}개 프레임 다운로드",
                        data=zip_data,
                        file_name=zip_filename,
                        mime="application/zip",
                        width='stretch'
                    )
    else:
        # 비디오가 업로드되지 않은 경우
        st.info("👈 왼쪽 사이드바에서 비디오 파일을 업로드하세요.")
        
        st.markdown("### 지원 형식")
        st.text("• MP4 (.mp4)")
        st.text("• AVI (.avi)")
        st.text("• MOV (.mov)")
        st.text("• MKV (.mkv)")
        st.text("• FLV (.flv)")
        st.text("• WMV (.wmv)")
        
        st.markdown("### 주요 기능")
        st.text("✅ 프레임 번호로 프레임 추출")
        st.text("✅ 시간(초)으로 프레임 추출")
        st.text("✅ 여러 프레임 일괄 추출")
        st.text("✅ 개별 이미지 다운로드")
        st.text("✅ ZIP 파일로 일괄 다운로드")

if __name__ == "__main__":
    main()

