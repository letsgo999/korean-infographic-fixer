"""
Korean Infographic Fixer - Streamlit Main App
한글 인포그래픽 교정 도구 (하이브리드 방식)
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from datetime import datetime
import tempfile
import os

# 모듈 임포트
from modules import (
    TextRegion,
    run_enhanced_ocr,
    apply_styles_and_colors,
    group_regions_by_lines,
    create_inpainter,
    MetadataBuilder,
    create_manual_region,
    CompositeRenderer,
    MultiFormatExporter
)

from config.settings import (
    AVAILABLE_FONTS,
    STYLE_TAGS,
    UI_CONFIG,
    EXPORT_CONFIG
)

# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="한글 인포그래픽 교정 도구",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 세션 상태 초기화
# ============================================
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'uploaded_image': None,
        'original_image': None,
        'processed_image': None,
        'text_regions': [],
        'edited_texts': {},
        'current_step': 1,
        'metadata': None,
        'background_image': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================
# 유틸리티 함수
# ============================================
def load_image(uploaded_file) -> np.ndarray:
    """업로드된 파일을 OpenCV 이미지로 변환"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

def draw_regions_on_image(image: np.ndarray, regions: list, edited_texts: dict = None) -> np.ndarray:
    """이미지에 텍스트 영역 표시"""
    result = image.copy()
    edited_texts = edited_texts or {}
    
    colors = {
        'normal': (0, 200, 0),      # 녹색
        'inverted': (255, 100, 0),  # 파란색 (BGR)
        'manual': (0, 165, 255),    # 주황색 (BGR)
        'edited': (255, 0, 255),    # 마젠타 (편집됨)
    }
    
    for region in regions:
        if isinstance(region, dict):
            b = region['bounds']
            region_id = region['id']
            is_inverted = region.get('is_inverted', False)
            is_manual = region.get('is_manual', False)
        else:
            b = region.bounds
            region_id = region.id
            is_inverted = region.is_inverted
            is_manual = region.is_manual
        
        # 색상 결정
        if region_id in edited_texts:
            color = colors['edited']
        elif is_manual:
            color = colors['manual']
        elif is_inverted:
            color = colors['inverted']
        else:
            color = colors['normal']
        
        # 사각형 그리기
        cv2.rectangle(
            result,
            (b['x'], b['y']),
            (b['x'] + b['width'], b['y'] + b['height']),
            color,
            2
        )
    
    return result

def regions_to_list(regions) -> list:
    """TextRegion 객체를 딕셔너리 리스트로 변환"""
    result = []
    for r in regions:
        if isinstance(r, TextRegion):
            result.append(r.to_dict())
        else:
            result.append(r)
    return result

# ============================================
# UI 컴포넌트
# ============================================
def render_header():
    """헤더 렌더링"""
    st.title("🖼️ 한글 인포그래픽 교정 도구")
    st.markdown("""
    **AI 생성 인포그래픽의 깨진 한글 텍스트를 교정합니다.**
    
    - 🔍 OCR로 텍스트 자동 감지
    - ✏️ 수동으로 텍스트 영역 추가/수정
    - 🎨 폰트, 크기, 색상 커스터마이징
    - 📤 PNG, PDF 다중 포맷 출력
    """)
    st.divider()

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 폰트 설정
        st.subheader("폰트 설정")
        font_family = st.selectbox(
            "기본 폰트",
            options=list(AVAILABLE_FONTS.keys()),
            index=0
        )
        
        default_font_size = st.slider(
            "기본 폰트 크기",
            min_value=8,
            max_value=72,
            value=16
        )
        
        # 색상 설정
        st.subheader("색상 설정")
        default_text_color = st.color_picker("기본 글자색", "#333333")
        default_bg_color = st.color_picker("기본 배경색", "#FFFFFF")
        
        # 출력 설정
        st.subheader("출력 설정")
        output_formats = st.multiselect(
            "출력 포맷",
            options=["PNG", "PDF"],
            default=["PNG"]
        )
        
        st.divider()
        
        # 현재 상태 표시
        st.subheader("📊 현재 상태")
        if st.session_state.text_regions:
            total = len(st.session_state.text_regions)
            edited = len(st.session_state.edited_texts)
            st.metric("감지된 텍스트 영역", total)
            st.metric("수정된 영역", edited)
        else:
            st.info("이미지를 업로드하세요")
        
        return {
            'font_family': font_family,
            'font_size': default_font_size,
            'text_color': default_text_color,
            'bg_color': default_bg_color,
            'output_formats': output_formats
        }

def render_step1_upload():
    """Step 1: 이미지 업로드"""
    st.header("📤 Step 1: 이미지 업로드")
    
    uploaded_file = st.file_uploader(
        "인포그래픽 이미지를 업로드하세요",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="PNG, JPG, WEBP 형식을 지원합니다."
    )
    
    if uploaded_file:
        image = load_image(uploaded_file)
        st.session_state.original_image = image
        st.session_state.uploaded_image = uploaded_file.name
        
        # 이미지 표시
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption=f"업로드된 이미지: {uploaded_file.name}",
                use_container_width=True
            )
        with col2:
            st.info(f"""
            **이미지 정보**
            - 파일명: {uploaded_file.name}
            - 크기: {image.shape[1]} x {image.shape[0]} px
            - 채널: {image.shape[2] if len(image.shape) > 2 else 1}
            """)
        
        if st.button("🔍 텍스트 자동 감지 시작", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

def render_step2_detect():
    """Step 2: 텍스트 감지"""
    st.header("🔍 Step 2: 텍스트 영역 감지")
    
    if st.session_state.original_image is None:
        st.warning("먼저 이미지를 업로드하세요.")
        return
    
    image = st.session_state.original_image
    
    # OCR 실행
    with st.spinner("텍스트 영역을 감지하는 중..."):
        try:
            # 향상된 OCR 실행
            ocr_results = run_enhanced_ocr(image)
            
            # 라인 단위 그룹핑
            all_regions = ocr_results['all_regions']
            
            # 일반 영역 그룹핑
            normal_grouped = group_regions_by_lines(ocr_results['normal_regions'])
            
            # 역상 영역은 그대로 (이미 파편화되어 있음)
            # 나중에 수동으로 병합 가능하도록 함
            
            # 스타일 및 색상 적용
            all_grouped = normal_grouped + ocr_results['inverted_regions']
            styled_regions = apply_styles_and_colors(image, all_grouped)
            
            # 세션에 저장
            st.session_state.text_regions = regions_to_list(styled_regions)
            
            st.success(f"✅ {len(styled_regions)}개의 텍스트 영역을 감지했습니다!")
            
        except Exception as e:
            st.error(f"OCR 실행 중 오류 발생: {e}")
            return
    
    # 결과 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 감지된 영역 시각화
        visualized = draw_regions_on_image(image, st.session_state.text_regions)
        st.image(
            cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB),
            caption="감지된 텍스트 영역 (🟢 일반 | 🔵 역상 | 🟠 수동)",
            use_container_width=True
        )
    
    with col2:
        st.subheader("감지 결과 요약")
        regions = st.session_state.text_regions
        
        normal_count = len([r for r in regions if not r.get('is_inverted', False)])
        inverted_count = len([r for r in regions if r.get('is_inverted', False)])
        
        st.metric("일반 텍스트", normal_count)
        st.metric("역상 텍스트", inverted_count)
        
        avg_conf = sum(r['confidence'] for r in regions) / len(regions) if regions else 0
        st.metric("평균 신뢰도", f"{avg_conf:.1f}%")
    
    st.divider()
    
    # 다음 단계 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 이전 단계"):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("✏️ 텍스트 편집으로 이동", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

def render_step3_edit():
    """Step 3: 텍스트 편집 (폰트 선택 및 장평 조절 기능 추가)"""
    st.header("✏️ Step 3: 텍스트 편집")
    
    if not st.session_state.text_regions:
        st.warning("먼저 텍스트 감지를 실행하세요.")
        return
    
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    
    # fonts 폴더의 폰트 파일 목록 읽어오기
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    if not os.path.exists(fonts_dir):
        os.makedirs(fonts_dir)
        
    available_fonts = [f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')]
    if not available_fonts:
        st.error("fonts 폴더에 .ttf 폰트 파일이 없습니다!")
        available_fonts = ["Default"]

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 텍스트 영역 목록")
        
        filter_option = st.radio("필터", ["전체", "일반", "역상", "수동 추가"], horizontal=True)
        
        if filter_option == "일반": filtered = [r for r in regions if not r.get('is_inverted') and not r.get('is_manual')]
        elif filter_option == "역상": filtered = [r for r in regions if r.get('is_inverted')]
        elif filter_option == "수동 추가": filtered = [r for r in regions if r.get('is_manual')]
        else: filtered = regions
        
        for i, region in enumerate(filtered):
            region_id = region['id']
            display_text = region['text'][:30] + "..." if len(region['text']) > 30 else region['text']
            
            with st.expander(f"📝 {i+1}. {display_text}", expanded=False):
                # 수정 텍스트 입력
                edited = st.text_area("수정된 텍스트", value=st.session_state.edited_texts.get(region_id, region['text']), key=f"text_{region_id}_{i}", height=80)
                
                # --- [UI 업데이트] 3단 레이아웃 (폰트선택 / 크기 / 장평) ---
                c1, c2, c3 = st.columns([2, 1, 1])
                
                with c1:
                    # 폰트 파일 선택 (기본값: 기존 설정 or 첫번째 폰트)
                    current_font = region.get('font_filename', available_fonts[0])
                    if current_font not in available_fonts: current_font = available_fonts[0]
                    
                    selected_font = st.selectbox("폰트 선택", options=available_fonts, index=available_fonts.index(current_font), key=f"font_{region_id}_{i}")
                
                with c2:
                    font_size = st.number_input("크기", min_value=8, max_value=200, value=int(region.get('suggested_font_size', 16)), key=f"size_{region_id}_{i}")
                    
                with c3:
                    # 장평 조절 슬라이더 (50% ~ 150%)
                    width_scale = st.number_input("장평(%)", min_value=50, max_value=200, value=int(region.get('width_scale', 100)), step=5, key=f"scale_{region_id}_{i}")

                # 글자색
                text_color = st.color_picker("글자색", value=region.get('text_color', '#333333'), key=f"color_{region_id}_{i}")
                
                if st.button("💾 저장", key=f"save_{region_id}_{i}"):
                    st.session_state.edited_texts[region_id] = edited
                    for r in st.session_state.text_regions:
                        if r['id'] == region_id:
                            r['text'] = edited
                            r['suggested_font_size'] = font_size
                            r['text_color'] = text_color
                            r['font_filename'] = selected_font # 폰트 저장
                            r['width_scale'] = width_scale     # 장평 저장
                            break
                    st.success("저장되었습니다!")
                    st.rerun()
    
    with col2:
        st.subheader("🖼️ 미리보기")
        visualized = draw_regions_on_image(image, regions, st.session_state.edited_texts)
        st.image(cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB), caption="편집 미리보기", use_container_width=True)
        
        st.divider()
        st.subheader("➕ 수동 영역 추가")
        with st.form("manual_region_form"):
            new_text = st.text_input("텍스트 내용")
            col_x, col_y = st.columns(2)
            with col_x: x = st.number_input("X 좌표", min_value=0, value=50); width = st.number_input("너비", min_value=10, value=200)
            with col_y: y = st.number_input("Y 좌표", min_value=0, value=50); height = st.number_input("높이", min_value=10, value=30)
            if st.form_submit_button("영역 추가"):
                if new_text:
                    from modules import create_manual_region
                    new_region = create_manual_region(x=x, y=y, width=width, height=height, text=new_text)
                    # 수동 영역 기본값 설정
                    new_region.font_filename = available_fonts[0]
                    st.session_state.text_regions.append(new_region.to_dict())
                    st.success("추가됨!"); st.rerun()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 이전 단계"): st.session_state.current_step = 2; st.rerun()
    with col2:
        if st.button("📤 내보내기로 이동", type="primary"): st.session_state.current_step = 4; st.rerun()

def render_step4_export(settings: dict):
    """Step 4: 내보내기 (수정된 영역만 반영 버전)"""
    st.header("📤 Step 4: 내보내기")
    
    if not st.session_state.text_regions:
        st.warning("먼저 텍스트 편집을 완료하세요.")
        return
    
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    
    # ------------------------------------------------------------------
    # [핵심 로직 변경] 수정된 영역만 골라내기
    # ------------------------------------------------------------------
    edited_ids = set(st.session_state.edited_texts.keys())
    
    target_regions = []
    target_objects = [] # 클래스 객체용
    
    for r in regions:
        # 1. 사용자가 내용을 수정하고 [저장]을 누른 영역
        is_edited = r['id'] in edited_ids
        # 2. 사용자가 [수동 영역 추가]로 만든 영역
        is_manual = r.get('is_manual', False)
        
        if is_edited or is_manual:
            target_regions.append(r)
            
            # TextRegion 객체 생성 (Inpainter/Renderer용)
            target_objects.append(TextRegion(
                id=r['id'],
                text=r['text'], # 이미 수정된 텍스트가 들어있음
                confidence=r['confidence'],
                bounds=r['bounds'],
                is_inverted=r.get('is_inverted', False),
                is_manual=r.get('is_manual', False),
                style_tag=r.get('style_tag', 'body'),
                suggested_font_size=r.get('suggested_font_size', 16),
                text_color=r.get('text_color', '#333333'),
                bg_color=r.get('bg_color', '#FFFFFF'),
                font_family=r.get('font_family', settings['font_family'])
            ))
            
    # 수정된 내역이 없으면 경고 표시
    if not target_regions:
        st.info("💡 수정된(저장된) 텍스트 영역이 없습니다. 원본 이미지를 그대로 사용합니다.")
        st.session_state.processed_image = image.copy()
        final_image = image.copy()
    else:
        st.success(f"✅ 총 {len(target_regions)}개의 수정된 영역만 이미지에 반영합니다.")
        
        # 1. 배경 지우기 (수정 대상 영역만 지움)
        inpainter = create_inpainter("simple_fill")
        background = inpainter.remove_all_text_regions(image, target_objects)
        st.session_state.background_image = background
        
        # 2. 텍스트 다시 쓰기 (수정 대상 영역만 씀)
        renderer = CompositeRenderer()
        final_image = renderer.composite(
            background,
            target_objects,
            st.session_state.edited_texts
        )
        st.session_state.processed_image = final_image

    # ------------------------------------------------------------------
    # 최종 미리보기 및 다운로드 UI
    # ------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("최종 미리보기")
        st.image(
            cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB),
            caption="최종 결과 (수정된 부분만 반영됨)",
            use_container_width=True
        )
    
    with col2:
        st.subheader("내보내기 옵션")
        output_formats = settings['output_formats']
        filename = st.text_input("파일명", value=f"infographic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        st.divider()
        
        if st.button("📥 파일 생성 및 다운로드", type="primary"):
            with st.spinner("파일 생성 중..."):
                exporter = MultiFormatExporter()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    results = exporter.export_all(
                        final_image,
                        tmp_dir,
                        filename,
                        formats=[f.lower() for f in output_formats]
                    )
                    for fmt, filepath in results.items():
                        if filepath and Path(filepath).exists():
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label=f"📥 {fmt.upper()} 다운로드",
                                    data=f.read(),
                                    file_name=f"{filename}.{fmt}",
                                    mime=f"application/{fmt}" if fmt == 'pdf' else f"image/{fmt}"
                                )
        
        st.divider()
        if st.button("📋 메타데이터 다운로드"):
            builder = MetadataBuilder()
            builder.set_image_info(filename=st.session_state.uploaded_image or "image", width=image.shape[1], height=image.shape[0])
            builder.metadata['text_regions'] = regions
            builder._update_summary()
            st.download_button(label="📥 JSON 메타데이터", data=builder.to_json(), file_name=f"{filename}_metadata.json", mime="application/json")
            
    st.divider()
    if st.button("⬅️ 이전 단계"):
        st.session_state.current_step = 3
        st.rerun()

# ============================================
# 메인 앱
# ============================================
def main():
    render_header()
    settings = render_sidebar()
    
    # 단계별 렌더링
    current_step = st.session_state.current_step
    
    # 진행 상태 표시
    steps = ["1. 업로드", "2. 감지", "3. 편집", "4. 내보내기"]
    cols = st.columns(4)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 == current_step:
                st.markdown(f"**🔵 {step}**")
            elif i + 1 < current_step:
                st.markdown(f"✅ {step}")
            else:
                st.markdown(f"⚪ {step}")
    
    st.divider()
    
    # 현재 단계 렌더링
    if current_step == 1:
        render_step1_upload()
    elif current_step == 2:
        render_step2_detect()
    elif current_step == 3:
        render_step3_edit()
    elif current_step == 4:
        render_step4_export(settings)

if __name__ == "__main__":
    main()
