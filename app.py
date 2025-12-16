import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

# [필수] 캔버스 라이브러리
from streamlit_drawable_canvas import st_canvas

# Modules
from modules import (
    TextRegion,
    extract_text_from_crop,
    apply_styles_and_colors,
    CompositeRenderer,
    MultiFormatExporter,
    MetadataBuilder,
    create_manual_region
)

# 페이지 설정
st.set_page_config(layout="wide", page_title="한글 인포그래픽 교정 도구")

def init_session_state():
    """세션 상태 초기화"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'text_regions' not in st.session_state:
        st.session_state.text_regions = []
    if 'edited_texts' not in st.session_state:
        st.session_state.edited_texts = {}
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

def draw_regions_on_image(image, regions, edited_texts):
    """미리보기용 이미지에 박스 그리기"""
    vis_image = image.copy()
    for region in regions:
        if isinstance(region, dict):
            r_id = region['id']
            bounds = region['bounds']
            text = region['text']
            is_inverted = region.get('is_inverted', False)
        else:
            r_id = region.id
            bounds = region.bounds
            text = region.text
            is_inverted = region.is_inverted

        x, y, w, h = bounds['x'], bounds['y'], bounds['width'], bounds['height']
        
        if r_id in edited_texts and edited_texts[r_id] != text:
            color = (255, 0, 255) # 수정됨 (Magenta)
            thickness = 3
        elif is_inverted:
            color = (255, 100, 0) # 역상 (Blue-ish)
            thickness = 2
        else:
            color = (0, 255, 0)   # 일반 (Green)
            thickness = 2
            
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
    return vis_image

def render_step1_upload():
    """Step 1: 이미지 업로드"""
    st.header("1. 이미지 업로드")
    
    uploaded_file = st.file_uploader("인포그래픽 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        st.session_state.original_image = image
        st.session_state.uploaded_filename = uploaded_file.name
        
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)
        
        if st.button("다음 단계로 이동", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

def render_step2_detect():
    """Step 2: 수동 영역 지정 (기본값 설정 적용: 16px, 90%, Black 폰트)"""
    st.header("Step 2: 텍스트 영역 지정")
    
    if st.session_state.original_image is None:
        st.warning("이미지를 먼저 업로드해주세요.")
        return

    original_image = st.session_state.original_image
    h_orig, w_orig = original_image.shape[:2]
    
    # [최적화] 이미지 리사이징 (화면 멈춤 방지)
    MAX_WIDTH = 800
    scale_factor = 1.0
    
    if w_orig > MAX_WIDTH:
        scale_factor = w_orig / MAX_WIDTH
        new_width = MAX_WIDTH
        new_height = int(h_orig / (w_orig / MAX_WIDTH))
        display_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        display_image = original_image
        new_width = w_orig
        new_height = h_orig

    try:
        if len(display_image.shape) == 3:
            img_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = display_image
        pil_image = Image.fromarray(img_rgb)
    except Exception as e:
        st.error(f"이미지 변환 오류: {e}")
        return

    st.info(f"🖱️ 마우스로 수정할 텍스트 영역을 박스로 그려주세요.")

    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=new_height,
            width=new_width,
            drawing_mode="rect",
            key="canvas_optimized_v2", # 키 변경으로 캔버스 리프레시 유도
            display_toolbar=True
        )
    except Exception as e:
        st.error(f"캔버스 로드 실패: {e}")
        st.stop()

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        if len(objects) > 0:
            st.success(f"✅ 총 {len(objects)}개의 영역이 지정되었습니다.")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ 영역 초기화"):
                    st.rerun()
            
            with col2:
                if st.button("📝 텍스트 추출 및 편집하기", type="primary"):
                    with st.spinner("텍스트 분석 및 기본값 적용 중..."):
                        regions = []
                        for i, obj in enumerate(objects):
                            # 좌표 복원
                            x = int(obj["left"] * scale_factor)
                            y = int(obj["top"] * scale_factor)
                            w = int(obj["width"] * scale_factor)
                            h = int(obj["height"] * scale_factor)
                            
                            x = max(0, min(x, w_orig))
                            y = max(0, min(y, h_orig))
                            w = min(w, w_orig - x)
                            h = min(h, h_orig - y)
                            
                            if w < 5 or h < 5: continue

                            # 1. 텍스트 추출
                            region = extract_text_from_crop(original_image, x, y, w, h)
                            region.id = f"manual_{i:03d}"
                            
                            # -------------------------------------------------------
                            # [핵심 변경] 요청하신 기본값 적용
                            # -------------------------------------------------------
                            region.suggested_font_size = 16                 # 기본 크기 16
                            region.width_scale = 90                         # 기본 장평 90%
                            region.font_filename = "NotoSansKR-Bold.ttf"   # 기본 폰트 Black
                            # -------------------------------------------------------
                            
                            regions.append(region.to_dict())
                        
                        st.session_state.text_regions = regions
                        st.session_state.current_step = 3
                        st.rerun()

def render_step3_edit():
    """Step 3: 텍스트 편집"""
    st.header("✏️ Step 3: 텍스트 편집")
    
    if not st.session_state.text_regions:
        st.warning("지정된 텍스트 영역이 없습니다.")
        return
    
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    
    # 폰트 로드
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    if not os.path.exists(fonts_dir):
        os.makedirs(fonts_dir)
        
    available_fonts = sorted([f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')])
    if not available_fonts:
        available_fonts = ["Default"]
        st.warning("⚠️ fonts 폴더에 폰트 파일이 없습니다.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 텍스트 영역 목록")
        for i, region in enumerate(regions):
            region_id = region['id']
            display_text = region['text'][:30] + "..." if len(region['text']) > 30 else region['text']
            
            with st.expander(f"📝 {i+1}. {display_text}", expanded=True):
                current_text = st.session_state.edited_texts.get(region_id, region['text'])
                edited = st.text_area("텍스트 내용", value=current_text, key=f"text_{region_id}_{i}", height=70)
                
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    curr_font = region.get('font_filename', available_fonts[0])
                    if curr_font not in available_fonts: curr_font = available_fonts[0]
                    selected_font = st.selectbox("폰트", options=available_fonts, index=available_fonts.index(curr_font), key=f"font_{region_id}_{i}")
                with c2:
                    curr_size = int(region.get('suggested_font_size', 14))
                    font_size = st.number_input("크기", min_value=5, max_value=200, value=curr_size, key=f"size_{region_id}_{i}")
                with c3:
                    curr_scale = int(region.get('width_scale', 80))
                    width_scale = st.number_input("장평(%)", min_value=50, max_value=200, value=curr_scale, step=5, key=f"scale_{region_id}_{i}")
                
                curr_color = region.get('text_color', '#333333')
                text_color = st.color_picker("글자색", value=curr_color, key=f"color_{region_id}_{i}")
                
                if st.button("💾 적용", key=f"save_{region_id}_{i}"):
                    st.session_state.edited_texts[region_id] = edited
                    for r in st.session_state.text_regions:
                        if r['id'] == region_id:
                            r['text'] = edited
                            r['suggested_font_size'] = font_size
                            r['text_color'] = text_color
                            r['font_filename'] = selected_font
                            r['width_scale'] = width_scale
                            break
                    st.success("적용되었습니다.")
                    st.rerun()
    
    with col2:
        st.subheader("🖼️ 편집 미리보기")
        visualized = draw_regions_on_image(image, regions, st.session_state.edited_texts)
        st.image(cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB), caption="영역 미리보기", use_container_width=True)

    st.divider()
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button("⬅️ 다시 영역 지정하기"):
            st.session_state.current_step = 2
            st.rerun()
    with c_next:
        if st.button("📤 내보내기 (Step 4)", type="primary"):
            st.session_state.current_step = 4
            st.rerun()

def render_step4_export(settings: dict):
    """Step 4: 내보내기"""
    st.header("📤 Step 4: 내보내기")
    
    if not st.session_state.text_regions:
        st.warning("데이터가 없습니다.")
        return
    
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    target_regions = regions # 수동 모드이므로 모든 영역 대상
    
    # 텍스트 객체 변환
    target_objects = []
    for r in target_regions:
        region_text = st.session_state.edited_texts.get(r['id'], r['text'])
        obj = TextRegion(
            id=r['id'],
            text=region_text,
            confidence=r['confidence'],
            bounds=r['bounds'],
            is_inverted=r.get('is_inverted', False),
            is_manual=True,
            suggested_font_size=r.get('suggested_font_size', 14),
            text_color=r.get('text_color', '#000000'),
            bg_color=r.get('bg_color', '#FFFFFF'),
            font_filename=r.get('font_filename', None),
            width_scale=r.get('width_scale', 80)
        )
        target_objects.append(obj)
        
    st.success(f"✅ 총 {len(target_objects)}개의 영역을 변환합니다.")
    
    try:
        from modules import create_inpainter
        inpainter = create_inpainter("simple_fill")
        background = inpainter.remove_all_text_regions(image, target_objects)
        
        renderer = CompositeRenderer()
        final_image = renderer.composite(
            background,
            target_objects,
            st.session_state.edited_texts
        )
        
        st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), caption="최종 결과물", use_container_width=True)
        
        filename = f"infographic_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        is_success, buffer = cv2.imencode(".png", final_image)
        
        if is_success:
            st.download_button(
                label="📥 최종 이미지 다운로드",
                data=buffer.tobytes(),
                file_name=filename,
                mime="image/png"
            )
            
    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")

    if st.button("⬅️ 편집 화면으로 돌아가기"):
        st.session_state.current_step = 3
        st.rerun()

def main():
    init_session_state()
    st.sidebar.title("⚙️ 설정")
    settings = {
        'font_family': st.sidebar.selectbox("기본 폰트", ["Noto Sans KR", "NanumGothic"]),
    }
    
    step = st.session_state.current_step
    
    if step == 1:
        render_step1_upload()
    elif step == 2:
        render_step2_detect()
    elif step == 3:
        render_step3_edit()
    elif step == 4:
        render_step4_export(settings)

if __name__ == "__main__":
    main()
