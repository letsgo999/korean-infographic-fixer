import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import uuid
import base64
from datetime import datetime

# [필수] 캔버스 라이브러리
from streamlit_drawable_canvas import st_canvas

# Modules
from modules import (
    TextRegion,
    extract_text_from_crop,
    apply_styles_and_colors,
    CompositeRenderer,
    MultiFormatExporter,
    MetadataBuilder
)

# 페이지 설정
st.set_page_config(layout="wide", page_title="한글 인포그래픽 교정 도구")

def init_session_state():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'text_regions' not in st.session_state:
        st.session_state.text_regions = []
    if 'edited_texts' not in st.session_state:
        st.session_state.edited_texts = {}
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = "canvas_v1"
    if 'scroll_y' not in st.session_state:
        st.session_state.scroll_y = 0

def draw_regions_on_image(image, regions, edited_texts):
    vis_image = image.copy()
    for region in regions:
        if isinstance(region, dict):
            r_id = region['id']; bounds = region['bounds']; text = region['text']; is_inverted = region.get('is_inverted', False)
        else:
            r_id = region.id; bounds = region.bounds; text = region.text; is_inverted = region.is_inverted
        x, y, w, h = bounds['x'], bounds['y'], bounds['width'], bounds['height']
        if r_id in edited_texts and edited_texts[r_id] != text: color = (255, 0, 255); thickness = 3
        elif is_inverted: color = (255, 100, 0); thickness = 2
        else: color = (0, 255, 0); thickness = 2
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
    return vis_image

def render_step1_upload():
    st.header("1. 이미지 업로드")
    uploaded_file = st.file_uploader("인포그래픽 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        st.session_state.original_image = image
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.scroll_y = 0
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)
        if st.button("다음 단계로 이동", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

def render_step2_detect():
    st.header("Step 2: 텍스트 영역 지정")
    if st.session_state.original_image is None:
        st.warning("이미지를 먼저 업로드해주세요."); return

    original_image = st.session_state.original_image
    h_orig, w_orig = original_image.shape[:2]
    
    # 뷰포트 설정
    VIEWPORT_HEIGHT = 1000
    CANVAS_WIDTH = 700
    
    if w_orig > CANVAS_WIDTH:
        scale_factor = w_orig / CANVAS_WIDTH
    else:
        scale_factor = 1.0

    current_scroll = st.session_state.scroll_y
    if h_orig > VIEWPORT_HEIGHT:
        st.info("💡 이미지가 길어서 부분적으로 표시합니다. 슬라이더로 이동하세요.")
        max_scroll = h_orig - VIEWPORT_HEIGHT
        current_scroll = st.slider("↕️ 작업 위치 이동", 0, max_scroll, st.session_state.scroll_y, step=100)
        st.session_state.scroll_y = current_scroll
    
    # 이미지 자르기
    crop_h = min(VIEWPORT_HEIGHT, h_orig - current_scroll)
    crop_img = original_image[current_scroll : current_scroll + crop_h, :]
    
    # 리사이징
    h_crop, w_crop = crop_img.shape[:2]
    disp_w = int(w_crop / scale_factor)
    disp_h = int(h_crop / scale_factor)
    display_img = cv2.resize(crop_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------
    # [최후의 수단] 직접 Base64 변환 (버전 호환성 문제 100% 회피)
    # Streamlit 함수를 거치지 않고, 우리가 직접 문자열을 만듭니다.
    # -------------------------------------------------------------
    try:
        # 1. BGR -> RGB
        if len(display_img.shape) == 3:
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = display_img
        
        pil_img = Image.fromarray(img_rgb)
        
        # 2. 메모리에 JPEG로 저장 후 문자열(Base64)로 변환
        with io.BytesIO() as buffer:
            pil_img.save(buffer, format="JPEG", quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            # 캔버스에 전달할 최종 문자열 URL
            bg_image_url = f"data:image/jpeg;base64,{img_str}"
            
    except Exception as e:
        st.error(f"이미지 변환 중 오류 발생: {e}")
        return

    st.caption(f"📍 현재 작업 위치: {current_scroll}px ~ {current_scroll + crop_h}px")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 캔버스 리셋"):
            st.session_state.canvas_key = f"canvas_{uuid.uuid4()}"
            st.rerun()

    # 캔버스 호출 (이미지 객체 대신 '문자열'을 전달 -> 에러 원천 차단)
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=bg_image_url,  # <--- [핵심] 문자열 전달
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=st.session_state.canvas_key,
            display_toolbar=True
        )
    except Exception as e:
        st.error(f"캔버스 로드 실패: {e}")
        st.stop()

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            st.success(f"✅ 선택된 영역: {len(objects)}개")
            
            if st.button("📝 텍스트 추출 및 편집하기 (Step 3)", type="primary"):
                with st.spinner("추출 중..."):
                    regions = []
                    for i, obj in enumerate(objects):
                        x_view = obj["left"] * scale_factor
                        y_view = obj["top"] * scale_factor
                        w_view = obj["width"] * scale_factor
                        h_view = obj["height"] * scale_factor
                        
                        x_real = int(x_view)
                        y_real = int(y_view + current_scroll)
                        w_real = int(w_view)
                        h_real = int(h_view)
                        
                        x_real = max(0, min(x_real, w_orig))
                        y_real = max(0, min(y_real, h_orig))
                        w_real = min(w_real, w_orig - x_real)
                        h_real = min(h_real, h_orig - y_real)
                        
                        if w_real < 5 or h_real < 5: continue
                        
                        region = extract_text_from_crop(original_image, x_real, y_real, w_real, h_real)
                        region.id = f"manual_{i:03d}"
                        region.suggested_font_size = 16
                        region.width_scale = 90
                        region.font_filename = "NotoSansKR-Black.ttf"
                        regions.append(region.to_dict())
                    
                    st.session_state.text_regions = regions
                    st.session_state.current_step = 3
                    st.rerun()

def render_step3_edit():
    st.header("✏️ Step 3: 텍스트 편집")
    if not st.session_state.text_regions: st.warning("데이터 없음"); return
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    if not os.path.exists(fonts_dir): os.makedirs(fonts_dir)
    available_fonts = sorted([f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')])
    if not available_fonts: available_fonts = ["Default"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("목록")
        for i, region in enumerate(regions):
            region_id = region['id']
            display_text = region['text'][:30]
            with st.expander(f"{i+1}. {display_text}", expanded=True):
                edited = st.text_area("내용", value=st.session_state.edited_texts.get(region_id, region['text']), key=f"t_{i}")
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1: 
                    curr_font = region.get('font_filename', available_fonts[0])
                    try: idx = available_fonts.index(curr_font)
                    except: idx = 0
                    font_sel = st.selectbox("폰트", available_fonts, index=idx, key=f"f_{i}")
                with c2: size_sel = st.number_input("크기", value=int(region.get('suggested_font_size', 16)), key=f"s_{i}")
                with c3: scale_sel = st.number_input("장평", value=int(region.get('width_scale', 90)), key=f"w_{i}")
                color_sel = st.color_picker("색상", value=region.get('text_color', '#000000'), key=f"c_{i}")
                
                if st.button("적용", key=f"b_{i}"):
                    st.session_state.edited_texts[region_id] = edited
                    for r in st.session_state.text_regions:
                        if r['id'] == region_id:
                            r['text'] = edited; r['suggested_font_size'] = size_sel
                            r['text_color'] = color_sel; r['font_filename'] = font_sel; r['width_scale'] = scale_sel
                    st.success("저장됨"); st.rerun()
    with col2:
        st.subheader("미리보기")
        visualized = draw_regions_on_image(image, regions, st.session_state.edited_texts)
        st.image(cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1: 
        if st.button("⬅️ 재지정"): st.session_state.current_step = 2; st.rerun()
    with c2:
        if st.button("📤 내보내기", type="primary"): st.session_state.current_step = 4; st.rerun()

def render_step4_export(settings):
    st.header("📤 Step 4: 결과물 생성")
    if not st.session_state.text_regions: return
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    target_objects = []
    for r in regions:
        region_text = st.session_state.edited_texts.get(r['id'], r['text'])
        obj = TextRegion(id=r['id'], text=region_text, confidence=r['confidence'], bounds=r['bounds'], is_inverted=r.get('is_inverted', False), is_manual=True, suggested_font_size=r.get('suggested_font_size', 16), text_color=r.get('text_color', '#000000'), bg_color=r.get('bg_color', '#FFFFFF'), font_filename=r.get('font_filename', None), width_scale=r.get('width_scale', 90))
        target_objects.append(obj)
    try:
        from modules import create_inpainter
        inpainter = create_inpainter("simple_fill")
        background = inpainter.remove_all_text_regions(image, target_objects)
        renderer = CompositeRenderer()
        final_image = renderer.composite(background, target_objects, st.session_state.edited_texts)
        st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), caption="완성본", use_container_width=True)
        is_success, buffer = cv2.imencode(".png", final_image)
        if is_success:
            st.download_button("다운로드", data=buffer.tobytes(), file_name=f"fixed_{datetime.now().strftime('%H%M%S')}.png", mime="image/png")
    except Exception as e: st.error(f"오류: {e}")
    if st.button("처음으로"): st.session_state.current_step = 1; st.rerun()

def main():
    init_session_state()
    step = st.session_state.current_step
    if step == 1: render_step1_upload()
    elif step == 2: render_step2_detect()
    elif step == 3: render_step3_edit()
    elif step == 4: render_step4_export({})

if __name__ == "__main__":
    main()
