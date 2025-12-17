import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
import uuid

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
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = "canvas_v1"
    # [NEW] 캔버스 스크롤 위치 저장용
    if 'scroll_y' not in st.session_state:
        st.session_state.scroll_y = 0

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
            color = (255, 0, 255) 
            thickness = 3
        elif is_inverted:
            color = (255, 100, 0) 
            thickness = 2
        else:
            color = (0, 255, 0)   
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
        # 스크롤 위치 초기화
        st.session_state.scroll_y = 0
        
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)
        
        if st.button("다음 단계로 이동", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

def render_step2_detect():
    """Step 2: 수동 영역 지정 (스크롤 뷰어 방식 적용)"""
    st.header("Step 2: 텍스트 영역 지정")
    
    if st.session_state.original_image is None:
        st.warning("이미지를 먼저 업로드해주세요.")
        return

    original_image = st.session_state.original_image
    h_orig, w_orig = original_image.shape[:2]
    
    # ---------------------------------------------------------
    # [핵심] 스크롤 뷰어 설정
    # 전체 이미지가 너무 크므로, 한 번에 1000px 높이만 보여줍니다.
    # ---------------------------------------------------------
    VIEWPORT_HEIGHT = 1000  # 화면에 보여줄 높이 (적당한 크기)
    
    # 가로폭 리사이징 (캔버스 폭 맞춤, 최대 800px)
    CANVAS_WIDTH = 800
    scale_factor = 1.0
    
    if w_orig > CANVAS_WIDTH:
        scale_factor = w_orig / CANVAS_WIDTH
        resized_w = CANVAS_WIDTH
        resized_h_total = int(h_orig / scale_factor)
        # 전체를 리사이징하면 느리므로, 크롭 후 리사이징할 비율만 계산해둠
    else:
        resized_w = w_orig
        resized_h_total = h_orig

    # 스크롤 슬라이더 (이미지가 뷰포트보다 클 때만 표시)
    current_scroll = st.session_state.scroll_y
    
    if h_orig > VIEWPORT_HEIGHT:
        st.info("💡 이미지가 길어서 **스크롤** 기능을 제공합니다. 슬라이더를 움직여 작업할 위치를 맞추세요.")
        # 슬라이더: 0부터 (전체높이 - 뷰포트높이)까지
        max_scroll = h_orig - VIEWPORT_HEIGHT
        
        # 슬라이더 값을 세션에 저장하여 리로드 되어도 유지
        scroll_val = st.slider(
            "↕️ 이미지 스크롤 (위/아래 이동)", 
            min_value=0, 
            max_value=max_scroll, 
            value=st.session_state.scroll_y,
            step=50,
            key="slider_scroll"
        )
        # 슬라이더 값이 바뀌면 세션 업데이트
        st.session_state.scroll_y = scroll_val
        current_scroll = scroll_val
    else:
        current_scroll = 0

    # 1. 현재 스크롤 위치에 맞춰 원본에서 잘라내기 (Crop)
    # 보여줄 높이는 뷰포트 높이 또는 남은 이미지 높이 중 작은 것
    crop_h = min(VIEWPORT_HEIGHT, h_orig - current_scroll)
    
    crop_img = original_image[current_scroll : current_scroll + crop_h, :]
    
    # 2. 잘라낸 조각을 화면 표시용으로 리사이징
    h_crop, w_crop = crop_img.shape[:2]
    
    if w_crop > CANVAS_WIDTH:
        # 가로폭을 800으로 맞춤
        disp_scale = CANVAS_WIDTH / w_crop
        disp_w = CANVAS_WIDTH
        disp_h = int(h_crop * disp_scale)
        display_img = cv2.resize(crop_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_scale = 1.0
        display_img = crop_img
        disp_w = w_crop
        disp_h = h_crop

    # 3. BGR -> RGB 변환
    try:
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
    except Exception as e:
        st.error(f"이미지 처리 오류: {e}")
        return

    st.write(f"📍 현재 위치: Y={current_scroll}px 부터 작업 중")

    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 캔버스 지우기"):
            st.session_state.canvas_key = f"canvas_{uuid.uuid4()}" 
            st.rerun()

    # 4. 캔버스 호출 (작아진 이미지 조각만 올림 -> 가벼움!)
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=st.session_state.canvas_key,
            display_toolbar=True
        )
    except Exception as e:
        st.error(f"캔버스 로딩 실패: {e}")
        st.stop()

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        if len(objects) > 0:
            st.success(f"✅ 현재 화면에서 {len(objects)}개의 영역을 지정했습니다.")
            
            # 주의 문구
            st.caption("⚠️ **주의:** '텍스트 추출' 버튼을 누르면 **현재 화면에 보이는 박스들만** 저장됩니다. 긴 이미지는 한 번에 한 구간씩 작업하거나, 여러 번 나누어 진행해주세요.")
            
            if st.button("📝 선택 영역 텍스트 추출하기 (Step 3)", type="primary"):
                with st.spinner("좌표 계산 및 텍스트 추출 중..."):
                    new_regions = []
                    
                    for i, obj in enumerate(objects):
                        # 1. 캔버스 좌표 -> 크롭 이미지 좌표 (리사이징 복원)
                        x_crop = int(obj["left"] / disp_scale)
                        y_crop = int(obj["top"] / disp_scale)
                        w_crop = int(obj["width"] / disp_scale)
                        h_crop = int(obj["height"] / disp_scale)
                        
                        # 2. 크롭 이미지 좌표 -> 전체 원본 좌표 (스크롤 오프셋 더하기)
                        x_real = x_crop
                        y_real = y_crop + current_scroll # [핵심] 스크롤 위치만큼 더해줌
                        w_real = w_crop
                        h_real = h_crop
                        
                        # 유효성 검사
                        x_real = max(0, min(x_real, w_orig))
                        y_real = max(0, min(y_real, h_orig))
                        w_real = min(w_real, w_orig - x_real)
                        h_real = min(h_real, h_orig - y_real)
                        
                        if w_real < 5 or h_real < 5: continue

                        # 3. 텍스트 추출 (원본 전체 이미지에서)
                        region = extract_text_from_crop(original_image, x_real, y_real, w_real, h_real)
                        
                        # ID 생성 (기존 목록이 있으면 이어서 번호 부여)
                        start_idx = len(st.session_state.text_regions)
                        region.id = f"manual_{start_idx + i:03d}"
                        
                        # 기본값
                        region.suggested_font_size = 16
                        region.width_scale = 90
                        region.font_filename = "NotoSansKR-Black.ttf"
                        
                        new_regions.append(region.to_dict())
                    
                    if not new_regions:
                        st.warning("유효한 영역이 없습니다.")
                    else:
                        # [중요] 기존에 작업한 내용에 '추가'할지, '덮어쓸지' 결정
                        # 여기서는 단순하게 매번 덮어쓰거나 추가하는 방식 중
                        # 사용자가 혼동하지 않게 '덮어쓰기(새로 시작)'로 처리하고
                        # 여러 구간 작업을 원하면 아래 로직을 'append'로 바꾸면 됩니다.
                        # 현재는 깔끔하게 이번에 선택한 것만 편집하도록 합니다.
                        st.session_state.text_regions = new_regions
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
                    try:
                        idx = available_fonts.index(curr_font)
                    except ValueError:
                        idx = 0
                    selected_font = st.selectbox("폰트", options=available_fonts, index=idx, key=f"font_{region_id}_{i}")
                with c2:
                    curr_size = int(region.get('suggested_font_size', 16))
                    font_size = st.number_input("크기", min_value=5, max_value=200, value=curr_size, key=f"size_{region_id}_{i}")
                with c3:
                    curr_scale = int(region.get('width_scale', 90))
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
    target_regions = regions
    
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
            suggested_font_size=r.get('suggested_font_size', 16),
            text_color=r.get('text_color', '#000000'),
            bg_color=r.get('bg_color', '#FFFFFF'),
            font_filename=r.get('font_filename', None),
            width_scale=r.get('width_scale', 90)
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
