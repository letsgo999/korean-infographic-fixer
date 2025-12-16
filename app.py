import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

# [NEW] 캔버스 라이브러리 (필수)
from streamlit_drawable_canvas import st_canvas

# Modules
from modules import (
    TextRegion,
    extract_text_from_crop, # 수동 추출 함수
    apply_styles_and_colors,
    CompositeRenderer,
    MultiFormatExporter,
    MetadataBuilder,
    create_manual_region
)
# 인페인터는 Step 4에서 직접 호출

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
        # 딕셔너리 호환 처리
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
        
        # 색상 설정 (수정됨: 마젠타, 기본: 초록/파랑)
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
        
        # ID 표시
        label = r_id
        cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return vis_image

def render_step1_upload():
    """Step 1: 이미지 업로드"""
    st.header("1. 이미지 업로드")
    
    uploaded_file = st.file_uploader("인포그래픽 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 이미지 로드
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
    """Step 2: 수동 영역 지정 (Canvas Drawing)"""
    st.header("Step 2: 텍스트 영역 지정")
    
    if st.session_state.original_image is None:
        st.warning("이미지를 먼저 업로드해주세요.")
        return

    image = st.session_state.original_image
    
    # 이미지 변환 (BGR -> RGB)
    try:
        if len(image.shape) == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        pil_image = Image.fromarray(img_rgb)
    except Exception as e:
        st.error(f"이미지 변환 오류: {e}")
        return

    st.info("🖱️ 수정하고 싶은 텍스트 영역을 마우스로 드래그하여 박스를 그려주세요.")

    # 캔버스 (수동 영역 지정)
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode="rect",
            key="canvas_manual",
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
                # [핵심] 수동 지정 영역 OCR 수행 및 이동
                if st.button("📝 텍스트 추출 및 편집하기", type="primary"):
                    with st.spinner("지정된 영역의 텍스트를 읽어오는 중..."):
                        regions = []
                        for i, obj in enumerate(objects):
                            # 좌표 보정
                            x = int(max(0, obj["left"]))
                            y = int(max(0, obj["top"]))
                            w = int(min(image.shape[1] - x, obj["width"]))
                            h = int(min(image.shape[0] - y, obj["height"]))
                            
                            if w < 5 or h < 5: continue

                            # OCR 수행
                            region = extract_text_from_crop(image, x, y, w, h)
                            
                            # ID 부여
                            region.id = f"manual_{i:03d}"
                            
                            # 기본 스타일 설정 (이전 대화에서 정한 값)
                            region.suggested_font_size = 16
                            region.width_scale = 90
                            
                            regions.append(region.to_dict())
                        
                        st.session_state.text_regions = regions
                        st.session_state.current_step = 3
                        st.rerun()

def render_step3_edit():
    """Step 3: 텍스트 편집 (폰트/장평 설정 포함)"""
    st.header("✏️ Step 3: 텍스트 편집")
    
    if not st.session_state.text_regions:
        st.warning("지정된 텍스트 영역이 없습니다.")
        return
    
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    
    # 폰트 폴더 스캔
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
        
        # 리스트 출력
        for i, region in enumerate(regions):
            region_id = region['id']
            # 긴 텍스트 말줄임
            display_text = region['text'][:30] + "..." if len(region['text']) > 30 else region['text']
            
            with st.expander(f"📝 {i+1}. {display_text}", expanded=False):
                # 텍스트 수정
                current_text = st.session_state.edited_texts.get(region_id, region['text'])
                edited = st.text_area("텍스트 내용", value=current_text, key=f"text_{region_id}_{i}", height=70)
                
                # 스타일 설정 (3단)
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    # 폰트 선택
                    curr_font = region.get('font_filename', available_fonts[0])
                    if curr_font not in available_fonts: curr_font = available_fonts[0]
                    selected_font = st.selectbox("폰트", options=available_fonts, index=available_fonts.index(curr_font), key=f"font_{region_id}_{i}")
                with c2:
                    # 크기
                    curr_size = int(region.get('suggested_font_size', 14))
                    font_size = st.number_input("크기", min_value=5, max_value=200, value=curr_size, key=f"size_{region_id}_{i}")
                with c3:
                    # 장평
                    curr_scale = int(region.get('width_scale', 80))
                    width_scale = st.number_input("장평(%)", min_value=50, max_value=200, value=curr_scale, step=5, key=f"scale_{region_id}_{i}")
                
                # 색상
                curr_color = region.get('text_color', '#333333')
                text_color = st.color_picker("글자색", value=curr_color, key=f"color_{region_id}_{i}")
                
                if st.button("💾 적용", key=f"save_{region_id}_{i}"):
                    # 세션 및 원본 데이터 업데이트
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
    """Step 4: 내보내기 (수정된 영역만 반영)"""
    st.header("📤 Step 4: 내보내기")
    
    if not st.session_state.text_regions:
        st.warning("데이터가 없습니다.")
        return
    
    image = st.session_state.original_image
    regions = st.session_state.text_regions
    
    # 수정된 내역이 있는 것 + 수동 지정한 모든 영역을 대상으로 함
    # (Step 2에서 수동으로 지정했다는 것 자체가 수정을 의도한 것이므로 모두 처리)
    target_regions = regions
    
    if not target_regions:
        st.info("수정할 영역이 없습니다.")
        return

    # TextRegion 객체 리스트로 변환 (Inpainter/Renderer 호환용)
    target_objects = []
    for r in target_regions:
        # 최신 수정 사항 반영 확인
        region_text = st.session_state.edited_texts.get(r['id'], r['text'])
        
        obj = TextRegion(
            id=r['id'],
            text=region_text,
            confidence=r['confidence'],
            bounds=r['bounds'],
            is_inverted=r.get('is_inverted', False),
            is_manual=r.get('is_manual', True),
            suggested_font_size=r.get('suggested_font_size', 16),
            text_color=r.get('text_color', '#000000'),
            bg_color=r.get('bg_color', '#FFFFFF'),
            font_filename=r.get('font_filename', None),
            width_scale=r.get('width_scale', 90)
        )
        target_objects.append(obj)
        
    st.success(f"✅ 총 {len(target_objects)}개의 영역을 변환합니다.")
    
    try:
        # 1. 배경 지우기
        from modules import create_inpainter
        inpainter = create_inpainter("simple_fill")
        background = inpainter.remove_all_text_regions(image, target_objects)
        
        # 2. 텍스트 쓰기
        renderer = CompositeRenderer()
        final_image = renderer.composite(
            background,
            target_objects,
            st.session_state.edited_texts
        )
        
        st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), caption="최종 결과물", use_container_width=True)
        
        # 다운로드
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
        st.exception(e)

    if st.button("⬅️ 편집 화면으로 돌아가기"):
        st.session_state.current_step = 3
        st.rerun()

def main():
    init_session_state()
    
    # 사이드바 설정
    st.sidebar.title("⚙️ 설정")
    settings = {
        'font_family': st.sidebar.selectbox("기본 폰트", ["Noto Sans KR", "NanumGothic"]),
        'output_formats': st.sidebar.multiselect("출력 포맷", ["PNG", "JPG", "PDF"], default=["PNG"])
    }
    
    # 단계별 라우팅
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
