import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

# [NEW] 캔버스 라이브러리 임포트
from streamlit_drawable_canvas import st_canvas

# Modules
from modules import (
    TextRegion, 
    extract_text_from_crop, # 새로 만든 함수
    apply_styles_and_colors,
    CompositeRenderer,
    MultiFormatExporter,
    MetadataBuilder
)

# ... (기존 설정 코드는 유지) ...

def render_step2_detect():
    """
    Step 2: 수동 영역 지정 (Canvas Drawing)
    자동 감지 대신, 사용자가 직접 마우스로 박스를 그립니다.
    """
    st.header("Step 2: 수정 영역 지정")
    st.info("🖱️ 마우스로 수정하고 싶은 텍스트 영역을 드래그해서 박스를 그려주세요.")
    
    image = st.session_state.original_image
    if image is None: return

    # 1. 캔버스 설정
    # 이미지 위에 그림을 그릴 수 있는 컴포넌트입니다.
    # 이미지가 너무 크면 스크롤이 생기므로, 가로폭에 맞춥니다.
    
    # 캔버스에서 그린 사각형 정보를 가져옵니다.
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",  # 박스 내부 색상 (주황색 투명)
        stroke_width=2,
        stroke_color="#FF0000",              # 박스 테두리 (빨강)
        background_image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="rect",                 # 사각형 그리기 모드
        key="canvas",
        display_toolbar=True                 # 그리기 취소/삭제 툴바 표시
    )

    # 2. 그려진 박스 데이터 실시간 처리
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        # 박스가 하나라도 그려졌다면 '다음 단계' 버튼 활성화
        if len(objects) > 0:
            st.success(f"✅ 총 {len(objects)}개의 영역이 지정되었습니다.")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ 영역 초기화"):
                    st.rerun() # 캔버스 리셋
            
            with col2:
                # [핵심] 이 버튼을 누르면 그려진 박스들의 좌표로 OCR을 돌립니다.
                if st.button("📝 텍스트 추출 및 편집하기", type="primary"):
                    with st.spinner("지정된 영역의 텍스트를 읽어오는 중..."):
                        regions = []
                        for i, obj in enumerate(objects):
                            # 캔버스 좌표 (left, top, width, height)
                            x = int(obj["left"])
                            y = int(obj["top"])
                            w = int(obj["width"])
                            h = int(obj["height"])
                            
                            # 해당 좌표로 OCR 수행
                            region = extract_text_from_crop(image, x, y, w, h)
                            
                            # ID 부여 (순서대로)
                            region.id = f"manual_{i:03d}"
                            regions.append(region.to_dict())
                        
                        # 세션에 저장하고 Step 3로 이동
                        st.session_state.text_regions = regions
                        st.session_state.current_step = 3
                        st.rerun()
        else:
            st.warning("이미지 위에 마우스로 박스를 그려주세요.")

# ... (Step 3, 4는 기존 코드 그대로 사용하면 됩니다. 데이터 구조가 같기 때문입니다) ...
