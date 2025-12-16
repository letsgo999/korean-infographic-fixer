"""
OCR Engine Module
수동 영역 지정(Manual Crop)을 위한 단순화된 엔진
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict
from dataclasses import dataclass, asdict
import re

@dataclass
class TextRegion:
    id: str
    text: str
    confidence: float
    bounds: Dict[str, int]
    is_inverted: bool = False
    style_tag: str = "body"
    suggested_font_size: int = 14  
    text_color: str = "#000000"
    bg_color: str = "#FFFFFF"
    font_family: str = "Noto Sans KR"
    font_filename: str = None
    width_scale: int = 80
    font_weight: str = "Regular"
    is_manual: bool = True # 기본적으로 수동 모드임
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ... (기존 OCREngine 등 클래스는 유지하되, 아래 함수가 핵심입니다)

def extract_text_from_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> TextRegion:
    """
    [NEW] 사용자가 드래그한 영역(Crop)만 콕 집어서 OCR 수행
    """
    # 1. 좌표 유효성 검사 및 크롭
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w)); y = max(0, min(y, img_h))
    w = max(1, min(w, img_w - x)); h = max(1, min(h, img_h - y))
    
    roi = image[y:y+h, x:x+w]
    
    # 2. OCR 수행 (단일 블록 모드 PSM 6 or 7)
    if len(roi.shape) == 3: roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else: roi_rgb = roi
    
    pil_roi = Image.fromarray(roi_rgb)
    
    # 한국어+영어, 단일 라인으로 가정
    config = r'--oem 3 --psm 6' 
    text = pytesseract.image_to_string(pil_roi, lang='kor+eng', config=config).strip()
    
    # 3. 텍스트가 없으면 빈 문자열 반환
    if not text:
        text = ""

    # 4. TextRegion 객체 생성
    return TextRegion(
        id=f"manual_{x}_{y}",
        text=text,
        confidence=100.0, # 수동 지정이므로 신뢰도 100
        bounds={'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
        is_manual=True
    )

# (호환성을 위해 기존 함수들은 빈 껍데기나 기본 로직으로 남겨두셔도 됩니다)
def run_enhanced_ocr(image): return {} 
def group_regions_by_lines(regions): return regions
def create_manual_region(**kwargs): pass
