"""
OCR Engine Module
수동 영역 지정(Manual Crop) + 자동 OCR 하이브리드 엔진
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
    is_manual: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


def extract_text_from_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> TextRegion:
    """
    사용자가 드래그한 영역(Crop)만 콕 집어서 OCR 수행
    """
    # 1. 좌표 유효성 검사 및 크롭
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w)); y = max(0, min(y, img_h))
    w = max(1, min(w, img_w - x)); h = max(1, min(h, img_h - y))
    
    roi = image[y:y+h, x:x+w]
    
    # 2. OCR 수행 (단일 블록 모드 PSM 6 or 7)
    if len(roi.shape) == 3: 
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else: 
        roi_rgb = roi
    
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
        confidence=100.0,
        bounds={'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
        is_manual=True
    )


def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """
    [수정됨] 전체 이미지에서 텍스트 영역 자동 감지
    
    Returns:
        dict: {
            'normal_regions': List[TextRegion],
            'inverted_regions': List[TextRegion],
            'all_regions': List[TextRegion]
        }
    """
    normal_regions = []
    inverted_regions = []
    
    try:
        # 이미지 전처리
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            gray = image
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(rgb)
        
        # Tesseract OCR 실행 (data output)
        config = r'--oem 3 --psm 3'
        data = pytesseract.image_to_data(pil_image, lang='kor+eng', config=config, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        region_id = 0
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            
            # 빈 텍스트나 낮은 신뢰도 스킵
            if not text or conf < 30:
                continue
            
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # 너무 작은 영역 스킵
            if w < 10 or h < 5:
                continue
            
            region = TextRegion(
                id=f"ocr_{region_id}",
                text=text,
                confidence=float(conf),
                bounds={'x': x, 'y': y, 'width': w, 'height': h},
                is_inverted=False,
                is_manual=False
            )
            normal_regions.append(region)
            region_id += 1
        
        # 역상 이미지에서도 OCR 시도 (어두운 배경의 밝은 텍스트)
        inverted_gray = cv2.bitwise_not(gray)
        inverted_rgb = cv2.cvtColor(cv2.cvtColor(inverted_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        pil_inverted = Image.fromarray(inverted_rgb)
        
        inv_data = pytesseract.image_to_data(pil_inverted, lang='kor+eng', config=config, output_type=pytesseract.Output.DICT)
        
        n_inv_boxes = len(inv_data['text'])
        for i in range(n_inv_boxes):
            text = inv_data['text'][i].strip()
            conf = int(inv_data['conf'][i]) if inv_data['conf'][i] != '-1' else 0
            
            if not text or conf < 30:
                continue
            
            x, y, w, h = inv_data['left'][i], inv_data['top'][i], inv_data['width'][i], inv_data['height'][i]
            
            if w < 10 or h < 5:
                continue
            
            # 이미 normal_regions에 비슷한 위치의 텍스트가 있으면 스킵
            is_duplicate = False
            for nr in normal_regions:
                nb = nr.bounds
                if abs(nb['x'] - x) < 20 and abs(nb['y'] - y) < 20:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            region = TextRegion(
                id=f"inv_{region_id}",
                text=text,
                confidence=float(conf),
                bounds={'x': x, 'y': y, 'width': w, 'height': h},
                is_inverted=True,
                is_manual=False
            )
            inverted_regions.append(region)
            region_id += 1
            
    except Exception as e:
        print(f"OCR Error: {e}")
    
    all_regions = normal_regions + inverted_regions
    
    return {
        'normal_regions': normal_regions,
        'inverted_regions': inverted_regions,
        'all_regions': all_regions
    }


def group_regions_by_lines(regions: List[TextRegion], y_threshold: int = 15) -> List[TextRegion]:
    """
    [수정됨] Y좌표가 비슷한 영역들을 같은 라인으로 그룹핑
    """
    if not regions:
        return []
    
    # TextRegion 객체 또는 dict 모두 처리
    def get_bounds(r):
        if isinstance(r, dict):
            return r['bounds']
        return r.bounds
    
    def get_text(r):
        if isinstance(r, dict):
            return r['text']
        return r.text
    
    def get_id(r):
        if isinstance(r, dict):
            return r['id']
        return r.id
    
    # Y좌표로 정렬
    sorted_regions = sorted(regions, key=lambda r: get_bounds(r)['y'])
    
    grouped = []
    current_line = [sorted_regions[0]]
    current_y = get_bounds(sorted_regions[0])['y']
    
    for region in sorted_regions[1:]:
        region_y = get_bounds(region)['y']
        
        if abs(region_y - current_y) <= y_threshold:
            current_line.append(region)
        else:
            # 현재 라인 병합
            if len(current_line) > 1:
                merged = merge_line_regions(current_line)
                grouped.append(merged)
            else:
                grouped.append(current_line[0])
            
            current_line = [region]
            current_y = region_y
    
    # 마지막 라인 처리
    if len(current_line) > 1:
        merged = merge_line_regions(current_line)
        grouped.append(merged)
    else:
        grouped.append(current_line[0])
    
    return grouped


def merge_line_regions(regions: List) -> TextRegion:
    """같은 라인의 영역들을 하나로 병합"""
    
    def get_bounds(r):
        if isinstance(r, dict):
            return r['bounds']
        return r.bounds
    
    def get_text(r):
        if isinstance(r, dict):
            return r['text']
        return r.text
    
    def get_confidence(r):
        if isinstance(r, dict):
            return r['confidence']
        return r.confidence
    
    # X좌표로 정렬
    sorted_regions = sorted(regions, key=lambda r: get_bounds(r)['x'])
    
    # 텍스트 병합
    merged_text = ' '.join(get_text(r) for r in sorted_regions)
    
    # 바운딩 박스 계산
    all_bounds = [get_bounds(r) for r in sorted_regions]
    min_x = min(b['x'] for b in all_bounds)
    min_y = min(b['y'] for b in all_bounds)
    max_x = max(b['x'] + b['width'] for b in all_bounds)
    max_y = max(b['y'] + b['height'] for b in all_bounds)
    
    # 평균 신뢰도
    avg_conf = sum(get_confidence(r) for r in sorted_regions) / len(sorted_regions)
    
    return TextRegion(
        id=f"merged_{min_x}_{min_y}",
        text=merged_text,
        confidence=avg_conf,
        bounds={
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        },
        is_inverted=False,
        is_manual=False
    )


def create_manual_region(x: int, y: int, width: int, height: int, text: str = "") -> TextRegion:
    """
    [수정됨] 수동으로 텍스트 영역 생성
    """
    return TextRegion(
        id=f"manual_{x}_{y}_{width}_{height}",
        text=text,
        confidence=100.0,
        bounds={'x': x, 'y': y, 'width': width, 'height': height},
        is_manual=True,
        suggested_font_size=14,
        width_scale=80
    )
