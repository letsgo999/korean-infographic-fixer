"""
OCR Engine Module
텍스트 추출 및 좌표 인식을 담당하는 핵심 모듈 (Row Force Merge - Rollback Version)
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
    suggested_font_size: int = 16
    text_color: str = "#000000"
    bg_color: str = "#FFFFFF"
    font_family: str = "Noto Sans KR"
    font_weight: str = "Regular"
    is_manual: bool = False
    block_num: int = 0
    line_num: int = 0
    word_count: int = 1
    
    def to_dict(self) -> Dict:
        return asdict(self)

class OCREngine:
    def __init__(self, lang: str = "kor+eng", min_confidence: int = 50):
        self.lang = lang
        self.min_confidence = min_confidence
        
    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        # PSM 6: 단일 블록으로 가정하여 파편화 방지
        custom_config = r'--oem 3 --psm 6'
        
        try:
            ocr_data = pytesseract.image_to_data(
                pil_image, lang=self.lang, config=custom_config, output_type=pytesseract.Output.DICT
            )
        except:
            return []
        
        regions = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            # 노이즈 필터 적용
            if conf >= self.min_confidence and is_valid_text(text):
                region = TextRegion(
                    id=f"raw_{len(regions)}",
                    text=text,
                    confidence=conf,
                    bounds={
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    },
                    is_inverted=False
                )
                regions.append(region)
        return regions
    
    def extract_from_inverted_region(self, image: np.ndarray, region_bounds: Dict[str, int], padding: int = 5) -> List[TextRegion]:
        x, y = region_bounds['x'], region_bounds['y']
        w, h = region_bounds['width'], region_bounds['height']
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2].copy()
        inverted_roi = cv2.bitwise_not(roi)
        
        if len(inverted_roi.shape) == 3:
            inverted_rgb = cv2.cvtColor(inverted_roi, cv2.COLOR_BGR2RGB)
        else:
            inverted_rgb = inverted_roi
            
        pil_roi = Image.fromarray(inverted_rgb)
        custom_config = r'--oem 3 --psm 7'
        
        try:
            ocr_data = pytesseract.image_to_data(
                pil_roi, lang=self.lang, config=custom_config, output_type=pytesseract.Output.DICT
            )
        except:
            return []
        
        regions = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            if conf >= self.min_confidence and is_valid_text(text):
                region = TextRegion(
                    id=f"inv_raw_{len(regions)}",
                    text=text,
                    confidence=conf,
                    bounds={
                        'x': x1 + ocr_data['left'][i],
                        'y': y1 + ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    },
                    is_inverted=True
                )
                regions.append(region)
        return regions

class InvertedRegionDetector:
    def __init__(self, dark_threshold=150, min_area=1000, min_width=40, min_height=15):
        self.dark_threshold = dark_threshold
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        
    def detect(self, image: np.ndarray) -> List[Dict[str, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        dark_mask = cv2.inRange(gray, 0, self.dark_threshold)
        combined_mask = cv2.bitwise_or(orange_mask, dark_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if w >= self.min_width and h >= self.min_height and area >= self.min_area:
                regions.append({'x': x, 'y': y, 'width': w, 'height': h})
        return regions

def is_valid_text(text: str) -> bool:
    """노이즈 필터: 특수문자만 있거나, 의미 없는 1글자 영문 등 제거"""
    t = text.strip()
    if not t: return False
    if not re.search(r'[가-힣a-zA-Z0-9]', t): return False
    if len(t) == 1 and re.match(r'[a-zA-Z\W]', t): return False
    return True

def merge_regions_by_row(regions: List[TextRegion]) -> List[TextRegion]:
    """
    [복구된 핵심 기능]
    Y좌표가 비슷하면 거리나 내용에 상관없이 무조건 하나의 행으로 강제 병합합니다.
    (9줄 인식 성공 버전)
    """
    if not regions:
        return []
        
    regions.sort(key=lambda r: r.bounds['y'])
    merged_rows = []
    
    while regions:
        current = regions.pop(0)
        current_cy = current.bounds['y'] + current.bounds['height'] // 2
        
        row_group = [current]
        others = []
        
        for r in regions:
            r_cy = r.bounds['y'] + r.bounds['height'] // 2
            height_ref = max(current.bounds['height'], r.bounds['height'])
            
            # 높이 차이가 크지 않으면 무조건 같은 줄로 간주
            if abs(current_cy - r_cy) < (height_ref * 0.5):
                row_group.append(r)
            else:
                others.append(r)
                
        regions = others
        
        # 합치기
        row_group.sort(key=lambda r: r.bounds['x'])
        
        min_x = min(r.bounds['x'] for r in row_group)
        min_y = min(r.bounds['y'] for r in row_group)
        max_x = max(r.bounds['x'] + r.bounds['width'] for r in row_group)
        max_y = max(r.bounds['y'] + r.bounds['height'] for r in row_group)
        
        # 텍스트 단순 연결 (공백 포함)
        full_text = " ".join([r.text for r in row_group])
        avg_conf = sum(r.confidence for r in row_group) / len(row_group)
        
        new_region = TextRegion(
            id="merged",
            text=full_text,
            confidence=avg_conf,
            bounds={'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y},
            is_inverted=row_group[0].is_inverted
        )
        merged_rows.append(new_region)
        
    merged_rows.sort(key=lambda r: (r.bounds['y'], r.bounds['x']))
    for i, r in enumerate(merged_rows):
        prefix = "inv" if r.is_inverted else "ocr"
        r.id = f"{prefix}_{i:03d}"
        
    return merged_rows

def run_enhanced_ocr(image: np.ndarray) -> Dict:
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    # 1. 추출
    normal_regions = ocr_engine.extract_text_regions(image)
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    all_raw_regions = normal_regions + inverted_regions
    
    # 2. 행 단위 강제 통합 (이전의 성공적인 로직)
    final_regions = merge_regions_by_row(all_raw_regions)
    
    final_normal = [r for r in final_regions if not r.is_inverted]
    final_inverted = [r for r in final_regions if r.is_inverted]
    
    return {
        'normal_regions': final_normal,
        'inverted_regions': final_inverted,
        'all_regions': final_regions,
        'image_info': {'width': image.shape[1], 'height': image.shape[0]}
    }

def group_regions_by_lines(regions: List[TextRegion]) -> List[TextRegion]:
    return regions
