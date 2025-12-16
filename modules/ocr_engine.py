"""
OCR Engine Module
텍스트 추출, 좌표 인식, [픽셀 기반 정밀 축소] 및 [행간 분리]가 적용된 모듈
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
        custom_config = r'--oem 3 --psm 6'
        try:
            ocr_data = pytesseract.image_to_data(pil_image, lang=self.lang, config=custom_config, output_type=pytesseract.Output.DICT)
        except:
            return []
        regions = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            if conf >= self.min_confidence and is_valid_text(text):
                region = TextRegion(
                    id=f"raw_{len(regions)}",
                    text=text,
                    confidence=conf,
                    bounds={'x': ocr_data['left'][i], 'y': ocr_data['top'][i], 'width': ocr_data['width'][i], 'height': ocr_data['height'][i]},
                    is_inverted=False
                )
                regions.append(region)
        return regions
    
    def extract_from_inverted_region(self, image: np.ndarray, region_bounds: Dict[str, int], padding: int = 5) -> List[TextRegion]:
        x, y = region_bounds['x'], region_bounds['y']
        w, h = region_bounds['width'], region_bounds['height']
        x1 = max(0, x - padding); y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding); y2 = min(image.shape[0], y + h + padding)
        roi = image[y1:y2, x1:x2].copy()
        inverted_roi = cv2.bitwise_not(roi)
        if len(inverted_roi.shape) == 3: inverted_rgb = cv2.cvtColor(inverted_roi, cv2.COLOR_BGR2RGB)
        else: inverted_rgb = inverted_roi
        pil_roi = Image.fromarray(inverted_rgb)
        custom_config = r'--oem 3 --psm 7'
        try:
            ocr_data = pytesseract.image_to_data(pil_roi, lang=self.lang, config=custom_config, output_type=pytesseract.Output.DICT)
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
                    bounds={'x': x1 + ocr_data['left'][i], 'y': y1 + ocr_data['top'][i], 'width': ocr_data['width'][i], 'height': ocr_data['height'][i]},
                    is_inverted=True
                )
                regions.append(region)
        return regions

class InvertedRegionDetector:
    def __init__(self, dark_threshold=150, min_area=1000, min_width=40, min_height=15):
        self.dark_threshold = dark_threshold; self.min_area = min_area; self.min_width = min_width; self.min_height = min_height
    def detect(self, image: np.ndarray) -> List[Dict[str, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 100, 100]); upper_orange = np.array([25, 255, 255])
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
    t = text.strip()
    if not t: return False
    if not re.search(r'[가-힣a-zA-Z0-9]', t): return False
    if len(t) == 1 and re.match(r'[a-zA-Z\W]', t): return False
    return True

def merge_regions_by_row(regions: List[TextRegion]) -> List[TextRegion]:
    if not regions: return []
    regions.sort(key=lambda r: r.bounds['y'])
    merged_rows = []
    while regions:
        current = regions.pop(0)
        current_cy = current.bounds['y'] + current.bounds['height'] // 2
        row_group = [current]; others = []
        for r in regions:
            r_cy = r.bounds['y'] + r.bounds['height'] // 2
            height_ref = max(current.bounds['height'], r.bounds['height'])
            if abs(current_cy - r_cy) < (height_ref * 0.5): row_group.append(r)
            else: others.append(r)
        regions = others
        row_group.sort(key=lambda r: r.bounds['x'])
        min_x = min(r.bounds['x'] for r in row_group); min_y = min(r.bounds['y'] for r in row_group)
        max_x = max(r.bounds['x'] + r.bounds['width'] for r in row_group); max_y = max(r.bounds['y'] + r.bounds['height'] for r in row_group)
        full_text = " ".join([r.text for r in row_group])
        avg_conf = sum(r.confidence for r in row_group) / len(row_group)
        new_region = TextRegion(
            id="merged", text=full_text, confidence=avg_conf, bounds={'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}, is_inverted=row_group[0].is_inverted
        )
        merged_rows.append(new_region)
    return merged_rows

def optimize_vertical_bounds(image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
    """
    [NEW] 픽셀 기반 박스 축소 (Tightening)
    박스 안의 실제 글자 픽셀을 분석하여, 위아래 빈 공간을 잘라냅니다.
    이것이 '제품의 매력...' 같은 박스가 위로 치솟는 것을 막아줍니다.
    """
    if not regions: return []
    
    # 그레이스케일 변환 및 이진화 (글자는 검정, 배경은 흰색 가정 또는 반대)
    # OCR 정확도를 위해 Otsu 이진화를 사용해 글자 영역을 확실히 찾습니다.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 글자가 어두운 색이라고 가정하고 이진화 (배경이 밝음)
    # 인포그래픽 특성상 글자가 밝을 수도 있으므로, 박스 내부의 분산 등을 볼 수 있지만
    # 여기서는 일반적인 adaptive thresholding을 사용합니다.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    for r in regions:
        x, y = r.bounds['x'], r.bounds['y']
        w, h = r.bounds['width'], r.bounds['height']
        
        # 이미지 범위를 벗어나지 않게 클핑
        x = max(0, x); y = max(0, y)
        w = min(w, image.shape[1] - x); h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0: continue
        
        # 해당 영역(ROI) 추출
        roi = binary[y:y+h, x:x+w]
        
        # 수평 투영 (Horizontal Projection): 각 행(row)의 픽셀 합을 구함
        # 글자가 있는 행은 값이 크고, 빈 공간은 0에 가까움
        row_sum = cv2.reduce(roi, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).flatten()
        
        # 픽셀이 있는 첫 번째 행(Top)과 마지막 행(Bottom) 찾기
        non_zero_indices = np.where(row_sum > (w * 0.05 * 255)) # 노이즈(5%) 제외하고 글자 픽셀 찾기
        
        if len(non_zero_indices[0]) > 0:
            top_offset = non_zero_indices[0][0]
            bottom_offset = non_zero_indices[0][-1]
            
            # 실제 글자 높이
            new_h = bottom_offset - top_offset + 1
            
            # 너무 작아지면(노이즈만 잡은 경우) 무시하고, 의미 있게 줄어들면 적용
            if new_h > 5 and new_h < h:
                # 좌표 업데이트 (Top은 아래로 내려가고, Height는 줄어듦)
                r.bounds['y'] = y + top_offset
                r.bounds['height'] = new_h
                
                # 디버깅용: "조여졌다"는 것을 알 수 있음
                # print(f"Region {r.id} tightened: y {y}->{r.bounds['y']}, h {h}->{new_h}")
                
    return regions

def prevent_vertical_overlaps(regions: List[TextRegion], buffer: int = 5) -> List[TextRegion]:
    """
    [강력한 행간 분리] 
    픽셀 축소 후에도 겹치는 박스가 있다면, '빈 공백'을 기준으로 강제 분할합니다.
    """
    if not regions: return []
    regions.sort(key=lambda r: r.bounds['y'])
    
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            upper = regions[i]
            lower = regions[j]
            
            if lower.bounds['y'] > (upper.bounds['y'] + upper.bounds['height'] + 10):
                continue

            u_left, u_right = upper.bounds['x'], upper.bounds['x'] + upper.bounds['width']
            l_left, l_right = lower.bounds['x'], lower.bounds['x'] + lower.bounds['width']
            
            intersect_x1 = max(u_left, l_left)
            intersect_x2 = min(u_right, l_right)
            
            # 가로로 겹치는 구간이 있다면 (같은 컬럼)
            if intersect_x2 > intersect_x1:
                u_bottom = upper.bounds['y'] + upper.bounds['height']
                l_top = lower.bounds['y']
                
                # 세로로 겹친다면
                if u_bottom >= l_top:
                    # 겹치는 구간의 중간 지점을 찾음
                    mid_y = (u_bottom + l_top) // 2
                    
                    # 윗 박스는 중간 지점보다 조금 위에서 끝내고
                    upper.bounds['height'] = max(10, (mid_y - buffer) - upper.bounds['y'])
                    
                    # 아랫 박스는 중간 지점보다 조금 아래에서 시작 (Top 이동, 높이 축소)
                    original_bottom = lower.bounds['y'] + lower.bounds['height']
                    new_top = mid_y + buffer
                    lower.bounds['y'] = new_top
                    lower.bounds['height'] = max(10, original_bottom - new_top)

    for i, r in enumerate(regions):
        prefix = "inv" if r.is_inverted else "ocr"
        r.id = f"{prefix}_{i:03d}"
        
    return regions

def run_enhanced_ocr(image: np.ndarray) -> Dict:
    ocr_engine = OCREngine(); inv_detector = InvertedRegionDetector()
    normal_regions = ocr_engine.extract_text_regions(image)
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    all_raw_regions = normal_regions + inverted_regions
    
    # 1. 행 단위 합치기
    merged_regions = merge_regions_by_row(all_raw_regions)
    
    # 2. [NEW] 픽셀 기반 박스 축소 (Tightening) - 여기서 공백을 확보합니다.
    tightened_regions = optimize_vertical_bounds(image, merged_regions)
    
    # 3. [UPDATED] 그래도 겹치면 강제 분리
    final_regions = prevent_vertical_overlaps(tightened_regions)
    
    final_normal = [r for r in final_regions if not r.is_inverted]
    final_inverted = [r for r in final_regions if r.is_inverted]
    
    return {'normal_regions': final_normal, 'inverted_regions': final_inverted, 'all_regions': final_regions, 'image_info': {'width': image.shape[1], 'height': image.shape[0]}}

def group_regions_by_lines(regions: List[TextRegion]) -> List[TextRegion]: return regions

def create_manual_region(x: int, y: int, width: int, height: int, text: str, style_tag: str = "body") -> TextRegion:
    return TextRegion(
        id=f"manual_{int(x)}_{int(y)}", text=text, confidence=100.0, 
        bounds={'x': x, 'y': y, 'width': width, 'height': height}, 
        is_inverted=False, is_manual=True, style_tag=style_tag,
        suggested_font_size=14, width_scale=80
    )
