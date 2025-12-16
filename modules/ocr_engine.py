"""
OCR Engine Module
텍스트 추출, 좌표 인식 및 [강력한 겹침 방지]가 적용된 모듈
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

def prevent_vertical_overlaps(regions: List[TextRegion], buffer: int = 5) -> List[TextRegion]:
    """
    [강력한 겹침 방지] 이중 루프 + 수평 교차 검사
    모든 박스 쌍을 전수 조사하여, 가로로 겹치는 박스들 사이의 세로 침범을 원천 차단합니다.
    """
    if not regions:
        return []
        
    # Y 좌표 순으로 정렬 (위 -> 아래)
    regions.sort(key=lambda r: r.bounds['y'])
    
    # 이중 루프: i는 윗박스, j는 아랫박스 후보
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            upper = regions[i]
            lower = regions[j]
            
            # 1. 아랫박스가 윗박스보다 훨씬 밑에 있어서 겹칠 가능성이 0이면 패스
            # (최적화를 위해 추가: 윗박스 바닥 + 10px보다 아랫박스 천장이 더 아래면 검사 불필요)
            if lower.bounds['y'] > (upper.bounds['y'] + upper.bounds['height'] + 10):
                continue

            # 2. [핵심] 수평(X축) 겹침 확인
            # 서로 다른 컬럼에 있는 박스라면 세로로 겹쳐도 상관없음 (예: 좌측 텍스트, 우측 텍스트)
            u_left, u_right = upper.bounds['x'], upper.bounds['x'] + upper.bounds['width']
            l_left, l_right = lower.bounds['x'], lower.bounds['x'] + lower.bounds['width']
            
            # 교차 구간 계산
            intersect_x1 = max(u_left, l_left)
            intersect_x2 = min(u_right, l_right)
            
            # 교차 폭이 존재하면 (= 같은 세로 라인 상에 있다면)
            if intersect_x2 > intersect_x1:
                # 3. 수직 겹침 확인 및 자르기
                u_bottom = upper.bounds['y'] + upper.bounds['height']
                l_top = lower.bounds['y']
                
                # 윗박스 바닥이 아랫박스 천장보다 아래거나, 여유공간(buffer) 안쪽으로 들어왔다면
                if u_bottom >= (l_top - buffer):
                    # 윗박스 높이 강제 조절
                    new_height = (l_top - buffer) - upper.bounds['y']
                    
                    # 높이가 너무 작아져서(15px 미만) 아예 사라지는 것 방지
                    upper.bounds['height'] = max(15, int(new_height))
                    
    # ID 재할당
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
    
    # 1. 행 단위로 합치기
    merged_regions = merge_regions_by_row(all_raw_regions)
    
    # 2. [강력한] 수직 겹침 해결
    final_regions = prevent_vertical_overlaps(merged_regions)
    
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
