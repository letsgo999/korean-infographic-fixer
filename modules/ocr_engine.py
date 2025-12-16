"""
OCR Engine Module
[Final] Edge-based Detection (색상 무관 경계선 인식) 적용
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

def get_content_mask(image: np.ndarray) -> np.ndarray:
    """
    [핵심 기술] 색상 무관, 경계선(Edge) 기반 콘텐츠 감지 마스크 생성
    글자 색깔이 무엇이든 '배경과 다르다면' 윤곽선을 잡아냅니다.
    """
    # 1. 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 2. 모폴로지 그라디언트 (경계선 추출)
    # 팽창(Dilation) - 침식(Erosion) = 경계선
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # 3. Otsu 이진화 (자동으로 임계값 설정하여 경계선만 흰색으로)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 4. 노이즈 제거 (아주 작은 점들은 무시)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    
    return binary

def refine_vertical_boundaries_by_projection(image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
    """[Updated] 경계선 기반 투영으로 세로 영역 정밀 절단"""
    if not regions: return []
    
    # [변경] 색상 무관 마스크 사용
    binary = get_content_mask(image)
    
    regions.sort(key=lambda r: r.bounds['y'])
    
    # 1. 개별 박스 최적화 (위아래 여백 제거)
    for r in regions:
        x, y, w, h = r.bounds['x'], r.bounds['y'], r.bounds['width'], r.bounds['height']
        # 이미지 범위 체크
        x = max(0, x); y = max(0, y)
        w = min(w, binary.shape[1] - x); h = min(h, binary.shape[0] - y)
        
        roi = binary[y:y+h, x:x+w]
        if roi.size == 0: continue
        
        row_sums = np.sum(roi, axis=1)
        # 픽셀이 조금이라도 있는 행 찾기
        non_empty_rows = np.where(row_sums > 0)[0]
        
        if len(non_empty_rows) > 0:
            top_trim = non_empty_rows[0]
            bottom_trim = non_empty_rows[-1]
            
            # 최소 높이 10px 보장 (너무 납작해지는 것 방지)
            new_h = bottom_trim - top_trim + 1
            if new_h > 10:
                r.bounds['y'] = y + top_trim
                r.bounds['height'] = new_h

    # 2. 박스 간 충돌 해결 (Split Line)
    for i in range(len(regions) - 1):
        upper = regions[i]
        lower = regions[i+1]
        
        u_bottom = upper.bounds['y'] + upper.bounds['height']
        l_top = lower.bounds['y']
        
        if u_bottom >= l_top - 5:
            search_y_start = upper.bounds['y'] + upper.bounds['height'] // 2
            search_y_end = lower.bounds['y'] + lower.bounds['height'] // 2
            
            # 순서가 꼬여서 start가 end보다 크면 스킵
            if search_y_start >= search_y_end: continue
            
            x_start = max(upper.bounds['x'], lower.bounds['x'])
            x_end = min(upper.bounds['x'] + upper.bounds['width'], lower.bounds['x'] + lower.bounds['width'])
            
            if x_end <= x_start: continue
            
            # 탐색
            search_roi = binary[search_y_start:search_y_end, x_start:x_end]
            if search_roi.size == 0: continue
            
            row_sums = np.sum(search_roi, axis=1)
            # 픽셀 합이 가장 작은(경계선이 없는) 곳 찾기
            min_indices = np.where(row_sums == row_sums.min())[0]
            split_line_local = min_indices[len(min_indices)//2]
            split_line_global = search_y_start + split_line_local
            
            upper.bounds['height'] = max(10, split_line_global - upper.bounds['y'] - 2)
            old_bottom = lower.bounds['y'] + lower.bounds['height']
            lower.bounds['y'] = split_line_global + 2
            lower.bounds['height'] = max(10, old_bottom - lower.bounds['y'])

    for i, r in enumerate(regions):
        prefix = "inv" if r.is_inverted else "ocr"
        r.id = f"{prefix}_{i:03d}"
    return regions

def expand_horizontal_bounds(image: np.ndarray, regions: List[TextRegion], max_lookahead: int = 200, gap_threshold: int = 30) -> List[TextRegion]:
    """[Updated] 경계선 기반 가로 영역 확장"""
    if not regions: return []
    
    # [변경] 색상 무관 마스크 사용
    binary = get_content_mask(image)
    img_h, img_w = binary.shape
    
    for r in regions:
        if r.is_inverted: continue
        x, y, w, h = r.bounds['x'], r.bounds['y'], r.bounds['width'], r.bounds['height']
        
        current_x = x + w
        consecutive_gap = 0
        extended_pixels = 0
        
        while current_x < img_w and extended_pixels < max_lookahead:
            # 안전한 ROI 슬라이싱
            y_start = max(0, y)
            y_end = min(img_h, y+h)
            if y_start >= y_end: break

            col_roi = binary[y_start:y_end, current_x:current_x+1]
            
            # 경계선(글자)이 있는가?
            if cv2.countNonZero(col_roi) > 0:
                consecutive_gap = 0
            else:
                consecutive_gap += 1
            
            if consecutive_gap > gap_threshold:
                break
                
            current_x += 1
            extended_pixels += 1
            
        final_extension = extended_pixels - consecutive_gap
        if final_extension > 0:
            r.bounds['width'] += final_extension
            
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
    merged_regions = merge_regions_by_row(all_raw_regions)
    vertical_refined = refine_vertical_boundaries_by_projection(image, merged_regions)
    final_regions = expand_horizontal_bounds(image, vertical_refined)
    
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
