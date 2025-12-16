"""
OCR Engine Module
[Final Polish] 
1. 엣지 기반 감지
2. 픽셀 투영 행 분리
3. [NEW] 터널 주행 기법(Tunnel Collision)으로 그림 영역 침범 원천 차단
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
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    return binary

def refine_vertical_boundaries_by_projection(image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
    if not regions: return []
    binary = get_content_mask(image)
    regions.sort(key=lambda r: r.bounds['y'])
    for r in regions:
        x, y, w, h = r.bounds['x'], r.bounds['y'], r.bounds['width'], r.bounds['height']
        x = max(0, x); y = max(0, y); w = min(w, binary.shape[1] - x); h = min(h, binary.shape[0] - y)
        roi = binary[y:y+h, x:x+w]
        if roi.size == 0: continue
        row_sums = np.sum(roi, axis=1)
        non_empty_rows = np.where(row_sums > 0)[0]
        if len(non_empty_rows) > 0:
            top_trim = non_empty_rows[0]; bottom_trim = non_empty_rows[-1]
            new_h = bottom_trim - top_trim + 1
            if new_h > 10:
                r.bounds['y'] = y + top_trim; r.bounds['height'] = new_h
    for i in range(len(regions) - 1):
        upper = regions[i]; lower = regions[i+1]
        u_bottom = upper.bounds['y'] + upper.bounds['height']; l_top = lower.bounds['y']
        if u_bottom >= l_top - 5:
            search_y_start = upper.bounds['y'] + upper.bounds['height'] // 2
            search_y_end = lower.bounds['y'] + lower.bounds['height'] // 2
            if search_y_start >= search_y_end: continue
            x_start = max(upper.bounds['x'], lower.bounds['x']); x_end = min(upper.bounds['x'] + upper.bounds['width'], lower.bounds['x'] + lower.bounds['width'])
            if x_end <= x_start: continue
            search_roi = binary[search_y_start:search_y_end, x_start:x_end]
            if search_roi.size == 0: continue
            row_sums = np.sum(search_roi, axis=1)
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
    """
    [NEW] 터널 주행 (Tunnel Collision) 방식 적용
    내 높이(Height)만큼의 터널을 뚫고 전진하되,
    터널 위아래로 튀어나온 '큰 물체(그림)'가 감지되면 즉시 정지합니다.
    """
    if not regions: return []
    
    # 엣지 마스크 생성
    binary = get_content_mask(image)
    img_h, img_w = binary.shape
    
    for r in regions:
        if r.is_inverted: continue
        
        x, y, w, h = r.bounds['x'], r.bounds['y'], r.bounds['width'], r.bounds['height']
        
        # 1. 터널 설정 (위아래로 약간의 여유를 둠, 너무 빡빡하면 노이즈에 걸림)
        # 텍스트 라인의 위쪽 경계와 아래쪽 경계
        tunnel_top = max(0, y)
        tunnel_bottom = min(img_h, y + h)
        
        current_x = x + w
        consecutive_gap = 0
        extended_pixels = 0
        
        while current_x < img_w and extended_pixels < max_lookahead:
            # 2. 터널 내부 검사 (In-Tunnel)
            # 현재 x위치의 터널 내부 픽셀들을 봅니다.
            col_roi_inner = binary[tunnel_top:tunnel_bottom, current_x:current_x+1]
            has_content_inside = cv2.countNonZero(col_roi_inner) > 0
            
            # 3. 충돌 감지 (Out-of-Tunnel Check) - 여기가 핵심!
            # 터널 바로 위(Head)와 바로 아래(Feet)를 검사합니다.
            # 만약 터널 내부에도 픽셀이 있고, 외부(위/아래)에도 픽셀이 연결되어 있다면? -> 큰 그림이다!
            
            # 위쪽 감시 구역 (터널 위 10px)
            check_top = max(0, tunnel_top - 10)
            col_roi_upper = binary[check_top:tunnel_top, current_x:current_x+1]
            
            # 아래쪽 감시 구역 (터널 아래 10px)
            check_bottom = min(img_h, tunnel_bottom + 10)
            col_roi_lower = binary[tunnel_bottom:check_bottom, current_x:current_x+1]
            
            has_content_upper = cv2.countNonZero(col_roi_upper) > 0
            has_content_lower = cv2.countNonZero(col_roi_lower) > 0
            
            # [STOP 조건] 
            # 내용물이 있는데(has_content_inside), 그게 위나 아래로 삐져나와 있다면(upper/lower)
            # 이것은 텍스트 라인이 아니라 세로로 긴 물체(사람, 아이콘 등)입니다.
            if has_content_inside and (has_content_upper or has_content_lower):
                # 단, 아주 미세한 연결(1~2픽셀)은 노이즈일 수 있으니 무시할 수도 있지만,
                # 안전을 위해 여기서는 발견 즉시 멈춥니다.
                break
            
            # 일반적인 확장 로직
            if has_content_inside:
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
    # [NEW] 터널 충돌 방지 적용
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
