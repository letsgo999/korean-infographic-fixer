"""
OCR Engine Module
[v4 - 원점 재설계] 
1. 행 병합 기본 유지 (같은 Y축 = 같은 행)
2. 수평 확장 시 "정사각형 덩어리"를 만나면 멈춤
3. 복잡한 컬러 분석 제거
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
    t = text.strip()
    if not t: return False
    if not re.search(r'[가-힣a-zA-Z0-9]', t): return False
    if len(t) == 1 and re.match(r'[a-zA-Z\W]', t): return False
    return True


def merge_regions_by_row(regions: List[TextRegion]) -> List[TextRegion]:
    """행 단위 병합 - 같은 Y축 라인은 하나의 행으로"""
    if not regions: return []
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
            if abs(current_cy - r_cy) < (height_ref * 0.5):
                row_group.append(r)
            else:
                others.append(r)
        
        regions = others
        row_group.sort(key=lambda r: r.bounds['x'])
        
        min_x = min(r.bounds['x'] for r in row_group)
        min_y = min(r.bounds['y'] for r in row_group)
        max_x = max(r.bounds['x'] + r.bounds['width'] for r in row_group)
        max_y = max(r.bounds['y'] + r.bounds['height'] for r in row_group)
        
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
    
    return merged_rows


def get_content_mask(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated


def refine_vertical_boundaries_by_projection(image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
    if not regions: return []
    binary = get_content_mask(image)
    regions.sort(key=lambda r: r.bounds['y'])
    
    for r in regions:
        x, y, w, h = r.bounds['x'], r.bounds['y'], r.bounds['width'], r.bounds['height']
        roi = binary[max(0, y):min(binary.shape[0], y+h), max(0, x):min(binary.shape[1], x+w)]
        if roi.size == 0: continue
        row_sums = np.sum(roi, axis=1)
        non_empty_rows = np.where(row_sums > 0)[0]
        if len(non_empty_rows) > 0:
            top_trim = non_empty_rows[0]
            bottom_trim = non_empty_rows[-1]
            new_h = bottom_trim - top_trim + 1
            if new_h > 10:
                r.bounds['y'] = y + top_trim
                r.bounds['height'] = new_h
            
    for i in range(len(regions) - 1):
        upper = regions[i]
        lower = regions[i+1]
        u_bottom = upper.bounds['y'] + upper.bounds['height']
        l_top = lower.bounds['y']
        if u_bottom >= l_top - 5:
            search_y_start = upper.bounds['y'] + upper.bounds['height'] // 2
            search_y_end = lower.bounds['y'] + lower.bounds['height'] // 2
            if search_y_start >= search_y_end: continue
            x_start = max(upper.bounds['x'], lower.bounds['x'])
            x_end = min(upper.bounds['x'] + upper.bounds['width'], lower.bounds['x'] + lower.bounds['width'])
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


def find_right_boundary(image: np.ndarray, region: TextRegion, max_lookahead: int = 500) -> int:
    """
    [핵심 함수] 텍스트 행의 오른쪽 끝을 어디서 자를지 결정
    
    원리: 오른쪽으로 스캔하면서 "정사각형에 가까운 큰 덩어리"를 만나면 그 전에서 멈춤
    - 텍스트: 가로로 길쭉함 (width >> height)
    - 아이콘/그림: 정사각형에 가까움 (width ≈ height)
    """
    binary = get_content_mask(image)
    img_h, img_w = binary.shape
    
    ref_h = region.bounds['height']
    start_x = region.bounds['x'] + region.bounds['width']
    y1 = max(0, region.bounds['y'] - 5)
    y2 = min(img_h, region.bounds['y'] + ref_h + 5)
    
    # 탐색 범위
    search_end = min(img_w, start_x + max_lookahead)
    
    if search_end <= start_x:
        return region.bounds['x'] + region.bounds['width']
    
    # 오른쪽 영역에서 컨투어 찾기
    roi = binary[y1:y2, start_x:search_end]
    if roi.size == 0:
        return region.bounds['x'] + region.bounds['width']
    
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 덩어리들을 왼쪽부터 정렬
    blobs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        blobs.append({'x': x, 'y': y, 'w': w, 'h': h, 'abs_x': start_x + x})
    
    blobs.sort(key=lambda b: b['x'])
    
    current_right = region.bounds['x'] + region.bounds['width']
    
    for blob in blobs:
        # 이전 끝과 너무 멀리 떨어져 있으면 (100px 이상 간격) 여기서 끝
        gap = blob['abs_x'] - current_right
        if gap > 100:
            break
        
        w, h = blob['w'], blob['h']
        
        # [핵심 판단] 정사각형에 가까운 큰 덩어리 = 아이콘/그림
        aspect_ratio = w / h if h > 0 else 0
        is_square_ish = 0.5 <= aspect_ratio <= 2.0  # 정사각형에 가까움
        is_big = w > ref_h * 0.8 and h > ref_h * 0.8  # 텍스트 높이보다 큰 덩어리
        
        if is_square_ish and is_big:
            # 아이콘/그림으로 판단 → 여기서 멈춤
            break
        
        # 텍스트로 판단 → 포함
        current_right = blob['abs_x'] + w
    
    return current_right


def expand_horizontal_bounds(image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
    """행 단위로 오른쪽 경계 확장 (아이콘/그림 전에서 멈춤)"""
    if not regions:
        return []
    
    for r in regions:
        if r.is_inverted:
            continue
        
        # 오른쪽 경계 찾기
        right_x = find_right_boundary(image, r)
        new_width = right_x - r.bounds['x']
        
        if new_width > r.bounds['width']:
            r.bounds['width'] = new_width
    
    return regions


def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """메인 OCR 함수"""
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    # 1. OCR로 텍스트 추출
    normal_regions = ocr_engine.extract_text_regions(image)
    
    # 2. 역상 영역 처리
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    # 3. 행 단위 병합 (핵심 - 같은 Y축은 하나의 행)
    all_raw_regions = normal_regions + inverted_regions
    merged_regions = merge_regions_by_row(all_raw_regions)
    
    # 4. 수직 경계 정제 (행간 분리)
    vertical_refined = refine_vertical_boundaries_by_projection(image, merged_regions)
    
    # 5. 수평 확장 (아이콘/그림 전에서 멈춤)
    final_regions = expand_horizontal_bounds(image, vertical_refined)
    
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


def create_manual_region(x: int, y: int, width: int, height: int, text: str, style_tag: str = "body") -> TextRegion:
    return TextRegion(
        id=f"manual_{int(x)}_{int(y)}",
        text=text,
        confidence=100.0,
        bounds={'x': x, 'y': y, 'width': width, 'height': height},
        is_inverted=False,
        is_manual=True,
        style_tag=style_tag,
        suggested_font_size=14,
        width_scale=80
    )
