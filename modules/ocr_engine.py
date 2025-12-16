"""
OCR Engine Module
텍스트 추출 및 좌표 인식을 담당하는 핵심 모듈 (Column Separation & Force Trim Version)
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
    t = text.strip()
    if not t: return False
    if not re.search(r'[가-힣a-zA-Z0-9]', t): return False
    if len(t) == 1 and re.match(r'[a-zA-Z\W]', t): return False
    return True

def merge_regions_by_row(regions: List[TextRegion]) -> List[TextRegion]:
    """
    [핵심 수정] 
    1. 행 단위로 묶되
    2. 글자 사이의 간격(Gap)이 너무 크면(그래픽 영역으로 넘어가는 경우) 끊어버립니다.
    """
    if not regions:
        return []
    
    # Y 좌표 순 정렬
    regions.sort(key=lambda r: r.bounds['y'])
    
    merged_rows = []
    
    while regions:
        current = regions.pop(0)
        current_cy = current.bounds['y'] + current.bounds['height'] // 2
        
        # 1. 같은 행 후보 찾기
        row_candidates = [current]
        others = []
        
        for r in regions:
            r_cy = r.bounds['y'] + r.bounds['height'] // 2
            height_ref = max(current.bounds['height'], r.bounds['height'])
            
            # 높이 차이가 크지 않으면 같은 행
            if abs(current_cy - r_cy) < (height_ref * 0.5):
                row_candidates.append(r)
            else:
                others.append(r)
        
        regions = others
        
        # 2. X 좌표 순 정렬 (좌 -> 우)
        row_candidates.sort(key=lambda r: r.bounds['x'])
        
        # 3. [Gap Check] 간격이 너무 넓으면 그룹 분리
        groups_in_row = []
        current_group = [row_candidates[0]]
        
        for i in range(1, len(row_candidates)):
            prev = row_candidates[i-1]
            curr = row_candidates[i]
            
            prev_right = prev.bounds['x'] + prev.bounds['width']
            curr_left = curr.bounds['x']
            gap = curr_left - prev_right
            
            # 허용 간격: 글자 높이의 2.5배 (이보다 멀면 다른 덩어리)
            max_gap = max(prev.bounds['height'], curr.bounds['height']) * 2.5
            
            if gap > max_gap:
                groups_in_row.append(current_group) # 그룹 마감
                current_group = [curr]              # 새 그룹 시작
            else:
                current_group.append(curr)
        
        groups_in_row.append(current_group)
        
        # 4. 각 그룹을 하나의 박스로 합치기
        for group in groups_in_row:
            min_x = min(r.bounds['x'] for r in group)
            min_y = min(r.bounds['y'] for r in group)
            max_x = max(r.bounds['x'] + r.bounds['width'] for r in group)
            max_y = max(r.bounds['y'] + r.bounds['height'] for r in group)
            
            # 텍스트 합치기
            full_text = " ".join([r.text for r in group])
            avg_conf = sum(r.confidence for r in group) / len(group)
            
            new_region = TextRegion(
                id="merged",
                text=full_text,
                confidence=avg_conf,
                bounds={'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y},
                is_inverted=group[0].is_inverted
            )
            merged_rows.append(new_region)

    return merged_rows

def force_trim_to_text_column(regions: List[TextRegion]) -> List[TextRegion]:
    """
    [신규 기능] 텍스트 컬럼의 우측 한계선을 계산하여 강제 절삭
    """
    if not regions:
        return []

    # 1. 우측 한계선(Column Limit) 계산
    # 모든 박스의 우측 끝 좌표(x2)를 수집
    right_edges = [(r.bounds['x'] + r.bounds['width']) for r in regions]
    
    # 상위 20% 지점을 '유효 텍스트 경계'로 간주 (너무 긴 튀는 값 제외)
    # 예: 대부분 텍스트가 500px에서 끝나는데 하나만 800px라면 500px 근처로 잡음
    right_edges.sort()
    limit_index = int(len(right_edges) * 0.85) # 85% 지점
    column_limit_x = right_edges[limit_index]
    
    # 약간의 여유(Padding)
    column_limit_x += 20 
    
    trimmed_regions = []
    for r in regions:
        r_right = r.bounds['x'] + r.bounds['width']
        
        # 박스가 한계선을 심하게 넘어가면 잘라버림
        if r.bounds['x'] < column_limit_x and r_right > column_limit_x:
            new_width = column_limit_x - r.bounds['x']
            
            # 너무 짧아지면(의미 없으면) 스킵, 아니면 수정
            if new_width > 20:
                r.bounds['width'] = int(new_width)
                trimmed_regions.append(r)
        # 아예 한계선 밖에 있는 박스(100% 노이즈)는 제거
        elif r.bounds['x'] >= column_limit_x:
            continue
        else:
            trimmed_regions.append(r)
            
    # 정렬 및 ID 재할당
    trimmed_regions.sort(key=lambda r: (r.bounds['y'], r.bounds['x']))
    for i, r in enumerate(trimmed_regions):
        prefix = "inv" if r.is_inverted else "ocr"
        r.id = f"{prefix}_{i:03d}"
        
    return trimmed_regions

def run_enhanced_ocr(image: np.ndarray) -> Dict:
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    normal_regions = ocr_engine.extract_text_regions(image)
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    all_raw_regions = normal_regions + inverted_regions
    
    # 1. 행 단위 통합 (Gap이 크면 분리)
    merged_regions = merge_regions_by_row(all_raw_regions)
    
    # 2. [NEW] 우측 강제 절삭 (그래픽 영역 침범 방지)
    final_regions = force_trim_to_text_column(merged_regions)
    
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
