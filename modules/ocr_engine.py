"""
OCR Engine Module
[Final Innovation] 
1. 엣지 기반 감지
2. 픽셀 투영 행 분리
3. [NEW] 컨투어(객체) 분석 기반 지능형 확장 (글자 vs 그림 형태학적 구분)
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
    width_scale: int = 90
    
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
    """
    같은 행(Y축)에 있는 텍스트 영역들을 병합합니다.
    [NEW] 마침표(".")로 끝나는 텍스트가 있으면 해당 지점에서 영역을 분리합니다.
    """
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
            if abs(current_cy - r_cy) < (height_ref * 0.5): 
                row_group.append(r)
            else: 
                others.append(r)
        regions = others
        
        # X좌표 기준 정렬
        row_group.sort(key=lambda r: r.bounds['x'])
        
        # [NEW] 마침표 기준 분리 로직
        # 마침표로 끝나는 텍스트가 있으면 해당 지점에서 분리
        split_groups = split_by_period(row_group)
        
        for group in split_groups:
            if not group: continue
            min_x = min(r.bounds['x'] for r in group)
            min_y = min(r.bounds['y'] for r in group)
            max_x = max(r.bounds['x'] + r.bounds['width'] for r in group)
            max_y = max(r.bounds['y'] + r.bounds['height'] for r in group)
            full_text = " ".join([r.text for r in group])
            avg_conf = sum(r.confidence for r in group) / len(group)
            new_region = TextRegion(
                id="merged", text=full_text, confidence=avg_conf, 
                bounds={'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}, 
                is_inverted=group[0].is_inverted
            )
            merged_rows.append(new_region)
    
    return merged_rows

def split_by_period(row_group: List[TextRegion]) -> List[List[TextRegion]]:
    """
    [NEW] 마침표(".")로 끝나는 텍스트를 기준으로 그룹을 분리합니다.
    예: ["안녕하세요.", "다음 문장"] -> [["안녕하세요."], ["다음 문장"]]
    """
    if not row_group: return []
    
    result = []
    current_group = []
    
    for r in row_group:
        current_group.append(r)
        text = r.text.strip()
        
        # 마침표, 물음표, 느낌표 등 문장 종결 부호로 끝나는지 확인
        if text and text[-1] in '.。?!？！':
            # 현재 그룹을 결과에 추가하고 새 그룹 시작
            result.append(current_group)
            current_group = []
    
    # 남은 그룹이 있으면 추가
    if current_group:
        result.append(current_group)
    
    return result

def get_content_mask(image: np.ndarray) -> np.ndarray:
    # 엣지 검출 (Canny 사용) - 색상 무관하게 형태만 봅니다.
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny Edge Detection (경계선 검출)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 경계선을 약간 팽창시켜서 연결성을 좋게 함
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    return dilated

def refine_vertical_boundaries_by_projection(image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
    # (이전의 성공적인 행 분리 로직 유지)
    if not regions: return []
    binary = get_content_mask(image) # Canny Edge 기반 마스크 사용
    regions.sort(key=lambda r: r.bounds['y'])
    
    # 개별 박스 트리밍
    for r in regions:
        x, y, w, h = r.bounds['x'], r.bounds['y'], r.bounds['width'], r.bounds['height']
        roi = binary[max(0, y):min(binary.shape[0], y+h), max(0, x):min(binary.shape[1], x+w)]
        if roi.size == 0: continue
        row_sums = np.sum(roi, axis=1)
        non_empty_rows = np.where(row_sums > 0)[0]
        if len(non_empty_rows) > 0:
            top_trim = non_empty_rows[0]; bottom_trim = non_empty_rows[-1]
            new_h = bottom_trim - top_trim + 1
            if new_h > 10: r.bounds['y'] = y + top_trim; r.bounds['height'] = new_h
            
    # 행간 분리
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

def expand_horizontal_bounds(image: np.ndarray, regions: List[TextRegion], max_lookahead: int = 400) -> List[TextRegion]:
    """
    [NEW] 컨투어(객체) 기반 지능형 확장
    단순 픽셀 확인이 아니라, '오른쪽에 있는 덩어리'를 찾아서
    그것이 '글자처럼 생겼으면' 먹고, '그림처럼 생겼으면' 뱉습니다.
    """
    if not regions: return []
    
    # 1. 엣지 마스크 (내용물 확인용)
    binary = get_content_mask(image)
    img_h, img_w = binary.shape
    
    for r in regions:
        if r.is_inverted: continue
        
        # 현재 텍스트 라인의 기준 높이 (이것과 비슷해야 글자임)
        ref_h = r.bounds['height']
        
        # 어디서부터 찾을까? (현재 박스 오른쪽 끝)
        start_x = r.bounds['x'] + r.bounds['width']
        
        # 탐색 영역 (오른쪽으로 max_lookahead 만큼, 위아래로 약간 여유)
        # 텍스트 라인의 Y축 범위 (위아래 10% 여유)
        y_margin = int(ref_h * 0.2)
        roi_y1 = max(0, r.bounds['y'] - y_margin)
        roi_y2 = min(img_h, r.bounds['y'] + r.bounds['height'] + y_margin)
        roi_x1 = start_x
        roi_x2 = min(img_w, start_x + max_lookahead)
        
        if roi_x2 <= roi_x1: continue
        
        # ROI 추출
        roi = binary[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # 2. 덩어리(Contours) 찾기
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 발견된 덩어리들을 왼쪽부터 순서대로 정렬
        blob_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 너무 작은 노이즈(점)는 무시
            if w < 3 or h < 3: continue 
            blob_list.append((x, y, w, h))
            
        # X좌표 기준 정렬
        blob_list.sort(key=lambda b: b[0])
        
        current_extension = 0
        
        # 3. 덩어리 심사 (글자니? 그림이니?)
        for bx, by, bw, bh in blob_list:
            # 덩어리의 절대 좌표 변환
            abs_x = roi_x1 + bx
            
            # --- 심사 기준 1: 거리 ---
            # 이전 글자와 너무 멀리 떨어져 있으면(100px 이상) 남남임
            gap = abs_x - (r.bounds['x'] + r.bounds['width'] + current_extension)
            if gap > 100: 
                break # 여기서 끝
            
            # --- 심사 기준 2: 높이 유사성 (가장 중요) ---
            # 글자라면 기존 줄 높이의 50% ~ 150% 사이여야 함
            # 그림(일러스트)은 보통 훨씬 큼
            if bh > (ref_h * 1.5):
                break # 너무 큰 놈이 나타났다 -> 그림이다 -> 정지!
                
            # --- 심사 기준 3: 수직 정렬 ---
            # 글자라면 기존 줄의 Y 범위 안에 들어와야 함
            # 덩어리의 중심이 텍스트 라인의 중심과 비슷해야 함
            blob_center_y = roi_y1 + by + bh/2
            line_center_y = r.bounds['y'] + r.bounds['height']/2
            
            if abs(blob_center_y - line_center_y) > (ref_h * 0.6):
                # 위나 아래로 너무 튀어 나간 놈 -> 그림의 일부일 가능성 큼
                break
            
            # --- 심사 기준 4: 형태 (비율) ---
            # 아이콘(페이스북 등)은 보통 정사각형에 가깝고 꽉 차 있음
            # 하지만 "글자"도 'ㅁ' 같은 건 그럴 수 있음.
            # 여기서는 "크기"와 "위치"가 맞으면 일단 글자로 간주하되,
            # 너무 넓은 덩어리(가로로 긴 박스)는 제외
            if bw > (ref_h * 3.0): # 높이보다 3배 이상 긴 덩어리?
                break # 그림 배경이나 배너일 확률 높음 -> 정지
            
            # 합격! 이 덩어리는 글자(파편)다. 포함시킨다.
            # 박스 확장
            new_right_edge = abs_x + bw
            current_w = new_right_edge - r.bounds['x']
            
            # 기존 너비보다 커지면 업데이트
            if current_w > r.bounds['width']:
                r.bounds['width'] = current_w
                current_extension = current_w - (start_x - r.bounds['x']) # 갱신
            
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
    # [NEW] 컨투어 기반 지능형 확장
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
        suggested_font_size=14, width_scale=90
    )
