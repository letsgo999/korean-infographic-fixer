"""
OCR Engine Module
[Balanced Version v3] 
1. 엣지 기반 감지
2. 픽셀 투영 행 분리
3. 컨투어 기반 지능형 확장
4. [BALANCED] 그래픽 보호 - 아이콘/일러스트만 타겟팅
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple
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


class GraphicProtector:
    """
    [v3] 균형 잡힌 그래픽 보호
    - 소셜 아이콘 그룹만 정확히 타겟팅
    - 큰 일러스트 영역만 보호
    - 텍스트는 최대한 보존
    """
    
    def __init__(self):
        self.protected_regions: List[Dict] = []
        self.mask: np.ndarray = None
    
    def detect_and_protect(self, image: np.ndarray, padding: int = 5) -> np.ndarray:
        h, w = image.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.protected_regions = []
        
        # 1. 소셜 아이콘 그룹 감지 (가장 중요)
        icon_groups = self._detect_social_icon_row(image)
        
        # 2. 큰 일러스트 영역 감지
        illustrations = self._detect_large_illustrations(image)
        
        all_regions = icon_groups + illustrations
        
        # 병합
        merged = self._merge_nearby_regions(all_regions)
        self.protected_regions = merged
        
        # 마스크에 그리기
        for region in merged:
            x = max(0, region['x'] - padding)
            y = max(0, region['y'] - padding)
            x2 = min(w, region['x'] + region['width'] + padding)
            y2 = min(h, region['y'] + region['height'] + padding)
            self.mask[y:y2, x:x2] = 255
        
        return self.mask
    
    def _detect_social_icon_row(self, image: np.ndarray) -> List[Dict]:
        """소셜 미디어 아이콘 행 감지 (인스타, 유튜브, 페이스북 등)"""
        regions = []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 채도가 높은 작은 사각형들 찾기
        saturation = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY)
        
        # 작은 커널로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 아이콘 후보 (20~60px 크기의 정사각형에 가까운 것들)
        icon_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 아이콘 크기 범위
            if 18 <= w <= 70 and 18 <= h <= 70:
                aspect = w / h if h > 0 else 0
                # 정사각형에 가까운 것
                if 0.6 <= aspect <= 1.7:
                    # 채워진 정도 확인
                    roi = sat_mask[y:y+h, x:x+w]
                    fill_ratio = np.sum(roi > 0) / (w * h) if w * h > 0 else 0
                    if fill_ratio > 0.3:
                        icon_candidates.append({'x': x, 'y': y, 'width': w, 'height': h})
        
        # 수평으로 나열된 아이콘들 그룹화
        if len(icon_candidates) >= 2:
            icon_candidates.sort(key=lambda c: (c['y'] // 40, c['x']))
            
            current_group = []
            for cand in icon_candidates:
                if not current_group:
                    current_group.append(cand)
                    continue
                
                last = current_group[-1]
                same_row = abs(cand['y'] - last['y']) < 30
                gap = cand['x'] - (last['x'] + last['width'])
                close = 0 < gap < 50
                
                if same_row and close:
                    current_group.append(cand)
                else:
                    if len(current_group) >= 2:
                        regions.append(self._merge_icons(current_group))
                    current_group = [cand]
            
            if len(current_group) >= 2:
                regions.append(self._merge_icons(current_group))
        
        return regions
    
    def _detect_large_illustrations(self, image: np.ndarray) -> List[Dict]:
        """큰 일러스트/캐릭터 영역 감지"""
        regions = []
        
        # 피부색 감지 (YCrCb)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # 모폴로지로 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # 큰 영역만 (일러스트는 보통 큼)
            if area > 3000 and w > 50 and h > 50:
                aspect = w / h if h > 0 else 0
                # 너무 길쭉하지 않은 것 (텍스트 배제)
                if 0.2 < aspect < 5:
                    regions.append({
                        'x': x, 'y': y, 'width': w, 'height': h, 'type': 'illustration'
                    })
        
        return regions
    
    def _merge_icons(self, icons: List[Dict]) -> Dict:
        x = min(i['x'] for i in icons)
        y = min(i['y'] for i in icons)
        x2 = max(i['x'] + i['width'] for i in icons)
        y2 = max(i['y'] + i['height'] for i in icons)
        return {'x': x, 'y': y, 'width': x2 - x, 'height': y2 - y, 'type': 'icon_group'}
    
    def _merge_nearby_regions(self, regions: List[Dict]) -> List[Dict]:
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = [False] * len(regions)
        
        for i, r1 in enumerate(regions):
            if used[i]:
                continue
            
            current = r1.copy()
            
            for j, r2 in enumerate(regions):
                if i == j or used[j]:
                    continue
                
                # 겹치거나 인접한 영역 병합
                if self._rects_close(current, r2, margin=20):
                    current = self._merge_two(current, r2)
                    used[j] = True
            
            merged.append(current)
            used[i] = True
        
        return merged
    
    def _rects_close(self, r1: Dict, r2: Dict, margin: int) -> bool:
        x1_1 = r1['x'] - margin
        y1_1 = r1['y'] - margin
        x2_1 = r1['x'] + r1['width'] + margin
        y2_1 = r1['y'] + r1['height'] + margin
        
        x1_2 = r2['x']
        y1_2 = r2['y']
        x2_2 = r2['x'] + r2['width']
        y2_2 = r2['y'] + r2['height']
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _merge_two(self, r1: Dict, r2: Dict) -> Dict:
        x = min(r1['x'], r2['x'])
        y = min(r1['y'], r2['y'])
        x2 = max(r1['x'] + r1['width'], r2['x'] + r2['width'])
        y2 = max(r1['y'] + r1['height'], r2['y'] + r2['height'])
        return {'x': x, 'y': y, 'width': x2 - x, 'height': y2 - y, 'type': 'merged'}
    
    def is_protected(self, bounds: Dict[str, int]) -> bool:
        """영역의 중심점이 보호 영역 안에 있는지 확인"""
        if self.mask is None:
            return False
        
        # 중심점 계산
        cx = bounds['x'] + bounds['width'] // 2
        cy = bounds['y'] + bounds['height'] // 2
        
        # 경계 체크
        if cy < 0 or cy >= self.mask.shape[0] or cx < 0 or cx >= self.mask.shape[1]:
            return False
        
        return self.mask[cy, cx] > 0
    
    def clip_bounds_at_protection(self, bounds: Dict[str, int]) -> Dict[str, int]:
        """보호 영역에서 잘라내기"""
        if self.mask is None:
            return bounds
        
        x, y, w, h = bounds['x'], bounds['y'], bounds['width'], bounds['height']
        
        # 오른쪽에서 보호 영역 찾기
        for check_x in range(x + w // 2, min(x + w, self.mask.shape[1])):
            y1 = max(0, y)
            y2 = min(self.mask.shape[0], y + h)
            col = self.mask[y1:y2, check_x]
            
            # 50% 이상이 보호 영역이면 여기서 자르기
            if np.sum(col > 0) > len(col) * 0.5:
                new_w = check_x - x - 3
                if new_w > 20:
                    return {'x': x, 'y': y, 'width': new_w, 'height': h}
                break
        
        return bounds


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


def merge_regions_by_row(regions: List[TextRegion], protector: GraphicProtector = None) -> List[TextRegion]:
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
                # 두 영역 사이에 보호 영역이 있으면 병합 안함
                should_merge = True
                
                if protector is not None:
                    left = current if current.bounds['x'] < r.bounds['x'] else r
                    right = r if current.bounds['x'] < r.bounds['x'] else current
                    gap_start = left.bounds['x'] + left.bounds['width']
                    gap_end = right.bounds['x']
                    
                    if gap_end > gap_start + 20:
                        # 간격 중간 지점 체크
                        mid_x = (gap_start + gap_end) // 2
                        mid_y = (left.bounds['y'] + right.bounds['y']) // 2
                        test_bounds = {'x': mid_x - 5, 'y': mid_y, 'width': 10, 'height': 10}
                        if protector.is_protected(test_bounds):
                            should_merge = False
                
                if should_merge:
                    row_group.append(r)
                else:
                    others.append(r)
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
        
        new_bounds = {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}
        
        # 보호 영역에서 클리핑
        if protector is not None:
            new_bounds = protector.clip_bounds_at_protection(new_bounds)
        
        new_region = TextRegion(
            id="merged",
            text=full_text,
            confidence=avg_conf,
            bounds=new_bounds,
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


def expand_horizontal_bounds(image: np.ndarray, regions: List[TextRegion], protector: GraphicProtector = None, max_lookahead: int = 400) -> List[TextRegion]:
    if not regions: return []
    
    binary = get_content_mask(image)
    img_h, img_w = binary.shape
    
    for r in regions:
        if r.is_inverted: continue
        
        ref_h = r.bounds['height']
        start_x = r.bounds['x'] + r.bounds['width']
        
        y_margin = int(ref_h * 0.2)
        roi_y1 = max(0, r.bounds['y'] - y_margin)
        roi_y2 = min(img_h, r.bounds['y'] + r.bounds['height'] + y_margin)
        roi_x1 = start_x
        roi_x2 = min(img_w, start_x + max_lookahead)
        
        if roi_x2 <= roi_x1: continue
        
        roi = binary[roi_y1:roi_y2, roi_x1:roi_x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blob_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 3 or h < 3: continue
            blob_list.append((x, y, w, h))
        
        blob_list.sort(key=lambda b: b[0])
        current_extension = 0
        
        for bx, by, bw, bh in blob_list:
            abs_x = roi_x1 + bx
            abs_y = roi_y1 + by
            
            # 보호 영역 체크
            if protector is not None:
                test_bounds = {'x': abs_x, 'y': abs_y, 'width': bw, 'height': bh}
                if protector.is_protected(test_bounds):
                    break
            
            gap = abs_x - (r.bounds['x'] + r.bounds['width'] + current_extension)
            if gap > 100: break
            
            if bh > (ref_h * 1.5): break
            
            blob_center_y = roi_y1 + by + bh/2
            line_center_y = r.bounds['y'] + r.bounds['height']/2
            if abs(blob_center_y - line_center_y) > (ref_h * 0.6): break
            
            if bw > (ref_h * 3.0): break
            
            new_right_edge = abs_x + bw
            current_w = new_right_edge - r.bounds['x']
            if current_w > r.bounds['width']:
                r.bounds['width'] = current_w
                current_extension = current_w - (start_x - r.bounds['x'])
    
    # 최종 클리핑
    if protector is not None:
        for r in regions:
            r.bounds = protector.clip_bounds_at_protection(r.bounds)
    
    return regions


def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """메인 OCR 함수 - 균형 잡힌 그래픽 보호 적용"""
    
    # 1. 그래픽 보호 영역 감지 (아이콘 그룹 + 큰 일러스트만)
    protector = GraphicProtector()
    protector.detect_and_protect(image, padding=8)
    
    # 2. OCR 수행
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    normal_regions = ocr_engine.extract_text_regions(image)
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    # 3. 영역 병합 (보호 영역 사이는 병합 안함)
    all_raw_regions = normal_regions + inverted_regions
    merged_regions = merge_regions_by_row(all_raw_regions, protector)
    
    # 4. 수직 경계 정제
    vertical_refined = refine_vertical_boundaries_by_projection(image, merged_regions)
    
    # 5. 수평 확장 (보호 영역에서 멈춤)
    final_regions = expand_horizontal_bounds(image, vertical_refined, protector)
    
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
