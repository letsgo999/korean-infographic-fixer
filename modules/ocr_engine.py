"""
OCR Engine Module
[Final Innovation + Graphic Protection v2] 
1. 엣지 기반 감지
2. 픽셀 투영 행 분리
3. 컨투어(객체) 분석 기반 지능형 확장
4. [ENHANCED] 그래픽/아이콘 영역 보호 - 더 민감한 감지 + 사전 필터링
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


# ============================================================
# [ENHANCED] 그래픽 영역 보호 시스템 v2
# ============================================================

class GraphicProtector:
    """
    그래픽(아이콘/이미지/일러스트) 영역을 감지하고 보호하는 통합 클래스
    v2: 더 민감한 감지 + 피부색/파스텔톤 포함
    """
    
    def __init__(
        self,
        min_icon_size: int = 12,          # 더 작은 아이콘도 감지
        max_icon_size: int = 120,         # 더 큰 아이콘도 감지
        color_diversity_threshold: float = 6.0,  # 더 낮은 임계값
        saturation_threshold: int = 25,    # 더 낮은 채도도 감지
        icon_gap_threshold: int = 50,
        min_icons_for_group: int = 2,
        skin_tone_detection: bool = True   # 피부색 감지 활성화
    ):
        self.min_icon_size = min_icon_size
        self.max_icon_size = max_icon_size
        self.color_diversity_threshold = color_diversity_threshold
        self.saturation_threshold = saturation_threshold
        self.icon_gap_threshold = icon_gap_threshold
        self.min_icons_for_group = min_icons_for_group
        self.skin_tone_detection = skin_tone_detection
        
        self.protected_regions: List[Dict] = []
        self.mask: np.ndarray = None
    
    def detect_and_protect(self, image: np.ndarray, padding: int = 12) -> np.ndarray:
        """
        이미지에서 그래픽 영역을 감지하고 보호 마스크 생성
        """
        h, w = image.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.protected_regions = []
        
        # 1. 색상 다양성 기반 그래픽 감지
        color_regions = self._detect_colorful_regions(image)
        
        # 2. 아이콘 그룹 감지
        icon_groups = self._detect_icon_groups(image)
        
        # 3. 일러스트/캐릭터 영역 감지 (피부색 포함)
        illustration_regions = self._detect_illustrations(image)
        
        # 4. [NEW] 피부색 영역 감지 (사람 일러스트)
        if self.skin_tone_detection:
            skin_regions = self._detect_skin_tone_regions(image)
        else:
            skin_regions = []
        
        # 5. [NEW] 비-텍스트 컬러 블롭 감지
        color_blob_regions = self._detect_color_blobs(image)
        
        # 모든 영역 합치기
        all_regions = color_regions + icon_groups + illustration_regions + skin_regions + color_blob_regions
        
        # 중복 제거 및 병합
        merged = self._merge_overlapping_regions(all_regions)
        self.protected_regions = merged
        
        # 마스크에 그리기
        for region in merged:
            x = max(0, region['x'] - padding)
            y = max(0, region['y'] - padding)
            x2 = min(w, region['x'] + region['width'] + padding)
            y2 = min(h, region['y'] + region['height'] + padding)
            self.mask[y:y2, x:x2] = 255
        
        return self.mask
    
    def _detect_colorful_regions(self, image: np.ndarray) -> List[Dict]:
        """채도가 있고 색상이 다양한 영역 감지"""
        regions = []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # 낮은 임계값으로 더 많은 영역 감지
        _, sat_mask = cv2.threshold(saturation, self.saturation_threshold, 255, cv2.THRESH_BINARY)
        
        # 모폴로지로 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            if w < 10 or h < 10 or area < 150:
                continue
            
            # 색상 다양성 계산
            roi = image[y:y+h, x:x+w]
            diversity = self._calculate_color_diversity(roi)
            
            if diversity >= self.color_diversity_threshold:
                regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'type': 'colorful', 'score': diversity
                })
        
        return regions
    
    def _detect_skin_tone_regions(self, image: np.ndarray) -> List[Dict]:
        """피부색 영역 감지 (사람 일러스트용)"""
        regions = []
        
        # YCrCb 색공간에서 피부색 감지 (더 정확함)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # 피부색 범위 (다양한 피부톤 포함)
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # HSV에서도 피부색 감지 (보완)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
        
        # 두 마스크 합치기
        combined_mask = cv2.bitwise_or(skin_mask, skin_mask_hsv)
        
        # 모폴로지 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # 피부색 영역은 보통 어느 정도 크기가 있음
            if area < 500 or w < 15 or h < 15:
                continue
            
            # 종횡비 체크 (너무 길쭉하면 텍스트일 수 있음)
            aspect = w / h if h > 0 else 0
            if aspect > 8 or aspect < 0.1:
                continue
            
            regions.append({
                'x': x, 'y': y, 'width': w, 'height': h,
                'type': 'skin_tone', 'score': 70
            })
        
        return regions
    
    def _detect_color_blobs(self, image: np.ndarray) -> List[Dict]:
        """특정 색상 블롭 감지 (빨강, 초록, 파랑, 노랑 등 주요 색상)"""
        regions = []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 주요 색상 범위 정의
        color_ranges = [
            # 빨간색 (두 범위)
            ((0, 70, 50), (10, 255, 255)),
            ((170, 70, 50), (180, 255, 255)),
            # 주황색
            ((10, 70, 50), (25, 255, 255)),
            # 노란색
            ((25, 70, 50), (35, 255, 255)),
            # 초록색
            ((35, 70, 50), (85, 255, 255)),
            # 파란색
            ((85, 70, 50), (130, 255, 255)),
            # 보라색
            ((130, 70, 50), (170, 255, 255)),
        ]
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 모폴로지
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # 아이콘 크기 범위
            if not (self.min_icon_size <= w <= self.max_icon_size * 2 and 
                    self.min_icon_size <= h <= self.max_icon_size * 2):
                continue
            
            if area < 200:
                continue
            
            # 어느 정도 밀도가 있어야 함 (빈 공간이 아니라 실제 색이 채워진 영역)
            roi_mask = combined_mask[y:y+h, x:x+w]
            fill_ratio = np.sum(roi_mask > 0) / area if area > 0 else 0
            
            if fill_ratio > 0.2:  # 20% 이상 채워져 있으면
                regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'type': 'color_blob', 'score': fill_ratio * 100
                })
        
        return regions
    
    def _detect_icon_groups(self, image: np.ndarray) -> List[Dict]:
        """수평으로 배열된 아이콘 그룹 감지"""
        groups = []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        _, sat_mask = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if not (self.min_icon_size <= w <= self.max_icon_size and 
                    self.min_icon_size <= h <= self.max_icon_size):
                continue
            
            aspect = w / h if h > 0 else 0
            if 0.3 <= aspect <= 3.0:
                candidates.append({'x': x, 'y': y, 'width': w, 'height': h})
        
        if len(candidates) < self.min_icons_for_group:
            return groups
        
        candidates.sort(key=lambda c: (c['y'] // 30, c['x']))
        
        current_group = []
        for cand in candidates:
            if not current_group:
                current_group.append(cand)
                continue
            
            last = current_group[-1]
            same_row = abs(cand['y'] - last['y']) < 30
            close = (cand['x'] - (last['x'] + last['width'])) < self.icon_gap_threshold
            
            if same_row and close:
                current_group.append(cand)
            else:
                if len(current_group) >= self.min_icons_for_group:
                    groups.append(self._merge_group(current_group))
                current_group = [cand]
        
        if len(current_group) >= self.min_icons_for_group:
            groups.append(self._merge_group(current_group))
        
        return groups
    
    def _detect_illustrations(self, image: np.ndarray) -> List[Dict]:
        """일러스트/캐릭터 영역 감지"""
        regions = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 80)  # 더 민감한 엣지 감지
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            if w < 30 or h < 30 or area < 1500:
                continue
            
            aspect = w / h if h > 0 else 0
            if aspect > 6 or aspect < 0.15:
                continue
            
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / area if area > 0 else 0
            
            roi = image[y:y+h, x:x+w]
            diversity = self._calculate_color_diversity(roi)
            
            # 조건 완화: 엣지 밀도 또는 색상 다양성 중 하나만 만족해도 됨
            if edge_density > 0.04 or diversity > 5:
                regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'type': 'illustration', 'score': edge_density * 100 + diversity
                })
        
        return regions
    
    def _calculate_color_diversity(self, roi: np.ndarray) -> float:
        """색상 다양성 점수 계산"""
        if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
            return 0
        
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].flatten()
            sat = hsv[:, :, 1].flatten()
            
            colored_mask = sat > 15  # 더 낮은 채도도 포함
            if np.sum(colored_mask) < 5:
                return 0
            
            colored_hue = hue[colored_mask]
            hist, _ = np.histogram(colored_hue, bins=18, range=(0, 180))
            
            threshold = max(1, len(colored_hue) * 0.005)  # 더 낮은 임계값
            used_colors = np.sum(hist > threshold)
            
            return used_colors * 3.0
        except:
            return 0
    
    def _merge_group(self, icons: List[Dict]) -> Dict:
        """아이콘 그룹을 하나의 영역으로 병합"""
        x = min(i['x'] for i in icons)
        y = min(i['y'] for i in icons)
        x2 = max(i['x'] + i['width'] for i in icons)
        y2 = max(i['y'] + i['height'] for i in icons)
        return {'x': x, 'y': y, 'width': x2 - x, 'height': y2 - y, 'type': 'icon_group', 'score': 90}
    
    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """겹치는 영역 병합"""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = [False] * len(regions)
        
        for i, r1 in enumerate(regions):
            if used[i]:
                continue
            
            current = r1.copy()
            changed = True
            
            while changed:
                changed = False
                for j, r2 in enumerate(regions):
                    if used[j] or i == j:
                        continue
                    
                    if self._rects_overlap(current, r2, margin=10):
                        current = self._merge_rects(current, r2)
                        used[j] = True
                        changed = True
            
            merged.append(current)
            used[i] = True
        
        return merged
    
    def _rects_overlap(self, r1: Dict, r2: Dict, margin: int = 0) -> bool:
        """두 사각형이 겹치는지 확인"""
        x1_1 = r1['x'] - margin
        y1_1 = r1['y'] - margin
        x2_1 = r1['x'] + r1['width'] + margin
        y2_1 = r1['y'] + r1['height'] + margin
        
        x1_2 = r2['x']
        y1_2 = r2['y']
        x2_2 = r2['x'] + r2['width']
        y2_2 = r2['y'] + r2['height']
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _merge_rects(self, r1: Dict, r2: Dict) -> Dict:
        """두 사각형 병합"""
        x = min(r1['x'], r2['x'])
        y = min(r1['y'], r2['y'])
        x2 = max(r1['x'] + r1['width'], r2['x'] + r2['width'])
        y2 = max(r1['y'] + r1['height'], r2['y'] + r2['height'])
        return {'x': x, 'y': y, 'width': x2 - x, 'height': y2 - y, 'type': 'merged', 'score': max(r1.get('score', 0), r2.get('score', 0))}
    
    def is_protected(self, bounds: Dict[str, int], threshold: float = 0.1) -> bool:
        """주어진 bounds가 보호 영역과 겹치는지 확인"""
        if self.mask is None:
            return False
        
        x, y = bounds['x'], bounds['y']
        w, h = bounds['width'], bounds['height']
        
        y1 = max(0, y)
        y2 = min(self.mask.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(self.mask.shape[1], x + w)
        
        if y2 <= y1 or x2 <= x1:
            return False
        
        roi = self.mask[y1:y2, x1:x2]
        overlap_ratio = np.sum(roi > 0) / roi.size if roi.size > 0 else 0
        
        # 10% 이상 겹치면 보호 영역과 충돌
        return overlap_ratio > threshold
    
    def get_overlap_ratio(self, bounds: Dict[str, int]) -> float:
        """보호 영역과의 겹침 비율 반환"""
        if self.mask is None:
            return 0.0
        
        x, y = bounds['x'], bounds['y']
        w, h = bounds['width'], bounds['height']
        
        y1 = max(0, y)
        y2 = min(self.mask.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(self.mask.shape[1], x + w)
        
        if y2 <= y1 or x2 <= x1:
            return 0.0
        
        roi = self.mask[y1:y2, x1:x2]
        return np.sum(roi > 0) / roi.size if roi.size > 0 else 0.0
    
    def clip_to_safe_bounds(self, bounds: Dict[str, int]) -> Dict[str, int]:
        """보호 영역을 피하도록 bounds 조정"""
        if self.mask is None:
            return bounds
        
        x, y = bounds['x'], bounds['y']
        w, h = bounds['width'], bounds['height']
        
        # 오른쪽에서 보호 영역 찾기
        for check_x in range(x, min(x + w, self.mask.shape[1])):
            y1 = max(0, y)
            y2 = min(self.mask.shape[0], y + h)
            col = self.mask[y1:y2, check_x]
            
            # 해당 열에서 보호 영역 픽셀이 일정 비율 이상이면
            if np.sum(col > 0) > len(col) * 0.3:
                new_w = check_x - x - 5
                if new_w > 15:
                    return {'x': x, 'y': y, 'width': new_w, 'height': h}
                break
        
        return bounds


# ============================================================
# 기존 클래스들
# ============================================================

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


# ============================================================
# 유틸리티 함수들
# ============================================================

def is_valid_text(text: str) -> bool:
    t = text.strip()
    if not t: return False
    if not re.search(r'[가-힣a-zA-Z0-9]', t): return False
    if len(t) == 1 and re.match(r'[a-zA-Z\W]', t): return False
    return True


def filter_regions_by_protection(regions: List[TextRegion], protector: GraphicProtector, max_overlap: float = 0.15) -> List[TextRegion]:
    """
    [NEW] 보호 영역과 많이 겹치는 OCR 결과를 사전 필터링
    """
    filtered = []
    for r in regions:
        overlap = protector.get_overlap_ratio(r.bounds)
        if overlap < max_overlap:
            filtered.append(r)
    return filtered


def merge_regions_by_row(regions: List[TextRegion], protector: GraphicProtector = None) -> List[TextRegion]:
    """
    [ENHANCED] 행 병합 시 보호 영역 고려
    """
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
            
            # 같은 행인지 확인
            if abs(current_cy - r_cy) < (height_ref * 0.5):
                # [NEW] 두 영역 사이에 보호 영역이 있는지 확인
                if protector is not None:
                    # 두 영역 사이의 간격 영역 체크
                    left_r = current if current.bounds['x'] < r.bounds['x'] else r
                    right_r = r if current.bounds['x'] < r.bounds['x'] else current
                    
                    gap_x = left_r.bounds['x'] + left_r.bounds['width']
                    gap_width = right_r.bounds['x'] - gap_x
                    
                    if gap_width > 10:  # 간격이 있는 경우만 체크
                        gap_bounds = {
                            'x': gap_x,
                            'y': min(left_r.bounds['y'], right_r.bounds['y']),
                            'width': gap_width,
                            'height': max(left_r.bounds['height'], right_r.bounds['height'])
                        }
                        
                        # 간격 영역에 보호 영역이 있으면 병합하지 않음
                        if protector.is_protected(gap_bounds, threshold=0.2):
                            others.append(r)
                            continue
                
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
        
        # [NEW] 병합된 영역이 보호 영역을 침범하면 클리핑
        if protector is not None:
            new_region.bounds = protector.clip_to_safe_bounds(new_region.bounds)
        
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


def expand_horizontal_bounds(
    image: np.ndarray, 
    regions: List[TextRegion], 
    protector: GraphicProtector = None,
    max_lookahead: int = 400
) -> List[TextRegion]:
    """
    [ENHANCED] 보호 영역을 피하면서 컨투어 기반 수평 확장
    """
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
        
        # [NEW] 먼저 오른쪽에 보호 영역이 있는지 확인
        if protector is not None:
            lookahead_bounds = {
                'x': roi_x1,
                'y': r.bounds['y'],
                'width': min(50, roi_x2 - roi_x1),  # 50px만 먼저 체크
                'height': r.bounds['height']
            }
            if protector.is_protected(lookahead_bounds, threshold=0.1):
                continue  # 바로 옆에 보호 영역이 있으면 확장 안함
        
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
                potential_bounds = {'x': abs_x, 'y': abs_y, 'width': bw, 'height': bh}
                if protector.is_protected(potential_bounds, threshold=0.05):
                    break
            
            gap = abs_x - (r.bounds['x'] + r.bounds['width'] + current_extension)
            if gap > 80: break  # 간격 임계값 축소
            
            if bh > (ref_h * 1.3): break  # 높이 임계값 더 엄격하게
            
            blob_center_y = roi_y1 + by + bh/2
            line_center_y = r.bounds['y'] + r.bounds['height']/2
            if abs(blob_center_y - line_center_y) > (ref_h * 0.5): break
            
            if bw > (ref_h * 2.5): break  # 너비 임계값 더 엄격하게
            
            new_right_edge = abs_x + bw
            if protector is not None:
                new_bounds = {
                    'x': r.bounds['x'],
                    'y': r.bounds['y'],
                    'width': new_right_edge - r.bounds['x'],
                    'height': r.bounds['height']
                }
                if protector.is_protected(new_bounds, threshold=0.05):
                    break
            
            current_w = new_right_edge - r.bounds['x']
            if current_w > r.bounds['width']:
                r.bounds['width'] = current_w
                current_extension = current_w - (start_x - r.bounds['x'])
    
    # 최종 클리핑
    if protector is not None:
        for r in regions:
            r.bounds = protector.clip_to_safe_bounds(r.bounds)
    
    return regions


# ============================================================
# 메인 함수
# ============================================================

def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """
    [ENHANCED v2] 그래픽 영역을 먼저 감지하고, OCR 결과를 필터링한 후 처리
    """
    # 1. 그래픽 보호 영역 감지 (더 민감한 설정)
    protector = GraphicProtector(
        min_icon_size=12,
        max_icon_size=120,
        color_diversity_threshold=5.0,
        saturation_threshold=20,
        skin_tone_detection=True
    )
    protector.detect_and_protect(image, padding=15)
    
    # 2. OCR 수행
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    normal_regions = ocr_engine.extract_text_regions(image)
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    # 3. [NEW] 보호 영역과 겹치는 OCR 결과 사전 필터링
    normal_regions = filter_regions_by_protection(normal_regions, protector, max_overlap=0.15)
    inverted_regions = filter_regions_by_protection(inverted_regions, protector, max_overlap=0.15)
    
    # 4. 영역 병합 (보호 영역 고려)
    all_raw_regions = normal_regions + inverted_regions
    merged_regions = merge_regions_by_row(all_raw_regions, protector)
    
    # 5. 수직 경계 정제
    vertical_refined = refine_vertical_boundaries_by_projection(image, merged_regions)
    
    # 6. 수평 확장 (보호 영역 피함)
    final_regions = expand_horizontal_bounds(image, vertical_refined, protector)
    
    # 7. 최종 클리핑
    for r in final_regions:
        r.bounds = protector.clip_to_safe_bounds(r.bounds)
    
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
