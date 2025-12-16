"""
OCR Engine Module
텍스트 추출 및 좌표 인식을 담당하는 핵심 모듈 (Noise Filter & Line Force Merge Version)
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
    """텍스트 영역 데이터 클래스"""
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
    def __init__(self, lang: str = "kor+eng", min_confidence: int = 50): # 신뢰도 기준 50으로 상향
        self.lang = lang
        self.min_confidence = min_confidence
        
    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        
        # psm 6: 단일 텍스트 블록으로 간주 (파편화 최소화 시도)
        custom_config = r'--oem 3 --psm 6'
        
        try:
            ocr_data = pytesseract.image_to_data(
                pil_image, 
                lang=self.lang, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
        except:
            return []
        
        regions = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            # [노이즈 필터 1] 신뢰도 체크 및 유효 텍스트 검증
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
        custom_config = r'--oem 3 --psm 7' # 한 줄로 인식 유도
        
        try:
            ocr_data = pytesseract.image_to_data(
                pil_roi,
                lang=self.lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
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
    """
    [핵심 필터] 그래픽 노이즈(아이콘 등)를 걸러냅니다.
    1. 빈 문자열 제외
    2. 특수문자로만 구성된 경우 제외
    3. 길이가 1인데 한글/숫자가 아닌 경우(영어 낱글자 등) 제외
    """
    t = text.strip()
    if not t: return False
    
    # 1. 특수문자만 있는 경우 (예: "===", "---") 제거
    # 한글(가-힣), 영문(a-zA-Z), 숫자(0-9)가 하나라도 없으면 노이즈
    if not re.search(r'[가-힣a-zA-Z0-9]', t):
        return False
        
    # 2. 길이가 짧은데 영어/특수문자인 경우 (아이콘 오인)
    # 한글은 1글자라도 의미가 있을 수 있으나, 영어 낱글자(i, l 등)는 보통 노이즈
    if len(t) == 1 and re.match(r'[a-zA-Z\W]', t):
        # 예외: A, I 같은 대문자는 살릴 수도 있으나 인포그래픽에선 보통 단어임.
        # 안전하게 1글자 영문은 버림.
        return False
        
    return True

def merge_regions_by_row(regions: List[TextRegion], y_threshold: int = 15) -> List[TextRegion]:
    """
    [핵심 병합] Y좌표가 유사한 영역들을 하나의 '행(Row)'으로 강제 통합합니다.
    X좌표 거리가 멀어도 같은 줄에 있으면 합칩니다.
    """
    if not regions:
        return []
        
    # Y 좌표(Top) 기준으로 정렬
    regions.sort(key=lambda r: r.bounds['y'])
    
    merged_rows = []
    
    while regions:
        current = regions.pop(0)
        
        # 현재 영역의 중심 Y 좌표
        current_cy = current.bounds['y'] + current.bounds['height'] // 2
        
        # 같은 행으로 볼 후보들 찾기
        row_group = [current]
        others = []
        
        for r in regions:
            r_cy = r.bounds['y'] + r.bounds['height'] // 2
            
            # 높이 차이가 크지 않으면 같은 줄로 간주
            # 기준: 두 영역 중 더 큰 높이의 50% 이내로 중심이 차이나면 같은 줄
            height_ref = max(current.bounds['height'], r.bounds['height'])
            
            if abs(current_cy - r_cy) < (height_ref * 0.5):
                row_group.append(r)
            else:
                others.append(r)
                
        regions = others # 처리 안 된 것들만 남김
        
        # 찾은 row_group을 하나로 합치기
        # X 좌표 순으로 정렬
        row_group.sort(key=lambda r: r.bounds['x'])
        
        # 좌표 계산
        min_x = min(r.bounds['x'] for r in row_group)
        min_y = min(r.bounds['y'] for r in row_group)
        max_x = max(r.bounds['x'] + r.bounds['width'] for r in row_group)
        max_y = max(r.bounds['y'] + r.bounds['height'] for r in row_group)
        
        # 텍스트 합치기
        full_text = " ".join([r.text for r in row_group])
        
        # 평균 신뢰도
        avg_conf = sum(r.confidence for r in row_group) / len(row_group)
        
        new_region = TextRegion(
            id="merged", # 나중에 재할당
            text=full_text,
            confidence=avg_conf,
            bounds={
                'x': min_x, 
                'y': min_y, 
                'width': max_x - min_x, 
                'height': max_y - min_y
            },
            is_inverted=row_group[0].is_inverted
        )
        merged_rows.append(new_region)
        
    # ID 재할당 및 정렬
    merged_rows.sort(key=lambda r: (r.bounds['y'], r.bounds['x']))
    for i, r in enumerate(merged_rows):
        prefix = "inv" if r.is_inverted else "ocr"
        r.id = f"{prefix}_{i:03d}"
        
    return merged_rows

def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """
    향상된 OCR 파이프라인 (노이즈 제거 -> 행 강제 통합)
    """
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    # 1. 추출 (여기서 이미 is_valid_text 필터 작동)
    normal_regions = ocr_engine.extract_text_regions(image)
    
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    all_raw_regions = normal_regions + inverted_regions
    
    # 2. 행 단위 강제 통합 (Fragmentation 해결)
    final_regions = merge_regions_by_row(all_raw_regions)
    
    final_normal = [r for r in final_regions if not r.is_inverted]
    final_inverted = [r for r in final_regions if r.is_inverted]
    
    return {
        'normal_regions': final_normal,
        'inverted_regions': final_inverted,
        'all_regions': final_regions,
        'image_info': {
            'width': image.shape[1],
            'height': image.shape[0]
        }
    }

def group_regions_by_lines(regions: List[TextRegion]) -> List[TextRegion]:
    """호환성 유지 함수"""
    return regions
