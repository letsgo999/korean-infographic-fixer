"""
OCR Engine Module
텍스트 추출 및 좌표 인식을 담당하는 핵심 모듈 (Final Fix)
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import math

@dataclass
class TextRegion:
    """텍스트 영역 데이터 클래스"""
    id: str
    text: str
    confidence: float
    bounds: Dict[str, int]  # x, y, width, height
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
    """OCR 엔진 클래스"""
    
    def __init__(self, lang: str = "kor+eng", min_confidence: int = 30):
        self.lang = lang
        self.min_confidence = min_confidence
        
    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """이미지에서 텍스트 영역 추출 (PSM 6 적용)"""
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        
        # 뭉쳐서 읽기 위해 PSM 6 사용
        custom_config = r'--oem 3 --psm 6'
        
        try:
            ocr_data = pytesseract.image_to_data(
                pil_image, 
                lang=self.lang, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
        except:
            ocr_data = pytesseract.image_to_data(
                pil_image, 
                lang=self.lang, 
                output_type=pytesseract.Output.DICT
            )
        
        regions = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if text and conf >= self.min_confidence:
                region = TextRegion(
                    id=f"ocr_{len(regions):03d}",
                    text=text,
                    confidence=conf,
                    bounds={
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    },
                    block_num=ocr_data['block_num'][i],
                    line_num=ocr_data['line_num'][i],
                    is_inverted=False
                )
                regions.append(region)
                
        return regions
    
    def extract_from_inverted_region(
        self, 
        image: np.ndarray, 
        region_bounds: Dict[str, int],
        padding: int = 5
    ) -> List[TextRegion]:
        """역상(반전) 영역에서 텍스트 추출"""
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
        
        # 한 줄 텍스트로 가정 (PSM 7)
        custom_config = r'--oem 3 --psm 7'
        
        try:
            ocr_data = pytesseract.image_to_data(
                pil_roi,
                lang=self.lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
        except:
            ocr_data = pytesseract.image_to_data(
                pil_roi,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )
        
        regions = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if text and conf >= self.min_confidence:
                region = TextRegion(
                    id=f"inv_{len(regions):03d}",
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
    """역상 텍스트 영역 감지 클래스"""
    
    def __init__(
        self,
        dark_threshold: int = 150,
        min_area: int = 1000,
        min_width: int = 40,
        min_height: int = 15
    ):
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


def smart_merge_regions(regions: List[TextRegion], x_tolerance: int = 20, y_tolerance: int = 15) -> List[TextRegion]:
    """물리적 거리가 가까운 텍스트 영역을 하나로 병합"""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: (r.bounds['y'] // y_tolerance, r.bounds['x']))
    
    merged = []
    current = sorted_regions[0]
    
    for next_reg in sorted_regions[1:]:
        curr_b = current.bounds
        next_b = next_reg.bounds
        
        y_diff = abs(curr_b['y'] - next_b['y'])
        h_diff = abs(curr_b['height'] - next_b['height'])
        is_same_line = y_diff < y_tolerance and h_diff < (max(curr_b['height'], next_b['height']) * 0.5)
        
        x_dist = next_b['x'] - (curr_b['x'] + curr_b['width'])
        is_close_x = -10 < x_dist < x_tolerance
        
        is_same_type = current.is_inverted == next_reg.is_inverted

        if is_same_line and is_close_x and is_same_type:
            new_x = min(curr_b['x'], next_b['x'])
            new_y = min(curr_b['y'], next_b['y'])
            new_w = max(curr_b['x'] + curr_b['width'], next_b['x'] + next_b['width']) - new_x
            new_h = max(curr_b['y'] + curr_b['height'], next_b['y'] + next_b['height']) - new_y
            
            combined_text = f"{current.text} {next_reg.text}".strip()
            new_conf = (current.confidence + next_reg.confidence) / 2
            
            current = TextRegion(
                id=current.id,
                text=combined_text,
                confidence=new_conf,
                bounds={'x': new_x, 'y': new_y, 'width': new_w, 'height': new_h},
                is_inverted=current.is_inverted,
                word_count=current.word_count + next_reg.word_count
            )
        else:
            merged.append(current)
            current = next_reg
            
    merged.append(current)
    
    for i, r in enumerate(merged):
        prefix = "inv" if r.is_inverted else "ocr"
        r.id = f"{prefix}_{i:03d}"
        
    return merged


def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """향상된 OCR 파이프라인 실행"""
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    normal_regions = ocr_engine.extract_text_regions(image)
    
    dark_regions = inv_detector.detect(image)
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    all_raw_regions = normal_regions + inverted_regions
    
    # 여기서 이미 강력한 병합을 수행합니다.
    merged_regions = smart_merge_regions(all_raw_regions)
    
    final_normal = [r for r in merged_regions if not r.is_inverted]
    final_inverted = [r for r in merged_regions if r.is_inverted]
    
    return {
        'normal_regions': final_normal,
        'inverted_regions': final_inverted,
        'all_regions': merged_regions,
        'image_info': {
            'width': image.shape[1],
            'height': image.shape[0]
        }
    }


def group_regions_by_lines(regions: List[TextRegion]) -> List[TextRegion]:
    """
    [호환성 유지 함수]
    app.py에서 호출하는 함수입니다.
    run_enhanced_ocr에서 이미 smart_merge_regions가 수행되었으므로,
    여기서는 별도의 작업 없이 입력받은 리스트를 그대로 반환합니다.
    """
    return regions
