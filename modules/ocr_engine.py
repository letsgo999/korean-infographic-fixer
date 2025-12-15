"""
OCR Engine Module
텍스트 추출 및 좌표 인식을 담당하는 핵심 모듈
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json

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
    is_manual: bool = False  # 수동 추가 여부
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
        """
        이미지에서 텍스트 영역 추출
        
        Args:
            image: OpenCV 이미지 (BGR)
            
        Returns:
            TextRegion 리스트
        """
        # BGR -> RGB 변환
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        
        # OCR 수행
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
        """
        역상(반전) 영역에서 텍스트 추출
        
        Args:
            image: 원본 이미지
            region_bounds: 역상 영역 좌표
            padding: 경계 여유 픽셀
            
        Returns:
            TextRegion 리스트
        """
        x, y = region_bounds['x'], region_bounds['y']
        w, h = region_bounds['width'], region_bounds['height']
        
        # 패딩 적용
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # 영역 추출 및 반전
        roi = image[y1:y2, x1:x2].copy()
        inverted_roi = cv2.bitwise_not(roi)
        
        # BGR -> RGB
        if len(inverted_roi.shape) == 3:
            inverted_rgb = cv2.cvtColor(inverted_roi, cv2.COLOR_BGR2RGB)
        else:
            inverted_rgb = inverted_roi
            
        pil_roi = Image.fromarray(inverted_rgb)
        
        # OCR 수행
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
                # 좌표를 원본 이미지 기준으로 변환
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
        min_width: int = 50,
        min_height: int = 15
    ):
        self.dark_threshold = dark_threshold
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        
    def detect(self, image: np.ndarray) -> List[Dict[str, int]]:
        """
        역상 텍스트가 있을 수 있는 어두운/색상 배경 영역 감지
        
        Args:
            image: OpenCV 이미지 (BGR)
            
        Returns:
            영역 좌표 리스트 [{'x', 'y', 'width', 'height'}, ...]
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # HSV 변환 (색상 배경 감지용)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 주황색 범위 마스크
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # 어두운 영역 마스크
        dark_mask = cv2.inRange(gray, 0, self.dark_threshold)
        
        # 마스크 합치기
        combined_mask = cv2.bitwise_or(orange_mask, dark_mask)
        
        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(
            combined_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if w >= self.min_width and h >= self.min_height and area >= self.min_area:
                regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h
                })
                
        return regions


def group_regions_by_lines(regions: List[TextRegion]) -> List[TextRegion]:
    """
    같은 라인의 텍스트 영역들을 그룹핑
    
    Args:
        regions: TextRegion 리스트
        
    Returns:
        라인 단위로 그룹핑된 TextRegion 리스트
    """
    if not regions:
        return []
    
    # block_num, line_num으로 그룹핑
    lines = {}
    
    for region in regions:
        key = (region.block_num, region.line_num)
        
        if key not in lines:
            lines[key] = {
                'texts': [],
                'bounds': [],
                'confidences': [],
                'is_inverted': region.is_inverted
            }
        
        lines[key]['texts'].append(region.text)
        lines[key]['bounds'].append(region.bounds)
        lines[key]['confidences'].append(region.confidence)
    
    # 라인별로 합치기
    line_regions = []
    for key, data in lines.items():
        if not data['bounds']:
            continue
            
        # 전체 바운딩 박스 계산
        min_x = min(b['x'] for b in data['bounds'])
        min_y = min(b['y'] for b in data['bounds'])
        max_x = max(b['x'] + b['width'] for b in data['bounds'])
        max_y = max(b['y'] + b['height'] for b in data['bounds'])
        
        line_region = TextRegion(
            id=f"line_{len(line_regions):03d}",
            text=' '.join(data['texts']),
            confidence=round(sum(data['confidences']) / len(data['confidences']), 1),
            bounds={
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            },
            word_count=len(data['texts']),
            is_inverted=data['is_inverted']
        )
        line_regions.append(line_region)
    
    # Y 좌표로 정렬
    line_regions.sort(key=lambda r: r.bounds['y'])
    
    return line_regions


def run_enhanced_ocr(image: np.ndarray) -> Dict:
    """
    향상된 OCR 파이프라인 실행 (일반 + 역상 영역)
    
    Args:
        image: OpenCV 이미지 (BGR)
        
    Returns:
        {
            'normal_regions': [...],
            'inverted_regions': [...],
            'all_regions': [...],
            'image_info': {...}
        }
    """
    ocr_engine = OCREngine()
    inv_detector = InvertedRegionDetector()
    
    # 1. 일반 OCR
    normal_regions = ocr_engine.extract_text_regions(image)
    
    # 2. 역상 영역 감지
    dark_regions = inv_detector.detect(image)
    
    # 3. 역상 영역 OCR
    inverted_regions = []
    for region_bounds in dark_regions:
        inv_texts = ocr_engine.extract_from_inverted_region(image, region_bounds)
        inverted_regions.extend(inv_texts)
    
    # 4. 결과 통합
    all_regions = normal_regions + inverted_regions
    
    return {
        'normal_regions': normal_regions,
        'inverted_regions': inverted_regions,
        'all_regions': all_regions,
        'image_info': {
            'width': image.shape[1],
            'height': image.shape[0]
        }
    }
