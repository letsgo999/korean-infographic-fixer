"""
Style Classifier Module
텍스트 스타일 자동 분류 및 색상 추출
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from .ocr_engine import TextRegion


class StyleClassifier:
    """텍스트 스타일 자동 분류기"""
    
    def __init__(self):
        self.style_thresholds = {}
        
    def classify(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        높이 기반으로 텍스트 스타일 자동 분류
        
        Args:
            regions: TextRegion 리스트
            
        Returns:
            스타일 태그가 추가된 TextRegion 리스트
        """
        if not regions:
            return regions
        
        # 높이 값 추출
        heights = [r.bounds['height'] for r in regions]
        
        # 통계 계산
        mean_height = np.mean(heights)
        std_height = np.std(heights) if len(heights) > 1 else 0
        
        # 분류 기준 설정
        title_threshold = mean_height + std_height * 0.8
        subtitle_threshold = mean_height + std_height * 0.3
        
        self.style_thresholds = {
            'title': title_threshold,
            'subtitle': subtitle_threshold,
            'mean': mean_height,
            'std': std_height
        }
        
        # 스타일 태그 할당
        for region in regions:
            h = region.bounds['height']
            
            if h >= title_threshold:
                region.style_tag = 'title'
                region.suggested_font_size = 32
            elif h >= subtitle_threshold:
                region.style_tag = 'subtitle'
                region.suggested_font_size = 24
            else:
                region.style_tag = 'body'
                region.suggested_font_size = 16
                
        return regions
    
    def get_thresholds(self) -> Dict:
        """현재 분류 기준 반환"""
        return self.style_thresholds


class ColorExtractor:
    """텍스트 영역 색상 추출기"""
    
    def extract_colors(
        self, 
        image: np.ndarray, 
        regions: List[TextRegion]
    ) -> List[TextRegion]:
        """
        텍스트 영역의 글자색과 배경색 추출
        
        Args:
            image: OpenCV 이미지 (BGR)
            regions: TextRegion 리스트
            
        Returns:
            색상 정보가 추가된 TextRegion 리스트
        """
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for region in regions:
            b = region.bounds
            
            # 영역 추출
            roi = image_rgb[b['y']:b['y']+b['height'], b['x']:b['x']+b['width']]
            
            if roi.size == 0:
                continue
            
            # 픽셀 색상 분석
            pixels = roi.reshape(-1, 3)
            
            # 밝기 기준으로 분류
            brightness = np.mean(pixels, axis=1)
            
            dark_pixels = pixels[brightness < 128]
            light_pixels = pixels[brightness >= 128]
            
            # 가장 많은 색상 추출
            if len(dark_pixels) > 0:
                text_color = np.median(dark_pixels, axis=0).astype(int)
            else:
                text_color = np.array([0, 0, 0])
                
            if len(light_pixels) > 0:
                bg_color = np.median(light_pixels, axis=0).astype(int)
            else:
                bg_color = np.array([255, 255, 255])
            
            # HEX 변환
            region.text_color = '#{:02x}{:02x}{:02x}'.format(*text_color)
            region.bg_color = '#{:02x}{:02x}{:02x}'.format(*bg_color)
        
        return regions
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """HEX 색상을 RGB 튜플로 변환"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """RGB 튜플을 HEX 색상으로 변환"""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    @staticmethod
    def get_contrast_color(hex_color: str) -> str:
        """배경색에 대비되는 텍스트 색상 반환"""
        r, g, b = ColorExtractor.hex_to_rgb(hex_color)
        # 밝기 계산 (YIQ 공식)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return "#000000" if brightness > 128 else "#FFFFFF"


def apply_styles_and_colors(
    image: np.ndarray, 
    regions: List[TextRegion]
) -> List[TextRegion]:
    """
    스타일 분류 및 색상 추출 적용
    
    Args:
        image: OpenCV 이미지 (BGR)
        regions: TextRegion 리스트
        
    Returns:
        스타일과 색상이 적용된 TextRegion 리스트
    """
    classifier = StyleClassifier()
    color_extractor = ColorExtractor()
    
    # 스타일 분류
    regions = classifier.classify(regions)
    
    # 색상 추출
    regions = color_extractor.extract_colors(image, regions)
    
    return regions
