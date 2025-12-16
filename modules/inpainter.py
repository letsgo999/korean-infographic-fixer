"""
Inpainter Module
텍스트 영역을 지우고 배경을 복원(Inpainting)하는 모듈
"""
import cv2
import numpy as np
from typing import List, Tuple, Union

class Inpainter:
    def __init__(self, method='telea'):
        self.method = method

    def hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Hex 문자열(#RRGGBB)을 OpenCV BGR 튜플로 변환"""
        if not isinstance(hex_color, str):
            return (255, 255, 255) # 기본값 흰색
            
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r) # OpenCV는 BGR 순서
        return (255, 255, 255)

    def remove_all_text_regions(self, image: np.ndarray, regions: List) -> np.ndarray:
        """지정된 모든 텍스트 영역을 지웁니다 (단순 색상 채우기)"""
        result = image.copy()
        for region in regions:
            self.remove_text_region(result, region)
        return result

    def remove_text_region(self, result: np.ndarray, region):
        """단일 영역 지우기"""
        # 객체 또는 딕셔너리 호환 처리
        if isinstance(region, dict):
            bounds = region['bounds']
            bg_color = region.get('bg_color', '#FFFFFF')
        else:
            bounds = region.bounds
            bg_color = getattr(region, 'bg_color', '#FFFFFF')
        
        x = int(bounds['x'])
        y = int(bounds['y'])
        w = int(bounds['width'])
        h = int(bounds['height'])
        
        # 색상 변환 (이 부분이 에러 해결의 핵심)
        fill_color = self.hex_to_bgr(bg_color)
            
        # 텍스트 영역을 배경색으로 덮어쓰기
        cv2.rectangle(result, (x, y), (x + w, y + h), fill_color, -1)

def create_inpainter(method='simple_fill'):
    return Inpainter(method)
