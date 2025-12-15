"""
Inpainter Module
텍스트 영역 제거 및 배경 복원 (단순 색상 채우기)
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from .ocr_engine import TextRegion


class SimpleInpainter:
    """단순 색상 채우기 방식의 인페인터"""
    
    def __init__(self, padding: int = 5, blur_kernel: int = 3):
        self.padding = padding
        self.blur_kernel = blur_kernel
        
    def remove_text_region(
        self, 
        image: np.ndarray, 
        region: TextRegion,
        fill_color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        단일 텍스트 영역 제거 및 배경색으로 채우기
        
        Args:
            image: OpenCV 이미지 (BGR)
            region: 제거할 TextRegion
            fill_color: 채울 색상 (BGR). None이면 자동 감지
            
        Returns:
            텍스트가 제거된 이미지
        """
        result = image.copy()
        b = region.bounds
        
        # 패딩 적용
        x1 = max(0, b['x'] - self.padding)
        y1 = max(0, b['y'] - self.padding)
        x2 = min(image.shape[1], b['x'] + b['width'] + self.padding)
        y2 = min(image.shape[0], b['y'] + b['height'] + self.padding)
        
        if fill_color is None:
            # 배경색 자동 감지 (영역 주변 픽셀에서)
            fill_color = self._detect_background_color(image, x1, y1, x2, y2)
        
        # 영역 채우기
        cv2.rectangle(result, (x1, y1), (x2, y2), fill_color, -1)
        
        # 경계 부드럽게 (선택적)
        if self.blur_kernel > 0:
            # 경계 영역만 블러 처리
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, self.blur_kernel * 2)
            cv2.rectangle(mask, (x1 + self.blur_kernel, y1 + self.blur_kernel), 
                         (x2 - self.blur_kernel, y2 - self.blur_kernel), 0, -1)
            
            blurred = cv2.GaussianBlur(result, (self.blur_kernel * 2 + 1, self.blur_kernel * 2 + 1), 0)
            result = np.where(mask[:, :, np.newaxis] > 0, blurred, result)
        
        return result
    
    def remove_all_text_regions(
        self, 
        image: np.ndarray, 
        regions: List[TextRegion]
    ) -> np.ndarray:
        """
        모든 텍스트 영역 제거
        
        Args:
            image: OpenCV 이미지 (BGR)
            regions: 제거할 TextRegion 리스트
            
        Returns:
            모든 텍스트가 제거된 이미지
        """
        result = image.copy()
        
        for region in regions:
            result = self.remove_text_region(result, region)
            
        return result
    
    def _detect_background_color(
        self, 
        image: np.ndarray, 
        x1: int, y1: int, x2: int, y2: int,
        sample_width: int = 10
    ) -> Tuple[int, int, int]:
        """
        영역 주변에서 배경색 감지
        
        Args:
            image: 이미지
            x1, y1, x2, y2: 영역 좌표
            sample_width: 샘플링할 테두리 너비
            
        Returns:
            배경색 (BGR)
        """
        h, w = image.shape[:2]
        samples = []
        
        # 상단 테두리
        if y1 - sample_width >= 0:
            samples.append(image[max(0, y1-sample_width):y1, x1:x2])
        
        # 하단 테두리
        if y2 + sample_width <= h:
            samples.append(image[y2:min(h, y2+sample_width), x1:x2])
        
        # 좌측 테두리
        if x1 - sample_width >= 0:
            samples.append(image[y1:y2, max(0, x1-sample_width):x1])
        
        # 우측 테두리
        if x2 + sample_width <= w:
            samples.append(image[y1:y2, x2:min(w, x2+sample_width)])
        
        if not samples:
            # 샘플이 없으면 영역 내부의 밝은 픽셀 사용
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                pixels = roi.reshape(-1, 3)
                brightness = np.mean(pixels, axis=1)
                light_pixels = pixels[brightness >= 128]
                if len(light_pixels) > 0:
                    return tuple(np.median(light_pixels, axis=0).astype(int))
            return (255, 255, 255)  # 기본값: 흰색
        
        # 모든 샘플에서 중간값 계산
        all_pixels = np.vstack([s.reshape(-1, 3) for s in samples if s.size > 0])
        
        if len(all_pixels) == 0:
            return (255, 255, 255)
        
        # 밝은 픽셀만 사용 (텍스트 픽셀 제외)
        brightness = np.mean(all_pixels, axis=1)
        light_pixels = all_pixels[brightness >= 100]
        
        if len(light_pixels) == 0:
            light_pixels = all_pixels
        
        bg_color = np.median(light_pixels, axis=0).astype(int)
        return tuple(bg_color)


class OpenCVInpainter:
    """OpenCV 인페인팅 알고리즘 사용"""
    
    def __init__(self, method: str = "telea", radius: int = 3):
        """
        Args:
            method: 'telea' 또는 'ns' (Navier-Stokes)
            radius: 인페인팅 반경
        """
        self.method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        self.radius = radius
        
    def remove_text_region(
        self, 
        image: np.ndarray, 
        region: TextRegion,
        padding: int = 5
    ) -> np.ndarray:
        """
        OpenCV 인페인팅으로 텍스트 영역 제거
        
        Args:
            image: OpenCV 이미지 (BGR)
            region: 제거할 TextRegion
            padding: 경계 여유 픽셀
            
        Returns:
            텍스트가 제거된 이미지
        """
        b = region.bounds
        
        # 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        x1 = max(0, b['x'] - padding)
        y1 = max(0, b['y'] - padding)
        x2 = min(image.shape[1], b['x'] + b['width'] + padding)
        y2 = min(image.shape[0], b['y'] + b['height'] + padding)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 인페인팅 수행
        result = cv2.inpaint(image, mask, self.radius, self.method)
        
        return result
    
    def remove_all_text_regions(
        self, 
        image: np.ndarray, 
        regions: List[TextRegion],
        padding: int = 5
    ) -> np.ndarray:
        """
        모든 텍스트 영역을 한 번에 제거 (더 자연스러운 결과)
        """
        # 모든 영역을 포함하는 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for region in regions:
            b = region.bounds
            x1 = max(0, b['x'] - padding)
            y1 = max(0, b['y'] - padding)
            x2 = min(image.shape[1], b['x'] + b['width'] + padding)
            y2 = min(image.shape[0], b['y'] + b['height'] + padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 인페인팅 수행
        result = cv2.inpaint(image, mask, self.radius, self.method)
        
        return result


def create_inpainter(method: str = "simple_fill", **kwargs):
    """
    인페인터 팩토리 함수
    
    Args:
        method: 'simple_fill', 'telea', 'ns'
        **kwargs: 인페인터별 추가 옵션
        
    Returns:
        인페인터 인스턴스
    """
    if method == "simple_fill":
        return SimpleInpainter(
            padding=kwargs.get('padding', 5),
            blur_kernel=kwargs.get('blur_kernel', 3)
        )
    elif method in ["telea", "ns"]:
        return OpenCVInpainter(
            method=method,
            radius=kwargs.get('radius', 3)
        )
    else:
        raise ValueError(f"Unknown inpainting method: {method}")
