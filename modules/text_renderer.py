"""
Text Renderer Module
한글 텍스트를 이미지에 렌더링
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .ocr_engine import TextRegion


class TextRenderer:
    """한글 텍스트 렌더러"""
    
    def __init__(self, fonts_dir: Optional[str] = None):
        """
        Args:
            fonts_dir: 폰트 파일 디렉토리 경로
        """
        self.fonts_dir = Path(fonts_dir) if fonts_dir else None
        self.font_cache = {}
        
    def get_font(
        self, 
        font_family: str, 
        font_size: int,
        font_weight: str = "Regular"
    ) -> ImageFont.FreeTypeFont:
        """
        폰트 객체 가져오기 (캐싱)
        
        Args:
            font_family: 폰트 패밀리 이름
            font_size: 폰트 크기
            font_weight: 폰트 굵기
            
        Returns:
            PIL ImageFont 객체
        """
        cache_key = f"{font_family}_{font_weight}_{font_size}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        # 폰트 파일 경로 결정
        font_files = {
            "Noto Sans KR": {
                "Regular": "NotoSansKR-Regular.ttf",
                "Bold": "NotoSansKR-Bold.ttf",
                "Light": "NotoSansKR-Light.ttf",
                "Medium": "NotoSansKR-Medium.ttf",
            },
            "Nanum Square": {
                "Regular": "NanumSquareR.ttf",
                "Bold": "NanumSquareB.ttf",
                "ExtraBold": "NanumSquareEB.ttf",
            },
            "Nanum Square acBold": {
                "Bold": "NanumSquareacB.ttf",
            }
        }
        
        # 폰트 파일 찾기
        font_file = None
        if font_family in font_files:
            weight_files = font_files[font_family]
            if font_weight in weight_files:
                font_file = weight_files[font_weight]
            elif "Regular" in weight_files:
                font_file = weight_files["Regular"]
            else:
                font_file = list(weight_files.values())[0]
        
        # 폰트 로드
        try:
            if font_file and self.fonts_dir:
                font_path = self.fonts_dir / font_file
                if font_path.exists():
                    font = ImageFont.truetype(str(font_path), font_size)
                    self.font_cache[cache_key] = font
                    return font
            
            # 시스템 폰트 시도
            system_fonts = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "C:/Windows/Fonts/malgun.ttf",
            ]
            
            for sys_font in system_fonts:
                if Path(sys_font).exists():
                    font = ImageFont.truetype(sys_font, font_size)
                    self.font_cache[cache_key] = font
                    return font
            
            # 기본 폰트 사용
            font = ImageFont.load_default()
            self.font_cache[cache_key] = font
            return font
            
        except Exception as e:
            print(f"폰트 로드 실패: {e}, 기본 폰트 사용")
            return ImageFont.load_default()
    
    def render_text_on_image(
        self,
        image: np.ndarray,
        region: TextRegion,
        text_override: Optional[str] = None
    ) -> np.ndarray:
        """
        이미지에 텍스트 렌더링
        
        Args:
            image: OpenCV 이미지 (BGR)
            region: TextRegion 객체
            text_override: 기존 텍스트 대신 사용할 텍스트
            
        Returns:
            텍스트가 렌더링된 이미지
        """
        # OpenCV -> PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 텍스트 및 스타일
        text = text_override if text_override is not None else region.text
        font = self.get_font(
            region.font_family,
            region.suggested_font_size,
            region.font_weight
        )
        
        # 색상 변환 (HEX -> RGB)
        text_color = self._hex_to_rgb(region.text_color)
        
        # 위치 계산
        x = region.bounds['x']
        y = region.bounds['y']
        
        # 텍스트 그리기
        draw.text((x, y), text, font=font, fill=text_color)
        
        # PIL -> OpenCV
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return result
    
    def render_all_regions(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        text_overrides: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        모든 텍스트 영역 렌더링
        
        Args:
            image: OpenCV 이미지 (BGR)
            regions: TextRegion 리스트
            text_overrides: {region_id: new_text} 형태의 텍스트 오버라이드
            
        Returns:
            텍스트가 렌더링된 이미지
        """
        result = image.copy()
        text_overrides = text_overrides or {}
        
        for region in regions:
            override_text = text_overrides.get(region.id)
            result = self.render_text_on_image(result, region, override_text)
        
        return result
    
    def create_text_layer(
        self,
        width: int,
        height: int,
        regions: List[TextRegion],
        text_overrides: Optional[Dict[str, str]] = None,
        background: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> Image.Image:
        """
        투명 배경의 텍스트 레이어 생성
        
        Args:
            width, height: 레이어 크기
            regions: TextRegion 리스트
            text_overrides: 텍스트 오버라이드
            background: 배경색 (RGBA)
            
        Returns:
            PIL Image (RGBA)
        """
        layer = Image.new('RGBA', (width, height), background)
        draw = ImageDraw.Draw(layer)
        text_overrides = text_overrides or {}
        
        for region in regions:
            text = text_overrides.get(region.id, region.text)
            font = self.get_font(
                region.font_family,
                region.suggested_font_size,
                region.font_weight
            )
            text_color = self._hex_to_rgb(region.text_color) + (255,)  # RGBA
            
            x = region.bounds['x']
            y = region.bounds['y']
            
            draw.text((x, y), text, font=font, fill=text_color)
        
        return layer
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """HEX -> RGB 변환"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class CompositeRenderer:
    """배경 + 텍스트 레이어 합성 렌더러"""
    
    def __init__(self, fonts_dir: Optional[str] = None):
        self.text_renderer = TextRenderer(fonts_dir)
        
    def composite(
        self,
        background: np.ndarray,
        regions: List[TextRegion],
        text_overrides: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        배경 이미지 위에 텍스트 합성
        
        Args:
            background: 배경 이미지 (텍스트 제거된 상태)
            regions: TextRegion 리스트
            text_overrides: 텍스트 오버라이드
            
        Returns:
            합성된 이미지
        """
        h, w = background.shape[:2]
        
        # 텍스트 레이어 생성
        text_layer = self.text_renderer.create_text_layer(
            w, h, regions, text_overrides
        )
        
        # 배경 이미지를 PIL로 변환
        bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        bg_pil = Image.fromarray(bg_rgb).convert('RGBA')
        
        # 합성
        composite = Image.alpha_composite(bg_pil, text_layer)
        
        # OpenCV로 변환
        result = cv2.cvtColor(np.array(composite.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        return result
    
    def preview_with_highlights(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        highlight_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        텍스트 영역을 하이라이트하여 미리보기 생성
        
        Args:
            image: 원본 이미지
            regions: TextRegion 리스트
            highlight_colors: 영역 유형별 색상
            
        Returns:
            하이라이트된 이미지
        """
        result = image.copy()
        
        default_colors = {
            'normal': (0, 200, 0),      # 녹색
            'inverted': (255, 100, 0),  # 파란색 (BGR)
            'manual': (0, 165, 255),    # 주황색 (BGR)
        }
        colors = highlight_colors or default_colors
        
        for region in regions:
            b = region.bounds
            
            # 색상 결정
            if region.is_manual:
                color = colors.get('manual', (0, 165, 255))
            elif region.is_inverted:
                color = colors.get('inverted', (255, 100, 0))
            else:
                color = colors.get('normal', (0, 200, 0))
            
            # 사각형 그리기
            cv2.rectangle(
                result,
                (b['x'], b['y']),
                (b['x'] + b['width'], b['y'] + b['height']),
                color,
                2
            )
            
            # 레이블 표시
            label = f"{region.style_tag}: {region.text[:15]}..."
            cv2.putText(
                result,
                label[:20],
                (b['x'], b['y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        return result
