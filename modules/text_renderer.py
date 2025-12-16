"""
Text Renderer Module
이미지에 텍스트를 합성(Rendering)하는 모듈 (장평 조절 및 다중 폰트 지원)
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class CompositeRenderer:
    def __init__(self):
        # fonts 폴더 경로 설정
        self.fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        
    def composite(self, background_image: np.ndarray, regions: list, edited_texts: dict) -> np.ndarray:
        """배경 이미지 위에 텍스트를 합성합니다."""
        
        # 1. 배경을 PIL 이미지로 변환 (RGBA 모드로 작업)
        if len(background_image.shape) == 3:
            # BGR -> RGB -> RGBA
            img_rgb = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).convert("RGBA")
        else:
            img_pil = Image.fromarray(background_image).convert("RGBA")
        
        # 텍스트를 그릴 투명 레이어 생성
        text_layer = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
        
        for region in regions:
            # 텍스트 내용
            text_content = edited_texts.get(region.id, region.text)
            
            # 스타일 정보
            font_size = int(getattr(region, 'suggested_font_size', 20))
            text_color = getattr(region, 'text_color', '#000000')
            width_scale = int(getattr(region, 'width_scale', 100)) # 장평 (기본 100)
            
            # 폰트 파일 로드
            font_file = getattr(region, 'font_filename', None)
            
            # 폰트 파일이 지정되지 않았거나 없으면 폴더 내 첫 번째 ttf 사용
            if not font_file:
                available_fonts = [f for f in os.listdir(self.fonts_dir) if f.lower().endswith('.ttf')]
                font_file = available_fonts[0] if available_fonts else 'NanumGothic.ttf'
                
            font_path = os.path.join(self.fonts_dir, font_file)
            
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
            
            # 색상 변환
            fill_color = self._hex_to_rgb(text_color) + (255,) # Alpha 255 추가
            
            # --- [핵심] 장평(Jangpyeong) 구현 로직 ---
            # 1. 글자의 원래 크기 측정
            dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
            bbox = dummy_draw.textbbox((0, 0), text_content, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 2. 텍스트만을 위한 임시 캔버스 생성 (여유 공간 포함)
            if text_w <= 0: text_w = 1
            if text_h <= 0: text_h = 1
            
            # 장평 적용 전 원본 캔버스
            temp_canvas = Image.new('RGBA', (text_w + 10, text_h + 20), (0,0,0,0))
            temp_draw = ImageDraw.Draw(temp_canvas)
            temp_draw.text((0, 0), text_content, font=font, fill=fill_color)
            
            # 3. 장평 적용 (Resize)
            # 100% -> 1.0, 90% -> 0.9
            scale_ratio = width_scale / 100.0
            new_width = int(temp_canvas.width * scale_ratio)
            new_height = temp_canvas.height
            
            if new_width > 0 and new_height > 0:
                # 고품질 리사이징 (LANCZOS)
                stretched_text = temp_canvas.resize((new_width, new_height), resample=Image.LANCZOS)
                
                # 4. 원래 위치에 붙여넣기
                x = region.bounds['x']
                y = region.bounds['y']
                
                # 합성 (Alpha channel 활용)
                text_layer.paste(stretched_text, (x, y), stretched_text)

        # 5. 배경과 텍스트 레이어 합치기
        final_pil = Image.alpha_composite(img_pil, text_layer)
        
        # 다시 OpenCV BGR로 변환
        final_image = cv2.cvtColor(np.array(final_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        return final_image

    def _hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (0, 0, 0)
