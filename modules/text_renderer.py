"""
Text Renderer Module
이미지에 텍스트를 합성(Rendering)하는 모듈 (폰트 파일 로드 수정판)
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class CompositeRenderer:
    def __init__(self):
        # 폰트 파일 경로 설정 (fonts 폴더 안의 NanumGothic.ttf)
        # 주의: 업로드한 폰트 파일명과 정확히 일치해야 합니다!
        self.font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts', 'NanumGothic.ttf')
        
        # 폰트 파일이 없으면 경고 출력
        if not os.path.exists(self.font_path):
            print(f"WARNING: Font file not found at {self.font_path}")

    def composite(self, background_image: np.ndarray, regions: list, edited_texts: dict) -> np.ndarray:
        """
        배경 이미지 위에 텍스트를 합성합니다.
        """
        # OpenCV 이미지를 PIL 이미지로 변환 (한글 출력을 위해)
        if len(background_image.shape) == 3:
            img_pil = Image.fromarray(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(background_image)
            
        draw = ImageDraw.Draw(img_pil)
        
        for region in regions:
            # 수정된 텍스트가 있으면 그것을 사용, 없으면 원본 사용
            text_content = edited_texts.get(region.id, region.text)
            
            # 스타일 정보 가져오기
            font_size = int(getattr(region, 'suggested_font_size', 20))
            text_color = getattr(region, 'text_color', '#000000')
            
            # 좌표 정보
            x = region.bounds['x']
            y = region.bounds['y']
            
            # 폰트 로드 (파일 경로 기반)
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except Exception as e:
                # 폰트 로드 실패 시 기본 폰트 사용 (한글 깨짐 발생 가능성 있음)
                print(f"Font Load Error: {e}")
                font = ImageFont.load_default()
            
            # 텍스트 그리기
            # (색상은 Hex 코드 -> RGB 튜플 변환)
            fill_color = self._hex_to_rgb(text_color)
            
            # 텍스트 그리기 (좌표 보정 없이 단순 출력)
            draw.text((x, y), text_content, font=font, fill=fill_color)
            
        # PIL 이미지를 다시 OpenCV 이미지로 변환
        final_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return final_image

    def _hex_to_rgb(self, hex_color: str):
        """Hex 코드를 RGB 튜플로 변환"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (0, 0, 0)
