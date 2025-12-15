"""
Korean Infographic Fixer - Configuration Settings
"""
import os
from pathlib import Path

# ============================================
# 경로 설정
# ============================================
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ============================================
# 폰트 설정
# ============================================
AVAILABLE_FONTS = {
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
        "Light": "NanumSquareL.ttf",
    },
    "Nanum Square ac Bold": {
        "Bold": "NanumSquareacB.ttf",
    }
}

DEFAULT_FONT_FAMILY = "Noto Sans KR"
DEFAULT_FONT_WEIGHT = "Regular"

# ============================================
# 스타일 분류 설정
# ============================================
STYLE_TAGS = {
    "title": {
        "name": "대제목",
        "default_font_size": 32,
        "font_weight": "Bold",
        "color": "#333333"
    },
    "subtitle": {
        "name": "소제목", 
        "default_font_size": 24,
        "font_weight": "Bold",
        "color": "#555555"
    },
    "body": {
        "name": "본문",
        "default_font_size": 16,
        "font_weight": "Regular",
        "color": "#666666"
    },
    "caption": {
        "name": "캡션",
        "default_font_size": 12,
        "font_weight": "Light",
        "color": "#888888"
    }
}

# ============================================
# OCR 설정
# ============================================
OCR_CONFIG = {
    "lang": "kor+eng",
    "min_confidence": 30,
    "psm": 3,  # Page Segmentation Mode: 3 = Fully automatic page segmentation
    "oem": 3,  # OCR Engine Mode: 3 = Default, based on what is available
}

# 역상 텍스트 감지 설정
INVERT_DETECTION_CONFIG = {
    "dark_threshold": 150,
    "min_area": 1000,
    "min_width": 50,
    "min_height": 15,
    # HSV 범위 (주황색 등 색상 배경 감지)
    "orange_hsv_lower": [5, 100, 100],
    "orange_hsv_upper": [25, 255, 255],
}

# ============================================
# 인페인팅 설정
# ============================================
INPAINT_CONFIG = {
    "method": "simple_fill",  # simple_fill, telea, ns
    "padding": 5,
    "blur_kernel": 3,
}

# ============================================
# 출력 설정
# ============================================
EXPORT_CONFIG = {
    "png": {
        "quality": 95,
        "dpi": 150,
    },
    "pdf": {
        "page_size": "A4",
        "margin": 20,
    },
    "psd": {
        "include_layers": True,
    }
}

# ============================================
# Figma API 설정
# ============================================
FIGMA_CONFIG = {
    "api_base_url": "https://api.figma.com/v1",
    "token_env_var": "FIGMA_TOKEN",
}

# ============================================
# UI 설정
# ============================================
UI_CONFIG = {
    "canvas_stroke_width": 2,
    "canvas_stroke_color": "#FF0000",
    "highlight_colors": {
        "normal": "#00C853",      # 녹색 - 일반 텍스트
        "inverted": "#2979FF",    # 파란색 - 역상 텍스트
        "manual": "#FF6D00",      # 주황색 - 수동 추가
        "selected": "#E91E63",    # 핑크 - 선택됨
    },
    "max_image_width": 800,
    "max_image_height": 600,
}

# ============================================
# 환경 변수 로드
# ============================================
def load_env():
    """환경 변수 로드"""
    from dotenv import load_dotenv
    load_dotenv()
    
def get_figma_token():
    """Figma API 토큰 반환"""
    return os.getenv(FIGMA_CONFIG["token_env_var"])
