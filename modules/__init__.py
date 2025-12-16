"""
Korean Infographic Fixer Modules Package
"""

# OCR 및 텍스트 영역 관련
from .ocr_engine import (
    TextRegion,
    run_enhanced_ocr,
    group_regions_by_lines,
    create_manual_region,
    extract_text_from_crop  # [핵심] 이 줄이 꼭 있어야 합니다!
)

# 스타일 및 색상 관련
from .style_classifier import apply_styles_and_colors

# 이미지 복원 (인페인팅) 관련
from .inpainter import (
    create_inpainter,
    Inpainter
)

# 텍스트 렌더링 관련
from .text_renderer import CompositeRenderer

# 내보내기 및 메타데이터 관련
from .exporter import MultiFormatExporter
from .metadata_builder import MetadataBuilder
