"""
Korean Infographic Fixer - Modules
"""
from .ocr_engine import (
    TextRegion,
    OCREngine,
    InvertedRegionDetector,
    group_regions_by_lines,
    run_enhanced_ocr
)

from .style_classifier import (
    StyleClassifier,
    ColorExtractor,
    apply_styles_and_colors
)

from .inpainter import (
    SimpleInpainter,
    OpenCVInpainter,
    create_inpainter
)

from .metadata_builder import (
    MetadataBuilder,
    create_manual_region,
    merge_overlapping_regions
)

from .text_renderer import (
    TextRenderer,
    CompositeRenderer
)

from .exporter import (
    PNGExporter,
    PDFExporter,
    MultiFormatExporter
)

__all__ = [
    # OCR
    'TextRegion',
    'OCREngine',
    'InvertedRegionDetector',
    'group_regions_by_lines',
    'run_enhanced_ocr',
    
    # Style
    'StyleClassifier',
    'ColorExtractor',
    'apply_styles_and_colors',
    
    # Inpainting
    'SimpleInpainter',
    'OpenCVInpainter',
    'create_inpainter',
    
    # Metadata
    'MetadataBuilder',
    'create_manual_region',
    'merge_overlapping_regions',
    
    # Rendering
    'TextRenderer',
    'CompositeRenderer',
    
    # Export
    'PNGExporter',
    'PDFExporter',
    'MultiFormatExporter',
]
