"""
Exporter Module
다중 포맷 출력 (PNG, PDF, PSD)
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from io import BytesIO
import json

# PSD 관련 임포트 (선택적)
try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    HAS_PSD_TOOLS = True
except ImportError:
    HAS_PSD_TOOLS = False

# PDF 관련 임포트
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from .ocr_engine import TextRegion


class PNGExporter:
    """PNG 이미지 출력"""
    
    def __init__(self, quality: int = 95, dpi: int = 150):
        self.quality = quality
        self.dpi = dpi
        
    def export(
        self, 
        image: np.ndarray, 
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        PNG 파일로 내보내기
        
        Args:
            image: OpenCV 이미지 (BGR)
            output_path: 출력 파일 경로
            metadata: 메타데이터 (PNG 메타데이터로 저장)
            
        Returns:
            저장된 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # DPI 설정
        pil_image.info['dpi'] = (self.dpi, self.dpi)
        
        # PNG 저장
        pil_image.save(
            str(output_path),
            'PNG',
            quality=self.quality,
            dpi=(self.dpi, self.dpi)
        )
        
        # 메타데이터를 별도 JSON으로 저장 (선택적)
        if metadata:
            meta_path = output_path.with_suffix('.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
    
    def export_to_bytes(self, image: np.ndarray) -> bytes:
        """메모리에서 PNG 바이트로 변환"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG', quality=self.quality)
        return buffer.getvalue()


class PDFExporter:
    """PDF 문서 출력"""
    
    def __init__(
        self, 
        page_size: str = "A4",
        margin: int = 20,
        dpi: int = 150
    ):
        if not HAS_REPORTLAB:
            raise ImportError("reportlab 패키지가 필요합니다: pip install reportlab")
            
        self.page_size = A4 if page_size == "A4" else letter
        self.margin = margin
        self.dpi = dpi
        
    def export(
        self, 
        image: np.ndarray, 
        output_path: str,
        title: str = "Infographic",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        PDF 파일로 내보내기
        
        Args:
            image: OpenCV 이미지 (BGR)
            output_path: 출력 파일 경로
            title: PDF 제목
            metadata: 메타데이터
            
        Returns:
            저장된 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # PDF 캔버스 생성
        c = canvas.Canvas(str(output_path), pagesize=self.page_size)
        
        # 페이지 크기
        page_width, page_height = self.page_size
        
        # 이미지 크기 계산 (마진 적용)
        available_width = page_width - (self.margin * 2)
        available_height = page_height - (self.margin * 2)
        
        img_width, img_height = pil_image.size
        
        # 비율 유지하며 크기 조정
        ratio = min(available_width / img_width, available_height / img_height)
        new_width = img_width * ratio
        new_height = img_height * ratio
        
        # 중앙 정렬
        x = (page_width - new_width) / 2
        y = (page_height - new_height) / 2
        
        # 이미지 삽입
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        c.drawImage(
            ImageReader(img_buffer),
            x, y,
            width=new_width,
            height=new_height
        )
        
        # 메타데이터 추가
        if metadata:
            c.setTitle(title)
            c.setAuthor("Korean Infographic Fixer")
        
        c.save()
        
        return str(output_path)
    
    def export_to_bytes(self, image: np.ndarray, title: str = "Infographic") -> bytes:
        """메모리에서 PDF 바이트로 변환"""
        buffer = BytesIO()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        c = canvas.Canvas(buffer, pagesize=self.page_size)
        page_width, page_height = self.page_size
        
        available_width = page_width - (self.margin * 2)
        available_height = page_height - (self.margin * 2)
        
        img_width, img_height = pil_image.size
        ratio = min(available_width / img_width, available_height / img_height)
        new_width = img_width * ratio
        new_height = img_height * ratio
        
        x = (page_width - new_width) / 2
        y = (page_height - new_height) / 2
        
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        c.drawImage(ImageReader(img_buffer), x, y, width=new_width, height=new_height)
        c.setTitle(title)
        c.save()
        
        return buffer.getvalue()


class PSDExporter:
    """PSD (Photoshop) 파일 출력 - 레이어 구조 포함"""
    
    def __init__(self):
        if not HAS_PSD_TOOLS:
            raise ImportError("psd-tools 패키지가 필요합니다: pip install psd-tools")
    
    def export(
        self,
        background: np.ndarray,
        text_layers: List[Tuple[str, np.ndarray]],  # [(layer_name, image), ...]
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        PSD 파일로 내보내기 (레이어 구조 포함)
        
        Note: psd-tools는 읽기 전용이므로, 
        대안으로 TIFF 레이어 또는 별도 방법 사용
        
        현재 구현: 합성된 단일 이미지로 저장
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 레이어 합성
        result = background.copy()
        for layer_name, layer_image in text_layers:
            # 알파 블렌딩
            if layer_image.shape[2] == 4:  # RGBA
                alpha = layer_image[:, :, 3] / 255.0
                for c in range(3):
                    result[:, :, c] = (
                        result[:, :, c] * (1 - alpha) + 
                        layer_image[:, :, c] * alpha
                    ).astype(np.uint8)
            else:
                result = layer_image
        
        # TIFF로 저장 (레이어 정보는 메타데이터로)
        image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # PSD 대신 TIFF로 저장 (레이어 지원)
        tiff_path = output_path.with_suffix('.tiff')
        pil_image.save(str(tiff_path), 'TIFF')
        
        # 레이어 정보를 JSON으로 저장
        if metadata:
            layer_info = {
                'layers': [name for name, _ in text_layers],
                'metadata': metadata
            }
            meta_path = output_path.with_suffix('.layers.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(layer_info, f, ensure_ascii=False, indent=2)
        
        return str(tiff_path)


class MultiFormatExporter:
    """다중 포맷 출력 통합 클래스"""
    
    def __init__(
        self,
        png_quality: int = 95,
        pdf_page_size: str = "A4",
        dpi: int = 150
    ):
        self.png_exporter = PNGExporter(quality=png_quality, dpi=dpi)
        
        try:
            self.pdf_exporter = PDFExporter(page_size=pdf_page_size, dpi=dpi)
        except ImportError:
            self.pdf_exporter = None
            
        try:
            self.psd_exporter = PSDExporter()
        except ImportError:
            self.psd_exporter = None
    
    def export_all(
        self,
        image: np.ndarray,
        output_dir: str,
        filename_base: str,
        formats: List[str] = ["png", "pdf"],
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        여러 포맷으로 동시 내보내기
        
        Args:
            image: OpenCV 이미지
            output_dir: 출력 디렉토리
            filename_base: 기본 파일명 (확장자 제외)
            formats: 출력 포맷 리스트 ["png", "pdf", "psd"]
            metadata: 메타데이터
            
        Returns:
            {format: filepath} 딕셔너리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for fmt in formats:
            try:
                if fmt.lower() == "png":
                    path = output_dir / f"{filename_base}.png"
                    results["png"] = self.png_exporter.export(image, str(path), metadata)
                    
                elif fmt.lower() == "pdf" and self.pdf_exporter:
                    path = output_dir / f"{filename_base}.pdf"
                    results["pdf"] = self.pdf_exporter.export(image, str(path), metadata=metadata)
                    
                elif fmt.lower() == "psd" and self.psd_exporter:
                    path = output_dir / f"{filename_base}.psd"
                    results["psd"] = self.psd_exporter.export(
                        image, [], str(path), metadata
                    )
                    
            except Exception as e:
                print(f"{fmt} 내보내기 실패: {e}")
                results[fmt] = None
        
        return results
    
    def get_available_formats(self) -> List[str]:
        """사용 가능한 포맷 목록 반환"""
        formats = ["png"]  # PNG는 항상 가능
        
        if self.pdf_exporter:
            formats.append("pdf")
        if self.psd_exporter:
            formats.append("psd")
            
        return formats
