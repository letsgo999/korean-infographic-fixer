"""
Metadata Builder Module
JSON 메타데이터 생성 및 관리
"""
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from .ocr_engine import TextRegion


class MetadataBuilder:
    """메타데이터 생성 및 관리 클래스"""
    
    def __init__(self):
        self.metadata = {
            'version': '1.0',
            'created_at': None,
            'updated_at': None,
            'image_info': {},
            'ocr_summary': {},
            'text_regions': []
        }
        
    def set_image_info(
        self, 
        filename: str, 
        width: int, 
        height: int,
        **kwargs
    ) -> 'MetadataBuilder':
        """이미지 정보 설정"""
        self.metadata['image_info'] = {
            'filename': filename,
            'width': width,
            'height': height,
            **kwargs
        }
        return self
    
    def set_regions(self, regions: List[TextRegion]) -> 'MetadataBuilder':
        """텍스트 영역 설정"""
        self.metadata['text_regions'] = [r.to_dict() for r in regions]
        self._update_summary()
        return self
    
    def add_region(self, region: TextRegion) -> 'MetadataBuilder':
        """텍스트 영역 추가"""
        self.metadata['text_regions'].append(region.to_dict())
        self._update_summary()
        return self
    
    def update_region(self, region_id: str, updates: Dict) -> 'MetadataBuilder':
        """특정 영역 업데이트"""
        for region in self.metadata['text_regions']:
            if region['id'] == region_id:
                region.update(updates)
                break
        self._update_summary()
        return self
    
    def remove_region(self, region_id: str) -> 'MetadataBuilder':
        """특정 영역 삭제"""
        self.metadata['text_regions'] = [
            r for r in self.metadata['text_regions'] 
            if r['id'] != region_id
        ]
        self._update_summary()
        return self
    
    def _update_summary(self):
        """요약 정보 업데이트"""
        regions = self.metadata['text_regions']
        
        if not regions:
            self.metadata['ocr_summary'] = {
                'total_regions': 0,
                'normal_regions': 0,
                'inverted_regions': 0,
                'manual_regions': 0,
                'avg_confidence': 0
            }
            return
        
        confidences = [r['confidence'] for r in regions]
        
        self.metadata['ocr_summary'] = {
            'total_regions': len(regions),
            'normal_regions': len([r for r in regions if not r.get('is_inverted', False) and not r.get('is_manual', False)]),
            'inverted_regions': len([r for r in regions if r.get('is_inverted', False)]),
            'manual_regions': len([r for r in regions if r.get('is_manual', False)]),
            'avg_confidence': round(sum(confidences) / len(confidences), 1),
            'style_distribution': {
                'title': len([r for r in regions if r.get('style_tag') == 'title']),
                'subtitle': len([r for r in regions if r.get('style_tag') == 'subtitle']),
                'body': len([r for r in regions if r.get('style_tag') == 'body']),
                'caption': len([r for r in regions if r.get('style_tag') == 'caption'])
            }
        }
    
    def build(self) -> Dict:
        """최종 메타데이터 생성"""
        now = datetime.now().isoformat()
        
        if self.metadata['created_at'] is None:
            self.metadata['created_at'] = now
        self.metadata['updated_at'] = now
        
        return self.metadata
    
    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.build(), ensure_ascii=False, indent=indent)
    
    def save(self, filepath: str) -> None:
        """파일로 저장"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'MetadataBuilder':
        """파일에서 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        builder = cls()
        builder.metadata = data
        return builder
    
    def get_regions(self) -> List[Dict]:
        """텍스트 영역 목록 반환"""
        return self.metadata['text_regions']
    
    def get_region_by_id(self, region_id: str) -> Optional[Dict]:
        """ID로 특정 영역 조회"""
        for region in self.metadata['text_regions']:
            if region['id'] == region_id:
                return region
        return None


def create_manual_region(
    x: int, 
    y: int, 
    width: int, 
    height: int,
    text: str = "",
    style_tag: str = "body",
    font_family: str = "Noto Sans KR",
    font_size: int = 16,
    text_color: str = "#000000",
    bg_color: str = "#FFFFFF"
) -> TextRegion:
    """
    수동으로 텍스트 영역 생성
    
    Args:
        x, y, width, height: 영역 좌표
        text: 텍스트 내용
        style_tag: 스타일 태그
        font_family: 폰트 패밀리
        font_size: 폰트 크기
        text_color: 글자색 (HEX)
        bg_color: 배경색 (HEX)
        
    Returns:
        TextRegion 객체
    """
    import uuid
    
    return TextRegion(
        id=f"manual_{uuid.uuid4().hex[:8]}",
        text=text,
        confidence=100.0,  # 수동 입력은 신뢰도 100%
        bounds={
            'x': x,
            'y': y,
            'width': width,
            'height': height
        },
        is_inverted=False,
        is_manual=True,
        style_tag=style_tag,
        suggested_font_size=font_size,
        font_family=font_family,
        text_color=text_color,
        bg_color=bg_color
    )


def merge_overlapping_regions(
    regions: List[TextRegion], 
    overlap_threshold: float = 0.5
) -> List[TextRegion]:
    """
    겹치는 영역 병합
    
    Args:
        regions: TextRegion 리스트
        overlap_threshold: 병합 기준 겹침 비율 (0~1)
        
    Returns:
        병합된 TextRegion 리스트
    """
    if len(regions) <= 1:
        return regions
    
    def calc_iou(r1: TextRegion, r2: TextRegion) -> float:
        """Intersection over Union 계산"""
        b1, b2 = r1.bounds, r2.bounds
        
        x1 = max(b1['x'], b2['x'])
        y1 = max(b1['y'], b2['y'])
        x2 = min(b1['x'] + b1['width'], b2['x'] + b2['width'])
        y2 = min(b1['y'] + b1['height'], b2['y'] + b2['height'])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = b1['width'] * b1['height']
        area2 = b2['width'] * b2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    merged = []
    used = set()
    
    for i, r1 in enumerate(regions):
        if i in used:
            continue
            
        # 겹치는 영역 찾기
        to_merge = [r1]
        for j, r2 in enumerate(regions[i+1:], start=i+1):
            if j in used:
                continue
            if calc_iou(r1, r2) >= overlap_threshold:
                to_merge.append(r2)
                used.add(j)
        
        if len(to_merge) == 1:
            merged.append(r1)
        else:
            # 영역 병합
            all_bounds = [r.bounds for r in to_merge]
            min_x = min(b['x'] for b in all_bounds)
            min_y = min(b['y'] for b in all_bounds)
            max_x = max(b['x'] + b['width'] for b in all_bounds)
            max_y = max(b['y'] + b['height'] for b in all_bounds)
            
            # 텍스트는 공백으로 연결
            merged_text = ' '.join(r.text for r in to_merge)
            
            merged_region = TextRegion(
                id=f"merged_{len(merged):03d}",
                text=merged_text,
                confidence=sum(r.confidence for r in to_merge) / len(to_merge),
                bounds={
                    'x': min_x,
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y
                },
                is_inverted=any(r.is_inverted for r in to_merge),
                style_tag=to_merge[0].style_tag,
                suggested_font_size=to_merge[0].suggested_font_size
            )
            merged.append(merged_region)
        
        used.add(i)
    
    return merged
