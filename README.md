# 🖼️ 한글 인포그래픽 교정 도구

AI 생성 인포그래픽의 깨진 한글 텍스트를 교정하는 Streamlit 웹앱입니다.

## ✨ 주요 기능

- **OCR 자동 감지**: Tesseract를 활용한 한글 텍스트 자동 인식
- **역상 텍스트 지원**: 색상 배경 위의 흰색 텍스트도 감지
- **하이브리드 편집**: OCR 결과 수정 + 수동 영역 추가
- **스타일 커스터마이징**: 폰트, 크기, 색상 자유 설정
- **다중 포맷 출력**: PNG, PDF 지원

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (또는 파일 다운로드)
cd korean-infographic-fixer

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Tesseract OCR 설치

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
1. [Tesseract 설치 파일](https://github.com/UB-Mannheim/tesseract/wiki) 다운로드
2. 설치 시 "Korean" 언어 데이터 선택
3. 환경 변수 PATH에 Tesseract 경로 추가

### 3. 폰트 설치

`assets/fonts/` 디렉토리에 한글 폰트 파일을 추가하세요:
- NotoSansKR-Regular.ttf
- NotoSansKR-Bold.ttf
- NanumSquareB.ttf (선택)

무료 폰트 다운로드:
- [Noto Sans KR](https://fonts.google.com/noto/specimen/Noto+Sans+KR)
- [나눔스퀘어](https://hangeul.naver.com/font/nanum)

### 4. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 📖 사용 방법

### Step 1: 이미지 업로드
교정할 인포그래픽 이미지(PNG, JPG, WEBP)를 업로드합니다.

### Step 2: 텍스트 감지
"텍스트 자동 감지 시작" 버튼을 클릭하면 OCR이 실행됩니다.
- 🟢 녹색: 일반 텍스트
- 🔵 파란색: 역상 텍스트 (색상 배경)
- 🟠 주황색: 수동 추가 영역

### Step 3: 텍스트 편집
- 각 영역을 클릭하여 텍스트 수정
- 폰트 크기, 색상 조정
- 필요시 수동으로 새 영역 추가

### Step 4: 내보내기
- 최종 결과 미리보기
- PNG, PDF 형식으로 다운로드
- 메타데이터 JSON 저장

## 📁 프로젝트 구조

```
korean-infographic-fixer/
├── app.py                  # Streamlit 메인 앱
├── requirements.txt        # Python 의존성
├── .env.example           # 환경변수 템플릿
│
├── config/
│   └── settings.py        # 전역 설정
│
├── modules/
│   ├── ocr_engine.py      # OCR 텍스트 추출
│   ├── style_classifier.py # 스타일 자동 분류
│   ├── inpainter.py       # 배경 복원
│   ├── metadata_builder.py # 메타데이터 관리
│   ├── text_renderer.py   # 텍스트 렌더링
│   └── exporter.py        # 다중 포맷 출력
│
├── assets/
│   └── fonts/             # 한글 폰트 파일
│
└── outputs/               # 생성된 파일 저장
```

## ⚙️ 설정 옵션

`config/settings.py`에서 다음 항목을 수정할 수 있습니다:

| 설정 | 설명 | 기본값 |
|------|------|--------|
| `DEFAULT_FONT_FAMILY` | 기본 폰트 | Noto Sans KR |
| `OCR_CONFIG.min_confidence` | OCR 최소 신뢰도 | 30 |
| `EXPORT_CONFIG.png.quality` | PNG 품질 | 95 |
| `EXPORT_CONFIG.png.dpi` | 출력 DPI | 150 |

## 🔧 문제 해결

### OCR이 텍스트를 인식하지 못할 때
1. Tesseract 한글 데이터 설치 확인: `tesseract --list-langs`
2. 이미지 해상도가 너무 낮으면 인식률 저하
3. 수동 영역 추가 기능 활용

### 폰트가 적용되지 않을 때
1. `assets/fonts/` 디렉토리에 폰트 파일 존재 확인
2. 폰트 파일명이 `settings.py`의 `AVAILABLE_FONTS`와 일치하는지 확인

### 역상 텍스트가 감지되지 않을 때
1. 배경색이 너무 밝으면 감지 어려움
2. `settings.py`의 `INVERT_DETECTION_CONFIG` 조정

## 📝 라이선스

MIT License

## 🤝 기여

이슈 및 PR 환영합니다!
