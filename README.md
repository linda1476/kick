# 자율주행용 객체 감지 시스템
# 본 글은 AI가 작성하였습니다.
PyTorch를 사용한 자율주행용 객체 감지 시스템입니다.

## 📋 주요 기능

- **실시간 객체 감지**: 차량, 보행자, 기타 장애물 실시간 감지
- **시각화**: 색상별 바운딩 박스와 신뢰도 표시
- **다중 클래스 지원**: 9가지 자율주행 관련 클래스
  - Person (보행자)
  - Bicycle (자전거)
  - Car (승용차)
  - Motorcycle (오토바이)
  - Bus (버스)
  - Train (기차)
  - Truck (트럭)
  - Traffic Light (신호등)
  - Stop Sign (정지 표지판)

## 🛠️ 기술 스택

- **PyTorch**: 딥러닝 프레임워크
- **Faster R-CNN**: 객체 감지 모델
- **COCO Dataset**: 학습 데이터
- **OpenCV**: 이미지 처리 및 시각화
# ( 추후 도커로 배포가능하게 할 예정 )
## 📦 설치 방법

### 1. 가상 환경 생성 (권장)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터셋 준비

#### 옵션 A: 전체 COCO 데이터셋 다운로드

```bash
# Validation 세트 (~1GB)
python download_data.py --split val

# Training 세트 (~18GB)
python download_data.py스

- [PyTorch](https://pytorch.org/)
- [COCO Dataset](https://cocodataset.org/)
- [Torchvision](https://pytorch.org/vision/)

## 📧 문의

문제가 발생하거나 질문이 있으시면 이슈 생성해주세요.

---

**⚡ 빠른 시작 예제**

```bash
# 1. 설치
pip install -r requirements.txt

# 2. 샘플 이미지 다운로드
python download_data.py --sample

# 3. 추론 실행
python detect.py --source sample_images/sample_1.jpg --output result.jpg

# 4. 결과 확인
# result.jpg 파일을 열어서 객체 감지 결과를 확인하세요!
```
