# 휠체어 자율주행용 사물 감지 / 판별 시스템
PyTorch / YOLO v5 Nano를 사용한 자율주행용 객체 감지 시스템입니다.

## 📋 주요 기능

- **실시간 객체 감지**: 차량, 보행자, 기타 장애물 실시간 감지
- **시각화**: 색상별 바운딩 박스와 신뢰도 표시
- **라벨링(사물 분류) 지원**: 9가지 자율주행 관련 클래스
  - Person (보행자)
  - Bicycle (자전거)
  - Car (승용차)
  - Motorcycle (오토바이)
  - Bus (버스)
  - Train (기차)
  - Truck (트럭)
  - Traffic Light (신호등)
  - Stop Sign (정지 표지판)
## 📦 설치 방법

### 1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터셋 준비

#### COCO 데이터셋 다운로드

```bash
# Validation 세트 (~1GB)
python download_data.py --split val

# Training 세트 (~18GB)
python download_data.py스

- [PyTorch](https://pytorch.org/)
- [COCO Dataset](https://cocodataset.org/)
- [Torchvision](https://pytorch.org/vision/)


