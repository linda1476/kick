# Tesla FSD 스타일 객체 감지 시스템

PyTorch를 사용한 Tesla FSD(Full Self-Driving) 스타일의 차량 자율주행용 객체 감지 시스템입니다.

![Tesla FSD Style Detection](demo.jpg)

## 📋 주요 기능

- **실시간 객체 감지**: 차량, 보행자, 기타 장애물 실시간 감지
- **Tesla FSD 스타일 시각화**: 색상별 바운딩 박스와 신뢰도 표시
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
python download_data.py --split train
```

#### 옵션 B: 샘플 이미지로 테스트

```bash
python download_data.py --sample
```

#### 🚀 사용 방법

### 1. 빠른 시작 (권장)

```bash
# 데모 실행 - 인터넷에서 샘플 이미지를 다운로드하여 자동 테스트
python demo.py
```

### 2. 실시간 웹캠 객체 감지 ⭐ NEW!

```bash
# 기본 사용법 (웹캠 0번)
python webcam.py

# 다른 카메라 사용
python webcam.py --camera 1

# 신뢰도 조정
python webcam.py --conf 0.7

# CPU 사용
python webcam.py --device cpu
```

**웹캠 조작법:**
- `q` 키: 종료
- `s` 키: 스크린샷 저장
- `+` 키: 신뢰도 증가
- `-` 키: 신뢰도 감소

### 3. 이미지 파일로 객체 감지

```bash
# 기본 사용법
python detect.py --source 이미지경로.jpg --output 결과.jpg

# 신뢰도 임계값 조정
python detect.py --source 이미지.jpg --conf 0.7 --output 결과.jpg

# CPU 사용
python detect.py --source 이미지.jpg --device cpu --output 결과.jpg
```

### 4. 비디오 파일 처리

```bash
# 기본 사용법
python detect.py --source 비디오경로.mp4 --output 결과.mp4

# 신뢰도 임계값 조정
python detect.py --source 비디오경로.mp4 --conf 0.7 --output 결과.mp4

# CPU 사용
python detect.py --source 비디오경로.mp4 --device cpu --output 결과.mp4
```

### 5. 모델 학습

```bash
python train.py \
  --data-root ./data/val2017 \
  --annotations ./data/annotations/instances_val2017.json \
  --epochs 10 \
  --batch-size 4 \
  --lr 0.001
```

### 3. 학습된 모델로 추론

```bash
python detect.py \
  --source sample.jpg \
  --weights checkpoints/best_model.pth \
  --output result.jpg
```

## 📁 프로젝트 구조

```
tesla_fsd_detection/
├── requirements.txt       # 의존성 패키지
├── README.md             # 프로젝트 문서
├── dataset.py            # COCO 데이터셋 로더
├── model.py              # 객체 감지 모델
├── train.py              # 학습 스크립트
├── detect.py             # 추론 및 시각화
├── download_data.py      # 데이터셋 다운로드
├── checkpoints/          # 모델 체크포인트
└── data/                 # 데이터셋
    ├── val2017/          # 이미지
    └── annotations/      # 어노테이션
```

## 🎨 시각화 스타일

각 클래스는 고유한 색상으로 표시됩니다:

- 🟢 **Person**: 초록색 - 보행자 보호 강조
- 🔵 **Car**: 파란색 - 주변 차량
- 🟡 **Bicycle**: 노란색 - 주의 필요
- 🟠 **Motorcycle**: 주황색 - 빠른 이동체
- 🟣 **Bus/Train**: 보라색 - 대형 차량
- 🔴 **Stop Sign**: 빨간색 - 필수 정지

## ⚙️ 고급 설정

### GPU 가속 사용

```bash
python detect.py --source sample.jpg --device cuda
```

### CPU만 사용

```bash
python detect.py --source sample.jpg --device cpu
```

### 클래스 수 변경

```bash
python train.py --num-classes 9 ...
```

## 📊 성능

- **모델**: Faster R-CNN (ResNet-50 백본)
- **입력 크기**: 640x640
- **처리 속도**: ~30 FPS (GPU) / ~5 FPS (CPU)
- **정확도**: COCO mAP ~0.60

## 🔧 문제 해결

### CUDA 오류

```bash
# CPU 모드로 전환
python detect.py --source sample.jpg --device cpu
```

### 메모리 부족

```bash
# 배치 크기 줄이기
python train.py --batch-size 2 ...
```

## 📝 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

## 🙏 감사의 말

- [PyTorch](https://pytorch.org/)
- [COCO Dataset](https://cocodataset.org/)
- [Torchvision](https://pytorch.org/vision/)

## 📧 문의

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.

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
# result.jpg 파일을 열어서 Tesla FSD 스타일 객체 감지 결과를 확인하세요!
```
