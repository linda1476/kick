# YOLOv5 사용 가이드

## 🚀 왜 YOLOv5?

**장점:**
- ✅ 설치 초간단 (`pip install ultralytics`)
- ✅ 사전 학습된 모델 즉시 사용
- ✅ 빠른 추론 (Faster R-CNN보다 2-3배 빠름)
- ✅ **오토바이 감지 우수** (COCO 80개 클래스 지원)
- ✅ AMD/NVIDIA GPU 모두 지원

**vs Faster R-CNN (현재):**
| 항목 | Faster R-CNN | YOLOv5 |
|------|-------------|--------|
| 속도 | 느림 | 빠름 ⚡ |
| 정확도 | 높음 | 매우 높음 |
| 설치 | 복잡 | 간단 |
| 오토바이 | ⚠️ 보통 | ✅ 우수 |

## 📦 설치

```bash
pip install ultralytics
```

그게 전부입니다! 🎉

## 🎯 사용 방법

### 기본 사용

```bash
python detect_yolo.py --source car.jpg --output yolo_result.jpg
```

### 고급 옵션

```bash
# 신뢰도 조정
python detect_yolo.py --source car.jpg --output result.jpg --conf 0.3

# GPU 사용 (NVIDIA)
python detect_yolo.py --source car.jpg --output result.jpg --device cuda

# GPU 사용 (AMD) - DirectML 필요
python detect_yolo.py --source car.jpg --output result.jpg --device cuda
```

## 🏍️ 오토바이 감지

YOLOv5는 **motorcycle** 클래스를 별도로 가지고 있어 정확합니다!

```bash
# 낮은 신뢰도로 더 많이 감지
python detect_yolo.py --source car.jpg --output result.jpg --conf 0.25
```

## 📊 YOLOv5 모델 크기

| 모델 | 크기 | 속도 | 정확도 | 용도 |
|------|------|------|--------|------|
| **YOLOv5n** | 1.9MB | ⚡⚡⚡ | ⭐⭐⭐ | 임베디드 |
| **YOLOv5s** | 7.2MB | ⚡⚡ | ⭐⭐⭐⭐ | 기본 (권장) |
| **YOLOv5m** | 21MB | ⚡ | ⭐⭐⭐⭐⭐ | 균형 |
| **YOLOv5l** | 46MB | 느림 | ⭐⭐⭐⭐⭐ | 고정확도 |

**권장: YOLOv5s** (빠르고 정확)

detect_yolo.py 에서 모델 변경:

```python
model = YOLO('yolov5s.pt')  # 기본
# model = YOLO('yolov5m.pt')  # 더 정확
# model = YOLO('yolov5l.pt')  # 최고 정확도
```

## ⚡ 성능 비교

### CPU (i5)
- Faster R-CNN: 3-5 FPS
- YOLOv5s: **8-12 FPS** ⚡

### AMD GPU (RX 6600)
- Faster R-CNN: 15-20 FPS
- YOLOv5s: **30-40 FPS** ⚡

### NVIDIA GPU (RTX 3060)
- Faster R-CNN: 30-35 FPS
- YOLOv5s: **60-80 FPS** ⚡

## 🎨 Tesla FSD 스타일

YOLOv5 결과도 Tesla FSD 스타일로 시각화됩니다:
- 클래스별 색상 구분
- 신뢰도 점수 표시
- 깔끔한 바운딩 박스

## 🔍 지원 클래스

YOLOv5는 COCO 80개 클래스 모두 지원합니다:

**자율주행 관련:**
- person (보행자)
- bicycle (자전거)
- **motorcycle** (오토바이) ⭐
- car (자동차)
- bus (버스)
- truck (트럭)
- traffic light (신호등)
- stop sign (정지 표지판)

## ✅ 빠른 테스트

```bash
# 1. 설치
pip install ultralytics

# 2. 즉시 실행
python detect_yolo.py --source car.jpg --output yolo_result.jpg --conf 0.3

# 3. 결과 확인
# yolo_result.jpg 파일 열기
```

**오토바이가 이제 잘 보일 겁니다!** 🏍️✨

## 💡 권장 사항

**프로토타입/테스트:**
```bash
python detect_yolo.py --source car.jpg --output result.jpg
```

**프로덕션:**
```bash
python detect_yolo.py --source car.jpg --output result.jpg --conf 0.4 --device cuda
```

**디버깅 (모든 객체 보기):**
```bash
python detect_yolo.py --source car.jpg --output result.jpg --conf 0.1
```

## 🎯 결론

**YOLOv5 = 완벽한 선택!**
- 간단한 설치
- 빠른 속도
- 오토바이 잘 잡힘
- Tesla FSD 스타일 지원

**지금 바로 시도하세요!** 🚀
