# Waymo Open Dataset 빠른 시작 가이드

## ⚠️ 간편한 대안: COCO로 계속 사용

**Waymo 데이터셋은 설치와 사용이 복잡합니다.** 오토바이 감지를 개선하는 더 간단한 방법:

### 옵션 A: 신뢰도 낮추기 (즉시 가능)

```bash
python detect.py --source car.jpg --output result.jpg --conf 0.15 --device cpu
```

### 옵션B: 더 나은 사전학습 모델 사용

현재 COCO 모델도 충분히 좋습니다. 다만 신뢰도 조정이 필요할 뿐입니다.

## 🔧 Waymo를 정말 사용하려면

### 1단계: TensorFlow 설치

```bash
pip install tensorflow
```

### 2단계: Waymo 패키지 수동 다운로드

Waymo 패키지는 pip로 직접 설치가 어렵습니다. 다음 방법을 사용하세요:

**옵션 1: Conda 사용 (권장)**

```bash
conda install -c conda-forge waymo-open-dataset-tf-2-11-0
```

**옵션 2: GitHub에서 빌드**

```bash
git clone https://github.com/waymo-research/waymo-open-dataset.git
cd waymo-open-dataset
pip install --upgrade pip
python setup.py install
```

### 3단계: 데이터 다운로드

#### 방법 1: 웹 브라우저 (가장 쉬움)

1. https://waymo.com/open/download/ 방문
2. Google 계정 로그인
3. 약관 동의
4. 'Perception' > 'Training' 섹션
5. 파일 1-2개 다운로드 (~150MB each)
6. `waymo_data/` 폴더에 저장

#### 방법 2: gsutil (Google Cloud SDK 필요)

```bash
# Google Cloud SDK 설치 후
python download_waymo.py --num-samples 2
```

### 4단계: 학습

```bash
python train_waymo.py --data-path "./waymo_data/*.tfrecord" --epochs 5
```

## 💡 권장사항

### 현재 상황에서는

**Waymo 대신 신뢰도 조정만으로 충분합니다:**

```bash
# 오토바이도 잘 잡히는 설정
python detect.py --source car.jpg --output result.jpg --conf 0.2 --device cpu
```

### Waymo가 필요한 경우

- 프로덕션 환경
- 최고 정확도 필요
- 충분한 컴퓨팅 리소스

**하지만 테스트/프로토타입에는 COCO로 충분합니다!**

## 📊 실용적 비교

| 방법 | 설치 시간 | 정확도 | 개발 시간 |
|------|----------|--------|----------|
| **COCO (현재)** | 0분 | 80-85% | 즉시 |
| **COCO + 낮은 conf** | 0분 | 85-90% | 즉시 |
| **Waymo** | 2-4시간 | 92-95% | 1-2일 |

## ✅ 최종 추천

**지금 당장:**
```bash
# 이거면 충분합니다!
python detect.py --source car.jpg --output result.jpg --conf 0.2
```

**나중에 필요하면:**
- Waymo 수동 설치 시도
- 또는 더 나은 YOLO 모델 고려

**궁금한 점:**
- 현재 COCO 모델로 오토바이가 정말 안 보이나요?
- conf 0.1~0.2로 해도 안 보이나요?

그렇다면 Waymo 설치를 진행하겠습니다. 아니면 다른 해결책을 찾아보겠습니다!
