# AMD GPU 실행 명령어

## ✅ DirectML 설치 확인됨!

```
torch-directml    0.2.5.dev240914
```

## 🚀 AMD GPU 실행 명령어

### 방법 1: AMD 전용 스크립트 (권장)

```bash
python webcam_amd.py --device dml
```

또는:

```bash
python webcam_amd.py --device amd
python webcam_amd.py --device directml
```

### 방법 2: 원본 스크립트 (CPU 모드)

```bash
python webcam.py --device cpu
```

## 📊 성능 비교

| 명령어 | GPU 사용 | 예상 FPS |
|--------|---------|----------|
| `python webcam_amd.py --device dml` | AMD GPU | 15-20 FPS |
| `python webcam.py --device cpu` | CPU | 5-10 FPS |

## 🎯 추천 명령어

AMD GPU를 최대한 활용하려면:

```bash
python webcam_amd.py --device dml --conf 0.5
```

## 📝 전체 옵션

```bash
# 기본 실행
python webcam_amd.py --device dml

# 신뢰도 조정
python webcam_amd.py --device dml --conf 0.7

# 다른 카메라 사용
python webcam_amd.py --device dml --camera 1

# 모든 옵션
python webcam_amd.py --device dml --conf 0.6 --camera 0
```

## ⚙️ 조작법

실행 후 창에서:
- `q`: 종료
- `s`: 스크린샷 저장
- `+`: 신뢰도 증가
- `-`: 신뢰도 감소

## 🔍 문제 해결

### DirectML이 작동하지 않으면:

```bash
# CPU 모드로 대체
python webcam_amd.py --device cpu
```

### FPS가 너무 낮으면:

```bash
# 신뢰도를 높여 처리량 감소
python webcam_amd.py --device dml --conf 0.7
```
