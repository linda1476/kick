# AMD GPU 사용 가이드

## 현재 상태

✅ PyTorch 설치 완료: **CPU 버전**
❌ GPU 지원: **AMD GPU는 CUDA 미지원**
✅ 시스템: **AMD GPU 보유**

## AMD GPU + Windows 해결 방법

### 옵션 1: DirectML 사용 (권장 - Windows 전용)

DirectML은 Microsoft가 제공하는 AMD GPU 가속 솔루션입니다.

```bash
# DirectML 설치
pip install torch-directml
```

**사용 방법:**
```bash
# webcam.py에서 --device dml 사용
python webcam.py --device dml
```

### 옵션 2: CPU 모드 사용 (가장 간단)

AMD GPU 가속이 복잡하다면 CPU 모드로 사용하세요:

```bash
python webcam.py --device cpu
```

**CPU 모드 성능:**
- FPS: 5-10 FPS
- 메모리: 약 2GB
- 느리지만 모든 기능 정상 동작

### 옵션 3: ROCm (Linux 전용)

Linux를 사용한다면 ROCm 버전의 PyTorch를 설치할 수 있습니다:

```bash
# Linux에서만 가능
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

## DirectML 설정 방법 (Windows)

### 1단계: torch-directml 설치

```bash
pip install torch-directml
```

### 2단계: 코드 수정 필요

webcam.py에서 DirectML을 사용하려면 디바이스 설정을 조정해야 합니다:

```python
import torch_directml
device = torch_directml.device()
```

### 3단계: 실행

```bash
python webcam.py --device dml
```

## AMD GPU 성능 비교

| 모드 | FPS | 지원 여부 |
|------|-----|----------|
| **CUDA (NVIDIA)** | ~30 FPS | ❌ AMD 미지원 |
| **DirectML (AMD)** | ~15-20 FPS | ✅ Windows |
| **ROCm (AMD)** | ~25 FPS | ✅ Linux만 |
| **CPU** | ~5-10 FPS | ✅ 항상 가능 |

## 권장 사항

### Windows + AMD GPU:
1. **DirectML 시도** (중간 성능)
2. DirectML이 안되면 **CPU 모드** (느리지만 안정적)

### Linux + AMD GPU:
1. **ROCm PyTorch** (최고 성능)
2. 안되면 **CPU 모드**

## 현실적인 선택

대부분의 경우 **CPU 모드가 가장 간단하고 안정적**입니다:

```bash
# 그냥 이렇게 사용하세요
python webcam.py --device cpu
```

AMD GPU 가속은 설정이 복잡하고 불안정할 수 있습니다. CPU 모드로도 충분히 테스트와 데모가 가능합니다.
