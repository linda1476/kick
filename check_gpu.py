"""
시스템 GPU/CUDA 지원 확인 스크립트
"""
import torch
import sys
import platform

print("="*60)
print("Tesla FSD - GPU/CUDA 진단 도구")
print("="*60)

# 시스템 정보
print(f"\n💻 시스템: {platform.system()} {platform.release()}")

# PyTorch 버전
print(f"\n📦 PyTorch 버전: {torch.__version__}")

# CUDA 지원 확인 (NVIDIA GPU)
cuda_available = torch.cuda.is_available()
print(f"\n🔍 CUDA 사용 가능 (NVIDIA): {'✅ 예' if cuda_available else '❌ 아니오'}")

if cuda_available:
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   cuDNN 버전: {torch.backends.cudnn.version()}")
    print(f"   GPU 개수: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      - 메모리: {props.total_memory / 1024**3:.2f} GB")
        print(f"      - 컴퓨팅 능력: {props.major}.{props.minor}")
    
    print(f"\n✅ NVIDIA GPU를 사용할 수 있습니다!")
    print(f"   웹캠 실행: python webcam.py --device cuda")
else:
    print(f"\n❌ CUDA를 사용할 수 없습니다.")
    
    # AMD GPU 확인
    print(f"\n🔍 AMD GPU 확인 중...")
    if platform.system() == "Windows":
        print(f"   ⚠️  AMD GPU를 사용하시나요?")
        print(f"   Windows에서 AMD GPU 가속:")
        print(f"   1. DirectML 설치: pip install torch-directml")
        print(f"   2. 또는 CPU 모드 사용 (권장)")
    elif platform.system() == "Linux":
        print(f"   ⚠️  AMD GPU를 사용하시나요?")
        print(f"   Linux에서 AMD GPU 가속:")
        print(f"   1. ROCm PyTorch 설치")
        print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
    
    print(f"\n가능한 원인:")
    print(f"   1. NVIDIA GPU가 없음 (AMD GPU일 수 있음)")
    print(f"   2. GPU 드라이버가 설치되지 않음")
    print(f"   3. PyTorch CPU 버전이 설치됨")
    
    print(f"\n해결 방법:")
    print(f"   [NVIDIA GPU를 사용하는 경우]")
    print(f"      pip uninstall torch torchvision")
    print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print(f"\n   [AMD GPU를 사용하는 경우]")
    print(f"      - Windows: pip install torch-directml (제한적 지원)")
    print(f"      - Linux: ROCm 버전 설치")
    print(f"\n   [GPU 없이 사용]")
    print(f"      python webcam.py --device cpu (가장 간단)")

# CPU 정보
print(f"\n💻 CPU 모드: 항상 사용 가능 ✅")
print(f"   웹캠 실행: python webcam.py --device cpu")
print(f"   권장: AMD GPU는 설정이 복잡하므로 CPU 모드 사용")

print("\n" + "="*60)

if cuda_available:
    print("✅ NVIDIA GPU 준비 완료! 빠른 추론이 가능합니다.")
    sys.exit(0)
else:
    print("⚠️  GPU 가속 불가: CPU 모드 사용을 권장합니다.")
    print("   python webcam.py --device cpu")
    sys.exit(1)
