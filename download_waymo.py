"""
Waymo Open Dataset 샘플 다운로드 스크립트
"""
import os
import subprocess
import sys
from pathlib import Path


def check_gsutil():
    """gsutil 설치 확인"""
    try:
        result = subprocess.run(['gsutil', '--version'], 
                              capture_output=True, 
                              text=True)
        print(f"✓ gsutil 설치됨: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ gsutil이 설치되지 않았습니다.")
        print("\n설치 방법:")
        print("1. Google Cloud SDK 다운로드: https://cloud.google.com/sdk/docs/install")
        print("2. 설치 후 'gcloud init' 실행")
        return False


def download_waymo_sample(output_dir='./waymo_data', num_samples=5):
    """
    Waymo 샘플 데이터 다운로드
    
    Args:
        output_dir: 저장 경로
        num_samples: 다운로드할 샘플 수 (1개 ~150MB)
    """
    
    print("\n" + "="*60)
    print("Waymo Open Dataset 샘플 다운로드")
    print("="*60)
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n저장 경로: {output_path.absolute()}")
    print(f"샘플 수: {num_samples}개 (각 ~150MB)")
    print(f"총 예상 크기: ~{num_samples * 150}MB\n")
    
    # gsutil 확인
    if not check_gsutil():
        print("\n대안: 웹 브라우저로 다운로드")
        print("https://waymo.com/open/download/")
        return False
    
    # Waymo 데이터셋 경로
    base_url = "gs://waymo_open_dataset_v_1_4_1/training/"
    
    # 샘플 파일 목록 (미리 선정된 파일들)
    sample_files = [
        "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord",
        "segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord",
        "segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord",
        "segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord",
        "segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord",
    ]
    
    # 지정된 수만큼 다운로드
    downloaded = 0
    for i, filename in enumerate(sample_files[:num_samples], 1):
        print(f"\n[{i}/{num_samples}] {filename}")
        
        # 이미 존재하는지 확인
        output_file = output_path / filename
        if output_file.exists():
            print(f"  ✓ 이미 존재함 (건너뜀)")
            downloaded += 1
            continue
        
        # 다운로드
        source = base_url + filename
        try:
            print(f"  다운로드 중...")
            result = subprocess.run(
                ['gsutil', 'cp', source, str(output_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10분 타임아웃
            )
            
            if result.returncode == 0:
                print(f"  ✓ 다운로드 완료")
                downloaded += 1
            else:
                print(f"  ❌ 다운로드 실패: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print(f"  ❌ 타임아웃 (파일이 너무 큼)")
        except Exception as e:
            print(f"  ❌ 오류: {e}")
    
    print("\n" + "="*60)
    print(f"✓ 완료! {downloaded}/{num_samples}개 파일 다운로드")
    print("="*60)
    
    if downloaded > 0:
        print(f"\n다음 단계:")
        print(f"python train_waymo.py --data-path {output_dir}")
    
    return downloaded > 0


def download_alternative():
    """
    gsutil 없이 다운로드하는 대안 방법 안내
    """
    print("\n" + "="*60)
    print("대안: 웹 브라우저로 다운로드")
    print("="*60)
    
    print("\n1. https://waymo.com/open/download/ 방문")
    print("2. Google 계정으로 로그인")
    print("3. 약관 동의")
    print("4. 'Perception' 탭 선택")
    print("5. 'Training' 섹션에서 원하는 파일 다운로드")
    print("6. 다운로드한 파일을 ./waymo_data/ 폴더에 저장")
    
    print("\n권장 파일 (1-2개만):")
    print("  - segment-10017090168044687777_*_with_camera_labels.tfrecord")
    print("  - segment-10023947602400723454_*_with_camera_labels.tfrecord")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Waymo 샘플 데이터 다운로드')
    parser.add_argument('--output', type=str, default='./waymo_data',
                        help='저장 경로')
    parser.add_argument('--num-samples', type=int, default=2,
                        help='다운로드할 샘플 수 (1개 ~150MB)')
    parser.add_argument('--manual', action='store_true',
                        help='수동 다운로드 방법 표시')
    
    args = parser.parse_args()
    
    if args.manual:
        download_alternative()
    else:
        success = download_waymo_sample(args.output, args.num_samples)
        if not success:
            print("\n자동 다운로드 실패. 수동 다운로드 방법 확인:")
            print("python download_waymo.py --manual")
