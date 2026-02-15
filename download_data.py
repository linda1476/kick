"""
COCO 데이터셋 다운로드 스크립트
"""
import os
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """다운로드 진행률 표시"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """URL에서 파일 다운로드"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def download_coco_dataset(data_dir='./data', split='val', year='2017'):
    """
    COCO 데이터셋 다운로드
    
    Args:
        data_dir: 데이터 저장 디렉토리
        split: 'train' 또는 'val'
        year: 데이터셋 연도 (2017 권장)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs
    base_url = 'http://images.cocodataset.org'
    
    if split == 'train':
        images_url = f'{base_url}/zips/train{year}.zip'
        annotations_url = f'http://images.cocodataset.org/annotations/annotations_trainval{year}.zip'
    else:  # val
        images_url = f'{base_url}/zips/val{year}.zip'
        annotations_url = f'http://images.cocodataset.org/annotations/annotations_trainval{year}.zip'
    
    # 이미지 다운로드
    images_zip = os.path.join(data_dir, f'{split}{year}.zip')
    if not os.path.exists(images_zip):
        print(f"\n이미지 다운로드 중: {split}{year}")
        download_url(images_url, images_zip)
        
        print(f"압축 해제 중...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"완료!")
    else:
        print(f"\n이미지 이미 존재: {images_zip}")
    
    # 어노테이션 다운로드
    annotations_zip = os.path.join(data_dir, f'annotations_trainval{year}.zip')
    if not os.path.exists(annotations_zip):
        print(f"\n어노테이션 다운로드 중:")
        download_url(annotations_url, annotations_zip)
        
        print(f"압축 해제 중...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"완료!")
    else:
        print(f"\n어노테이션 이미 존재: {annotations_zip}")
    
    print(f"\n{'='*50}")
    print(f"COCO {year} {split} 데이터셋 준비 완료!")
    print(f"이미지 경로: {os.path.join(data_dir, f'{split}{year}')}")
    print(f"어노테이션: {os.path.join(data_dir, 'annotations', f'instances_{split}{year}.json')}")
    print(f"{'='*50}\n")


def download_sample_images(data_dir='./sample_images'):
    """
    테스트용 샘플 이미지 다운로드
    (COCO 데이터셋이 너무 클 경우 사용)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n샘플 이미지 다운로드...")
    
    # 무료 이미지 URL들 (자율주행 관련)
    sample_urls = [
        'https://images.unsplash.com/photo-1449965408869-eaa3f722e40d',  # 도로
        'https://images.unsplash.com/photo-1502877338535-766e1452684a',  # 차량
        'https://images.unsplash.com/photo-1486428128344-5413e434ad35',  # 보행자
    ]
    
    for i, url in enumerate(sample_urls):
        output_path = os.path.join(data_dir, f'sample_{i+1}.jpg')
        if not os.path.exists(output_path):
            try:
                print(f"다운로드 중: sample_{i+1}.jpg")
                download_url(url, output_path)
            except Exception as e:
                print(f"오류: {e}")
    
    print(f"\n샘플 이미지 저장 위치: {data_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='COCO 데이터셋 다운로드')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='데이터 저장 디렉토리')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='데이터셋 분할')
    parser.add_argument('--year', type=str, default='2017',
                        help='데이터셋 연도')
    parser.add_argument('--sample', action='store_true',
                        help='샘플 이미지만 다운로드')
    
    args = parser.parse_args()
    
    if args.sample:
        download_sample_images()
    else:
        print("\n⚠️  경고: COCO 데이터셋은 크기가 매우 큽니다!")
        print(f"  - train2017: ~18GB")
        print(f"  - val2017: ~1GB")
        print(f"  - annotations: ~241MB\n")
        
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            download_coco_dataset(args.data_dir, args.split, args.year)
        else:
            print("취소되었습니다.")


if __name__ == '__main__':
    main()
