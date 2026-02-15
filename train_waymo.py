"""
Waymo Open Dataset으로 모델 학습
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import argparse
from tqdm import tqdm
import os

from waymo_dataset import WaymoDetectionDataset, collate_fn
from model import YOLOv5Detector, save_model


def train_epoch(model, dataloader, optimizer, device, epoch):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        
        # 타겟 준비
        targets = []
        for i in range(len(batch['boxes'])):
            target = {
                'boxes': batch['boxes'][i].to(device),
                'labels': batch['labels'][i].to(device)
            }
            targets.append(target)
        
        # Forward
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        
        # Loss 계산
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # 진행률 표시
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Waymo 데이터로 객체 감지 모델 학습')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Waymo TFRecord 파일 경로 (glob 패턴 가능)')
    parser.add_argument('--output-dir', type=str, default='./waymo_checkpoints',
                        help='체크포인트 저장 경로')
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='학습 디바이스')
    parser.add_argument('--num-classes', type=int, default=9,
                        help='클래스 수')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='최대 샘플 수 (테스트용)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Waymo 데이터셋으로 모델 학습")
    print("="*60)
    
    # 데이터셋 로드
    print("\n데이터셋 로딩 중...")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = WaymoDetectionDataset(
            tfrecord_path=args.data_path,
            transform=transform,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        print("\nTensorFlow 및 Waymo 패키지가 설치되었는지 확인하세요:")
        print("pip install tensorflow waymo-open-dataset-tf-2-12-0")
        return
    
    print(f"✓ 데이터셋 크기: {len(dataset)}")
    
    # 데이터로더
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows에서 문제 방지
    )
    
    # 모델 생성
    print("\n모델 생성 중...")
    model = YOLOv5Detector(num_classes=args.num_classes, pretrained=True)
    model.to(device)
    print("✓ 모델 생성 완료")
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습
    print(f"\n학습 시작 ({args.epochs} 에폭)")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
        
        # 체크포인트 저장
        checkpoint_path = output_dir / f'waymo_epoch_{epoch}.pth'
        save_model(model, str(checkpoint_path))
        print(f"체크포인트 저장: {checkpoint_path}")
        
        # 최고 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / 'waymo_best.pth'
            save_model(model, str(best_path))
            print(f"✓ 최고 모델 저장: {best_path}")
    
    print("\n" + "="*60)
    print("✓ 학습 완료!")
    print("="*60)
    print(f"\n최고 모델: {output_dir / 'waymo_best.pth'}")
    print(f"\n추론 실행:")
    print(f"python detect.py --source car.jpg --weights {output_dir / 'waymo_best.pth'} --output waymo_result.jpg")


if __name__ == '__main__':
    main()
