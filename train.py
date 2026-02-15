"""
모델 학습 스크립트
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import os

from model import YOLOv5Detector, save_model
from dataset import COCODetectionDataset, collate_fn


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['images'].to(device)
        
        # 타겟 준비
        targets = []
        for i in range(len(images)):
            target = {
                'boxes': batch['boxes'][i].to(device),
                'labels': batch['labels'][i].to(device)
            }
            targets.append(target)
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # 진행 상황 업데이트
        progress_bar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='객체 감지 모델 학습')
    parser.add_argument('--data-root', type=str, required=True,
                        help='이미지 디렉토리 경로')
    parser.add_argument('--annotations', type=str, required=True,
                        help='COCO 어노테이션 파일 경로')
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--num-classes', type=int, default=9,
                        help='클래스 수')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n=== 학습 시작 ===")
    print(f"디바이스: {device}")
    print(f"에폭: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.lr}\n")
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 데이터셋 및 데이터로더
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    print("데이터셋 로딩...")
    dataset = COCODetectionDataset(
        root_dir=args.data_root,
        annotation_file=args.annotations,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"총 샘플 수: {len(dataset)}")
    print(f"배치 수: {len(dataloader)}\n")
    
    # 모델 초기화
    print("모델 초기화...")
    model = YOLOv5Detector(num_classes=args.num_classes, pretrained=True)
    model.to(device)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 루프
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # 학습
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch} 완료 - 평균 Loss: {avg_loss:.4f}")
        
        # 체크포인트 저장
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        save_model(model, optimizer, epoch, avg_loss, checkpoint_path)
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_model(model, optimizer, epoch, avg_loss, best_model_path)
            print(f"★ 최고 성능 모델 업데이트! Loss: {best_loss:.4f}")
    
    print(f"\n{'='*50}")
    print("학습 완료!")
    print(f"최고 성능: Loss {best_loss:.4f}")
    print(f"체크포인트 디렉토리: {args.checkpoint_dir}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
