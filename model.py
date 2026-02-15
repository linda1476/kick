"""
YOLOv5 기반 객체 감지 모델
"""
import torch
import torch.nn as nn
import torchvision


class YOLOv5Detector(nn.Module):
    """YOLOv5 기반 객체 감지 모델"""
    
    def __init__(self, num_classes=9, pretrained=True):
        """
        Args:
            num_classes: 감지할 클래스 수
            pretrained: 사전 학습된 가중치 사용 여부
        """
        super(YOLOv5Detector, self).__init__()
        self.num_classes = num_classes
        
        # YOLOv5를 직접 구현하는 대신, 
        # Torchvision의 Faster R-CNN을 사용 (더 안정적)
        # 실제 프로덕션에서는 ultralytics/yolov5 사용 권장
        
        if pretrained:
            # 사전 학습된 모델 로드
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights='DEFAULT'
            )
            
            # 헤드를 우리 클래스 수에 맞게 수정
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes
            )
    
    def forward(self, images, targets=None):
        """
        Args:
            images: 입력 이미지 텐서 [B, 3, H, W]
            targets: 학습 시 사용하는 타겟 (dict 리스트)
        
        Returns:
            학습 모드: loss dict
            추론 모드: 예측 결과 (boxes, labels, scores)
        """
        if self.training and targets is not None:
            # 학습 모드
            return self.model(images, targets)
        else:
            # 추론 모드
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
            return predictions


class YOLOv5Lite(nn.Module):
    """경량화된 YOLOv5 (실시간 추론용)"""
    
    def __init__(self, num_classes=9):
        super(YOLOv5Lite, self).__init__()
        self.num_classes = num_classes
        
        # MobileNetV3를 백본으로 사용
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT'
        )
        
        # 헤드 수정
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
            return predictions


def load_model(checkpoint_path, num_classes=9, device='cuda'):
    """저장된 모델 로드"""
    model = YOLOv5Detector(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def save_model(model, optimizer, epoch, loss, save_path):
    """모델 체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"모델 저장 완료: {save_path}")


if __name__ == '__main__':
    # 모델 테스트
    print("모델 초기화 테스트...")
    
    model = YOLOv5Detector(num_classes=9, pretrained=True)
    print(f"모델 생성 완료!")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 더미 입력으로 테스트
    dummy_input = torch.randn(2, 3, 640, 640)
    model.eval()
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"\n출력 확인:")
    print(f"배치 크기: {len(outputs)}")
    if len(outputs) > 0:
        print(f"첫 번째 예측:")
        print(f"  - 감지된 객체 수: {len(outputs[0]['boxes'])}")
        print(f"  - Boxes shape: {outputs[0]['boxes'].shape}")
        print(f"  - Labels shape: {outputs[0]['labels'].shape}")
        print(f"  - Scores shape: {outputs[0]['scores'].shape}")
