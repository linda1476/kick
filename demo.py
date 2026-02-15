"""
간단한 데모 스크립트 - 사전 학습된 모델로 빠른 테스트
"""
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import cv2
from model import YOLOv5Detector


# 클래스 정의
CLASS_NAMES = {
    0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle',
    4: 'Bus', 5: 'Train', 6: 'Truck', 7: 'Traffic Light', 8: 'Stop Sign'
}

CLASS_COLORS = {
    0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 0, 0), 3: (255, 128, 0),
    4: (255, 0, 255), 5: (128, 0, 255), 6: (0, 128, 255),
    7: (0, 255, 255), 8: (0, 0, 255)
}


def demo_detection():
    """데모 실행"""
    print("="*60)
    print("Tesla FSD 스타일 객체 감지 데모")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    
    # 모델 로드
    print("모델 로딩 중...")
    model = YOLOv5Detector(num_classes=9, pretrained=True)
    model.to(device)
    model.eval()
    print("✓ 모델 로드 완료!")
    
    # 샘플 이미지 URL (도로 환경)
    sample_url = "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800"
    
    print(f"\n샘플 이미지 다운로드 중...")
    try:
        response = requests.get(sample_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print("✓ 이미지 다운로드 완료!")
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        print("로컬 이미지를 사용하려면 detect.py를 사용하세요.")
        return
    
    # 전처리
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 추론
    print("\n객체 감지 중...")
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 결과 처리
    pred = predictions[0]
    keep = pred['scores'] > 0.5
    
    boxes = pred['boxes'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    
    print(f"\n✓ 감지 완료! 총 {len(boxes)}개 객체 발견")
    
    # 감지된 객체 출력
    if len(boxes) > 0:
        print("\n감지된 객체:")
        for i, (label, score) in enumerate(zip(labels, scores)):
            class_name = CLASS_NAMES.get(label, f'Class {label}')
            print(f"  {i+1}. {class_name}: {score:.3f}")
    
    # 시각화
    print("\n시각화 중...")
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # 바운딩 박스
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
        
        # 레이블
        class_name = CLASS_NAMES.get(label, f'Class {label}')
        label_text = f'{class_name} {score:.2f}'
        
        cv2.putText(cv_image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 저장
    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    result_image.save('demo_result.jpg')
    
    print("\n" + "="*60)
    print("✓ 완료! 결과가 'demo_result.jpg'에 저장되었습니다.")
    print("="*60)


if __name__ == '__main__':
    demo_detection()
