"""
YOLOv5를 사용한 간단한 객체 감지 - Tesla FSD 스타일
"""
import cv2
import torch
from ultralytics import YOLO
import argparse
import numpy as np


# Tesla FSD 스타일 색상
CLASS_COLORS = {
    'person': (0, 255, 0),      # 녹색
    'bicycle': (255, 255, 0),   # 노란색
    'car': (255, 0, 0),         # 파란색
    'motorcycle': (255, 128, 0), # 주황색
    'bus': (255, 0, 255),       # 보라색
    'train': (128, 0, 255),     # 진보라
    'truck': (0, 128, 255),     # 하늘색
    'traffic light': (0, 255, 255), # 청록색
    'stop sign': (0, 0, 255),   # 빨간색
}


def detect_yolo(image_path, output_path, conf=0.5, device='cpu'):
    """
    YOLOv5로 객체 감지
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        conf: 신뢰도 임계값
        device: 'cuda' 또는 'cpu'
    """
    
    print("\n" + "="*60)
    print("YOLOv5 객체 감지")
    print("="*60)
    
    # YOLOv5 모델 로드 (사전 학습된 모델)
    print("\n모델 로딩 중...")
    model = YOLO('yolov5s.pt')  # small 모델 (빠름)
    # model = YOLO('yolov5m.pt')  # medium 모델 (균형)
    # model = YOLO('yolov5l.pt')  # large 모델 (정확)
    print("✓ YOLOv5s 모델 로드 완료!")
    
    # 이미지 로드
    print(f"\n이미지 로딩: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 열 수 없습니다: {image_path}")
        return
    
    # 추론
    print(f"객체 감지 중 (디바이스: {device}, 신뢰도: {conf})...")
    results = model(image, conf=conf, device=device)
    
    # 결과 추출
    detections = results[0].boxes
    num_objects = len(detections)
    
    print(f"\n✓ 감지 완료! 총 {num_objects}개 객체 발견")
    
    # Tesla FSD 스타일로 시각화
    output_image = image.copy()
    
    for detection in detections:
        # 바운딩 박스
        box = detection.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        # 클래스 및 신뢰도
        cls_id = int(detection.cls[0])
        conf_score = float(detection.conf[0])
        class_name = model.names[cls_id]
        
        # 색상
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # 바운딩 박스 그리기
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
        
        # 레이블
        label = f'{class_name} {conf_score:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 레이블 배경
        cv2.rectangle(output_image,
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     color, -1)
        
        # 레이블 텍스트
        cv2.putText(output_image, label,
                   (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"  - {class_name}: {conf_score:.2f}")
    
    # 저장
    cv2.imwrite(output_path, output_image)
    print(f"\n✓ 저장 완료: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 객체 감지 (Tesla FSD 스타일)')
    parser.add_argument('--source', type=str, required=True,
                        help='입력 이미지 경로')
    parser.add_argument('--output', type=str, required=True,
                        help='출력 이미지 경로')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='디바이스')
    
    args = parser.parse_args()
    
    detect_yolo(args.source, args.output, args.conf, args.device)


if __name__ == '__main__':
    main()
