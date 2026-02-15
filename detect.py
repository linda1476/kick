"""
Tesla FSD 스타일 객체 감지 및 시각화
"""
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import argparse
import os
from model import YOLOv5Detector, load_model


# 클래스 이름 및 색상
CLASS_NAMES = {
    0: 'Person',
    1: 'Bicycle',
    2: 'Car',
    3: 'Motorcycle',
    4: 'Bus',
    5: 'Train',
    6: 'Truck',
    7: 'Traffic Light',
    8: 'Stop Sign',
}

# Tesla FSD 스타일 색상 (BGR 형식)
CLASS_COLORS = {
    0: (0, 255, 0),      # 보행자 - 초록색
    1: (255, 255, 0),    # 자전거 - 노란색
    2: (255, 0, 0),      # 승용차 - 파란색
    3: (255, 128, 0),    # 오토바이 - 주황색
    4: (255, 0, 255),    # 버스 - 마젠타
    5: (128, 0, 255),    # 기차 - 보라색
    6: (0, 128, 255),    # 트럭 - 하늘색
    7: (0, 255, 255),    # 신호등 - 시안
    8: (0, 0, 255),      # 정지 표지판 - 빨간색
}


def load_image(image_path, device='cuda'):
    """이미지 로드 및 전처리"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 변환
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor, image, original_size


def detect_objects(model, image_tensor, device='cuda', conf_threshold=0.5):
    """객체 감지 수행"""
    model.eval()
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 신뢰도 필터링
    pred = predictions[0]
    keep = pred['scores'] > conf_threshold
    
    boxes = pred['boxes'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    
    return boxes, labels, scores


def draw_tesla_style_boxes(image, boxes, labels, scores, original_size):
    """Tesla FSD 스타일로 바운딩 박스 그리기"""
    # PIL Image를 OpenCV 형식으로 변환
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    
    # 원본 이미지 크기로 바운딩 박스 스케일링
    for box, label, score in zip(boxes, labels, scores):
        # 좌표 추출 (640x640 -> 원본 크기)
        x1, y1, x2, y2 = box
        
        # 스케일 조정
        x1 = int(x1 * width / 640)
        y1 = int(y1 * height / 640)
        x2 = int(x2 * width / 640)
        y2 = int(y2 * height / 640)
        
        # 클래스에 따른 색상
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # 바운딩 박스 그리기 (두께 3)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
        
        # 레이블 텍스트
        class_name = CLASS_NAMES.get(label, f'Class {label}')
        label_text = f'{class_name} {score:.2f}'
        
        # 텍스트 배경
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 배경 사각형
        cv2.rectangle(
            cv_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 텍스트 (흰색)
        cv2.putText(
            cv_image,
            label_text,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    # OpenCV -> PIL 변환
    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    
    return result_image


def process_image(model, image_path, output_path, device='cuda', conf_threshold=0.5):
    """단일 이미지 처리"""
    print(f"\n처리 중: {image_path}")
    
    # 이미지 로드
    image_tensor, original_image, original_size = load_image(image_path, device)
    
    # 객체 감지
    boxes, labels, scores = detect_objects(model, image_tensor, device, conf_threshold)
    
    print(f"감지된 객체: {len(boxes)}개")
    for label, score in zip(labels, scores):
        class_name = CLASS_NAMES.get(label, f'Class {label}')
        print(f"  - {class_name}: {score:.3f}")
    
    # 시각화
    result_image = draw_tesla_style_boxes(
        original_image, boxes, labels, scores, original_size
    )
    
    # 저장
    result_image.save(output_path)
    print(f"저장 완료: {output_path}")
    
    return result_image


def process_video(model, video_path, output_path, device='cuda', conf_threshold=0.5):
    """비디오 처리"""
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 속성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 비디오 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"\n비디오 처리 중: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV -> PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 변환
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 감지
        boxes, labels, scores = detect_objects(model, image_tensor, device, conf_threshold)
        
        # 시각화
        result_image = draw_tesla_style_boxes(pil_image, boxes, labels, scores, (width, height))
        
        # PIL -> OpenCV
        result_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        
        out.write(result_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"처리된 프레임: {frame_count}")
    
    cap.release()
    out.release()
    
    print(f"비디오 저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Tesla FSD 스타일 객체 감지')
    parser.add_argument('--weights', type=str, default='best_model.pth',
                        help='모델 가중치 파일 경로')
    parser.add_argument('--source', type=str, required=True,
                        help='입력 이미지/비디오 경로')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='출력 파일 경로')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='신뢰도 임계값')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스 (cuda/cpu)')
    parser.add_argument('--num-classes', type=int, default=9,
                        help='클래스 수')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    if os.path.exists(args.weights):
        print(f"모델 로딩: {args.weights}")
        model = load_model(args.weights, args.num_classes, device)
    else:
        print("사전 학습된 모델 사용 (COCO 데이터셋)")
        model = YOLOv5Detector(num_classes=args.num_classes, pretrained=True)
        model.to(device)
        model.eval()
    
    # 파일 타입 확인
    ext = os.path.splitext(args.source)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # 이미지 처리
        process_image(model, args.source, args.output, device, args.conf)
    elif ext in ['.mp4', '.avi', '.mov']:
        # 비디오 처리
        process_video(model, args.source, args.output, device, args.conf)
    else:
        print(f"지원하지 않는 파일 형식: {ext}")


if __name__ == '__main__':
    main()
