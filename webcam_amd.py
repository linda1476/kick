"""
AMD GPU용 실시간 웹캠 객체 감지 - DirectML 지원
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import argparse
from model import YOLOv5Detector, load_model
import time
import sys


# 클래스 이름 및 색상
CLASS_NAMES = {
    0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle',
    4: 'Bus', 5: 'Train', 6: 'Truck', 7: 'Traffic Light', 8: 'Stop Sign'
}

CLASS_COLORS = {
    0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 0, 0), 3: (255, 128, 0),
    4: (255, 0, 255), 5: (128, 0, 255), 6: (0, 128, 255),
    7: (0, 255, 255), 8: (0, 0, 255)
}


def get_device(device_name='cuda'):
    """디바이스 설정 - DirectML(AMD), CUDA(NVIDIA), CPU 지원"""
    
    # DirectML 시도 (AMD GPU)
    if device_name.lower() in ['dml', 'directml', 'amd']:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"✅ DirectML (AMD GPU) 사용")
            return device, 'directml'
        except ImportError:
            print("❌ torch-directml이 설치되지 않았습니다.")
            print("설치: pip install torch-directml")
            print("CPU 모드로 전환합니다...")
            return torch.device('cpu'), 'cpu'
    
    # CUDA 시도 (NVIDIA GPU)
    elif device_name.lower() == 'cuda':
        if torch.cuda.is_available():
            print(f"✅ CUDA (NVIDIA GPU) 사용")
            return torch.device('cuda'), 'cuda'
        else:
            print("❌ CUDA를 사용할 수 없습니다. CPU 모드로 전환합니다...")
            return torch.device('cpu'), 'cpu'
    
    # CPU
    else:
        print(f"✅ CPU 사용")
        return torch.device('cpu'), 'cpu'


def detect_objects(model, image_tensor, device, device_type, conf_threshold=0.5):
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


def draw_tesla_style_boxes(frame, boxes, labels, scores):
    """Tesla FSD 스타일로 바운딩 박스 그리기"""
    height, width = frame.shape[:2]
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1 = int(x1 * width / 640)
        y1 = int(y1 * height / 640)
        x2 = int(x2 * width / 640)
        y2 = int(y2 * height / 640)
        
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        class_name = CLASS_NAMES.get(label, f'Class {label}')
        label_text = f'{class_name} {score:.2f}'
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1), color, -1)
        
        cv2.putText(frame, label_text, (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def webcam_detection(model, device, device_type, conf_threshold=0.5, camera_id=0):
    """웹캠 실시간 객체 감지"""
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 카메라 {camera_id}를 열 수 없습니다!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Tesla FSD 스타일 실시간 객체 감지")
    print("="*60)
    print(f"카메라 ID: {camera_id}")
    print(f"디바이스: {device_type}")
    print(f"신뢰도 임계값: {conf_threshold}")
    print("\n조작법:")
    print("  - 'q' 키: 종료")
    print("  - 's' 키: 스크린샷 저장")
    print("  - '+' 키: 신뢰도 증가")
    print("  - '-' 키: 신뢰도 감소")
    print("="*60 + "\n")
    
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    fps_time = time.time()
    fps = 0
    frame_count = 0
    screenshot_count = 0
    current_conf = conf_threshold
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            boxes, labels, scores = detect_objects(model, image_tensor, device, device_type, current_conf)
            
            frame = draw_tesla_style_boxes(frame, boxes, labels, scores)
            
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            info_text = f"FPS: {fps} | Objects: {len(boxes)} | Conf: {current_conf:.2f} | {device_type.upper()}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_offset = 60
            for label in set(labels):
                count = sum(1 for l in labels if l == label)
                class_name = CLASS_NAMES.get(label, f'Class {label}')
                obj_text = f"{class_name}: {count}"
                cv2.putText(frame, obj_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
            
            cv2.imshow('Tesla FSD - Real-time Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n종료합니다...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"스크린샷 저장: {filename}")
            elif key == ord('+') or key == ord('='):
                current_conf = min(0.95, current_conf + 0.05)
                print(f"신뢰도 임계값: {current_conf:.2f}")
            elif key == ord('-') or key == ord('_'):
                current_conf = max(0.1, current_conf - 0.05)
                print(f"신뢰도 임계값: {current_conf:.2f}")
    
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n웹캠이 종료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='Tesla FSD 스타일 실시간 웹캠 객체 감지 (AMD GPU 지원)')
    parser.add_argument('--weights', type=str, default=None,
                        help='모델 가중치 파일 경로')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='신뢰도 임계값')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'dml', 'directml', 'amd'],
                        help='디바이스: cpu, cuda(NVIDIA), dml/directml/amd(AMD GPU)')
    parser.add_argument('--camera', type=int, default=0,
                        help='카메라 ID')
    parser.add_argument('--num-classes', type=int, default=9,
                        help='클래스 수')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device, device_type = get_device(args.device)
    
    # 모델 로드
    print("모델 로딩 중...")
    if args.weights:
        print(f"가중치 파일: {args.weights}")
        model = load_model(args.weights, args.num_classes, device)
    else:
        print("사전 학습된 모델 사용 (COCO 데이터셋)")
        model = YOLOv5Detector(num_classes=args.num_classes, pretrained=True)
        model.to(device)
        model.eval()
    
    print("✓ 모델 로드 완료!")
    
    # 웹캠 감지 시작
    webcam_detection(model, device, device_type, args.conf, args.camera)


if __name__ == '__main__':
    main()
