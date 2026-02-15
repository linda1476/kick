"""
동적 객체 추적이 포함된 실시간 웹캠 감지 - AMD GPU 지원
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import argparse
from model import YOLOv5Detector, load_model
from tracker import ObjectTracker
import time
import math


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
    """디바이스 설정"""
    if device_name.lower() in ['dml', 'directml', 'amd']:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"✅ DirectML (AMD GPU) 사용")
            return device, 'directml'
        except ImportError:
            print("❌ torch-directml이 설치되지 않았습니다.")
            print("CPU 모드로 전환합니다...")
            return torch.device('cpu'), 'cpu'
    elif device_name.lower() == 'cuda':
        if torch.cuda.is_available():
            print(f"✅ CUDA (NVIDIA GPU) 사용")
            return torch.device('cuda'), 'cuda'
        else:
            print("❌ CUDA를 사용할 수 없습니다. CPU 모드로 전환합니다...")
            return torch.device('cpu'), 'cpu'
    else:
        print(f"✅ CPU 사용")
        return torch.device('cpu'), 'cpu'


def detect_objects(model, image_tensor, device, device_type, conf_threshold=0.5):
    """객체 감지 수행"""
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred = predictions[0]
    keep = pred['scores'] > conf_threshold
    
    boxes = pred['boxes'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    
    return boxes, labels, scores


def draw_trajectory(frame, trajectory, color, thickness=2):
    """이동 경로 그리기"""
    if len(trajectory) < 2:
        return
    
    points = np.array(trajectory, dtype=np.int32)
    
    # 경로 선 그리기
    for i in range(1, len(points)):
        # 이전 점에서 현재 점까지 선
        alpha = i / len(points)  # 투명도 (최근일수록 진함)
        thickness_scaled = max(1, int(thickness * alpha))
        cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness_scaled)
    
    # 현재 위치에 점 표시
    if len(points) > 0:
        cv2.circle(frame, tuple(points[-1]), 5, color, -1)


def draw_velocity_arrow(frame, center, velocity, color):
    """속도 화살표 그리기"""
    vx, vy = velocity
    speed = math.sqrt(vx**2 + vy**2)
    
    if speed < 1:  # 거의 정지
        return
    
    # 화살표 끝점 계산 (속도에 비례)
    scale = 3
    end_x = int(center[0] + vx * scale)
    end_y = int(center[1] + vy * scale)
    
    # 화살표 그리기
    cv2.arrowedLine(frame, 
                   (int(center[0]), int(center[1])),
                   (end_x, end_y),
                   color, 3, tipLength=0.3)


def draw_tracked_boxes(frame, tracked_objects, tracker, width, height):
    """추적된 객체들을 Tesla FSD 스타일로 그리기"""
    
    for object_id, (box, label, score) in tracked_objects.items():
        # 좌표 스케일링
        x1, y1, x2, y2 = box
        x1 = int(x1 * width / 640)
        y1 = int(y1 * height / 640)
        x2 = int(x2 * width / 640)
        y2 = int(y2 * height / 640)
        
        # 색상
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # 이동 경로 그리기
        trajectory = tracker.get_trajectory(object_id)
        if len(trajectory) > 1:
            # 스케일링된 경로
            scaled_trajectory = [
                (int(p[0] * width / 640), int(p[1] * height / 640))
                for p in trajectory
            ]
            draw_trajectory(frame, scaled_trajectory, color)
        
        # 속도 화살표
        velocity = tracker.get_velocity(object_id)
        if len(trajectory) > 0:
            center = trajectory[-1]
            scaled_center = (int(center[0] * width / 640), int(center[1] * height / 640))
            scaled_velocity = (velocity[0] * width / 640, velocity[1] * height / 640)
            draw_velocity_arrow(frame, scaled_center, scaled_velocity, color)
        
        # 레이블 텍스트 (ID 포함)
        class_name = CLASS_NAMES.get(label, f'Class {label}')
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        label_text = f'ID{object_id} {class_name} {score:.2f}'
        
        if speed > 1:
            label_text += f' {speed:.1f}px/f'
        
        # 텍스트 배경
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1), color, -1)
        
        cv2.putText(frame, label_text, (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def webcam_tracking(model, device, device_type, conf_threshold=0.5, camera_id=0):
    """웹캠 실시간 객체 추적"""
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 카메라 {camera_id}를 열 수 없습니다!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Tesla FSD 스타일 동적 객체 추적")
    print("="*60)
    print(f"카메라 ID: {camera_id}")
    print(f"디바이스: {device_type}")
    print(f"신뢰도 임계값: {conf_threshold}")
    print("\n기능:")
    print("  ✅ 객체 ID 부여")
    print("  ✅ 이동 경로 표시")
    print("  ✅ 속도 및 방향 화살표")
    print("\n조작법:")
    print("  - 'q' 키: 종료")
    print("  - 's' 키: 스크린샷 저장")
    print("  - '+' 키: 신뢰도 증가")
    print("  - '-' 키: 신뢰도 감소")
    print("="*60 + "\n")
    
    # 객체 추적기 초기화
    tracker = ObjectTracker(max_disappeared=30, max_distance=50)
    
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
            
            height, width = frame.shape[:2]
            
            # 전처리
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # 객체 감지
            boxes, labels, scores = detect_objects(model, image_tensor, device, device_type, current_conf)
            
            # 객체 추적 업데이트
            tracked_objects = tracker.update(boxes, labels, scores)
            
            # 시각화
            frame = draw_tracked_boxes(frame, tracked_objects, tracker, width, height)
            
            # FPS 계산
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # 정보 표시
            info_text = f"FPS: {fps} | Tracked: {len(tracked_objects)} | Conf: {current_conf:.2f} | {device_type.upper()}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 프레임 표시
            cv2.imshow('Tesla FSD - Dynamic Object Tracking', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n종료합니다...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'tracking_{screenshot_count}.jpg'
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
    parser = argparse.ArgumentParser(description='Tesla FSD 스타일 동적 객체 추적')
    parser.add_argument('--weights', type=str, default=None,
                        help='모델 가중치 파일 경로')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='신뢰도 임계값')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'dml', 'directml', 'amd'],
                        help='디바이스')
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
        model = load_model(args.weights, args.num_classes, device)
    else:
        print("사전 학습된 모델 사용 (COCO 데이터셋)")
        model = YOLOv5Detector(num_classes=args.num_classes, pretrained=True)
        model.to(device)
        model.eval()
    
    print("✓ 모델 로드 완료!")
    
    # 웹캠 추적 시작
    webcam_tracking(model, device, device_type, args.conf, args.camera)


if __name__ == '__main__':
    main()
