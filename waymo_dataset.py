"""
Waymo Open Dataset 로더 - 자율주행 특화
"""
import tensorflow as tf
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import torch
from torch.utils.data import Dataset
from PIL import Image


class WaymoDetectionDataset(Dataset):
    """Waymo Open Dataset for PyTorch"""
    
    # Waymo 클래스 매핑
    WAYMO_CLASSES = {
        0: 'Unknown',
        1: 'Vehicle',      # 차량 (자동차, 트럭, 버스 등)
        2: 'Pedestrian',   # 보행자
        3: 'Sign',         # 표지판
        4: 'Cyclist',      # 자전거/오토바이 탑승자
    }
    
    # Tesla FSD 클래스로 매핑
    CLASS_MAPPING = {
        1: 2,  # Vehicle -> Car
        2: 0,  # Pedestrian -> Person
        3: 8,  # Sign -> Stop Sign
        4: 1,  # Cyclist -> Bicycle (또는 Motorcycle)
    }
    
    def __init__(self, tfrecord_path, transform=None, max_samples=None):
        """
        Args:
            tfrecord_path: Waymo TFRecord 파일 경로
            transform: 이미지 변환
            max_samples: 최대 샘플 수 (None이면 전체)
        """
        self.tfrecord_path = tfrecord_path
        self.transform = transform
        self.max_samples = max_samples
        
        # TFRecord 데이터셋 생성
        self.dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        
        # 샘플 수 제한
        if max_samples:
            self.dataset = self.dataset.take(max_samples)
        
        # 데이터를 메모리에 로드 (작은 데이터셋용)
        self.frames = []
        print(f"Waymo 데이터셋 로딩 중...")
        
        for data in self.dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.frames.append(frame)
            
            if len(self.frames) % 100 == 0:
                print(f"  로드됨: {len(self.frames)} 프레임")
        
        print(f"✓ 총 {len(self.frames)} 프레임 로드 완료")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # 첫 번째 카메라 이미지 사용 (FRONT)
        camera_image = None
        for image in frame.images:
            if image.name == open_dataset.CameraName.FRONT:
                camera_image = image
                break
        
        if camera_image is None:
            camera_image = frame.images[0]
        
        # 이미지 디코딩
        image_array = tf.image.decode_jpeg(camera_image.image).numpy()
        image = Image.fromarray(image_array)
        orig_w, orig_h = image.size
        
        # 라벨 추출
        boxes = []
        labels = []
        
        for camera_label in frame.camera_labels:
            if camera_label.name == camera_image.name:
                for label in camera_label.labels:
                    # 바운딩 박스 (픽셀 좌표)
                    box = label.box
                    
                    x1 = box.center_x - box.length / 2
                    y1 = box.center_y - box.width / 2
                    x2 = box.center_x + box.length / 2
                    y2 = box.center_y + box.width / 2
                    
                    # 정규화
                    x1_norm = max(0, x1 / orig_w)
                    y1_norm = max(0, y1 / orig_h)
                    x2_norm = min(1, x2 / orig_w)
                    y2_norm = min(1, y2 / orig_h)
                    
                    if x2_norm > x1_norm and y2_norm > y1_norm:
                        boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                        
                        # Waymo 클래스 -> Tesla FSD 클래스
                        waymo_class = label.type
                        if waymo_class in self.CLASS_MAPPING:
                            labels.append(self.CLASS_MAPPING[waymo_class])
                        else:
                            labels.append(0)  # Unknown -> Person (기본값)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # Tensor 변환
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'frame_id': frame.context.name
        }


def collate_fn(batch):
    """배치 데이터 처리"""
    images = []
    boxes = []
    labels = []
    frame_ids = []
    
    for item in batch:
        images.append(item['image'])
        boxes.append(item['boxes'])
        labels.append(item['labels'])
        frame_ids.append(item['frame_id'])
    
    images = torch.stack(images, 0)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'frame_ids': frame_ids
    }


if __name__ == '__main__':
    # 테스트 코드
    import torchvision.transforms as transforms
    
    print("Waymo 데이터셋 로더 테스트")
    print("=" * 60)
    
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    # 실제 사용 시 경로 지정
    # dataset = WaymoDetectionDataset(
    #     tfrecord_path='path/to/waymo/tfrecord',
    #     transform=transform,
    #     max_samples=100
    # )
    # print(f"데이터셋 크기: {len(dataset)}")
