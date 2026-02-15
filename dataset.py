"""
COCO 데이터셋 로더 - 차량 자율주행용 객체 감지
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO


class COCODetectionDataset(Dataset):
    """COCO 형식 데이터셋을 위한 PyTorch Dataset"""
    
    # 자율주행 관련 클래스만 필터링
    DRIVING_CLASSES = {
        'person': 0,      # 보행자
        'bicycle': 1,     # 자전거
        'car': 2,         # 승용차
        'motorcycle': 3,  # 오토바이
        'bus': 4,         # 버스
        'train': 5,       # 기차
        'truck': 6,       # 트럭
        'traffic light': 7,  # 신호등
        'stop sign': 8,   # 정지 표지판
    }
    
    def __init__(self, root_dir, annotation_file, transform=None, img_size=640):
        """
        Args:
            root_dir: 이미지 디렉토리 경로
            annotation_file: COCO 형식 JSON 파일 경로
            transform: 이미지 변환 함수
            img_size: 출력 이미지 크기
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        
        # COCO API 초기화
        self.coco = COCO(annotation_file)
        
        # 자율주행 관련 카테고리 ID 가져오기
        self.category_ids = []
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            if cat['name'] in self.DRIVING_CLASSES:
                self.category_ids.append(cat['id'])
        
        # 해당 카테고리를 포함하는 이미지 ID 필터링
        self.image_ids = []
        for cat_id in self.category_ids:
            self.image_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        self.image_ids = list(set(self.image_ids))  # 중복 제거
        
        print(f"로드된 이미지 수: {len(self.image_ids)}")
        print(f"대상 클래스: {list(self.DRIVING_CLASSES.keys())}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 이미지 ID 가져오기
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 이미지 로드
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 어노테이션 가져오기
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)
        
        # 바운딩 박스 처리
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO 형식: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # 이미지 범위 내로 클리핑
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(orig_w, x + w)
            y2 = min(orig_h, y + h)
            
            # 유효한 박스만 추가
            if x2 > x1 and y2 > y1:
                # 정규화 (0~1 범위)
                boxes.append([
                    x1 / orig_w,
                    y1 / orig_h,
                    x2 / orig_w,
                    y2 / orig_h
                ])
                
                # 클래스 레이블
                cat_name = self.coco.loadCats(ann['category_id'])[0]['name']
                if cat_name in self.DRIVING_CLASSES:
                    labels.append(self.DRIVING_CLASSES[cat_name])
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # Tensor로 변환
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
            'image_id': img_id
        }


def collate_fn(batch):
    """배치 데이터 처리"""
    images = []
    boxes = []
    labels = []
    image_ids = []
    
    for item in batch:
        images.append(item['image'])
        boxes.append(item['boxes'])
        labels.append(item['labels'])
        image_ids.append(item['image_id'])
    
    images = torch.stack(images, 0)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids
    }


if __name__ == '__main__':
    # 테스트 코드
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    # 데이터셋 로드 (경로는 실제 환경에 맞게 수정)
    print("데이터셋 로드 테스트...")
    # dataset = COCODetectionDataset(
    #     root_dir='./data/coco/val2017',
    #     annotation_file='./data/coco/annotations/instances_val2017.json',
    #     transform=transform
    # )
    # print(f"총 샘플 수: {len(dataset)}")
