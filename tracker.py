"""
객체 추적 유틸리티 - 간단한 IoU 기반 트래커
"""
import numpy as np
from collections import defaultdict, deque


class ObjectTracker:
    """간단한 IoU 기반 객체 추적기"""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Args:
            max_disappeared: 객체가 사라진 후 ID를 유지할 최대 프레임 수
            max_distance: 같은 객체로 간주할 최대 거리
        """
        self.next_object_id = 0
        self.objects = {}  # ID -> (box, label, score)
        self.disappeared = {}  # ID -> disappeared count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # 이동 경로 저장
        self.trajectories = defaultdict(lambda: deque(maxlen=30))
    
    def register(self, box, label, score):
        """새 객체 등록"""
        self.objects[self.next_object_id] = (box, label, score)
        self.disappeared[self.next_object_id] = 0
        center = self._get_center(box)
        self.trajectories[self.next_object_id].append(center)
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """객체 등록 해제"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]
    
    def _get_center(self, box):
        """바운딩 박스 중심 계산"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, center1, center2):
        """두 중심점 간 거리 계산"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_iou(self, box1, box2):
        """IoU (Intersection over Union) 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, boxes, labels, scores):
        """
        새 프레임의 감지 결과로 추적 업데이트
        
        Returns:
            tracked_objects: {object_id: (box, label, score)}
        """
        # 감지된 객체가 없으면
        if len(boxes) == 0:
            # 모든 기존 객체의 disappeared 증가
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # 너무 오래 사라진 객체 제거
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # 추적 중인 객체가 없으면
        if len(self.objects) == 0:
            for i in range(len(boxes)):
                self.register(boxes[i], labels[i], scores[i])
        else:
            # 기존 객체와 새 감지 결과 매칭
            object_ids = list(self.objects.keys())
            object_boxes = [self.objects[oid][0] for oid in object_ids]
            
            # IoU 행렬 계산
            iou_matrix = np.zeros((len(object_boxes), len(boxes)))
            for i, obj_box in enumerate(object_boxes):
                for j, det_box in enumerate(boxes):
                    iou_matrix[i, j] = self._calculate_iou(obj_box, det_box)
            
            # 가장 높은 IoU로 매칭
            used_rows = set()
            used_cols = set()
            
            # IoU가 높은 순서로 매칭
            matches = []
            while True:
                max_iou = 0
                max_pos = None
                
                for i in range(len(object_boxes)):
                    if i in used_rows:
                        continue
                    for j in range(len(boxes)):
                        if j in used_cols:
                            continue
                        if iou_matrix[i, j] > max_iou and iou_matrix[i, j] > 0.3:
                            max_iou = iou_matrix[i, j]
                            max_pos = (i, j)
                
                if max_pos is None:
                    break
                
                i, j = max_pos
                matches.append((object_ids[i], j))
                used_rows.add(i)
                used_cols.add(j)
            
            # 매칭된 객체 업데이트
            for object_id, det_idx in matches:
                self.objects[object_id] = (boxes[det_idx], labels[det_idx], scores[det_idx])
                self.disappeared[object_id] = 0
                center = self._get_center(boxes[det_idx])
                self.trajectories[object_id].append(center)
            
            # 매칭되지 않은 기존 객체
            for i, object_id in enumerate(object_ids):
                if i not in used_rows:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # 매칭되지 않은 새 감지 결과 (새 객체 등록)
            for j in range(len(boxes)):
                if j not in used_cols:
                    self.register(boxes[j], labels[j], scores[j])
        
        return self.objects
    
    def get_trajectory(self, object_id):
        """객체의 이동 경로 반환"""
        return list(self.trajectories.get(object_id, []))
    
    def get_velocity(self, object_id):
        """객체의 속도 계산 (픽셀/프레임)"""
        trajectory = self.trajectories.get(object_id, [])
        if len(trajectory) < 2:
            return 0, 0
        
        # 최근 5개 프레임의 평균 속도
        n = min(5, len(trajectory))
        dx = trajectory[-1][0] - trajectory[-n][0]
        dy = trajectory[-1][1] - trajectory[-n][1]
        
        return dx / n, dy / n
