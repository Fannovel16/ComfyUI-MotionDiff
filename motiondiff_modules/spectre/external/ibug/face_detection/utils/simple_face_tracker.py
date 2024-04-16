import numpy as np
from typing import List, Optional
from scipy.optimize import linear_sum_assignment


__all__ = ['SimpleFaceTracker']


class SimpleFaceTracker(object):
    def __init__(self, iou_threshold: float = 0.4, minimum_face_size: float = 0.0) -> None:
        self._iou_threshold = iou_threshold
        self._minimum_face_size = minimum_face_size
        self._tracklets = []
        self._tracklet_counter = 0

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, threshold: float) -> None:
        self._iou_threshold = threshold

    @property
    def minimum_face_size(self) -> float:
        return self._minimum_face_size

    @minimum_face_size.setter
    def minimum_face_size(self, face_size: float) -> None:
        self._minimum_face_size = face_size

    def __call__(self, face_boxes: np.ndarray) -> List[Optional[int]]:
        if face_boxes.size <= 0:
            self._tracklets = []
            return []

        # Calculate area of the faces
        face_areas = np.abs((face_boxes[:, 2] - face_boxes[:, 0]) * (face_boxes[:, 3] - face_boxes[:, 1]))

        # Prepare tracklets
        for tracklet in self._tracklets:
            tracklet['tracked'] = False

        # Calculate the distance matrix based on IOU
        iou_distance_threshold = np.clip(1.0 - self._iou_threshold, 0.0, 1.0)
        min_face_area = max(self._minimum_face_size ** 2, np.finfo(float).eps)
        distances = np.full(shape=(face_boxes.shape[0], len(self._tracklets)),
                            fill_value=2.0 * min(face_boxes.shape[0], len(self._tracklets)), dtype=float)
        for row, face_box in enumerate(face_boxes):
            if face_areas[row] >= min_face_area:
                for col, tracklet in enumerate(self._tracklets):
                    x_left = max(min(face_box[0], face_box[2]), min(tracklet['bbox'][0], tracklet['bbox'][2]))
                    y_top = max(min(face_box[1], face_box[3]), min(tracklet['bbox'][1], tracklet['bbox'][3]))
                    x_right = min(max(face_box[2], face_box[0]), max(tracklet['bbox'][2], tracklet['bbox'][0]))
                    y_bottom = min(max(face_box[3], face_box[1]), max(tracklet['bbox'][3], tracklet['bbox'][1]))
                    if x_right <= x_left or y_bottom <= y_top:
                        distance = 1.0
                    else:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        distance = 1.0 - intersection_area / float(face_areas[row] + tracklet['area'] -
                                                                   intersection_area)
                    if distance <= iou_distance_threshold:
                        distances[row, col] = distance

        # ID assignment
        tracked_ids = [None] * face_boxes.shape[0]
        for row, col in zip(*linear_sum_assignment(distances)):
            if distances[row, col] <= iou_distance_threshold:
                tracked_ids[row] = self._tracklets[col]['id']
                self._tracklets[col]['bbox'] = face_boxes[row, :4].copy()
                self._tracklets[col]['area'] = face_areas[row]
                self._tracklets[col]['tracked'] = True

        # Remove expired tracklets
        self._tracklets = [x for x in self._tracklets if x['tracked']]

        # Register new faces
        for idx, face_box in enumerate(face_boxes):
            if face_areas[idx] >= min_face_area and tracked_ids[idx] is None:
                self._tracklet_counter += 1
                self._tracklets.append({'bbox': face_box[:4].copy(), 'area': face_areas[idx],
                                        'id': self._tracklet_counter, 'tracked': True})
                tracked_ids[idx] = self._tracklets[-1]['id']

        return tracked_ids

    def reset(self, reset_tracklet_counter: bool = True) -> None:
        self._tracklets = []
        if reset_tracklet_counter:
            self._tracklet_counter = 0
