import cv2
import numpy as np
from typing import Optional, Sequence, Tuple


__all__ = ['get_landmark_connectivity', 'plot_landmarks']


def get_landmark_connectivity(num_landmarks: int) -> Optional[Sequence[Tuple[int, int]]]:
    if num_landmarks == 68:
        return ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24),
                (24, 25), (25, 26), (27, 28), (28, 29), (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35),
                (36, 37), (37, 38), (38, 39), (40, 41), (41, 36), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
                (47, 42), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57),
                (57, 58), (58, 59), (59, 48), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
                (67, 60), (39, 40))
    elif num_landmarks == 100:
        return ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24),
                (24, 25), (25, 26), (68, 69), (69, 70), (70, 71), (72, 73), (73, 74), (74, 75), (36, 76), (76, 37),
                (37, 77), (77, 38), (38, 78), (78, 39), (39, 40), (40, 79), (79, 41), (41, 36), (42, 80), (80, 43),
                (43, 81), (81, 44), (44, 82), (82, 45), (45, 46), (46, 83), (83, 47), (47, 42), (27, 28), (28, 29),
                (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35), (84, 85), (86, 87), (48, 49), (49, 88),
                (88, 50), (50, 51), (51, 52), (52, 89), (89, 53), (53, 54), (54, 55), (55, 90), (90, 56), (56, 57),
                (57, 58), (58, 91), (91, 59), (59, 48), (60, 92), (92, 93), (93, 61), (61, 62), (62, 63), (63, 94),
                (94, 95), (95, 64), (64, 96), (96, 97), (97, 65), (65, 66), (66, 67), (67, 98), (98, 99), (99, 60),
                (17, 68), (21, 71), (22, 72), (26, 75))
    else:
        return None


def plot_landmarks(image: np.ndarray, landmarks: np.ndarray, landmark_scores: Optional[Sequence[float]] = None,
                   threshold: float = 0.2, line_colour: Tuple[int, int, int] = (0, 255, 0),
                   pts_colour: Tuple[int, int, int] = (0, 0, 255), line_thickness: int = 1, pts_radius: int = 1,
                   landmark_connectivity: Optional[Sequence[Tuple[int, int]]] = None) -> None:
    num_landmarks = len(landmarks)
    if landmark_scores is None:
        landmark_scores = np.full((num_landmarks,), threshold + 1.0, dtype=float)
    if landmark_connectivity is None:
        landmark_connectivity = get_landmark_connectivity(len(landmarks))
    if landmark_connectivity is not None:
        for (idx1, idx2) in landmark_connectivity:
            if (idx1 < num_landmarks and idx2 < num_landmarks and
                    landmark_scores[idx1] >= threshold and landmark_scores[idx2] >= threshold):
                cv2.line(image, tuple(landmarks[idx1].astype(int).tolist()),
                         tuple(landmarks[idx2].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
    for landmark, score in zip(landmarks, landmark_scores):
        if score >= threshold:
            cv2.circle(image, tuple(landmark.astype(int).tolist()), pts_radius, pts_colour, -1)
