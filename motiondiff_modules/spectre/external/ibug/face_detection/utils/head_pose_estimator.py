import os
import cv2
import math
import numpy as np
from typing import Optional, Tuple


__all__ = ['HeadPoseEstimator']


class HeadPoseEstimator(object):
    def __init__(self, mean_shape_path: str = os.path.join(os.path.dirname(__file__),
                                                           'data', 'bfm_lms.npy')) -> None:
        # Load the 68-point mean shape derived from BFM
        mean_shape = np.load(mean_shape_path)

        # Calculate the 5-points mean shape
        left_eye = mean_shape[[37, 38, 40, 41]].mean(axis=0)
        right_eye = mean_shape[[43, 44, 46, 47]].mean(axis=0)
        self._mean_shape_5pts = np.vstack((left_eye, right_eye, mean_shape[[30, 48, 54]]))

        # Flip the y coordinates of the mean shape to match that of the image coordinate system
        self._mean_shape_5pts[:, 1] = -self._mean_shape_5pts[:, 1]

    def __call__(self, landmarks: np.ndarray, image_width: int = 0, image_height: int = 0,
                 camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None,
                 output_preference: int = 0) -> Tuple[float, float, float]:
        # Form the camera matrix
        if camera_matrix is None:
            if image_width <= 0 or image_height <= 0:
                raise ValueError(
                    'image_width and image_height must be specified when camera_matrix is not given directly')
            else:
                camera_matrix = np.array([[image_width + image_height, 0, image_width / 2.0],
                                          [0, image_width + image_height, image_height / 2.0],
                                          [0, 0, 1]], dtype=float)

        # Prepare the landmarks
        if landmarks.shape[0] == 68:
            landmarks = landmarks[17:]
        if landmarks.shape[0] in [49, 51]:
            left_eye = landmarks[[20, 21, 23, 24]].mean(axis=0)
            right_eye = landmarks[[26, 27, 29, 30]].mean(axis=0)
            landmarks = np.vstack((left_eye, right_eye, landmarks[[13, 31, 37]]))

        # Use EPnP to estimate pitch, yaw, and roll
        _, rvec, _ = cv2.solvePnP(self._mean_shape_5pts, np.expand_dims(landmarks, axis=1),
                                  camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        rot_mat, _ = cv2.Rodrigues(rvec)
        if 1.0 + rot_mat[2, 0] < 1e-9:
            pitch = 0.0
            yaw = 90.0
            roll = -math.atan2(rot_mat[0, 1], rot_mat[0, 2]) / math.pi * 180.0
        elif 1.0 - rot_mat[2, 0] < 1e-9:
            pitch = 0.0
            yaw = -90.0
            roll = math.atan2(-rot_mat[0, 1], -rot_mat[0, 2]) / math.pi * 180.0
        else:
            pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2]) / math.pi * 180.0
            yaw = -math.asin(rot_mat[2, 0]) / math.pi * 180.0
            roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0]) / math.pi * 180.0

        # Respond to output_preference:
        # output_preference == 1: limit pitch to the range of -90.0 ~ 90.0
        # output_preference == 2: limit yaw to the range of -90.0 ~ 90.0 (already satisfied)
        # output_preference == 3: limit roll to the range of -90.0 ~ 90.0
        # otherwise: minimise total rotation, min(abs(pitch) + abs(yaw) + abs(roll))
        if output_preference != 2:
            alt_pitch = pitch - 180.0 if pitch > 0.0 else pitch + 180.0
            alt_yaw = -180.0 - yaw if yaw < 0.0 else 180.0 - yaw
            alt_roll = roll - 180.0 if roll > 0.0 else roll + 180.0
            if (output_preference == 1 and -90.0 < alt_pitch < 90.0 or
                    output_preference == 3 and -90.0 < alt_roll < 90.0 or
                    output_preference not in (1, 2, 3) and
                    abs(alt_pitch) + abs(alt_yaw) + abs(alt_roll) < abs(pitch) + abs(yaw) + abs(roll)):
                pitch, yaw, roll = alt_pitch, alt_yaw, alt_roll

        return -pitch, yaw, roll
