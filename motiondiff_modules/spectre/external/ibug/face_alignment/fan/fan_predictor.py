import os
import cv2
import torch
import numpy as np
from types import SimpleNamespace
from typing import Union, Optional, Tuple
from .fan import FAN


__all__ = ['FANPredictor']



class FANPredictor(object):
    def __init__(self, device: Union[str, torch.device] = 'cuda:0', model: Optional[SimpleNamespace] = None,
                 config: Optional[SimpleNamespace] = None) -> None:
        self.device = device
        if model is None:
            model = FANPredictor.get_model()
        if config is None:
            config = FANPredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = FAN(config=self.config).to(self.device)
        self.net.load_state_dict(torch.load(model.weights, map_location=self.device))
        self.net.eval()
        if self.config.use_jit:
            self.net = torch.jit.trace(self.net, torch.rand(1, 3, self.config.input_size,
                                                            self.config.input_size).to(self.device))

    @staticmethod
    def get_model(name: str = '2dfan2') -> SimpleNamespace:
        from motiondiff_modules import CKPT_DIR_PATH, download_models
        name = name.lower()
        FACE_ALIGNMENT_PREFIX = "https://github.com/hhj1897/face_alignment/raw/9cf5494e443f26d567972f3f50f6212d65b76c01/ibug/face_alignment/fan/weights/"
        download_models({f'{name}.pth': FACE_ALIGNMENT_PREFIX + f'{name}.pth'})
        if name == '2dfan2':
            return SimpleNamespace(weights=os.path.join(str(CKPT_DIR_PATH), '2dfan2.pth'),
                                   config=SimpleNamespace(crop_ratio=0.55, input_size=256, num_modules=2,
                                                          hg_num_features=256, hg_depth=4, use_avg_pool=False,
                                                          use_instance_norm=False, stem_conv_kernel_size=7,
                                                          stem_conv_stride=2, stem_pool_kernel_size=2,
                                                          num_landmarks=68))
        elif name == '2dfan4':
            return SimpleNamespace(weights=os.path.join(str(CKPT_DIR_PATH), '2dfan4.pth'),
                                   config=SimpleNamespace(crop_ratio=0.55, input_size=256, num_modules=4,
                                                          hg_num_features=256, hg_depth=4, use_avg_pool=True,
                                                          use_instance_norm=False, stem_conv_kernel_size=7,
                                                          stem_conv_stride=2, stem_pool_kernel_size=2,
                                                          num_landmarks=68))
        elif name == '2dfan2_alt':
            return SimpleNamespace(weights=os.path.join(str(CKPT_DIR_PATH), '2dfan2_alt.pth'),
                                   config=SimpleNamespace(crop_ratio=0.55, input_size=256, num_modules=2,
                                                          hg_num_features=256, hg_depth=4, use_avg_pool=False,
                                                          use_instance_norm=False, stem_conv_kernel_size=7,
                                                          stem_conv_stride=2, stem_pool_kernel_size=2,
                                                          num_landmarks=68))
        else:
            raise ValueError('name must be set to either 2dfan2, 2dfan4, or 2dfan2_alt')

    @staticmethod
    def create_config(gamma: float = 1.0, radius: float = 0.1, use_jit: bool = True) -> SimpleNamespace:
        return SimpleNamespace(gamma=gamma, radius=radius, use_jit=use_jit)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, face_boxes: np.ndarray, rgb: bool = True,
                 return_features: bool = False) -> Union[Tuple[np.ndarray, np.ndarray],
                                                         Tuple[np.ndarray, np.ndarray, torch.Tensor]]:
        if face_boxes.size > 0:
            if not rgb:
                image = image[..., ::-1]
            if face_boxes.ndim == 1:
                face_boxes = face_boxes[np.newaxis, ...]

            # Crop the faces
            face_patches = []
            centres = (face_boxes[:, [0, 1]] + face_boxes[:, [2, 3]]) / 2.0
            face_sizes = (face_boxes[:, [3, 2]] - face_boxes[:, [1, 0]]).mean(axis=1)
            enlarged_face_box_sizes = (face_sizes / self.config.crop_ratio)[:, np.newaxis].repeat(2, axis=1)
            enlarged_face_boxes = np.zeros_like(face_boxes[:, :4])
            enlarged_face_boxes[:, :2] = np.round(centres - enlarged_face_box_sizes / 2.0)
            enlarged_face_boxes[:, 2:] = np.round(enlarged_face_boxes[:, :2] + enlarged_face_box_sizes) + 1
            enlarged_face_boxes = enlarged_face_boxes.astype(int)
            outer_bounding_box = np.hstack((enlarged_face_boxes[:, :2].min(axis=0),
                                            enlarged_face_boxes[:, 2:].max(axis=0)))
            pad_widths = np.zeros(shape=(3, 2), dtype=int)
            if outer_bounding_box[0] < 0:
                pad_widths[1][0] = -outer_bounding_box[0]
            if outer_bounding_box[1] < 0:
                pad_widths[0][0] = -outer_bounding_box[1]
            if outer_bounding_box[2] > image.shape[1]:
                pad_widths[1][1] = outer_bounding_box[2] - image.shape[1]
            if outer_bounding_box[3] > image.shape[0]:
                pad_widths[0][1] = outer_bounding_box[3] - image.shape[0]
            if np.any(pad_widths > 0):
                image = np.pad(image, pad_widths)
            for left, top, right, bottom in enlarged_face_boxes:
                left += pad_widths[1][0]
                top += pad_widths[0][0]
                right += pad_widths[1][0]
                bottom += pad_widths[0][0]
                face_patches.append(cv2.resize(image[top: bottom, left: right, :],
                                               (self.config.input_size, self.config.input_size)))
            face_patches = torch.from_numpy(np.array(face_patches).transpose(
                (0, 3, 1, 2)).astype(np.float32)).to(self.device) / 255.0

            # Get heatmaps
            heatmaps, stem_feats, hg_feats = self.net(face_patches)

            # Get landmark coordinates and scores
            landmarks, landmark_scores = self._decode(heatmaps)

            # Rectify landmark coordinates
            hh, hw = heatmaps.size(2), heatmaps.size(3)
            for landmark, (left, top, right, bottom) in zip(landmarks, enlarged_face_boxes):
                landmark[:, 0] = landmark[:, 0] * (right - left) / hw + left
                landmark[:, 1] = landmark[:, 1] * (bottom - top) / hh + top

            if return_features:
                return landmarks, landmark_scores, torch.cat((stem_feats, torch.cat(hg_feats, dim=1) *
                                                              torch.sum(heatmaps, dim=1, keepdim=True)), dim=1)
            else:
                return landmarks, landmark_scores
        else:
            landmarks = np.empty(shape=(0, 68, 2), dtype=np.float32)
            landmark_scores = np.empty(shape=(0, 68), dtype=np.float32)
            if return_features:
                return landmarks, landmark_scores, torch.Tensor([])
            else:
                return landmarks, landmark_scores

    def _decode(self, heatmaps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        heatmaps = heatmaps.contiguous()
        scores = heatmaps.max(dim=3)[0].max(dim=2)[0]

        if (self.config.radius ** 2 * heatmaps.shape[2] * heatmaps.shape[3] <
                heatmaps.shape[2] ** 2 + heatmaps.shape[3] ** 2):
            # Find peaks in all heatmaps
            m = heatmaps.view(heatmaps.shape[0] * heatmaps.shape[1], -1).argmax(1)
            all_peaks = torch.cat(
                [(m / heatmaps.shape[3]).trunc().view(-1, 1), (m % heatmaps.shape[3]).view(-1, 1)], dim=1
            ).reshape((heatmaps.shape[0], heatmaps.shape[1], 1, 1, 2)).repeat(
                1, 1, heatmaps.shape[2], heatmaps.shape[3], 1).float()

            # Apply masks created from the peaks
            all_indices = torch.zeros_like(all_peaks) + torch.stack(
                [torch.arange(0.0, all_peaks.shape[2],
                              device=all_peaks.device).unsqueeze(-1).repeat(1, all_peaks.shape[3]),
                 torch.arange(0.0, all_peaks.shape[3],
                              device=all_peaks.device).unsqueeze(0).repeat(all_peaks.shape[2], 1)], dim=-1)
            heatmaps = heatmaps * ((all_indices - all_peaks).norm(dim=-1) <= self.config.radius *
                                   (heatmaps.shape[2] * heatmaps.shape[3]) ** 0.5).float()

        # Prepare the indices for calculating centroids
        x_indices = (torch.zeros((*heatmaps.shape[:2], heatmaps.shape[3]), device=heatmaps.device) +
                     torch.arange(0.5, heatmaps.shape[3], device=heatmaps.device))
        y_indices = (torch.zeros(heatmaps.shape[:3], device=heatmaps.device) +
                     torch.arange(0.5, heatmaps.shape[2], device=heatmaps.device))

        # Finally, find centroids as landmark locations
        heatmaps = heatmaps.clamp_min(0.0)
        if self.config.gamma != 1.0:
            heatmaps = heatmaps.pow(self.config.gamma)
        m00s = heatmaps.sum(dim=(2, 3)).clamp_min(torch.finfo(heatmaps.dtype).eps)
        xs = heatmaps.sum(dim=2).mul(x_indices).sum(dim=2).div(m00s)
        ys = heatmaps.sum(dim=3).mul(y_indices).sum(dim=2).div(m00s)

        lm_info = torch.stack((xs, ys, scores), dim=-1).cpu().numpy()
        return lm_info[..., :-1], lm_info[..., -1]
