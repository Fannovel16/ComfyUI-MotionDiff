import os
import torch
import numpy as np
from types import SimpleNamespace
from typing import Union, Optional
from .s3fd_net import S3FDNet


__all__ = ['S3FDPredictor']


class S3FDPredictor(object):
    def __init__(self, threshold: float = 0.8, device: Union[str, torch.device] = 'cuda:0',
                 model: Optional[SimpleNamespace] = None, config: Optional[SimpleNamespace] = None) -> None:
        self.threshold = threshold
        self.device = device
        if model is None:
            model = S3FDPredictor.get_model()
        if config is None:
            config = S3FDPredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = S3FDNet(config=self.config, device=self.device).to(self.device)
        self.net.load_state_dict(torch.load(model.weights, map_location=self.device))
        self.net.eval()

    @staticmethod
    def get_model(name: str = 's3fd') -> SimpleNamespace:
        from motiondiff_modules import CKPT_DIR_PATH, download_models
        S3FD_FACE_PREFIX = "https://github.com/hhj1897/face_detection/raw/71852f00b815f568f3b51f045a418ae84cbe162a/ibug/face_detection/s3fd/weights/"
        name = name.lower().strip()
        if name == 's3fd':
            download_models({'s3fd_weights.pth': S3FD_FACE_PREFIX + 's3fd_weights.pth'})
            return SimpleNamespace(weights=os.path.realpath(CKPT_DIR_PATH, 's3fd_weights.pth'),
                                   config=SimpleNamespace(num_classes=2, variance=(0.1, 0.2),
                                                          prior_min_sizes=(16, 32, 64, 128, 256, 512),
                                                          prior_steps=(4, 8, 16, 32, 64, 128), prior_clip=False))
        else:
            raise ValueError('name must be set to s3fd')

    @staticmethod
    def create_config(top_k: int = 750, conf_thresh: float = 0.05,nms_thresh: float = 0.3,
                      nms_top_k: int = 5000, use_nms_np: bool = True) -> SimpleNamespace:
        return SimpleNamespace(top_k=top_k, conf_thresh=conf_thresh, nms_thresh=nms_thresh,
                               nms_top_k=nms_top_k, use_nms_np=use_nms_np)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, rgb: bool = True) -> np.ndarray:
        w, h = image.shape[1], image.shape[0]
        if not rgb:
            image = image[..., ::-1]
        image = image.astype(int) - np.array([123, 117, 104])
        image = image.transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        image = torch.from_numpy(image).float().to(self.device)

        bboxes = []
        detections = self.net(image)
        scale = torch.Tensor([w, h, w, h]).to(detections.device)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= self.threshold:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                bbox = (pt[0], pt[1], pt[2], pt[3], score)
                bboxes.append(bbox)
                j += 1
        if len(bboxes) > 0:
            return np.array(bboxes)
        else:
            return np.empty(shape=(0, 5), dtype=np.float32)
