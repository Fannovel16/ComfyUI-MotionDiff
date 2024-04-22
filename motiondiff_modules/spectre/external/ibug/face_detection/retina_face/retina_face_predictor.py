import os
import torch
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from typing import Union, Optional
from .prior_box import PriorBox
from .py_cpu_nms import py_cpu_nms
from .retina_face import RetinaFace
from .config import cfg_mnet, cfg_re50
from .box_utils import decode, decode_landm

__all__ = ['RetinaFacePredictor']


class RetinaFacePredictor(object):
    def __init__(self, threshold: float = 0.8, device: Union[str, torch.device] = 'cuda:0',
                 model: Optional[SimpleNamespace] = None, config: Optional[SimpleNamespace] = None) -> None:
        self.threshold = threshold
        self.device = device
        if model is None:
            model = RetinaFacePredictor.get_model()
        if config is None:
            config = RetinaFacePredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = RetinaFace(cfg=self.config.__dict__, phase='test').to(self.device)
        pretrained_dict = torch.load(model.weights, map_location=self.device)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = {key.split('module.', 1)[-1] if key.startswith('module.') else key: value
                               for key, value in pretrained_dict['state_dict'].items()}
        else:
            pretrained_dict = {key.split('module.', 1)[-1] if key.startswith('module.') else key: value
                               for key, value in pretrained_dict.items()}
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()
        self.priors = None
        self.previous_size = None

    @staticmethod
    def get_model(name: str = 'resnet50') -> SimpleNamespace:
        from motiondiff_modules import CKPT_DIR_PATH, download_models
        RETINA_FACE_PREFIX = "https://github.com/hhj1897/face_detection/raw/71852f00b815f568f3b51f045a418ae84cbe162a/ibug/face_detection/retina_face/weights/"
        name = name.lower().strip()
        if name == 'resnet50':
            download_models({'Resnet50_Final.pth': RETINA_FACE_PREFIX + 'Resnet50_Final.pth'})
            return SimpleNamespace(weights=os.path.join(str(CKPT_DIR_PATH), 'Resnet50_Final.pth'),
                                   config=SimpleNamespace(**deepcopy(cfg_re50)))
        elif name == 'mobilenet0.25':
            download_models({'Resnet50_Final.pth': RETINA_FACE_PREFIX + 'mobilenet0.25_Final.pth'})
            return SimpleNamespace(weights=os.path.join(str(CKPT_DIR_PATH), 'mobilenet0.25_Final.pth'),
                                   config=SimpleNamespace(**deepcopy(cfg_mnet)))
        else:
            raise ValueError('name must be set to either resnet50 or mobilenet0.25')

    @staticmethod
    def create_config(top_k: int = 750, conf_thresh: float = 0.02,
                      nms_thresh: float = 0.4, nms_top_k: int = 5000) -> SimpleNamespace:
        return SimpleNamespace(top_k=top_k, conf_thresh=conf_thresh, nms_thresh=nms_thresh, nms_top_k=nms_top_k)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, rgb: bool = True) -> np.ndarray:
        im_height, im_width, _ = image.shape
        if rgb:
            image = image[..., ::-1]
        image = image.astype(int) - np.array([104, 117, 123])
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).float().to(self.device)
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)
        loc, conf, landms = self.net(image)
        image_size = (im_height, im_width)
        if self.priors is None or self.previous_size != image_size:
            self.priors = PriorBox(self.config.__dict__, image_size=image_size).forward().to(self.device)
            self.previous_size = image_size
        prior_data = self.priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.config.variance)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.config.variance)
        scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2]]).to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.config.conf_thresh)[0]
        if len(inds) == 0:
            return np.empty(shape=(0, 15), dtype=np.float32)
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config.nms_thresh, self.config.nms_top_k)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K
        dets = dets[:self.config.top_k, :]
        landms = landms[:self.config.top_k, :]
        dets = np.concatenate((dets, landms), axis=1)

        # further filter by confidence
        inds = np.where(dets[:, 4] >= self.threshold)[0]
        if len(inds) == 0:
            return np.empty(shape=(0, 15), dtype=np.float32)
        else:
            return dets[inds]
