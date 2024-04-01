from pathlib import Path
import tempfile
import torch
import argparse
import os
import numpy as np
import sys
from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from motiondiff_modules.hmr2.configs import CACHE_DIR_4DHUMANS
from motiondiff_modules.hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from motiondiff_modules.hmr2.utils import recursive_to
from motiondiff_modules.hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from motiondiff_modules.hmr2.utils.renderer import cam_crop_to_full
from ultralytics import YOLO, NAS, RTDETR
from comfy.model_management import get_torch_device
from types import SimpleNamespace
from torch.utils.data import DataLoader
from ..md_config import get_smpl_models_dict
from motiondiff_modules.hmr2.utils.render_openpose import render_openpose as _render_openpose
from functools import partial

smpl_models_dict = get_smpl_models_dict()

class Humans4DLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "det_filename": ("STRING", {"default": "yolov8x.pt"}), 
                "fp16": ("BOOLEAN", {"default": False}) 
            }
        }

    RETURN_TYPES = ("HUMAN4D_MODEL", )
    FUNCTION = "load"
    CATEGORY = "MotionDiff"

    def load(self, det_filename, fp16):
        download_models(CACHE_DIR_4DHUMANS, {det_filename: "https://github.com/ultralytics/assets/releases/latest/download/" + det_filename})
        model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        device = get_torch_device()
        model = model.to(device)
        if fp16:
            model = model.half()
        detector_cls = YOLO
        if "nas" in det_filename:
            detector_cls = NAS
        elif "rtdetr" in det_filename:
            detector_cls = RTDETR
        detector = detector_cls(str(Path(CACHE_DIR_4DHUMANS) / det_filename))
        return (SimpleNamespace(human4d=model, model_cfg=model_cfg, detector=detector, fp16=fp16), )

# kps_2d_frames: #List of [num_subjects, 44, 3]
def render_openpose(kps_2d_frames, boxes_frames, frame_width, frame_height):
    openpose_frames = []
    for subjects_kps, xyxy_boxes_batch in zip(kps_2d_frames, boxes_frames):
        print(xyxy_boxes_batch.shape)
        canvas = np.zeros([frame_height, frame_width, 3], dtype=np.uint8)
        subjects_kps = subjects_kps.numpy() # [num_subjects, 44, 3]
        subjects_kps = np.concatenate((subjects_kps, np.ones_like(subjects_kps)[:, :, [0]]), axis=-1)
        keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9), (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
        for i in range(subjects_kps.shape[0]):
            subject_xyxy_box = xyxy_boxes_batch[i]
            print(subject_xyxy_box.shape)
            x0, y0, x1, y1 = subject_xyxy_box.astype(np.int32)
            _width, _height = x1-x0+1, y1-y0+1
            subjects_kps[i, :, 0] = _width * (subjects_kps[i, :, 0] + 0.5)
            subjects_kps[i, :, 1] = _height * (subjects_kps[i, :, 1] + 0.5)

            pred_keypoints_img = np.zeros([_height, _width, 3], dtype=np.uint8)
            body_keypoints = subjects_kps[i, :25]
            extra_keypoints = subjects_kps[i, -19:]
            for pair in keypoint_matches:
                body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
            pred_keypoints_img = _render_openpose(pred_keypoints_img, body_keypoints)
            canvas[y0:y1+1, x0:x1+1, :] = pred_keypoints_img
        openpose_frames.append(canvas)
    return torch.from_numpy(np.stack(openpose_frames))

class Human4D_Img2SMPL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "human4d_model": ("HUMAN4D_MODEL", ),
                "image": ("IMAGE",),
                "det_confidence_thresh": ("FLOAT", {"min": 0.1, "max": 1, "step": 0.05, "default": 0.25}),
                "det_iou_thresh": ("FLOAT", {"min": 0.1, "max": 1, "step": 0.05, "default": 0.7}),
                "det_batch_size": ("INT", {"min": 1, "max": 20, "default": 10}),
                "hmr_batch_size": ("INT", {"min": 1, "max": 20, "default": 8})
            },
            "optional": {
                "opt_scorehmr_refiner": ("SCORE_HMR_MODEL", )
            }
        }

    RETURN_TYPES = ("SMPL_MULTIPLE_SUBJECTS", )
    FUNCTION = "sample"
    CATEGORY = "MotionDiff"

    def get_boxes(self, detector, image, batch_size, **kwargs):
        boxes_images = []
        for img_batch in DataLoader(image, shuffle=False, batch_size=batch_size, num_workers=0):
            det_results = detector.predict([img.numpy() for img in img_batch], classes=[0], **kwargs)
            boxes_images.extend([det_result.boxes.xyxy.cpu().numpy() for det_result in det_results])
        return boxes_images

    def sample(self, human4d_model, image, det_confidence_thresh, det_iou_thresh, det_batch_size, hmr_batch_size, opt_scorehmr_refiner=None):
        models = human4d_model
        if opt_scorehmr_refiner is not None:
            raise NotImplementedError()
        image = image.__mul__(255.).to(torch.uint8)
        boxes_images = self.get_boxes(models.detector, image, conf=det_confidence_thresh, iou=det_iou_thresh, batch_size=det_batch_size)
        verts_frames = []
        cam_t_frames = []
        kps_2d_frames = []
        for img_pt, boxes in zip(image, boxes_images):
            img_cv2 = img_pt.numpy()[:, :, ::-1].copy()

            # Run HMR2.0 on all detected humans
            dataset = ViTDetDataset(models.model_cfg, img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=hmr_batch_size, shuffle=False, num_workers=0)
            _all_verts = []
            _all_cam_t = []
            _all_kps_2d = []

            for batch in dataloader:
                batch = recursive_to(batch, get_torch_device())
                if models.fp16:
                    batch = recursive_to(batch, torch.float16)
                with torch.no_grad():
                    out = models.human4d(batch)

                pred_cam = out['pred_cam']
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = models.model_cfg.EXTRA.FOCAL_LENGTH / models.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu()

                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    verts = out['pred_vertices'][n].detach().cpu() #Shape [num_verts, 3]
                    cam_t = pred_cam_t_full[n]
                    kps_2d = out['pred_keypoints_2d'][n].detach().cpu() #Shape [44, 3]
                    _all_verts.append(verts)
                    _all_cam_t.append(cam_t)
                    _all_kps_2d.append(kps_2d)
            
            verts_frames.append(
                torch.stack(_all_verts) #Shape [num_subjects, num_verts, 3]
            )
            cam_t_frames.append(
                torch.stack(_all_cam_t) #Shape [num_subjects, 3]
            )
            kps_2d_frames.append(
                torch.stack(_all_kps_2d) #Shape [num_subjects, 44, 3]
            )
        verts_frames #List of [num_subjects, num_verts, 3]
        cam_t_frames #List of [num_frames, num_subjects, 3]
        kps_2d_frames #List of [num_subjects, 44, 3]
        return ((
            smpl_models_dict["SMPL_NEUTRAL.pkl"], verts_frames, 
            {"normalized_to_vertices": True, 'cam': cam_t_frames, 
            "frame_width": int(img_size[0, 0].item()), "frame_height": int(img_size[0, 1].item()), 
            "focal_length": scaled_focal_length, 
            "render_openpose": partial(render_openpose, kps_2d_frames, boxes_images, int(img_size[0, 0].item()), int(img_size[0, 1].item()))}
            # In Comfy, IMAGE is a batched Tensor so all frames always share the same size
        ), )

NODE_CLASS_MAPPINGS = {
    "Humans4DLoader": Humans4DLoader,
    "Human4D_Img2SMPL": Human4D_Img2SMPL
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Humans4DLoader": "Human4D Loader",
    "Human4D_Img2SMPL": "Human4D Image2SMPL"
}
