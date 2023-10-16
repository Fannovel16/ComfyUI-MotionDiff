import torch
import os
import sys
from pathlib import Path
EXTENSION_PATH = Path(__file__).parent
sys.path.insert(0, str(EXTENSION_PATH.resolve()))

import custom_mmpkg.custom_mmcv as mmcv
import numpy as np
import torch
from mogen.models import build_architecture
from custom_mmpkg.custom_mmcv.runner import load_checkpoint
from custom_mmpkg.custom_mmcv.parallel import MMDataParallel
from mogen.utils.plot_utils import (
    recover_from_ric,
    plot_3d_motion,
    t2m_kinematic_chain
)
from scipy.ndimage import gaussian_filter
from IPython.display import Image
import comfy.model_management as model_management
from .config import get_model_dataset_dict
import warnings
from .utils import *

def create_mdm_model(model_config):
    cfg = mmcv.Config.fromstring(model_config.config_code, '.py')
    mdm = build_architecture(cfg.model)
    load_checkpoint(mdm, str(model_config.ckpt_path), map_location='cpu')
    mdm.eval().cpu()
    return mdm

class MotionDiffModelWrapper(torch.nn.Module): #Anything beside CLIP (mdm.model)
    def __init__(self, mdm, dataset) -> None:
        super(MotionDiffModelWrapper, self).__init__()
        self.loss_recon=mdm.loss_recon
        self.diffusion_train=mdm.diffusion_train
        self.diffusion_test=mdm.diffusion_test
        self.sampler=mdm.sampler
        self.dataset = dataset

    def forward(self, clip, cond_dict, **kwargs):
        motion, motion_mask = kwargs['motion'].float(), kwargs['motion_mask'].float()
        sample_idx = kwargs.get('sample_idx', None)
        clip_feat = kwargs.get('clip_feat', None)
        sampler = kwargs.get('sampler', 'ddpm')
        B, T = motion.shape[:2]

        dim_pose = kwargs['motion'].shape[-1]
        model_kwargs = cond_dict
        model_kwargs['motion_mask'] = motion_mask
        model_kwargs['sample_idx'] = sample_idx
        inference_kwargs = kwargs.get('inference_kwargs', {})
        if sampler == 'ddpm':
            output = self.diffusion_test.p_sample_loop(
                clip,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=False,
                model_kwargs=model_kwargs,
                **inference_kwargs
            )
        else:
            output = self.diffusion_test.ddim_sample_loop(
                clip,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=False,
                model_kwargs=model_kwargs,
                eta=0,
                **inference_kwargs
            )
        if getattr(clip, "post_process") is not None:
            output = clip.post_process(output)
        results = kwargs
        results['pred_motion'] = output
        results = self.split_results(results)
        return results
    
    def split_results(self, results):
        B = results['motion'].shape[0]
        output = []
        for i in range(B):
            batch_output = dict()
            batch_output['motion'] = to_cpu(results['motion'][i])
            batch_output['pred_motion'] = to_cpu(results['pred_motion'][i])
            batch_output['motion_length'] = to_cpu(results['motion_length'][i])
            batch_output['motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'pred_motion_length' in results.keys():
                batch_output['pred_motion_length'] = to_cpu(results['pred_motion_length'][i])
            else:
                batch_output['pred_motion_length'] = to_cpu(results['motion_length'][i])
            if 'pred_motion_mask' in results:
                batch_output['pred_motion_mask'] = to_cpu(results['pred_motion_mask'][i])
            else:
                batch_output['pred_motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'motion_metas' in results.keys():
                motion_metas = results['motion_metas'][i]
                if 'text' in motion_metas.keys():
                    batch_output['text'] = motion_metas['text']
                if 'token' in motion_metas.keys():
                    batch_output['token'] = motion_metas['token']
            output.append(batch_output)
        return output

class MotionDiffCLIPWrapper(torch.nn.Module):
    def __init__(self, mdm):
        super(MotionDiffCLIPWrapper, self).__init__()
        self.model = mdm.model
    
    def forward(self, text, motion_data):
        self.model.to(model_management.get_torch_device())
        B, T = motion_data["motion"].shape[:2]
        texts = []
        for _ in range(B):
            texts.append(text)
        out = self.model.get_precompute_condition(device=model_management.get_torch_device(), text=texts, **motion_data)
        self.model.cpu()
        return out

model_dataset_dict = None

class MotionDiffLoader:
    @classmethod
    def INPUT_TYPES(s):
        global model_dataset_dict
        model_dataset_dict = get_model_dataset_dict()
        return {
            "required": {
                "model_dataset": (
                    list(model_dataset_dict.keys()), 
                    { "default": "remodiffuse-human_ml3d" }
                )
            },
        }

    RETURN_TYPES = ("MD_MODEL", "MD_CLIP")
    CATEGORY = "MotionDiff"
    FUNCTION = "load_mdm"

    def load_mdm(self, model_dataset):
        global model_dataset_dict
        model_config = model_dataset_dict[model_dataset]()
        mdm = create_mdm_model(model_config)
        return (MotionDiffModelWrapper(mdm, dataset=model_config.dataset), MotionDiffCLIPWrapper(mdm))

class MotionDiffTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("MD_CLIP", ),
                "motion_data": ("MOTION_DATA", ),
                "text": ("STRING", {"default": '' ,"multiline": False})
            },
        }

    RETURN_TYPES = ("MD_CONDITIONING",)
    CATEGORY = "MotionDiff"
    FUNCTION = "encode_text"

    def encode_text(self, clip, motion_data, text):
        return (clip(text, motion_data), )

class EmptyMotionData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("INT", {"default": 1, "min": 1, "max": 196})
            }
        }

    RETURN_TYPES = ("MOTION_DATA", )
    CATEGORY = "MotionDiff"
    FUNCTION = "encode_text"

    def encode_text(self, frames):
        return ({
            'motion': torch.zeros(1, frames, 263),
            'motion_mask': torch.ones(1, frames),
            'motion_length': torch.Tensor([frames]).long(),
        }, )

class MotionDiffSimpleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (["ddpm", "ddim"], ),
                "model": ("MD_MODEL", ),
                "clip": ("MD_CLIP", ),
                "md_cond": ("MD_CONDITIONING", ),
                "motion_data": ("MOTION_DATA",)
            }
        }

    RETURN_TYPES = ("MOTION_DATA",)
    CATEGORY = "MotionDiff"
    FUNCTION = "sample"

    def sample(self, sampler_name, model: MotionDiffModelWrapper, clip, md_cond, motion_data):
        model.to(model_management.get_torch_device())
        clip.to(model_management.get_torch_device())
        for key in motion_data:
            motion_data[key] = to_gpu(motion_data[key])

        kwargs = {
            **motion_data,
            'inference_kwargs': {},
            'sampler': sampler_name,
        }

        with torch.no_grad():
            output = model(clip.model, cond_dict=md_cond, **kwargs)[0]['pred_motion']
            pred_motion = output * model.dataset.std + model.dataset.mean
            pred_motion = pred_motion.cpu().detach()
        
        model.cpu(), clip.cpu()
        for key in motion_data:
            motion_data[key] = to_cpu(motion_data[key])
        return ({
            'motion': pred_motion,
            'motion_mask': motion_data['motion_mask'],
            'motion_length': motion_data['motion_length'],
        }, )

class MotionDiffVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("MOTION_DATA", ),
                "title": ("STRING", {"default": '' ,"multiline": False}),
                "visualization": (["original", "pseudo-openpose"], {"default": "pseudo-openpose"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MotionDiff"
    FUNCTION = "visualize"

    def visualize(self, motion_data, title, visualization):
        pred_motion = motion_data["motion"]
        joint = recover_from_ric(pred_motion, 22).numpy()
        joint = motion_temporal_filter(joint, sigma=2.5)
        pil_frames = plot_3d_motion(None, t2m_kinematic_chain, joint, title=title, fps=1, save_as_pil_lists=True, visualization=visualization)
        tensor_frames = []
        for pil_image in pil_frames:
            np_image = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
            tensor_frames.append(torch.from_numpy(np_image))
        return (torch.stack(tensor_frames, dim=0), )

NODE_CLASS_MAPPINGS = {
    "MotionDiffLoader": MotionDiffLoader,
    "MotionDiffTextEncode": MotionDiffTextEncode,
    "MotionDiffSimpleSampler": MotionDiffSimpleSampler,
    "EmptyMotionData": EmptyMotionData,
    "MotionDiffVisualizer": MotionDiffVisualizer
}