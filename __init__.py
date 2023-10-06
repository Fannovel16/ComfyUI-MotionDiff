import matplotlib
from .mogen import digit_version
assert digit_version(matplotlib.__version__) == digit_version("3.3.1"), "This extension requires matplotlib==3.3.1, otherwise the visualization won't work."

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

import yaml
CONFIGS = yaml.load((EXTENSION_PATH / "config.yml").resolve())
DATASET_PATH = Path(CONFIGS.dataset)
mean_path = DATASET_PATH / "mean.npy"
std_path = DATASET_PATH / "std.npy"
assert os.path.exists(mean_path) and os.path.exists(std_path), "Dataset folder not found or lost mean.npy, std.npy"
mean = np.load(mean_path)
std = np.load(std_path)

def create_mdm_model(config_path, ckpt_path):
    cfg = mmcv.Config.fromfile(config_path)
    mdm = build_architecture(cfg.model)
    load_checkpoint(mdm, ckpt_path, map_location='cpu')
    motion_module = motion_module.to(model_management.unet_offload_device())
    mdm.eval()
    return mdm

def is_model_available(mdm_config):
    return os.path.exists(mdm_config["config"]) and os.path.exists(mdm_config["ckpt"])

class MotionDiffModel(torch.nn.Module): #Anything beside CLIP (mdm.model)
    def __init__(self, **kwargs) -> None:
        super(MotionDiffModel).__init__()
        self.loss_recon = kwargs["loss_recon"]
        self.diffusion_train = kwargs["diffusion_train"]
        self.diffusion_test = kwargs["sampler"]

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

class MotionDiffCLIP(torch.nn.Module):
    def __init__(self, model):
        super(MotionDiffCLIP).__init__()
        self.model = model
    
    def forward(self, text):
        return self.model.get_precompute_condition(device=model_management.get_torch_device(), text=text)

class MotionDiffLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mdm_name": (
                    list(
                        filter(CONFIGS.keys(), lambda key: (key != "dataset") and is_model_available(CONFIGS[key]))
                    ), 
                    { "default": "remodiffuse" }
                )
            },
        }

    RETURN_TYPES = ("MD_MODEL", "MD_CLIP")
    CATEGORY = "Human Motion Diff"
    FUNCTION = "load_mdm"

    def load_mdm(self, mdm_name):
        mdm = create_mdm_model(mdm_name)
        model = MotionDiffModel(loss_recon=mdm.loss_recon, diffusion_train=mdm.diffusion_train, diffusion_test=mdm.diffusion_test, sampler=mdm.sampler)
        clip = mdm.model
        return (model, clip)

class MotionDiffTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("MD_CLIP", ),
                "text": ("STRING", {"multiline": True})
            },
        }

    RETURN_TYPES = ("MD_CONDITIONING",)
    CATEGORY = "Human Motion Diff"
    FUNCTION = "encode_text"

    def encode_text(self, clip, text):
        return (clip(text), )

class MotionDiffSimpleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (["ddpm", "ddim"], ),
                "model": ("MD_MODEL", ),
                "clip": ("MD_CLIP", ),
                "cond": ("MD_CONDITIONING", )
            },
        }

    RETURN_TYPES = ("MOTION_DATA",)
    CATEGORY = "Human Motion Diff"
    FUNCTION = "sample"

    def sample(self, sampler_name, model, clip, cond):
        device = model_management.get_torch_device()
        motion = torch.zeros(1, motion_length, 263).to(device)
        motion_mask = torch.ones(1, motion_length).to(device)
        motion_length = torch.Tensor([motion_length]).long().to(device)
        model = model.to(device)
        kwargs = {
            'motion': motion,
            'motion_mask': motion_mask,
            'motion_length': motion_length,
            'inference_kwargs': {},
            'sampler': sampler_name,

        }

        with torch.no_grad():
            output = model(clip, cond_dict=cond, **kwargs)[0]['pred_motion']
            pred_motion = output.cpu().detach().numpy()
            pred_motion = pred_motion * std + mean
        
        return pred_motion
