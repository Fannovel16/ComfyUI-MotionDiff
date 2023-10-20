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
from mogen.utils.plot_utils import (
    plot_3d_motion,
    t2m_kinematic_chain
)
from comfy.model_management import get_torch_device, soft_empty_cache
from .md_config import get_model_dataset_dict, get_smpl_models_dict
from .utils import *
from mogen.smpl.simplify_loc2rot import joints2smpl
from mogen.smpl.rotation2xyz import Rotation2xyz
#from custom_mmpkg.custom_mmhuman3d.core.conventions.keypoints_mapping import convert_kps
#from custom_mmpkg.custom_mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mogen.smpl.render_mesh import render_from_smpl
import gc
from pathlib import Path

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
        self.model.to(get_torch_device())
        B, T = motion_data["motion"].shape[:2]
        texts = []
        for _ in range(B):
            texts.append(text)
        out = self.model.get_precompute_condition(device=get_torch_device(), text=texts, **motion_data)
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
        if model_dataset_dict is None:
            model_dataset_dict = get_model_dataset_dict() #In case of API users
        model_config = model_dataset_dict[model_dataset]()
        mdm = create_mdm_model(model_config)
        return (MotionDiffModelWrapper(mdm, dataset=model_config.dataset), MotionDiffCLIPWrapper(mdm))

class MotionCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "md_clip": ("MD_CLIP", ),
                "motion_data": ("MOTION_DATA", ),
                "text": ("STRING", {"default": "a person performs a cartwheel" ,"multiline": True})
            },
        }

    RETURN_TYPES = ("MD_CONDITIONING",)
    CATEGORY = "MotionDiff"
    FUNCTION = "encode_text"

    def encode_text(self, md_clip, motion_data, text):
        return (md_clip(text, motion_data), )

class EmptyMotionData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("INT", {"default": 196, "min": 1, "max": 196})
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
                "md_model": ("MD_MODEL", ),
                "md_clip": ("MD_CLIP", ),
                "md_cond": ("MD_CONDITIONING", ),
                "motion_data": ("MOTION_DATA",)
            }
        }

    RETURN_TYPES = ("MOTION_DATA",)
    CATEGORY = "MotionDiff"
    FUNCTION = "sample"

    def sample(self, sampler_name, md_model: MotionDiffModelWrapper, md_clip, md_cond, motion_data):
        md_model.to(get_torch_device())
        md_clip.to(get_torch_device())
        for key in motion_data:
            motion_data[key] = to_gpu(motion_data[key])

        kwargs = {
            **motion_data,
            'inference_kwargs': {},
            'sampler': sampler_name,
        }

        with torch.no_grad():
            output = md_model(md_clip.model, cond_dict=md_cond, **kwargs)[0]['pred_motion']
            pred_motion = output * md_model.dataset.std + md_model.dataset.mean
            pred_motion = pred_motion.cpu().detach()
        
        md_model.cpu(), md_clip.cpu()
        for key in motion_data:
            motion_data[key] = to_cpu(motion_data[key])
        return ({
            'motion': pred_motion,
            'motion_mask': motion_data['motion_mask'],
            'motion_length': motion_data['motion_length'],
        }, )

class MotionDataVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("MOTION_DATA", ),
                "visualization": (["original", "pseudo-openpose"], {"default": "pseudo-openpose"}),
                "distance": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "elevation": ("FLOAT", {"default": 120, "min": 0.0, "max": 300.0, "step": 0.1}),
                "rotation": ("FLOAT", {"default": -90, "min": -180, "max": 180, "step": 1}),
                "poselinewidth": ("FLOAT", {"default": 2, "min": 0, "max": 50, "step": 0.1}),
            },
            "optional": {
                "opt_title": ("STRING", {"default": '' ,"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MotionDiff"
    FUNCTION = "visualize"

    def visualize(self, motion_data, visualization, distance, elevation, rotation, poselinewidth, opt_title=None):
        joint = motion_data_to_joints(motion_data["motion"])
        pil_frames = plot_3d_motion(
            None, t2m_kinematic_chain, joint, distance, elevation, rotation, poselinewidth,
            title=opt_title if opt_title is not None else '',
            fps=1,  save_as_pil_lists=True, visualization=visualization
        )
        tensor_frames = []
        for pil_image in pil_frames:
            np_image = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
            tensor_frames.append(torch.from_numpy(np_image))
        return (torch.stack(tensor_frames, dim=0), )

smpl_model_dicts = None
class SmplifyMotionData:
    @classmethod
    def INPUT_TYPES(s):
        global smpl_model_dicts
        smpl_model_dicts = get_smpl_models_dict()
        return {
            "required": {
                "motion_data": ("MOTION_DATA", ),
                "num_smplify_iters": ("INT", {"min": 10, "max": 1000, "default": 150}),
                "smplify_step_size": ("FLOAT", {"min": 1e-4, "max": 5e-1, "step": 1e-4, "default": 1e-2}),
                "smpl_model": (list(smpl_model_dicts.keys()), {"default": "SMPL_NEUTRAL.pkl"})
            }
        }

    RETURN_TYPES = ("SMPL",)
    CATEGORY = "MotionDiff/smpl"
    FUNCTION = "convent"
    
    def convent(self, motion_data, num_smplify_iters, smplify_step_size, smpl_model):
        global smpl_model_dicts
        if smpl_model_dicts is None:
            smpl_model_dicts = get_smpl_models_dict()
        smpl_model_path = smpl_model_dicts[smpl_model]
        joints = motion_data_to_joints(motion_data["motion"])
        with torch.inference_mode(False):
            convention = joints2smpl(
                num_frames=joints.shape[0], 
                device=get_torch_device(), 
                num_smplify_iters=num_smplify_iters, 
                smplify_step_size=smplify_step_size,
                smpl_model_path = smpl_model_path
            )
            motion_tensor, meta = convention.joint2smpl(joints)
        motion_tensor = motion_tensor.cpu().detach()
        for key in meta:
            meta[key] = meta[key].cpu().detach()
        gc.collect()
        soft_empty_cache()
        return ((smpl_model_path, motion_tensor, meta), ) #Caching

class RenderOpenPoseFromSMPL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl": ("SMPL", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MotionDiff/smpl"
    FUNCTION = "convent"
    
    def convent(self, smpl):
        kps = smpl[1]["pose"]
        kp3d_openpose, _ = convert_kps(kps, src='smpl_45', dst='openpose_25')
        cv2_frames = visualize_kp3d(kp3d_openpose.cpu().numpy(), data_source='openpose_25', return_array=True, resolution=(1024, 1024))
        return (torch.from_numpy(cv2_frames), )

class RenderSMPLMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl": ("SMPL", ),
                "draw_platform": ("BOOLEAN", {"default": False}),
                "depth_only": ("BOOLEAN", {"default": False}),
                "yfov": ("FLOAT", {"default": np.pi / 3.0, "min": 0.1, "max": 10, "step": 0.1}),
                "move_x": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.1}),
                "move_y": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.1}),
                "move_z": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "DEPTH_MAP")
    CATEGORY = "MotionDiff/smpl"
    FUNCTION = "render"
    def render(self, smpl, yfov, move_x, move_y, move_z, draw_platform, depth_only):
        smpl_model_path, motion_tensor, _ = smpl
        color_frames, depth_frames = render_from_smpl(
            motion_tensor.to(get_torch_device()),
            yfov, move_x, move_y, move_z, draw_platform,depth_only, 
            smpl_model_path=smpl_model_path
        )
        color_frames = torch.from_numpy(color_frames[..., :3].astype(np.float32) / 255.)

        #Normalize to [0, 1]
        normalized_depth = (depth_frames - depth_frames.min()) / (depth_frames.max() - depth_frames.min())
        #Pyrender's depths are the distance in meters to the camera, which is the inverse of depths in normal context
        #Ref: https://github.com/mmatl/pyrender/issues/10#issuecomment-468995891
        normalized_depth[normalized_depth != 0] = 1 - normalized_depth[normalized_depth != 0]
        #https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/controlnet_aux/util.py#L24
        depth_frames = [torch.from_numpy(np.concatenate([x, x, x], axis=2)) for x in normalized_depth[..., None]]
        depth_frames = torch.stack(depth_frames, dim=0)
        return (color_frames, depth_frames,)

NODE_CLASS_MAPPINGS = {
    "MotionDiffLoader": MotionDiffLoader,
    "MotionCLIPTextEncode": MotionCLIPTextEncode,
    "MotionDiffSimpleSampler": MotionDiffSimpleSampler,
    "EmptyMotionData": EmptyMotionData,
    "MotionDataVisualizer": MotionDataVisualizer,
    "SmplifyMotionData": SmplifyMotionData,
    "RenderSMPLMesh": RenderSMPLMesh
    #"RenderOpenPoseFromSMPL": RenderOpenPoseFromSMPL
}
