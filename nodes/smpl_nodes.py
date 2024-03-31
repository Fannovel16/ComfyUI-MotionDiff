import torch
import os

import numpy as np
import torch
from comfy.model_management import get_torch_device, soft_empty_cache
from ..md_config import get_smpl_models_dict
from ..utils import *
from motiondiff_modules.mogen.smpl.simplify_loc2rot import joints2smpl
from motiondiff_modules.mogen.smpl.rotation2xyz import Rotation2xyz
#from custom_mmpkg.custom_mmhuman3d.core.conventions.keypoints_mapping import convert_kps
#from custom_mmpkg.custom_mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from motiondiff_modules.mogen.smpl.render_mesh import render_from_smpl, render_from_smpl_multiple_subjects
import gc
from PIL import ImageColor
import folder_paths
from trimesh import Trimesh
from trimesh.exchange.load import mesh_formats

smpl_model_dicts = None
class SmplifyMotionData:
    @classmethod
    def INPUT_TYPES(s):
        global smpl_model_dicts
        smpl_model_dicts = get_smpl_models_dict()
        return {
            "required": {
                "motion_data": ("MOTION_DATA", ),
                "num_smplify_iters": ("INT", {"min": 1, "max": 1000, "default": 20}),
                "smplify_step_size": ("FLOAT", {"min": 1e-4, "max": 5e-1, "step": 1e-4, "default": 1e-1}),
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
        if "joints" in motion_data:
            joints = motion_data["joints"]
        else:
            joints = motion_data_to_joints(motion_data["motion"])
        with torch.inference_mode(False):
            convention = joints2smpl(
                num_frames=joints.shape[0], 
                device=get_torch_device(), 
                num_smplify_iters=num_smplify_iters, 
                smplify_step_size=smplify_step_size,
                smpl_model_path = smpl_model_path
            )
            thetas, meta = convention.joint2smpl(joints)
        thetas = thetas.cpu().detach()
        for key in meta:
            meta[key] = meta[key].cpu().detach()
        gc.collect()
        soft_empty_cache()
        return ((smpl_model_path, thetas, meta), ) #thetas after normalized to vertices is 1N3B with N, B being number of vertices and number of frames respectively

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
                "yfov": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 10, "step": 0.01}),
                "move_x": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.01}),
                "move_y": ("FLOAT", {"default": -0.1,"min": -500, "max": 500, "step": 0.01}),
                "move_z": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.01}),
                "rotate_x": ("FLOAT", {"default": 0,"min": -180, "max": 180, "step": 0.1}),
                "rotate_y": ("FLOAT", {"default": 0,"min": -180, "max": 180, "step": 0.1}),
                "rotate_z": ("FLOAT", {"default": 0,"min": -180, "max": 180, "step": 0.1}),
                "background_hex_color": ("STRING", {"default": "#000000", "mutiline": False}),
                "frame_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "frame_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "normals": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "DEPTH_MAP")
    CATEGORY = "MotionDiff/smpl"
    FUNCTION = "render"
    def render(self, smpl, yfov, move_x, move_y, move_z, rotate_x, rotate_y, rotate_z, frame_width, frame_height, draw_platform, depth_only, background_hex_color, normals=False):
        smpl_model_path, thetas, meta = smpl
        color_frames, depth_frames = render_from_smpl(
            thetas.to(get_torch_device()),
            yfov, move_x, move_y, move_z, rotate_x, rotate_y, rotate_z, frame_width, frame_height, draw_platform,depth_only, normals,
            smpl_model_path=smpl_model_path, shape_parameters=smpl[2].get("shape_parameters", None),
            normalized_to_vertices=meta.get("normalized_to_vertices", False)
        )
        bg_color = ImageColor.getcolor(background_hex_color, "RGB")
        color_frames = torch.from_numpy(color_frames[..., :3].astype(np.float32) / 255.)
        white_mask = [
            (color_frames[..., 0] == 1.) & 
            (color_frames[..., 1] == 1.) & 
            (color_frames[..., 2] == 1.)
        ]
        color_frames[..., :3][white_mask] = torch.Tensor(bg_color)
        white_mask_tensor = torch.stack(white_mask, dim=0)
        white_mask_tensor = white_mask_tensor.float() / white_mask_tensor.max()
        white_mask_tensor = 1.0 - white_mask_tensor.permute(1, 2, 3, 0).squeeze(dim=-1)
        #Normalize to [0, 1]
        normalized_depth = (depth_frames - depth_frames.min()) / (depth_frames.max() - depth_frames.min())
        #Pyrender's depths are the distance in meters to the camera, which is the inverse of depths in normal context
        #Ref: https://github.com/mmatl/pyrender/issues/10#issuecomment-468995891
        normalized_depth[normalized_depth != 0] = 1 - normalized_depth[normalized_depth != 0]
        #https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/controlnet_aux/util.py#L24
        depth_frames = [torch.from_numpy(np.concatenate([x, x, x], axis=2)) for x in normalized_depth[..., None]]
        depth_frames = torch.stack(depth_frames, dim=0)
        return (color_frames, depth_frames, white_mask_tensor,)

class RenderMultipleSubjectsSMPLMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl_multi_subjects": ("SMPL_MULTIPLE_SUBJECTS", ),
                "draw_platform": ("BOOLEAN", {"default": False}),
                "depth_only": ("BOOLEAN", {"default": False}),
                "fx_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "fy_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "move_x": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.01}),
                "move_y": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.01}),
                "move_z": ("FLOAT", {"default": 0,"min": -500, "max": 500, "step": 0.01}),
                "rotate_x": ("FLOAT", {"default": 0,"min": -180, "max": 180, "step": 0.1}),
                "rotate_y": ("FLOAT", {"default": 0,"min": -180, "max": 180, "step": 0.1}),
                "rotate_z": ("FLOAT", {"default": 0,"min": -180, "max": 180, "step": 0.1}),
                "background_hex_color": ("STRING", {"default": "#000000", "mutiline": False}),
            },
            "optional": {
                "normals": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "DEPTH_MAP")
    CATEGORY = "MotionDiff/smpl"
    FUNCTION = "render"
    def render(self, smpl_multi_subjects, fx_offset, fy_offset, move_x, move_y, move_z, rotate_x, rotate_y, rotate_z, draw_platform, depth_only, background_hex_color, normals=False):
        smpl_model_path, verts_frames, meta = smpl_multi_subjects
        color_frames, depth_frames = render_from_smpl_multiple_subjects(
            verts_frames, meta["cam"], meta["focal_length"],
            fx_offset, fy_offset, move_x, move_y, move_z, rotate_x, rotate_y, rotate_z, meta["frame_width"], meta["frame_height"], draw_platform,depth_only, normals,
            smpl_model_path=smpl_model_path
        )
        bg_color = ImageColor.getcolor(background_hex_color, "RGB")
        color_frames = torch.from_numpy(color_frames[..., :3].astype(np.float32) / 255.)
        white_mask = [
            (color_frames[..., 0] == 1.) & 
            (color_frames[..., 1] == 1.) & 
            (color_frames[..., 2] == 1.)
        ]
        color_frames[..., :3][white_mask] = torch.Tensor(bg_color)
        white_mask_tensor = torch.stack(white_mask, dim=0)
        white_mask_tensor = white_mask_tensor.float() / white_mask_tensor.max()
        white_mask_tensor = 1.0 - white_mask_tensor.permute(1, 2, 3, 0).squeeze(dim=-1)
        #Normalize to [0, 1]
        normalized_depth = (depth_frames - depth_frames.min()) / (depth_frames.max() - depth_frames.min())
        #Pyrender's depths are the distance in meters to the camera, which is the inverse of depths in normal context
        #Ref: https://github.com/mmatl/pyrender/issues/10#issuecomment-468995891
        normalized_depth[normalized_depth != 0] = 1 - normalized_depth[normalized_depth != 0]
        #https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/controlnet_aux/util.py#L24
        depth_frames = [torch.from_numpy(np.concatenate([x, x, x], axis=2)) for x in normalized_depth[..., None]]
        depth_frames = torch.stack(depth_frames, dim=0)
        return (color_frames, depth_frames, white_mask_tensor,)

class SMPLLoader:
    @classmethod
    def INPUT_TYPES(s):
        global smpl_model_dicts
        smpl_model_dicts = get_smpl_models_dict()
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_extensions(files, ['.pt'])
        return {
            "required": {
                "smpl": (files, ),
                "smpl_model": (list(smpl_model_dicts.keys()), {"default": "SMPL_NEUTRAL.pkl"})
            }
        }
    
    RETURN_TYPES = ("SMPL", )
    FUNCTION = "load_smpl"
    CATEGORY = "MotionDiff/smpl"

    def load_smpl(self, smpl, smpl_model):
        input_dir = folder_paths.get_input_directory()
        smpl_dict = torch.load(os.path.join(input_dir, smpl))
        thetas, meta = smpl_dict["thetas"], smpl_dict["meta"]
        global smpl_model_dicts
        if smpl_model_dicts is None:
            smpl_model_dicts = get_smpl_models_dict()
        smpl_model_path = smpl_model_dicts[smpl_model]

        return ((smpl_model_path, thetas, meta), )

class SaveSMPL:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = "_smpl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl": ("SMPL", ),
                "filename_prefix": ("STRING", {"default": "motiondiff_pt"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_smpl"

    OUTPUT_NODE = True

    CATEGORY = "MotionDiff/smpl"

    def save_smpl(self, smpl, filename_prefix):
        _, thetas, meta = smpl
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, 196, 24)
        file = f"{filename}_{counter:05}_.pt"
        torch.save({ "thetas": thetas, "meta": meta }, os.path.join(full_output_folder, file))
        return {}

class ExportSMPLTo3DSoftware:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = "_smpl"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl": ("SMPL", ),
                "foldername_prefix": ("STRING", {"default": "motiondiff_meshes"}),
                "format": (list(mesh_formats()), {"default": 'glb'})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_smpl"

    OUTPUT_NODE = True

    CATEGORY = "MotionDiff/smpl"

    def create_trimeshs(self, smpl_model_path, thetas, normalized_to_vertices=False):
        rot2xyz = Rotation2xyz(device=get_torch_device(), smpl_model_path=smpl_model_path)
        faces = rot2xyz.smpl_model.faces
        vertices = rot2xyz(thetas.clone().detach().to(get_torch_device()), mask=None,
                                        pose_rep='xyz' if normalized_to_vertices else 'rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)
        frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
        return [Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces) for i in range(frames)]
    
    def save_smpl(self, smpl, foldername_prefix, format):
        smpl_model_path, thetas, meta = smpl
        foldername_prefix += self.prefix_append
        full_output_folder, foldername, counter, subfolder, foldername_prefix = folder_paths.get_save_image_path(foldername_prefix, self.output_dir, 196, 24)
        folder = os.path.join(full_output_folder, f"{foldername}_{counter:05}_")
        os.makedirs(folder, exist_ok=True)
        trimeshs = self.create_trimeshs(smpl_model_path, thetas, normalized_to_vertices=meta.get("normalized_to_vertices", False))
        for i, trimesh in enumerate(trimeshs):
            trimesh.export(os.path.join(folder, f'{i}.{format}'))
        return {}

class SMPLShapeParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl": ("SMPL", ),
                "size": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "thickness": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "upper_body_height": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "lower_body_height": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "muscle_mass": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "legs": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "chest": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "waist_height": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "waist_width": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
                "arms": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SMPL",)
    CATEGORY = "MotionDiff/smpl"
    FUNCTION = "setparams"
    def setparams(self, smpl, size, thickness, upper_body_height, lower_body_height, muscle_mass, legs, chest, waist_height, waist_width, arms):
        shape_parameters = [size, thickness, upper_body_height, lower_body_height, muscle_mass, legs, chest, waist_height, waist_width, arms]
        smpl[2]["shape_parameters"] = shape_parameters
        return (smpl,)

""" 
class Render_OpenPose_From_SMPL_Mesh_Multiple_Subjects:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl_multi_subjects": ("SMPL_MULTIPLE_SUBJECTS", )
            },
        }
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MotionDiff/smpl"

    def render(self, smpl_multi_subjects):
        meta = smpl_multi_subjects[2]
        kps_2d_frames = meta['keypoints_2d']
        
"""
   
NODE_CLASS_MAPPINGS = {
    "SmplifyMotionData": SmplifyMotionData,
    "RenderSMPLMesh": RenderSMPLMesh,
    "SMPLLoader": SMPLLoader,
    "SaveSMPL": SaveSMPL,
    "ExportSMPLTo3DSoftware": ExportSMPLTo3DSoftware,
    "SMPLShapeParameters": SMPLShapeParameters,
    "RenderMultipleSubjectsSMPLMesh": RenderMultipleSubjectsSMPLMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmplifyMotionData": "Smplify Motion Data",
    "RenderSMPLMesh": "Render SMPL Mesh",
    "SMPLLoader": "SMPL Loader",
    "SaveSMPL": "Save SMPL",
    "ExportSMPLTo3DSoftware": "Export SMPL to 3DCGI Software",
    "SMPLShapeParameters": "SMPL Shape Parameters",
    "RenderMultipleSubjectsSMPLMesh": "Render Mutiple Subjects from SMPL Mesh"
}