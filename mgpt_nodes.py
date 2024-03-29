import os
import torch

from .mGPT.models.build_model import build_model
from .mGPT.config import instantiate_from_config
from omegaconf import OmegaConf
from model_management import get_torch_device

class mgpt_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        },
        }

    RETURN_TYPES = ("MGPTMODEL", )
    RETURN_NAMES = ("mgpt_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "MotionDiff/mGPT"

    def loadmodel(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_directory, "mGPT/configs/inference.yaml")
        cfg = OmegaConf.load(config_path)
        if not hasattr(self, 'model') or self.model == None:
            base_bath = os.path.join(script_directory, 'mGPT','checkpoints','MotionGPT-base')
            model_path = os.path.join(script_directory, base_bath,'motiongpt_s3_h3d.tar')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
            else:
                #https://huggingface.co/OpenMotionLab/MotionGPT-base/tree/main to ComfyUI/custom_nodes/ComfyUI-MotionDiff/mGPT/checkpoints
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="OpenMotionLab/MotionGPT-base",
                                    local_dir=base_bath, local_dir_use_symlinks=False, allow_patterns=['motiongpt_s3_h3d.tar'])  
                state_dict = torch.load(model_path, map_location="cpu")["state_dict"]

            data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
            data_config['params'] = {'cfg': cfg, 'phase': 'test'}
            datamodule = instantiate_from_config(data_config)
            self.model = build_model(cfg, datamodule)
            
            self.model.load_state_dict(state_dict)
        
        return (self.model,)
    
class mgpt_t2m:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mgpt_model": ("MGPTMODEL",),
            "motion_length": ("INT", {"default": 196, "min": 1, "max": 196, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "text": ("STRING", {"multiline": True,"default": 'make the person jump and turn around'}),
            }
        }

    RETURN_TYPES = ("MOTION_DATA",)
    RETURN_NAMES = ("motion_data", )
    FUNCTION = "process"
    CATEGORY = "MotionDiff/mGPT"

    def process(self, seed, text, mgpt_model, motion_length):
        device = get_torch_device()
        torch.manual_seed(seed)

        prompt = mgpt_model.lm.placeholder_fulfill(text, motion_length, "", "")
        batch = {
            "length": [motion_length], #I don't know what this is supposed to do if anything? Lenght seems to be determined by the prompt up to the max of 196
            "text": [prompt],
        }
        mgpt_model.to(device)
        outputs = mgpt_model(batch, task="t2m")
        #out_feats = outputs["feats"][0]
        #print("out_feats_shape: ",out_feats.shape)
        out_lengths = outputs["length"][0]
        #print("out_lengths: ",out_lengths)
        out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
        mgpt_model.cpu()
        return ({"joints": out_joints.squeeze(0)},)

NODE_CLASS_MAPPINGS = {
    "mgpt_model_loader": mgpt_model_loader,
    "mgpt_t2m": mgpt_t2m,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "mgpt_model_loader": "MotionGPT Model Loader",
    "mgpt_t2m": "MotionGPT Text2Motion",
}
