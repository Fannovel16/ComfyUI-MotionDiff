import os
import torch

from motiondiff_modules.mGPT.models.build_model import build_model
from motiondiff_modules.mGPT.config import instantiate_from_config
from omegaconf import OmegaConf
from model_management import get_torch_device
import inspect
import motiondiff_modules
from pathlib import Path
from comfy.utils import load_torch_file

class mgpt_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                [   
                    "MotionGPT",
                    "AnimationGPT",
                ],
                {
                "default": "MotionGPT-base"
                    }),
        },
        }

    RETURN_TYPES = ("MGPTMODEL", )
    RETURN_NAMES = ("mgpt_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "MotionDiff/mGPT"

    def loadmodel(self, model):
        script_directory = os.path.dirname(inspect.getabsfile(motiondiff_modules))
        config_path = os.path.join(script_directory, "mGPT/configs/inference.yaml")
        cfg = OmegaConf.load(config_path)

        if not hasattr(self, "model") or self.model == None or model != self.model_name:
            self.model_name = f"{model}_fp16.safetensors"
            base_bath = os.path.join(script_directory, "mGPT","checkpoints")
            model_path = os.path.join(script_directory, base_bath, self.model_name)

            if not os.path.exists(model_path):
                #https://huggingface.co/Kijai/AnimationGPT_pruned/ to ComfyUI/custom_nodes/ComfyUI-MotionDiff/motiondiff_modules/mGPT/checkpoints
                print(f"Downloading model to {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="Kijai/AnimationGPT_pruned",
                                    local_dir=model_path, local_dir_use_symlinks=False, allow_patterns=[f"*{self.model_name}*"])  
            
            state_dict = load_torch_file(model_path)

            data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
            data_config["params"] = {"cfg": cfg, "phase": "test"}
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
            "text": ("STRING", {"multiline": True,"default": "make the person jump and turn around"}),
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
            "length": [motion_length], #I don"t know what this is supposed to do if anything? Lenght seems to be determined by the prompt up to the max of 196
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
