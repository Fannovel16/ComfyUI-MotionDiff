import yaml
from pathlib import Path
import os
import numpy as np
from .utils import load_file_from_url, HF_PREFIX

EXTENSION_PATH = Path(__file__).parent
CONFIG = yaml.load(open((EXTENSION_PATH / "config.yml").resolve(), "r"), Loader=yaml.FullLoader)
DATASET_CONFIGS = CONFIG["dataset"]
MODEL_CONFIGS = CONFIG["model"]

DB_ALIAS = {
    "human_ml3d": "t2m",
    "kit_ml": "kit"
}

class MotionDataset:
    def __init__(self, name):
        self.name = name
        self.raw_config = DATASET_CONFIGS[name]
        self.path = (EXTENSION_PATH / self.raw_config["path"]).resolve().absolute()
        self.retrieval_db_path = (EXTENSION_PATH / self.raw_config["retrieval_db"]).resolve().absolute()
        self.base_config_path = (EXTENSION_PATH / self.raw_config["base"]).resolve().absolute()

    def __call__(self):
        print(f"Checking motion dataset {self.name}...")
        std_path, mean_path = (self.path / "std.npy").resolve(), (self.path / "mean.npy").resolve()
        
        if not (os.path.exists(std_path) and os.path.exists(mean_path)):
            print(f"Dataset {self.name}'s folder not found or lost mean.npy, std.npy: {self.path}")
            print("Downloading...")
            load_file_from_url(HF_PREFIX + f"data/datasets/{self.name}/mean.npy", model_dir=self.path, file_name="mean.npy")
            load_file_from_url(HF_PREFIX + f"data/datasets/{self.name}/std.npy", model_dir=self.path, file_name="std.npy")
        
        assert os.path.exists(self.base_config_path), f"Dataset {self.name}'s base config not found: {self.base_config_path}\n"
        
        if not os.path.exists(self.retrieval_db_path):
            print(f"Dataset {self.name}'s retrieval DB not found: {self.retrieval_db_path}\n")
            print("Downloading...")
            load_file_from_url(
                HF_PREFIX + f"data/database/{DB_ALIAS[self.name]}_text_train.npz", 
                model_dir=self.retrieval_db_path.parent, 
                file_name=self.retrieval_db_path.name
            )

        self.std = np.load(std_path)
        self.mean = np.load(mean_path)
        return self

class MotionDiffConfig:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.config = MODEL_CONFIGS[name]
        self.config_path = (EXTENSION_PATH / self.config["config"][dataset.name]).resolve().absolute()
        self.ckpt_path = (EXTENSION_PATH / self.config["ckpt"][dataset.name]).resolve().absolute()

    def __call__(self):
        self.dataset = self.dataset()
        assert os.path.exists(self.config_path), f"Config for MotionDiff model {self.name} not found: {self.config_path}"
        if not os.path.exists(self.ckpt_path):
            print(f"Weight of MotionDiff model {self.name} not found")
            print("Downloading...")
            load_file_from_url(
                HF_PREFIX + f"logs/{self.name}/{self.name}_{DB_ALIAS[self.dataset.name]}/latest.pth",
                model_dir=self.ckpt_path.parent,
                file_name=self.ckpt_path.name
            )

        with open(self.config_path, 'r') as f:
            self.config_code = f.read() \
                .replace("MOTION_DIFF_RETRIEVAL_FILE", str(self.dataset.retrieval_db_path)) \
                .replace("MOTION_DIFF_BASE_DATASET", str(self.dataset.base_config_path)) \
                .replace("\\", "\\\\") #Windows path sucks
        return self

def get_model_dataset_dict() -> dict[str, MotionDiffConfig]:
    datasets = {}
    for key in ["human_ml3d"]: #TODO: Include KIT motion dataset #DATASET_CONFIGS:
        dataset = MotionDataset(key)
        if dataset: datasets[key] = dataset
    
    dataset_models = {}
    for model_key in MODEL_CONFIGS:
        for dataset_key in datasets:
            model_config = MotionDiffConfig(model_key, datasets[dataset_key])
            if model_config: dataset_models[f"{model_key}-{dataset_key}"] = model_config
    return dataset_models
