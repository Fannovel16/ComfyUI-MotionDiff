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
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

HF_PREFIX = "https://huggingface.co/spaces/mingyuan/ReMoDiffuse/resolve/main/"

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def get_motion_length(motion_data):
    return motion_data["samples"].shape[1]

def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x

def to_gpu(x):
    if isinstance(x, torch.Tensor):
        return x.to(model_management.get_torch_device())
    return x

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

__all__ = ["motion_temporal_filter", "get_motion_length", "to_cpu", "to_gpu"]
