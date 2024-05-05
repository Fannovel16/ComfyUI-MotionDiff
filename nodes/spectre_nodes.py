from motiondiff_modules.spectre.tracker.face_tracker import FaceTracker
from motiondiff_modules.spectre.tracker.utils import get_landmarks
from comfy.model_management import get_torch_device
import numpy as np
from skimage.transform import estimate_transform, warp
from motiondiff_modules.spectre.src.spectre import SPECTRE
from motiondiff_modules import CKPT_DIR_PATH, download_models
from motiondiff_modules.spectre.config import cfg as spectre_cfg
from motiondiff_modules.spectre.src.utils import util
import collections
import torch
import trimesh
import math
from einops import rearrange
import comfy.utils
from tqdm import tqdm

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# https://github.com/filby89/spectre/blob/master/datasets/data_utils.py#L17
def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def crop_face(frame, landmarks, scale=1.0):
    image_size = 224
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

class SpectreFaceReconLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"fp16": ("BOOLEAN", {"default": False})}
        }

    RETURN_TYPES = ("SPECTRE_MODEL", )
    FUNCTION = "load"
    CATEGORY = "MotionDiff"

    def load(self, fp16):
        face_tracker = FaceTracker(get_torch_device())
        download_models({"spectre_model.tar": "https://github.com/Fannovel16/ComfyUI-MotionDiff/releases/download/latest/spectre_model.tar"})
        spectre_cfg.pretrained_modelpath = str(CKPT_DIR_PATH / "spectre_model.tar")
        spectre_cfg.model.use_tex = False
        spectre = SPECTRE(spectre_cfg, get_torch_device())
        spectre.eval()
        return ((face_tracker, spectre), )

class SpectreImg2SMPL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "spectre_model": ("SPECTRE_MODEL", ),
                "image": ("IMAGE", ),
                "chunk_size": ("INT", {"default": 50, "min": 10, "max": 100})
            }
        }
    
    RETURN_TYPES = ("SMPL_MULTIPLE_SUBJECTS", "IMAGE")
    RETURN_NAMES = ("SMPL_MULTIPLE_SUBJECTS", "CROPPED_FACE_IMAGE")
    FUNCTION = "sample"
    CATEGORY = "MotionDiff"

    def get_landmarks(self, face_tracker, image_batch):
        face_info = collections.defaultdict(list)
        pbar = comfy.utils.ProgressBar(len(image_batch))
        for image in tqdm(image_batch):    
            detected_faces = face_tracker.face_detector(image, rgb=True)
            # -- face alignment
            landmarks, scores = face_tracker.landmark_detector(image, detected_faces, rgb=True)
            face_info['bbox'].append(detected_faces)
            face_info['landmarks'].append(landmarks)
            face_info['landmarks_scores'].append(scores)
            pbar.update(1)
        pbar.update_absolute(0, 0)
        return get_landmarks(face_info)

    def sample(self, spectre_model, image, chunk_size):
        face_tracker, spectre = spectre_model
        image = image.numpy().__mul__(255.).astype(np.uint8)
        landmarks = self.get_landmarks(face_tracker, image)
        landmarks = landmarks_interpolate(landmarks)
        images_list = list(image)
        
        """ SPECTRE uses a temporal convolution of size 5. 
        Thus, in order to predict the parameters for a contiguous video with need to 
        process the video in chunks of overlap 2, dropping values which were computed from the 
        temporal kernel which uses pad 'same'. For the start and end of the video we
        pad using the first and last frame of the video. 
        e.g., consider a video of size 48 frames and we want to predict it in chunks of 20 frames 
        (due to memory limitations). We first pad the video two frames at the start and end using
        the first and last frames correspondingly, making the video 52 frames length.
        
        Then we process independently the following chunks:
        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        [16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
        [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51]]
        
        In the first chunk, after computing the 3DMM params we drop 0,1 and 18,19, since they were computed 
        from the temporal kernel with padding (we followed the same procedure in training and computed loss 
        only from valid outputs of the temporal kernel) In the second chunk, we drop 16,17 and 34,35, and in 
        the last chunk we drop 32,33 and 50,51. As a result we get:
        [2..17], [18..33], [34..49] (end included) which correspond to all frames of the original video 
        (removing the initial padding).     
        """

        # pad
        images_list.insert(0,images_list[0])
        images_list.insert(0,images_list[0])
        images_list.append(images_list[-1])
        images_list.append(images_list[-1])

        landmarks.insert(0,landmarks[0])
        landmarks.insert(0,landmarks[0])
        landmarks.append(landmarks[-1])
        landmarks.append(landmarks[-1])

        landmarks = np.stack(landmarks)

        L = chunk_size

        # create lists of overlapping indices
        indices = list(range(len(images_list)))
        overlapping_indices = [indices[i: i + L] for i in range(0, len(indices), L-4)]

        if len(overlapping_indices[-1]) < 5:
            # if the last chunk has less than 5 frames, pad it with the semilast frame
            overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
            overlapping_indices[-2] = np.unique(overlapping_indices[-2]).astype(np.int32).tolist()
            overlapping_indices = overlapping_indices[:-1]

        # doesn't work if length is not divisible to chunk_size
        # overlapping_indices = np.array(overlapping_indices)

        images_list = np.stack(images_list) # do this to index with multiple indices
        all_verts = []
        all_cams = []
        all_cropped_images = []

        # torch.no_grad() isn't needed as the context of ComfyUI's nodes already has torch.inference_mode(True)
        pbar = comfy.utils.ProgressBar(len(overlapping_indices))
        for chunk_id in tqdm(range(len(overlapping_indices))):
            print('Processing frames {} to {}'.format(overlapping_indices[chunk_id][0], overlapping_indices[chunk_id][-1]))
            images_chunk = images_list[overlapping_indices[chunk_id]]

            landmarks_chunk = landmarks[overlapping_indices[chunk_id]]

            _images_list = []

            """ load each image and crop it around the face if necessary """
            for j in range(len(images_chunk)):
                frame = images_chunk[j]
                kpt = landmarks_chunk[j]

                tform = crop_face(frame,kpt,scale=1.6)
                cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))

                _images_list.append(cropped_image.transpose(2,0,1))

            images_array = torch.from_numpy(np.stack(_images_list)).type(dtype = torch.float32).to(get_torch_device()) #K.3,224,224

            codedict, initial_deca_exp, initial_deca_jaw = spectre.encode(images_array)
            codedict['exp'] = codedict['exp'] + initial_deca_exp
            codedict['pose'][..., 3:] = codedict['pose'][..., 3:] + initial_deca_jaw
            
            opdict = spectre.decode(codedict, rendering=False, vis_lmk=False, return_vis=False)

            for key in codedict.keys():
                """ filter out invalid indices - see explanation at the top of the function """

                if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    codedict[key] = codedict[key][:-2]
                elif chunk_id == len(overlapping_indices) - 1:
                    codedict[key] = codedict[key][2:]
                else:
                    codedict[key] = codedict[key][2:-2]
            
            for key in opdict.keys():
                """ filter out invalid indices - see explanation at the top of the function """

                if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    opdict[key] = opdict[key][:-2]
                elif chunk_id == len(overlapping_indices) - 1:
                    opdict[key] = opdict[key][2:]
                else:
                    opdict[key] = opdict[key][2:-2]

            all_verts.append(opdict["verts"].cpu().detach())
            all_cams.append(codedict["cam"].cpu().detach())
            all_cropped_images.append(codedict["images"].cpu().detach())
            pbar.update(1)
        
        all_verts, all_cams, all_cropped_images = torch.cat(all_verts)[2:-2], torch.cat(all_cams)[2:-2], torch.cat(all_cropped_images)[2:-2]
        trans_verts = util.batch_orth_proj(all_verts, all_cams)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        trans_verts[:,:,2] = trans_verts[:,:,2] + 10

        # from observation
        for i in range(len(trans_verts)):
            mesh = trimesh.Trimesh(vertices=trans_verts[i])
            rot_matrix = trimesh.transformations.rotation_matrix(math.pi, direction=[1, 1, 0], point=[0, 0, 0])
            mesh.apply_transform(rot_matrix)
            trans_verts[i] = torch.from_numpy(mesh.vertices)
        all_verts = list(all_verts.unsqueeze(1))

        # Match PerspectiveCamera with IntrinsicsCamera
        # https://github.com/mmatl/pyrender/blob/master/pyrender/camera.py
        # cx, cy = width / 2, height / 2 by default
        yfov = torch.Tensor([0.06]) #From observation again
        focal_length = (1/torch.tan(yfov/2)) * (224 / 2) # P[0][0] = 2.0 * fx / width = 1.0 / (aspect_ratio * np.tan(self.yfov / 2.0))
        all_cropped_images = rearrange(all_cropped_images, "n c h w -> n h w c")

        return ((all_verts, {
            "frame_width": 224, "frame_height": 224, "vertical_flip": True,
            "focal_length": focal_length,
            "faces": spectre.flame.faces_tensor.cpu().detach()
        }), all_cropped_images)

NODE_CLASS_MAPPINGS = {
    "SpectreFaceReconLoader": SpectreFaceReconLoader,
    "SpectreImg2SMPL": SpectreImg2SMPL
}
