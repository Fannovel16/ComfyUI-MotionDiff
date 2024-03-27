#Based on https://github.com/Mael-zys/T2M-GPT/blob/main/render_final.py
from mogen.smpl.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os

#https://stackoverflow.com/a/45756291
if os.name == 'posix' and "DISPLAY" not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = "egl"

import torch
import comfy.utils
from mogen.smpl.simplify_loc2rot import joints2smpl
import pyrender
from pyrender.shader_program import ShaderProgramCache
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
from comfy.model_management import get_torch_device
from tqdm import tqdm

shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def render(motions):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device=get_torch_device())
    rot2xyz = Rotation2xyz(device=get_torch_device())
    faces = rot2xyz.smpl_model.faces

    print(f'Running SMPLify, it may take a few minutes.')
    motion_tensor, opt_dict = j2s.forward(motions)  # [nframes, njoints, 3]

    vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)

    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5


    out_list = []
    
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []
    for i in range(frames):
        if i % 10 == 0:
            print(i)

        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        base_color = (0.11, 0.53, 0.8, 0.5)
        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )


        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)

        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        c = np.pi / 2

        scene.add(polygon_render, pose=np.array([[ 1, 0, 0, 0],

        [ 0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],

        [ 0, np.sin(c), np.cos(c), 0],

        [ 0, 0, 0, 1]]))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())


        c = -np.pi / 6

        scene.add(camera, pose=[[ 1, 0, 0, (minx+maxx).cpu().numpy()/2],

                                [ 0, np.cos(c), -np.sin(c), 1.5],

                                [ 0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())],

                                [ 0, 0, 0, 1]
                                ])
        
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        vid.append(color)

        r.delete()

    out = np.stack(vid, axis=0)
    return out

def render_from_smpl(thetas, yfov, move_x, move_y, move_z, x_rot, y_rot, z_rot, frame_width, frame_height, draw_platform=True, depth_only=False, normals=False, smpl_model_path=None, shape_parameters=None):
    if shape_parameters is not None:
        betas_tensor = torch.tensor([shape_parameters], dtype=torch.float32)
        batch_size = thetas.shape[3]  
        betas_batch = betas_tensor.repeat(batch_size, 1)  # Replicates the single sample across the batch
        betas_batch = betas_batch.to(device=get_torch_device())
    else:
        betas_batch = None

    rot2xyz = Rotation2xyz(device=get_torch_device(), smpl_model_path=smpl_model_path, betas=betas_batch)
    faces = rot2xyz.smpl_model.faces

    vertices = rot2xyz(thetas.clone().to(get_torch_device()).detach(), mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)

    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    #print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5

    if draw_platform:
        polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
        polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)
        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
        c = np.pi / 2
        platform_pose=np.array([[ 1, 0, 0, 0],
                                [ 0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],
                                [ 0, np.sin(c), np.cos(c), 0],
                                [ 0, 0, 0, 1]])

    base_color = (0.11, 0.53, 0.8, 0.5)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )

    x_translation = move_x  #X-axis translation value
    y_translation = move_y # Y-axis translation value
    z_translation = move_z  # Z-axis translation value

    initial_pos = [(minx+maxx).cpu().numpy()/2 + x_translation,
                y_translation,
                max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy()) + z_translation]

    alpha = np.radians(x_rot)
    beta = np.radians(y_rot)
    gamma = np.radians(z_rot)

    # Rotation matrix around X-axis
    R_x = [[1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]]

    # Rotation matrix around Y-axis
    R_y = [[np.cos(beta), 0, np.sin(beta), 0],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1]]

    # Rotation matrix around Z-axis
    R_z = [[np.cos(gamma), -np.sin(gamma), 0, 0],
        [np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]

    # Combine rotations, order of multiplication depends on the desired rotation order
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Now, R is a 4x4 matrix that represents the rotation around X, Y, and Z

    # Translation vector
    T = [initial_pos[0], initial_pos[1], initial_pos[2], 1]

    # Combine the rotation and translation into the final transformation matrix
    camera_pose = np.dot(R, np.array([[1, 0, 0, T[0]],
                               [0, 1, 0, T[1]],
                               [0, 0, 1, T[2]],
                               [0, 0, 0, 1]]))
    if normals and not depth_only:
        r = pyrender.OffscreenRenderer(frame_width, frame_height)
        r._renderer._program_cache = ShaderProgramCache(shader_dir=shader_dir)
    else:
        r = pyrender.OffscreenRenderer(frame_width, frame_height)
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)
        light_positions = [
            [0, -1, 1],
            [0, 1, 1],
            [1, 1, 2] 
        ]
        # Create transformation matrices for each light
        light_poses = [np.eye(4) for _ in light_positions]
        for i, position in enumerate(light_positions):
            light_poses[i][:3, 3] = position
     
    #Build the scene
    camera = pyrender.PerspectiveCamera(yfov)
    bg_color = [1, 1, 1, 0.8]
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
    scene.add(camera, pose=camera_pose)
    if draw_platform:
            scene.add(polygon_render, pose=platform_pose)
    if not normals:    
        for pose in light_poses:
            scene.add(light, pose=pose)
    
    # Render loop
    vid = []
    vid_depth = []
    print("Rendering SMPL human mesh...")
    pbar = comfy.utils.ProgressBar(frames)
    for i in tqdm(range(frames)):
        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)

        if depth_only:
            depth = r.render(scene, flags=RenderFlags.DEPTH_ONLY)
            color = np.zeros([frame_width, frame_height, 3])
        else:
            color, depth = r.render(scene, flags=RenderFlags.RGBA)

        vid.append(color)
        vid_depth.append(depth)
        pbar.update(1)
    r = None

    return np.stack(vid, axis=0), np.stack(vid_depth, axis=0)