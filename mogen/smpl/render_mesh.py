#Based on https://github.com/Mael-zys/T2M-GPT/blob/main/render_final.py
from mogen.smpl.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os

#https://stackoverflow.com/a/45756291
if os.name == 'posix' and "DISPLAY" not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = "egl"

import torch
from mogen.smpl.simplify_loc2rot import joints2smpl
import pyrender
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
from comfy.model_management import get_torch_device
from tqdm import tqdm

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

def render_from_smpl(thetas, yfov, move_x, move_y, move_z, draw_platform=True, depth_only=False, smpl_model_path=None):
    rot2xyz = Rotation2xyz(device=get_torch_device(), smpl_model_path=smpl_model_path)
    faces = rot2xyz.smpl_model.faces

    vertices = rot2xyz(thetas.clone().to(get_torch_device()).detach(), mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)

    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5
    
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []
    vid_depth = []
    print("Rendering SMPL human mesh...")
    for i in tqdm(range(frames)):

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

        camera = pyrender.PerspectiveCamera(yfov)
        
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        c = np.pi / 2

        if draw_platform:
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

        x_translation = move_x  # Your X-axis translation value
        y_translation = move_y # Your Y-axis translation value
        z_translation = move_z  # Your Z-axis translation value

        initial_pos = [(minx+maxx).cpu().numpy()/2 + x_translation,
                    y_translation,
                    max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy()) + z_translation]

        pose = [[ 1, 0, 0, initial_pos[0]],
                [ 0, np.cos(c), -np.sin(c), 1.5 + initial_pos[1]],
                [ 0, np.sin(c), np.cos(c), initial_pos[2]],
                [ 0, 0, 0, 1]
            ]

        # Add the camera to the scene with the modified pose
        scene.add(camera, pose=pose)

        # Add the camera to the scene with the modified pose
        scene.add(camera, pose=pose)
        
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        if depth_only:
            depth = r.render(scene, flags=RenderFlags.DEPTH_ONLY)
            color = np.zeros([960, 960, 3])
        else:
            color, depth = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        vid.append(color)
        vid_depth.append(depth)

        r.delete()

    return np.stack(vid, axis=0), np.stack(vid_depth, axis=0)