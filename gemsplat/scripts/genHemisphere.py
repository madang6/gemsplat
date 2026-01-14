# %%
from __future__ import annotations

import json
import os, sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
import open3d as o3d
from enum import Enum
from PIL import Image
from nerfstudio.utils.rich_utils import Console

from nerfstudio.cameras.camera_utils import quaternion_from_matrix

import copy
import gc

console = Console()

from utils.nerf_utils import *
from utils.scene_editing_utils import *

# # #
# # # Generate a hemisphere centered at the look-at point
# # #
from scipy.spatial.transform import Rotation as rot

def SE3error(T, That):
    Terr = np.linalg.inv(T) @ That

    r = rot.from_matrix(Terr[:3, :3])
    axis_angle = r.as_rotvec()
    axis_angle = axis_angle / np.linalg.norm(axis_angle)

    rerr = abs(np.arccos(min(max(((Terr[0:3,0:3]).trace() - 1) / 2, -1.0), 1.0)))

    terr = np.linalg.norm(Terr[0:3,3])
    return (rerr*180/np.pi, terr, axis_angle[0], axis_angle[1], axis_angle[2])

# inspired by LERF-TOGO
def point_camera_at(cam_center, look_at_point):
    # z-direction
    z_dir = look_at_point - cam_center
    z_dir = z_dir / np.linalg.norm(z_dir)
    
    # orthogonal unit vector for the x-axis
    x_dir = -np.cross(np.array([0, 0, 1]), z_dir)
    
    # error-checking
    if np.linalg.norm(x_dir) < 1e-10:
        x_dir = np.array([0, 1, 0])
        
    # normalize the vector
    x_dir = x_dir / np.linalg.norm(x_dir)
    
    # y-direction
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    
    # pose of the camera
    pose = np.eye(4)
    pose[:3, :3] = np.hstack((x_dir[:, None], y_dir[:, None], z_dir[:, None]))
    pose[:3, -1] = cam_center
    
    return pose


def generate_hemisphere(center, radius, theta_intervals, phi_intervals, look_at_point, sweep_phi_before_theta=False):
    # poses
    poses = []
    
    # sweep through the angles
    if sweep_phi_before_theta:
        for theta_idx, theta in enumerate(theta_intervals):
            # poses
            pose_sweep = [point_camera_at(cam_center=np.array([center[0] + radius * np.cos(theta) * np.sin(phi),
                                                               center[1] + radius * np.sin(theta) * np.sin(phi),
                                                               center[2] + radius * np.cos(phi)]),
                                          look_at_point=look_at_point)
                          for phi in phi_intervals]
            
            # reverse the order of the poses at every odd sweep to get a smooth trajectory for the cameras.
            if theta_idx % 2 == 1:
                pose_sweep.reverse()
                
            # add to poses
            poses.extend(pose_sweep)
    else:
        for phi_idx, phi in enumerate(phi_intervals):
            # poses
            pose_sweep = [point_camera_at(cam_center=np.array([center[0] + radius * np.cos(theta) * np.sin(phi),
                                                           center[1] + radius * np.sin(theta) * np.sin(phi),
                                                           center[2] + radius * np.cos(phi)]),
                                          look_at_point=look_at_point)
                          for theta in theta_intervals]
            # pose_sweep = [point_camera_at(cam_center=np.array([center[0] + radius * np.cos(theta) * np.sin(phi),
            #                                                center[1] + radius * np.sin(theta) * np.sin(phi),
            #                                                center[2]]),
            #                               look_at_point=look_at_point)
            #               for theta in theta_intervals]
            
            # reverse the order of the poses at every odd sweep to get a smooth trajectory for the cameras.
            if phi_idx % 2 == 1:
                pose_sweep.reverse()
                
            # add to poses
            poses.extend(pose_sweep)
            
    return poses


def get_hemisphere(nerf, env_pts, target_look_at_point, hemisphere_radius, theta_intervals, phi_intervals, exclusion_radius=None):
    from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # name of the scene
    scene_name = 'scene'

    # target points to look at
    # target_points = "Please enter your target point."

    # point to look at in the target Gaussian Splatting scene
    # target_look_at_point = target_points.mean(dim=0).cpu().numpy()

    # generate a candidate pose for the camera in the target Gaussian Splatting scene
    # parameters for the hemisphere
    # hemisphere_radius = [0.2, 0.25, 0.3]
    # theta_intervals = [-np.pi / 2, -np.pi / 3]
    # phi_intervals = [np.pi / 8, np.pi / 4]
    render_target_poses = []
    env_pts = env_pts[:3, :]
    for h_rad in hemisphere_radius:
        poses = generate_hemisphere(center=target_look_at_point,
                                                    radius=h_rad,
                                                    theta_intervals=theta_intervals,
                                                    phi_intervals=phi_intervals,
                                                    look_at_point=target_look_at_point)
        
        #TODO: check within a ball if the pcd has points
        if exclusion_radius is not None:
            pose_xyz = np.array([pose[:3, -1] for pose in poses])
            for pose, xyz in zip(poses, pose_xyz):
                distances = np.linalg.norm(env_pts - xyz[:,np.newaxis], axis=0)
                if not np.any(distances <= exclusion_radius):
                    render_target_poses.append(pose)
        else:
            render_target_poses.extend(poses)

        print()
        # render_target_poses.extend(poses)

    # path to save the images
    img_path: Path = Path(f'./data/{scene_name}_refinement/images')

    # create directory, if necessary
    img_path.mkdir(exist_ok=True, parents=True)

    # camera Intrinsics
    cam_fx = torch.tensor([[635]])
    cam_fy = cam_fx
    cam_cx = torch.tensor([[646]])
    cam_cy = torch.tensor([[371]])
    cam_type = torch.tensor([[1]])

    # image index
    img_idx = 0

    # Generate the images
    pose_targs = []
    for pose_idx, targ_pose in enumerate(render_target_poses):
        # pose to render from
        render_pose_target = torch.tensor(targ_pose,
                                        device=device).float()

        # convert to OpenGL
        render_pose_target_gl = render_pose_target.clone()
        render_pose_target_gl[:, 1:3] *= -1

        pose_targs.append(render_pose_target_gl)
        
        # create a camera
        camera = Cameras(
            fx=cam_fx,
            fy=cam_fy,
            cx=cam_cx,
            cy=cam_cy,
            camera_type=cam_type,
            camera_to_worlds=render_pose_target_gl[None][:, :3, :]
        )
        
        # Target Model
        
        # RGB image from the target Gaussian Splatting Map
        # target_rgb = nerf.render(cameras=camera, pose=None)['rgb']
        target_rgb = nerf.render(cameras=camera, pose=None, compute_semantics=True)['similarity_GUI']
        # target_rgb = nerf.render(pose=render_pose_target_gl)['rgb']
        
        # render an image
        img_rgb = Image.fromarray((255 * target_rgb).detach().cpu().numpy().astype(np.uint8))
        
        # image relative path
        img_rel_path = f'frame_{img_idx:05}.png'
        
        # save the image
        img_rgb.save(f'{img_path}/{img_rel_path}')
        
        # increment the image index
        img_idx += 1

    return pose_targs