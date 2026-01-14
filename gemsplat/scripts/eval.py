# %%
from __future__ import annotations

import json
import os
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
import open3d as o3d

from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.splatfacto import SplatfactoModel

from utils.nerf_utils import *

# # # # #
# # # # # Config Path
# # # # #

# # mode
gaussian_splatting = True

if gaussian_splatting:
    # Gaussian Splatting
    config_path = Path(f"Enter the path to your config file.")
else:
    # Nerfacto
    config_path = Path(f"Enter the path to your config file.")

# %%
 # rescale factor
res_factor = None

# option to enable visualization of the environment point cloud
enable_visualization_pcd = False

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize NeRF
nerf = NeRF(config_path=config_path,
            res_factor=res_factor,
            test_mode="test", #"inference", "val"
            dataset_mode="val",
            device=device)

# camera intrinsics
H, W, K = nerf.get_camera_intrinsics()
K = K.to(device)

# poses in test dataset
poses = nerf.get_poses()

# images for evaluation
eval_imgs = nerf.get_images()

# generate the point cloud of the environment
env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True)

if enable_visualization_pcd:
    # visualize point cloud
    o3d.visualization.draw_plotly([env_pcd]) 

# %%
# list of positives
# e.g., kitchen: ['babynurser bottle', 'red apple', 'kettle']
positives = 'red apple'

# update list of negatives ['things', 'stuff', 'object', 'texture'] -> 'object, things, stuff, texture'
negatives = 'object, things, stuff, texture'

# option to render the point cloud of the entire environment or from a camera
camera_semantic_pcd = False

if camera_semantic_pcd:
    # camera pose
    cam_pose = poses[9]
    
    # generate semantic RGB-D point cloud
    cam_rgb, cam_pcd_points, gem_pcd, depth_mask, outputs = nerf.generate_RGBD_point_cloud(pose=cam_pose,
                                                                                           save_image=True,
                                                                                           filename='figures/eval.png',
                                                                                           compute_semantics=True,
                                                                                           positives=positives,
                                                                                           negatives=negatives)
    
    
    # apply the depth mask to the semantic outputs
    semantic_info = outputs
    
    if depth_mask is not None:
        semantic_info['similarity'] = semantic_info['similarity'][depth_mask]
else:   
    # get the semantic outputs
    semantic_info = nerf.get_semantic_point_cloud(positives=positives,
                                                  negatives=negatives,
                                                  pcd_attr=env_attr)
    
    # initial point cloud for semantic-conditioning
    gem_pcd = env_pcd
# %%
# # #
# # # Generating a Semantic-Conditioned Point Cloud
# # # 

# threshold for masking the point cloud
threshold_mask = 0.88

# scaled similarity
sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
sc_sim = sc_sim / (sc_sim.max() + 1e-6)

# mask
similarity_mask = (sc_sim > threshold_mask).squeeze().reshape(-1,).cpu().numpy()

# masked point cloud
masked_pcd_pts = np.asarray(gem_pcd.points)[similarity_mask, ...]
masked_pcd_color = np.asarray(gem_pcd.colors)[similarity_mask, ...]

# %%
# # #
# # # Visualizing  a Semantic-Conditioned Point Cloud
# # # 

# semantic-conditioned point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(masked_pcd_pts)
pcd.colors = o3d.utility.Vector3dVector(masked_pcd_color)

if enable_visualization_pcd:
    # visualize point cloud
    o3d.visualization.draw_plotly([pcd])
# %%
