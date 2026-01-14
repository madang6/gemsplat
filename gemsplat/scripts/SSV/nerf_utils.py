from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Tuple
import synthesize.trajectory_helper as th

from typing_extensions import Annotated
import pickle
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
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
# from nerfstudio.scripts.render import RenderInterpolated

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.data.datasets.base_dataset import InputDataset

# # # # #
# # # # # Utils
# # # # #

# From INRIA
# Base auxillary coefficient
C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5

def normalized_quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4, quat.shape
    return normalized_quat_to_rotmat(torch.nn.functional.normalize(quat, dim=-1))


def scale_rot_to_cov3d(scale: torch.Tensor, glob_scale: float, quat: torch.Tensor) -> torch.Tensor:
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    R = normalized_quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * glob_scale * scale[..., None, :]  # (..., 3, 3)
    # TODO: save upper right because symmetric
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


class NeRF():
    def __init__(self, config_path: Path, res_factor=None, 
                 width:int=640, height:int=360,
                 test_mode: Literal["test", "val", "inference"] = "inference",
                 dataset_mode: Literal["train", "val", "test"] = 'test',
                 device: Union[torch.device, str] = "cpu"
                 ) -> None:
        # config path
        self.config_path = config_path

        # camera rescale resolution factor
        self.res_factor = res_factor

        # device
        self.device = device

        # initialize pipeline
        self.init_pipeline(test_mode)

        # load dataset
        self.load_dataset(dataset_mode)

        # load cameras
        self.get_cameras()

        # Get reference camera
        self.camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]

        # Render parameters
        self.channels = 3
        self.camera_out,self.width,self.height = self.generate_output_camera(width,height)

        with open("/home/admin/StanfordMSL/SFTI-Program/nerf_data/sv_917_3_left_gemsplat/transforms.json", "r") as f:
            self.transforms_nerf = json.load(f)

    def init_pipeline(self,
        test_mode: Literal["test", "val", "inference"]
    ):
        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path, 
            test_mode=test_mode,
        )

    def load_dataset(self,
        dataset_mode: Literal["train", "val", "test"]
    ):
        # return dataset
        if dataset_mode == "train":
            self.dataset = self.pipeline.datamanager.train_dataset
        elif dataset_mode in ["val", "test"]:
            self.dataset = self.pipeline.datamanager.eval_dataset
        else:
            ValueError('Incorrect value for datset_mode. Accepted values include: dataset_mode: Literal["train", "val", "test"].')

    def get_cameras(self):
        # Camera object contains camera intrinsics and extrinsics
        self.cameras = self.dataset.cameras

        if self.res_factor is not None:
            self.cameras.rescale_output_resolution(self.res_factor)

    def get_poses(self):
        return self.cameras.camera_to_worlds
    
    def get_images(self):
        # images
        images = [self.dataset.get_image_float32(image_idx)
                  for image_idx 
                  in range(len(self.dataset._dataparser_outputs.image_filenames))]
        
        return images
    
    def generate_output_camera(self,width:int,height:int):
        fx,fy = 462.956,463.002
        cx,cy = 323.076,181.184
        
        camera_out = Cameras(
            camera_to_worlds=1.0*self.camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out,width,height
    
    def get_camera_intrinsics(self):
        K = self.cameras[0].get_intrinsics_matrices().squeeze()
        # width and height
        W = int(self.cameras[0].width.item())
        H = int(self.cameras[0].height.item())
        return H, W, K

    def render(self, xcr:np.ndarray, xpr:np.ndarray=None,
               visual_mode:Literal["static","dynamic","semantic_depth","semantic"]="static",
               positives: Optional[str] = "",
               negatives: Optional[str] = "object, things, stuff, texture",
               pose=None, cameras=None, normalized=False,
               sample_semantic_embeds: Optional[bool] = False,
               compute_semantics: Optional[bool] = True,
               debug_mode=False):
        
        if visual_mode == "static":
            image = self.static_render(xcr)
            image_rgb = image["rgb"]
            image_depth = image["depth"]
            image_sem = image["semantic"]

            return {"rgb":image_rgb}
        elif visual_mode == "dynamic":
            image = self.dynamic_render(xcr,xpr)
            image_rgb = image
        elif visual_mode == "semantic_depth":
            image = self.static_render(xcr, normalized, visual_mode, compute_semantics, positives, negatives, sample_semantic_embeds, debug_mode)
            image_rgb = image["rgb"]
            image_depth = image["depth"]
            image_sem = image["semantic"]

            return {"rgb":image_rgb, "depth":image_depth, "sem":image_sem}
        else:
            raise ValueError(f"Invalid visual mode: {visual_mode}")
        
        # Convert to numpy
        # image = image.cpu().numpy()

        # Convert to uint8
        # image = (255*image).astype(np.uint8)

        
        
    def static_render(self, xcr:np.ndarray, normalized=False, visual_mode="static", compute_semantics=False,
                       positives="", negatives="object, things, stuff, texture", sample_semantic_embeds=False, debug_mode=False) -> torch.Tensor:
        # Extract the pose
        T_c2n = pose2nerf_transform(np.hstack((xcr[0:3],xcr[6:10])))
        P_c2n = torch.tensor(T_c2n[0:3,:]).float()
        
        #TODO Fix this by making it more programmatic - currently only use if the nerf you're loading wasn't scaled to the flightroom
        if normalized==True:
            # Compatibility with gemsplat nerf scaling:
            dataparser_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
            dataparser_transform = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

            transform = torch.eye(4)
            transform[:3,:3] = dataparser_transform[:3,:3]
            # invtransform = torch.linalg.inv(transform)

            T_c2n = dataparser_scale * transform @ np.linalg.inv(np.asarray(self.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])) @ T_c2n
            P_c2n = T_c2n[0:3,:].clone().detach().float().requires_grad_(False)
            # P_c2n = torch.tensor(T_c2n[0:3,:]).float()
            P_c2n[:3,:3]*=1/dataparser_scale

            # print("Transformed pose: ", P_c2n)
            # print("Transformation matrix: ", T_c2n)

        # if cameras is None and pose is not None:
            # Render from a single pose
            # camera_to_world = pose[None,:3, ...]
        render_pose_target = P_c2n.clone().detach().float().requires_grad_(False)
        # render_pose_target = torch.tensor(P_c2n,
                                            # device=self.device).float()
        # convert to OpenGL
        render_pose_target_gl = render_pose_target.clone()

        # render_pose_target_gl = render_pose_target_gl[None][:, :3, :]
        # render_pose_target_gl[:,:,:3]*=1/dataparser_scale

        cameras = Cameras(
            # camera_to_worlds=camera_to_world,
            fx=self.camera_out.fx,
            fy=self.camera_out.fy,
            cx=self.camera_out.cx,
            cy=self.camera_out.cy,
            width=self.camera_out.width,
            height=self.camera_out.height,
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=render_pose_target_gl[None][:, :3, :]
        )
            
        cameras = cameras.to(self.device)
        
        if visual_mode == "semantic" or visual_mode == "semantic_depth":
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                        is_positive=True)
                
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                        is_positive=False)
            # render outputs
            if isinstance(self.pipeline.model, NerfactoModel):
                aabb_box = None
                camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=aabb_box)

                tnow = time.perf_counter()
                with torch.no_grad():
                    outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                if debug_mode:
                    print('Rendering time: ', time.perf_counter() - tnow)

                # insert ray bundles
                outputs['ray_bundle'] = camera_ray_bundle
            # elif isinstance(self.pipeline.model, SplatfactoModel):
            else:
                obb_box = None
                
                tnow = time.perf_counter()
                with torch.no_grad():
                    try:
                        outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box,
                                                                            sample_semantic_embeds=sample_semantic_embeds,
                                                                            compute_semantics=compute_semantics)
                    except:  
                        try:
                            outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box,
                                                                                compute_semantics=compute_semantics)
                        except:
                            outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box)

                if debug_mode:
                    print('Rendering time: ', time.perf_counter() - tnow)

            if "similarity_GUI" not in outputs.keys():
                print("No semantic output found")
                image_sem = outputs['rgb']
            else:
                image_sem = outputs['similarity_GUI']
            image_d = outputs['depth']
            image_rgb = outputs['rgb']

        
            # image_sem = np.squeeze(image_sem).cpu().numpy()
            # sem_norm = cv2.normalize(image_sem,None,0,255,cv2.NORM_MINMAX)
            # sem_norm = sem_norm.astype(np.uint8)
            # sem_colored = cv2.applyColorMap(sem_norm, cv2.COLORMAP_JET)
            # sem_colored_tensor = torch.from_numpy(sem_colored).float().cuda() / 255.0
            # image_sem = sem_colored_tensor
            image_sem = image_sem.cpu().numpy()
            image_sem = (255*image_sem).astype(np.uint8)

            #TODO make sure to only use this when testing depth
            # image_d = image['depth']
            depth_image = np.squeeze(image_d).cpu().numpy()
            depth_normalized = cv2.normalize(depth_image,None,0,255,cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_colored_tensor = torch.from_numpy(depth_colored).float().cuda() / 255.0
            image_depth = depth_colored_tensor
            image_depth = image_depth.cpu().numpy()
            image_depth = (255*image_depth).astype(np.uint8)

            # image_rgb = image['rgb']
            image_rgb = image_rgb.cpu().numpy()
            image_rgb = (255*image_rgb).astype(np.uint8)

            return {"semantic":image_sem, "depth":image_depth, "rgb":image_rgb}
        
        else:
            # render outputs
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=None)

            image_rgb = outputs["rgb"]
            image_rgb = image_rgb.cpu().numpy()
            image_rgb = (255*image_rgb).astype(np.uint8)

            image_depth = np.zeros_like(image_rgb)
            image_sem = np.zeros_like(image_rgb)

            return {"semantic": image_sem,"depth": image_depth,"rgb":image_rgb}
    
    def generate_point_cloud(self,
                             use_bounding_box: bool = False,
                             bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1),
                             bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1),
                             densify_scene: bool = False,
                             split_params: Dict = {
                                 'n_split_samples': 2
                             },
                             cull_scene: bool = False,
                             cull_params: Dict = {
                                 'cull_alpha_thresh': 0.1,
                                 'cull_scale_thresh': 0.5
                             }
                             )->None:
        if densify_scene:
            if cull_scene:
                # cache the previous values of all parameters
                means_prev = self.pipeline.model.means.clone()
                scales_prev = self.pipeline.model.scales.clone()
                quats_prev = self.pipeline.model.quats.clone() 
                features_dc_prev = self.pipeline.model.features_dc.clone()
                features_rest_prev = self.pipeline.model.features_rest.clone()
                opacities_prev = self.pipeline.model.opacities.clone()
                
                try:
                    clip_embeds_prev = self.pipeline.model.clip_embeds.clone()
                except AttributeError:
                    pass
        
                # cull Gaussians
                self.pipeline.model.cull_gaussians_refinement(cull_alpha_thresh=cull_params['cull_alpha_thresh'],
                                                              cull_scale_thresh=cull_params['cull_scale_thresh'],
                                                              )
                
            # split mask
            split_mask = torch.ones(len(self.pipeline.model.scales),
                                    dtype=torch.bool).to(self.device)

            # split Gaussians
            try:
                means, features_dc, features_rest, opacities, scales, quats, clip_embeds \
                    = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])
            except AttributeError:
                means, features_dc, features_rest, opacities, scales, quats \
                    = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])
                    
                # extract the semantic embeddings
                clip_embeds = self.pipeline.model.clip_field(self.pipeline.model.means).float().detach()

            
            # 3D points
            pcd_points = means

            # colors computed from the term of order 0 in the Spherical Harmonic basis
            # coefficient of the order 0th-term in the Spherical Harmonics basis
            pcd_colors_coeff = features_dc
        else:
            # 3D points
            pcd_points = self.pipeline.model.means

            # colors computed from the term of order 0 in the Spherical Harmonic basis
            # coefficient of the order 0th-term in the Spherical Harmonics basis
            pcd_colors_coeff = self.pipeline.model.features_dc
            
            # other attributes of the Gaussian
            opacities, scales, quats \
                = self.pipeline.model.opacities, self.pipeline.model.scales, self.pipeline.model.quats
                   
            try:
                clip_embeds =  self.pipeline.model.clip_embeds
            except AttributeError:
                # extract the semantic embeddings
                clip_embeds = self.pipeline.model.clip_field(self.pipeline.model.means).float().detach()

        # color computed from the Spherical Harmonics
        pcd_colors = SH2RGB(pcd_colors_coeff).squeeze()
        
        # mask points using a bounding box
        if use_bounding_box:
            mask = ((pcd_points[:, 0] > bounding_box_min[0]) & (pcd_points[:, 0] < bounding_box_max[0])
                    & (pcd_points[:, 1] > bounding_box_min[1]) & (pcd_points[:, 1] < bounding_box_max[1])
                    & (pcd_points[:, 2] > bounding_box_min[2]) & (pcd_points[:, 2] < bounding_box_max[2])
            )
            
            pcd_points = pcd_points[mask]
            pcd_colors = pcd_colors[mask]
            
            # other attributes of the Gaussian
            opacities, scales, quats, clip_embeds \
                = opacities[mask], scales, quats[mask], clip_embeds[mask]
        else:
            mask = None
            
        # apply transformation to the opacities and scales
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)

        # enviromment attributes
        env_attr = {'means': pcd_points,
                    'quats': quats,
                    'scales': scales,
                    'opacities': opacities,
                    'clip_embeds': clip_embeds
                    }

        # create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points.double().cpu().detach().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors.double().cpu().detach().numpy())
        
        # reset the values of all parameters
        if cull_scene:
            self.pipeline.model.means = torch.nn.Parameter(means_prev)
            self.pipeline.model.scales = torch.nn.Parameter(scales_prev)
            self.pipeline.model.quats = torch.nn.Parameter(quats_prev)
            self.pipeline.model.features_dc = torch.nn.Parameter(features_dc_prev)
            self.pipeline.model.features_rest = torch.nn.Parameter(features_rest_prev)
            self.pipeline.model.opacities = torch.nn.Parameter(opacities_prev)
            
            try:
                self.pipeline.model.clip_embeds = torch.nn.Parameter(clip_embeds_prev)
            except AttributeError:
                pass

        return pcd, mask, env_attr
    
    def get_semantic_point_cloud(self,
                                 positives: str = "",
                                 negatives: str = "object, things, stuff, texture",
                                 pcd_attr: Dict[str, torch.Tensor] = {}):
        # update the list of positives (objects)
        self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                 is_positive=True)
        
        # update the list of positives (objects)
        self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                 is_positive=False)
        
        # semantic point cloud
        # CLIP features
        if 'clip_embeds' in pcd_attr.keys():
            pcd_clip = pcd_attr['clip_embeds']
        else:
            try:
                pcd_clip = self.pipeline.model.clip_embeds
            except AttributeError:
                # extract the semantic embeddings
                pcd_clip = self.pipeline.model.clip_field(self.pipeline.model.means).float().detach()

        # get the semantic outputs
        pcd_clip = {'clip': pcd_clip}
        semantic_pcd = self.pipeline.model.get_semantic_outputs(pcd_clip)
        
        return semantic_pcd
    
    # Generate RGB-D Image
    def generate_RGBD_point_cloud(self, pose,
                                  save_image: bool=False,
                                  filename: Optional[str]='/',
                                  compute_semantics: Optional[bool] = False,
                                  max_depth: Optional[float] = 1.0,
                                  return_pcd: Optional[bool] = True,
                                  positives: str = "",
                                  negatives: str = "object, things, stuff, texture"
                                 ):
        # update the semantic-query information
        if compute_semantics:
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                     is_positive=True)
            
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                     is_positive=False)
            
        # pose to render from
        pose = pose.to(self.device)
        
        # render
        outputs = self.render(pose,
                              compute_semantics=compute_semantics)
        
        print(outputs.keys())
        
        if save_image:
            # figure
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            plt.tight_layout()

            # plot rendering
            axs[0].imshow(outputs['rgb'].cpu().numpy())
            axs[1].imshow(outputs['depth'].cpu().numpy())

            for ax in axs:
                ax.set_axis_off()
            plt.show()
            
            # create directory, if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # save figure
            fig.savefig(filename)

        # create a point cloud from RGB-D image
        # depth channel
        cam_depth = outputs['depth'].squeeze()
        cam_rgb = outputs['rgb']
        
        # depth mask
        if max_depth is None:
            depth_mask = torch.ones_like(cam_depth, dtype=bool, device=cam_depth.device)
        else:
            depth_mask = cam_depth < max_depth 
        
        # start time
        t0 = time.perf_counter()
        
        # camera intrinsics
        H, W, K = self.get_camera_intrinsics()

        # unnormalized pixel coordinates
        u_coords = torch.arange(W, device=self.device)
        v_coords = torch.arange(H, device=self.device)

        # meshgrid
        U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')

        # transformed points in camera frame
        # [u, v, 1] = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]] @ [x/z, y/z, 1]
        cam_pts_x = (U_grid - K[0, 2]) * cam_depth / K[0, 0]
        cam_pts_y = (V_grid - K[1, 2]) * cam_depth / K[1, 1]
        
        cam_pcd_points = torch.stack((cam_pts_x, cam_pts_y, cam_depth), axis=-1)

        if return_pcd:
            # point cloud
            cam_pcd = o3d.geometry.PointCloud()
            cam_pcd.points = o3d.utility.Vector3dVector(cam_pcd_points[depth_mask, ...].view(-1, 3).double().cpu().detach().numpy())
            cam_pcd.colors = o3d.utility.Vector3dVector(cam_rgb[depth_mask, ...].view(-1, 3).double().cpu().detach().numpy())
            
        else:
            cam_pcd = None
        
        return cam_rgb, cam_pcd_points, cam_pcd, depth_mask, outputs
    
def load_dataset(data_path: Path,
    dataset_mode: Literal["train", "val", "test", "all"] # 'all' uses the entire dataset.
):
    # init Nerfstudio dataset config
    nerfstudio_data_parser_config = NerfstudioDataParserConfig(data=data_path,
                                                               eval_mode="all" if dataset_mode == "all" else "fraction")

    # init data parser
    nerfstudio_data_parser = Nerfstudio(nerfstudio_data_parser_config)

    # data parser outputs
    data_parser_ouputs = nerfstudio_data_parser._generate_dataparser_outputs(split=dataset_mode
                                                                             if dataset_mode != "all" else "val")

    # load dataset
    dataset = InputDataset(data_parser_ouputs)

def get_nerf(map:Literal["sv_701","sv_712","gate","crossroads"]) -> NeRF:
    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main_dir_path = os.getcwd()
    nerf_dir_path = os.path.join(workspace_path,"nerf_data")

    maps = {
        "gate_left":"sv_917_3_left_nerfstudio",
        "gate_left_gemsplat":"sv_917_3_left_gemsplat",
        "gate_right":"sv_917_3_right_nerfstudio",
        "gate_mid":"sv_917_3_mid_nerfstudio",
        "clutter":"sv_712_nerfstudio",
    }

    map_folder = os.path.join(nerf_dir_path,'outputs',maps[map])
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')

    # Go into NeRF data folder and get NeRF object (because the NeRF instantation
    # requires the current working directory to be the NeRF data folder)
    os.chdir(nerf_dir_path)
    nerf = NeRF(Path(nerf_cfg_path))
    os.chdir(main_dir_path)

    return nerf

def pose2nerf_transform(pose):

    # Realsense to Drone Frame
    T_r2d = np.array([
        [ 0.989, 0.021, 0.145, 0.156],
        [-0.021, 1.000,-0.000,-0.033],
        [-0.145,-0.003, 0.989,-0.035],
        [ 0.000, 0.000, 0.000, 1.000]
    ])

    # Drone to Flightroom Frame
    T_d2f = np.eye(4)
    T_d2f[0:3,:] = np.hstack((R.from_quat(pose[3:]).as_matrix(),pose[0:3].reshape(-1,1)))

    # Flightroom Frame to NeRF world frame
    T_f2n = np.array([
        [ 1.000, 0.000, 0.000, 0.000],
        [ 0.000,-1.000, 0.000, 0.000],
        [ 0.000, 0.000,-1.000, 0.000],
        [ 0.000, 0.000, 0.000, 1.000]
    ])

    # Camera convention frame to realsense frame
    T_c2r = np.array([
        [ 0.0, 0.0,-1.0, 0.0],
        [ 1.0, 0.0, 0.0, 0.0],
        [ 0.0,-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ])

    # Get image transform
    T_c2n = T_f2n@T_d2f@T_r2d@T_c2r

    return T_c2n