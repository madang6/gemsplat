#%%
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import open3d as o3d
from gemsplat.scripts.utils.nerf_utils import NeRF
from gemsplat.scripts.utils.scene_editing_utils import get_centroid


#%%
def rescale_point_cloud(nerf,viz=False,cull=False,verbose=False):
    viz = True
    verbose = True

    cull = False
    
    # Generate the point cloud of the environment
    env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True)
    # env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True,bounding_box_max=(5.50,2.75,2.5),bounding_box_min=(-5.50,-2.75,0.0))

    cl, ind = env_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    filtered_pcd = env_pcd.select_by_index(ind)

    dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

    transform = torch.eye(4)
    transform[:3,:3] = dataparser_transform[:3,:3]
    invtransform = torch.linalg.inv(transform)

    env_pcd_scaled = np.asarray(filtered_pcd.points).T / dataparser_scale
    env_pcd_scaled = np.vstack((env_pcd_scaled, np.ones((1, env_pcd_scaled.shape[1]))))

    # env_pcd_scaled = invtransform @ env_pcd_scaled
    # env_pcd_scaled = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(env_pcd_scaled)

    env_pcd_scaled = np.asarray(env_pcd_scaled) @ np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ invtransform

    env_pcd_scaled = env_pcd_scaled[:3, :].T

    epcds = o3d.geometry.PointCloud()
    epcds.points=o3d.utility.Vector3dVector(env_pcd_scaled)
    epcds.colors=o3d.utility.Vector3dVector(np.asarray(filtered_pcd.colors))

    minbound = np.percentile(env_pcd_scaled,5, axis=0).tolist()
    maxbound = np.percentile(env_pcd_scaled,95, axis=0).tolist()
    if cull:
        cullbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=minbound, max_bound=maxbound)
        epcds = epcds.crop(cullbox)

    epcds_aabb = epcds.get_axis_aligned_bounding_box()
    bx, by, bz = epcds_aabb.get_extent()

    if verbose:
        if not cull:
            print("Theoretical Bounding Box:")
        print(f"Bounding Box: {bx}, {by}, {bz}")
        print(f"Minbound: {minbound}", f"Maxbound: {maxbound}")

    if viz:
        o3d.visualization.draw_plotly([epcds])

    epcds_bounds = {"minbound": minbound, "maxbound": maxbound}

    return epcds, env_pcd_scaled.T, epcds_bounds, env_pcd, env_pcd_mask, env_attr

#%%
def get_points(path: Path, positives: str, negatives: str, threshold: float, filter_radius: float, enable_visualization_pcd=False):

    # # # # #
    # # # # # Config Path
    # # # # #

    # rootdir = Path("/home/admin/StanfordMSL/")
    # SFTIpth = rootdir / "SFTI-Program"
    # outpth  = SFTIpth / "nerf_data/outputs/"
    # scnpth  = outpth / "sv_806_3_nerfstudio/gemsplat"
    modelpth = path

    # # mode
    gaussian_splatting = True

    if gaussian_splatting:
        # Gaussian Splatting
        config_path = Path(f"modelpth / config.yml")
    else:
        # Nerfacto
        config_path = Path(f"<Enter the path to your config file.>")

    # rescale factor
    res_factor = None

    # option to enable visualization of the environment point cloud
    # enable_visualization_pcd = True

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize NeRF
    nerf = NeRF(config_path=modelpth / "config.yml",
                test_mode="inference", #"inference", "eval"
                dataset_mode="test",
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
    # env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True,bounding_box_max=(6.5,2.75,2.5),bounding_box_min=(-6.50,-2.75,0.0))

    if enable_visualization_pcd:
        # visualize point cloud
        o3d.visualization.draw_plotly([env_pcd]) 

    # list of positives
    # e.g., kitchen: ['babynurser bottle', 'red apple', 'kettle']

    # update list of negatives ['things', 'stuff', 'object', 'texture'] -> 'object, things, stuff, texture'

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

    # # #
    # # # Generating a Semantic-Conditioned Point Cloud
    # # # 

    # threshold for masking the point cloud
    threshold_mask = threshold

    # scaled similarity
    sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
    sc_sim = sc_sim / (sc_sim.max() + 1e-6)

    # mask
    similarity_mask = (sc_sim > threshold_mask).squeeze().reshape(-1,).cpu().numpy()

    # masked point cloud
    masked_pcd_pts = np.asarray(gem_pcd.points)[similarity_mask, ...]
    masked_pcd_color = np.asarray(gem_pcd.colors)[similarity_mask, ...]

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

    # # # # #
    # # # # # Scene Editing from the Gaussian Means: Objects to Target
    # # # # #  

    # option to print debug info
    print_debug_info: bool = True

    # Set of all fruits
    # fruits = {'fruit', 'apple', 'orange', 'pear', 'tomato'}

    # objects to move
    objects: List[str] = [positives]


    # # # # # #
    # # # # # # Scene Editing from the Gaussian Means: Generate a Sample Trajectory and Task
    # # # # # #  

    # # objects: List[str] = ['saucepan', 'glass lid', 'knife', 'orange']
    # # object_to_target: Dict[str, str] = {'saucepan': targets[0],
    # #                                     'glass lid': targets[0],
    # #                                     'knife': targets[1],
    # #                                     'orange': targets[1] 
    # #                                     }

    # outputs for each object
    obj_outputs = {}

    # offset for each object
    offsets = np.zeros((len(objects), 3))

    # filter-size - radius
    # filter_radius = [0.075, 0.05, 0.05, 0.05]
    filter_radius = [filter_radius]
    # filter_radius = [0.25, 0.15, 0.1, 0.05]
    # TODO
    filter_radius.extend([filter_radius[-1]] * (len(objects) - len(filter_radius)))

    # similarity threshold - multiple objects
    # threshold_obj = [0.6, 0.9, 0.9, 0.97]
    threshold_obj = [threshold]
    # TODO
    threshold_obj.extend([threshold_obj[-1]] * (len(objects) - len(threshold_obj)))

    for idx, obj in enumerate(objects):
    # if idx < 2:
        # continue

        print('*' * 50)
        print(f'Processing Object: {obj}')
        print('*' * 50)
        
    # prior information on the object masks
    # obj_priors: Dict = {
    #     'mask_prior': table_attr['raw_similarity']
    # }

        # source location
        src_centroid, src_z_bounds, scene_pcd, similarity_mask, other_attr = get_centroid(nerf=nerf,
                                                                                        env_pcd=env_pcd,
                                                                                        pcd_attr=env_attr,
                                                                                        positives=objects[idx],
                                                                                        negatives='object, things, stuff, texture',
                                                                                        threshold=threshold_obj[idx],
                                                                                        visualize_pcd=False,
                                                                                        enable_convex_hull=True,
                                                                                        enable_spherical_filter=True,
                                                                                        enable_clustering=False,
                                                                                        filter_radius=filter_radius[idx],
                                                                                        obj_priors={},#obj_priors,
                                                                                        use_Mahalanobis_distance=True)

        # object
        object_pcd_points = np.asarray(scene_pcd.points)[similarity_mask]
        object_pcd_colors = np.asarray(scene_pcd.colors)[similarity_mask]
        object_pcd_sim = other_attr['raw_similarity'][similarity_mask].cpu().numpy()

        # if any(item in obj for item in ['pot', 'pan', 'lid']):
        # plane-fitting
        pcd_clus = o3d.geometry.PointCloud()
        pcd_clus.points = o3d.utility.Vector3dVector(object_pcd_points[:, :3])
        pcd_clus.colors = o3d.utility.Vector3dVector(object_pcd_colors)

        if enable_visualization_pcd:
            fig = o3d.visualization.draw_plotly([pcd_clus])

    # remove outliers
        # pcd, inlier_ind = pcd_clus.remove_radius_outlier(nb_points=30, radius=0.03) # r=0.03 maybe nb_points=5
        pcd, inlier_ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
            
        if enable_visualization_pcd:
            print('*' * 50)
            print('Post-Outlier-Removal')
            fig = o3d.visualization.draw_plotly([pcd])
            # fig.show()
            print('*' * 50)
    
    return nerf, {"env_pcd": env_pcd, "pcd": pcd, "inlier_ind": inlier_ind, "pcd_clus": pcd_clus, "src_centroid": src_centroid, "src_z_bounds": src_z_bounds, "scene_pcd": scene_pcd, "similarity_mask": similarity_mask, "other_attr": other_attr, "object_pcd_points": object_pcd_points, "object_pcd_colors": object_pcd_colors, "object_pcd_sim": object_pcd_sim}

#%%
if __name__ == '__main__':
    #%%
    # path = Path("/home/admin/StanfordMSL/SFTI-Program/nerf_data/outputs/sv_701_nerfstudio/gemsplat/2024-08-07_134148")
    path = Path("/home/admin/StanfordMSL/SFTI-Program/nerf_data/outputs/sv_701_nerfstudio_gemsplat_2/gemsplat/2024-09-04_142608")
    positives = "ladder"
    negatives = "object, things, stuff, texture"
    threshold = 0.6
    filter_radius = 0.05
    nrf, pts = get_points(path, positives, negatives, threshold, filter_radius, enable_visualization_pcd=True)

    np.asarray(pts["env_pcd"].points).T

    print(pts.keys())
# %%