#%%
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import open3d as o3d
from gemsplat.scripts.utils.nerf_utils import NeRF
from gemsplat.scripts.utils.scene_editing_utils import get_centroid

# # # # #
# # # # # Config Path
# # # # #

rootdir = Path("/home/admin/StanfordMSL/")
SFTIpth = rootdir / "SFTI-Program"
outpth  = SFTIpth / "nerf_data/outputs/"
# scnpth  = outpth / "sv_701_nerfstudio/gemsplat"
# modelpth = scnpth / "2024-08-07_134148"
# scnpth  = outpth / "sv_701_nerfstudio_gemsplat/gemsplat"
# modelpth = scnpth / "2024-08-31_144408"
scnpth  = outpth / "sv_1007_gemsplat/gemsplat"
modelpth = scnpth / "2024-10-09_113200"
# scnpth  = outpth / "sv_917_3_left_gemsplat/gemsplat"
# modelpth = scnpth / "2024-09-26_114823"
with open("/home/admin/StanfordMSL/SFTI-Program/nerf_data/sv_1007_gemsplat/transforms.json", "r") as f:
# with open("/home/admin/StanfordMSL/SFTI-Program/nerf_data/sv_917_3_left_gemsplat/transforms.json", "r") as f:
        transforms_nerf = json.load(f)

# # mode
gaussian_splatting = True

if gaussian_splatting:
    # Gaussian Splatting
    config_path = Path(f"modelpth / config.yml")
else:
    # Nerfacto
    config_path = Path(f"<Enter the path to your config file.>")

# %%
 # rescale factor
res_factor = None

# option to enable visualization of the environment point cloud
enable_visualization_pcd = True

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
# env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True,bounding_box_max=(5.50,2.75,2.5),bounding_box_min=(-5.50,-2.75,0.0))

dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

transform = torch.eye(4)
transform[:3,:3] = dataparser_transform[:3,:3]
invtransform = torch.linalg.inv(transform)

env_pcd_scaled = np.asarray(env_pcd.points).T / dataparser_scale
env_pcd_scaled = np.vstack((env_pcd_scaled, np.ones((1, env_pcd_scaled.shape[1]))))

env_pcd_scaled = invtransform @ env_pcd_scaled
env_pcd_scaled = np.asarray(transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(env_pcd_scaled)

pcd_maxs = env_pcd_scaled.max(axis=1)
pcd_mins = env_pcd_scaled.min(axis=1)
pcd_ranges = pcd_maxs - pcd_mins
print(pcd_ranges)

env_pcd_scaled = env_pcd_scaled[:3, :].T
epcds = o3d.geometry.PointCloud()
epcds.points=o3d.utility.Vector3dVector(env_pcd_scaled)
epcds.colors=o3d.utility.Vector3dVector(np.asarray(env_pcd.colors))

#%%
if enable_visualization_pcd:
    # visualize point cloud
    # fig = o3d.visualization.draw_plotly([env_pcd])
    fig = o3d.visualization.draw_plotly([epcds])


# %%
# list of positives
# e.g., kitchen: ['babynurser bottle', 'red apple', 'kettle']
positives = 'computer'

# update list of negatives ['things', 'stuff', 'object', 'texture'] -> 'object, things, stuff, texture'
negatives = 'a'

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
threshold_mask = 0.4

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
# %%

# #  %%    
# # # # #
# # # # # Scene Editing from the Gaussian Means: Objects to Move
# # # # #  

# option to print debug info
print_debug_info: bool = True

# Set of all fruits
# fruits = {'fruit', 'apple', 'orange', 'pear', 'tomato'}

# objects to move
objects: List[str] = [positives]
# targets: List[str] = ['electric burner coil', 'cutting board']
# object_to_target: Dict[str, str] = {'saucepan': targets[0],
#                                     'glass lid': targets[0],
#                                     'knife': targets[1],
#                                     'orange': targets[1] 
#                                     }

# # table location
# table_centroid, table_z_bounds, scene_pcd, table_sim_mask, table_attr = get_centroid(nerf=nerf,
#                                                                                     env_pcd=env_pcd,
#                                                                                     pcd_attr=env_attr,
#                                                                                     positives='ground',
#                                                                                     threshold=0.7,
#                                                                                     enable_convex_hull=True,
#                                                                                     enable_spherical_filter=False,
#                                                                                     visualize_pcd=False)

# # table data
# table_pcd_points = np.asarray(scene_pcd.points)[table_sim_mask]
# table_pcd_colors = np.asarray(scene_pcd.colors)[table_sim_mask]
# table_pcd_sim = table_attr['raw_similarity'][table_sim_mask].cpu().numpy()

# # # # # #
# # # # # # Plane Fitting for the Table
# # # # # # 
# pcd_clus = o3d.geometry.PointCloud()
# pcd_clus.points = o3d.utility.Vector3dVector(table_pcd_points[:, :3])
# pcd_clus.colors = o3d.utility.Vector3dVector(table_pcd_colors)

# # plane
# plane_model, inliers = pcd_clus.segment_plane(distance_threshold=0.001,
#                                         ransac_n=3,
#                                         num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# # normal to the plane
# normal_plane = plane_model[:3]

# # threshold for the distance to the plane
# dist_plane_threshold = np.amax(np.abs(normal_plane @ table_pcd_points[inliers, :3].T + plane_model[-1]))

# if enable_visualization_pcd:
#     inlier_cloud = pcd_clus.select_by_index(inliers)
#     inlier_cloud.paint_uniform_color([1.0, 0, 0])
#     outlier_cloud = pcd_clus.select_by_index(inliers, invert=True)
#     fig = o3d.visualization.draw_plotly([inlier_cloud, outlier_cloud],
#                                        )
#     fig.show()

# # target location
# target_data = dict()

# # similarity threshold
# threshold_targ = [0.95, 0.9]

# # filter radius
# filter_radius_targ = 0.05

# # clearance for the z-component at the target place point (m)
# target_z_clearance = 0.007

# for idx, obj in enumerate(targets):
#     if idx != 1:
#         continue

#     # target
#     target_centroid, target_z_bounds, _, target_sim_mask, target_attr = get_centroid(nerf=nerf,
#                                                                                     env_pcd=env_pcd,
#                                                                                     pcd_attr=env_attr,
#                                                                                     positives=obj,
#                                                                                     threshold=threshold_targ[idx],
#                                                                                     filter_radius=filter_radius_targ,
#                                                                                     enable_convex_hull=True,
#                                                                                     enable_spherical_filter=True,
#                                                                                     visualize_pcd=False)

#     # target data
#     target_pcd_points = np.asarray(scene_pcd.points)[target_sim_mask]
#     target_pcd_colors = np.asarray(scene_pcd.colors)[target_sim_mask]
#     target_pcd_sim = target_attr['raw_similarity'][target_sim_mask].cpu().numpy()

#     # # # # #
#     # # # # # Desired Location for Placing the Objects
#     # # # # # 

#     pcd_clus = o3d.geometry.PointCloud()
#     pcd_clus.points = o3d.utility.Vector3dVector(target_pcd_points[:, :3])
#     pcd_clus.colors = o3d.utility.Vector3dVector(target_pcd_colors)

#     if enable_visualization_pcd:
#         o3d.visualization.draw_plotly([pcd_clus])

#     plane_model_target, inliers = pcd_clus.segment_plane(distance_threshold=0.001,
#                                                         ransac_n=3,
#                                                         num_iterations=1000)
#     # [a, b, c, d] = plane_model_obj
#     # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

#     inlier_cloud = pcd_clus.select_by_index(inliers)
#     inlier_cloud.paint_uniform_color([1.0, 0, 0])
#     outlier_cloud = pcd_clus.select_by_index(inliers, invert=True)

#     if enable_visualization_pcd:
#         fig = o3d.visualization.draw_plotly([inlier_cloud, outlier_cloud],
#                                             )
#         fig.show()
    
#     # place-task desired location
#     target_place_point = np.mean(target_pcd_points[inliers], axis=0)

#     # incorporate the clearance for the target z-point
#     target_place_point += (target_z_clearance * normal_plane)

#     # store the data
#     target_data[obj] = {'place_point': target_place_point}

# # # %%
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
filter_radius = [0.075, 0.05, 0.05, 0.05]
# filter_radius = [0.5, 0.15, 0.1, 0.05]
# TODO
filter_radius.extend([filter_radius[-1]] * (len(objects) - len(filter_radius)))

# similarity threshold - multiple objects
threshold_obj = [0.9, 0.9, 0.9, 0.97]
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

    # Scene - transformed
    spcd = np.asarray(scene_pcd.points).T / dataparser_scale
    spcd = np.vstack((spcd, np.ones((1, spcd.shape[1]))))

    spcd = invtransform @ spcd
    spcd = np.asarray(transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(spcd)

    pcd_maxs = spcd.max(axis=1)
    pcd_mins = spcd.min(axis=1)
    pcd_ranges = pcd_maxs - pcd_mins
    print(pcd_ranges)

    spcd = spcd[:3, :].T
    spcds = o3d.geometry.PointCloud()
    spcds.points=o3d.utility.Vector3dVector(spcd)

    # Object - transformed
    obpcd = object_pcd_points.T / dataparser_scale
    obpcd = np.vstack((obpcd, np.ones((1, obpcd.shape[1]))))

    obpcd = invtransform @ obpcd
    obpcd = np.asarray(transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(obpcd)
    obpcd_centroid = np.mean(obpcd[:3, :].T, axis=0)
    print(obpcd_centroid)
    obpcd_clus = o3d.geometry.PointCloud()
    obpcd_clus.points=o3d.utility.Vector3dVector(obpcd[:3, :].T)
    obpcd_clus.colors=o3d.utility.Vector3dVector(object_pcd_colors)

    pcd_maxs = obpcd.max(axis=1)
    pcd_mins = obpcd.min(axis=1)
    pcd_ranges = pcd_maxs - pcd_mins
    print(pcd_ranges)

    if enable_visualization_pcd:
        fig = o3d.visualization.draw_plotly([pcd_clus])
        fig = o3d.visualization.draw_plotly([spcds])
        fig = o3d.visualization.draw_plotly([obpcd_clus])
        # fig.show()
    ## %%

    # TODO: Refactor into a function

    # distance threshold
    # dist_threshold_plane_fitting = 0.008 if 'knife' in obj else 0.001 # (default)

    # plane_model_obj, inliers_obj = pcd_clus.segment_plane(distance_threshold=dist_threshold_plane_fitting,
    #                                                     ransac_n=3,
    #                                                     num_iterations=1000)
    # # [a, b, c, d] = plane_model_obj
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # inlier_cloud = pcd_clus.select_by_index(inliers_obj)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd_clus.select_by_index(inliers_obj, invert=True)

    # if enable_visualization_pcd or enable_minimal_visualization:
    #     print('*' * 50)
    #     print('Plane-Fitting: Stage 1')
    #     fig = o3d.visualization.draw_plotly([inlier_cloud, outlier_cloud],
    #                                        )
    #     fig.show()
    #     print('*' * 50)
        
    # if not 'lid' in obj and not 'knife' in obj:
    #     # # normal to the plane
    #     # normal_plane = plane_model[:3]

    #     # distance along the normal to the plane
    #     dist_plane = normal_plane @ object_pcd_points[:, :3].T + plane_model[-1]

    #     # threshold for the distance to the plane
    #     dist_plane_threshold_obj = dist_plane_threshold + (1e0 * dist_plane_threshold)
    # elif 'knife' in obj:
    #     # normal to the plane
    #     normal_plane_obj = plane_model_obj[:3]
        
    #     # distance along the normal to the plane
    #     dist_plane = normal_plane_obj @ object_pcd_points[:, :3].T + plane_model_obj[-1]

    #     # threshold for the distance to the plane
    #     dist_plane_threshold_obj = np.mean(np.abs(normal_plane_obj @ object_pcd_points[inliers_obj, :3].T + plane_model_obj[-1]))
    # elif 'lid' in obj:
    #     # # # # #
    #     # # # # # Scene Editing for Elevated Objects: Two-Stage Process
    #     # # # # # 
        
    #     # Stage 1
        
    #     # normal to the plane
    #     normal_plane_obj = plane_model_obj[:3]

    #     # distance along the normal to the plane
    #     dist_plane = normal_plane_obj @ object_pcd_points[:, :3].T + plane_model_obj[-1]

    #     # threshold for the distance to the plane
    #     dist_plane_threshold_obj = np.amax(np.abs(normal_plane_obj @ object_pcd_points[inliers_obj, :3].T + plane_model_obj[-1]))
    #     dist_plane_threshold_obj = dist_plane_threshold_obj + (1e-1 * dist_plane_threshold_obj)
        
    #     # filter the points
    #     pcd_mask = dist_plane > dist_plane_threshold_obj
    #     vis_pcd_pts = object_pcd_points[:, :3][pcd_mask]
    #     vis_colors = object_pcd_colors[pcd_mask]
        
    # # visualize the point cloud
    # # mask for the selected points
    # pcd_mask = dist_plane > dist_plane_threshold_obj
    # if 'knife' in obj:
    #     # include the inliers
    #     pcd_mask[inliers_obj] = True
        
    # # selected points with colors
    # vis_pcd_pts = object_pcd_points[:, :3][pcd_mask]
    # vis_colors = object_pcd_colors[pcd_mask]

    # # point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
    # pcd.colors = o3d.utility.Vector3dVector(vis_colors)

    # if enable_visualization_pcd:
    #     fig = o3d.visualization.draw_plotly([pcd])
    #     fig.show()
        
    # remove outliers
    pcd, inlier_ind = pcd_clus.remove_radius_outlier(nb_points=30, radius=0.03) # r=0.03 maybe nb_points=5
    # pcd, inlier_ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        
    if enable_visualization_pcd:
        print('*' * 50)
        print('Post-Outlier-Removal')
        fig = o3d.visualization.draw_plotly([pcd])
        # fig.show()
        print('*' * 50)

    # if 'knife' in obj:
    #     # compute the object's base
    #     obj_base = np.mean(pcd.points, axis=0)
        
    #     # modify the height of the base
    #     b_argmin = np.argmin(np.abs(normal_plane_obj @ object_pcd_points[inliers_obj, :3].T + plane_model_obj[-1]))
    #     obj_base[-1] = object_pcd_points[inliers_obj, -1][b_argmin]
        
    #     # object's width
    #     knife_dim = np.max(pcd.points, axis=0) - np.min(pcd.points, axis=0)
    # elif any(item in obj for item in fruits):
    #     # compute the object's base
    #     obj_base = np.mean(pcd.points, axis=0)
        
    #     # modify the height of the base
    #     base_pts = object_pcd_points[dist_plane < dist_plane_threshold_obj, :]
    #     obj_base[-1] = np.mean(base_pts[:, -1])

    # # # Utilize Convex hull
    # # if 'lid' in obj:
    # #     # points classified as being part of the object
    # #     pts_cond = np.asarray(pcd.points)
        
    # #     # compute the convex hull
    # #     convex_hull = ConvexHull(pts_cond)
        
    # #     # examine the convex hull
    # #     convex_hull_mask = in_convex_hull(np.asarray(scene_pcd.points), pts_cond[convex_hull.vertices])
        
    # #     if print_debug_info:
    # #         print(f'Convex Hull Proc. Before : {len(pts_cond)}, After: {len(convex_hull_mask.nonzero()[0])}')

    # #     # update the similarity mask
    # #     similarity_mask = np.logical_or(similarity_mask, convex_hull_mask)

    # new_obj_mask = np.copy(similarity_mask)
    # new_obj_mask[similarity_mask] = pcd_mask

    # # # incorporate the mask after outlier removal
    # out_rem_mask = np.zeros_like(new_obj_mask[new_obj_mask], dtype=bool)
    # out_rem_mask[inlier_ind] = True
    # new_obj_mask[new_obj_mask] = out_rem_mask

    # # composite mask
    # comp_mask = env_pcd_mask.clone()
    # comp_mask[comp_mask == True] = torch.tensor(new_obj_mask).to(device)

    # # print('*' * 50)
    # # print(f'Num. points: {comp_mask.count_nonzero()}')
    # # print('*' * 50)

    # # # mask for the selected points based on opacity
    # # opac_mask = comp_mask[env_pcd_mask]
    # # opac_mask[(env_attr['opacities'] < 0.1).squeeze()] = False
    # # comp_mask[env_pcd_mask] = opac_mask

    # # print('*' * 50)
    # # print(f'Num. points: {comp_mask.count_nonzero()}')
    # # print('*' * 50)

    # # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(nerf.pipeline.model.means[comp_mask].clone().detach().cpu().numpy())
    # # pcd.colors = o3d.utility.Vector3dVector(nerf.pipeline.model.features_dc[comp_mask].clone().detach().cpu().numpy())

    # # o3d.visualization.draw_plotly([pcd])

    # # update the local point-cloud mask
    # pcd_mask_update = np.zeros_like(pcd_mask[pcd_mask], dtype=bool)
    # pcd_mask_update[inlier_ind] = True
    # pcd_mask[pcd_mask] = pcd_mask_update


    # # pcd_mask = torch.logical_and(pcd_mask, env_attr['opacities'][comp_mask]  > 0.1)

    # # update the translation
    # object_pcd_points_upd = np.asarray(nerf.pipeline.model.means[comp_mask].clone().detach().cpu().numpy())
    # object_pcd_colors_upd = np.asarray(nerf.pipeline.model.features_dc[comp_mask].clone().detach().cpu().numpy())

    # if any(item in obj for item in ['pot', 'pan']):
    #     # using the plane fitted to the rim of the pot
    #     dist_plane_inliers = normal_plane @ object_pcd_points[inliers_obj, :3].T + plane_model[-1]

    #     # height of the pot along the normal to the table
    #     pot_height = np.mean(dist_plane_inliers) - dist_plane_threshold_obj

    # if any(item in obj for item in ['pot', 'pan', 'lid']):
    #     # base of the object
    #     pcd_mask_base = np.abs(dist_plane[pcd_mask] - dist_plane_threshold_obj) < 1e0 * dist_plane_threshold_obj
    #     vis_pcd_pts = object_pcd_points_upd.copy()
    #     vis_colors = object_pcd_colors_upd.copy()

    #     vis_colors[pcd_mask_base, :3] = [1, 0, 0]

    #     # point cloud
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
    #     pcd.colors = o3d.utility.Vector3dVector(vis_colors)

    #     if enable_visualization_pcd:
    #         fig = o3d.visualization.draw_plotly([pcd])
    #         fig.show()

    #     # point cloud for the base of the pot
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(object_pcd_points_upd[pcd_mask_base, :3])
    #     pcd.colors = o3d.utility.Vector3dVector(object_pcd_colors_upd[pcd_mask_base, :3])

    #     # remove outliers
    #     pcd, inlier_ind = pcd.remove_radius_outlier(nb_points=3, radius=0.01) # r=0.03

    #     if enable_visualization_pcd:
    #         o3d.visualization.draw_plotly([pcd])
            
    #     # the centroid the object's base
    #     obj_base = np.mean(pcd.points, axis=0)
        
    # # target place-point
    # target_place_point = target_data[object_to_target[obj]]['place_point']

    # if any(item in obj for item in ['pot', 'pan']):
    #     # translation
    #     translation = target_place_point - obj_base
    # elif 'lid' in obj:
    #     # translation
    #     translation = (target_place_point - obj_base) + pot_height * normal_plane
    # elif 'knife' in obj:
    #     # translation
    #     translation = target_place_point - obj_base
    # elif any(item in obj for item in fruits):
    #     # translation
    #     translation = target_place_point - obj_base
        
    #     try:
    #         # translation (plus a further translation given the width of the knife)
    #         translation += 0.5 * np.array([knife_dim[0], 0, 0])
    #     except NameError:
    #         pass
    # else:
    #     pass

    # # outputs
    # obj_outputs[obj] = {
    #     'centroid': src_centroid,
    #     'z_bounds': src_z_bounds,
    #     'pcd': scene_pcd,
    #     'similarity_mask': similarity_mask,
    #     'other_attr': other_attr,
    #     'translation': translation,
    #     'comp_mask': comp_mask
    # }
# %%