#%%
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import getSemanticPoints as gsp
import genHemisphere as gh
import torch
import json


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
#%%
# nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
if __name__ == '__main__':
#%%
    # # # # #
    # # # # # Paths
    # # # # #

    rootdir = Path("/home/admin/StanfordMSL/")
    SFTIpth = rootdir / "SFTI-Program"
    outpth  = SFTIpth / "nerf_data/outputs/"
    scnpth  = outpth / "sv_917_3_left_gemsplat/gemsplat"
    modelpth = scnpth / "2024-09-26_114823"

    with open("/home/admin/StanfordMSL/SFTI-Program/nerf_data/sv_917_3_left_gemsplat/transforms.json", "r") as f:
        transforms_nerf = json.load(f)


    semantic_query = "microwave"
    semantic_negatives = "a"
    
    threshold = 0.6 #Currently may be overloaded within get_points()
    filter_radius = 0.075 #Currently may be overloaded within get_points()
    nerf, pts = gsp.get_points(modelpth, semantic_query, semantic_negatives, threshold, filter_radius)


    # hemisphere_radius = [1.0, 1.0, 1.0] #TODO how to get a height that makes sense for the drone
    hemisphere_radius = [0.1, 0.1, 0.1]
    theta_intervals = [-np.pi / 2, -np.pi, -3*np.pi/2, 0.00] #TODO (azimuth) get this dynamically from the drone?
    phi_intervals = [80*np.pi/180] #elevation
    exclusion_radius = 0.03 #Determines how close the camera can get to any point in the environment
    pcdarr = np.asarray(pts["env_pcd"].points).T
    print(pcdarr.shape)
    print(pcdarr[:3, :].shape)
    pose_targets = gh.get_hemisphere(nerf, np.asarray(pts["env_pcd"].points).T, pts["src_centroid"], hemisphere_radius, theta_intervals, phi_intervals, exclusion_radius)
    print(pose_targets)


    #%% - Transform the poses to Mocap world frame
    # dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    # dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
    # pcd = np.asarray(pts["env_pcd"].points)
    # pcd = pcd/dataparser_scale
    # transform = torch.eye(4)
    # transform[:3,:3] = dataparser_transform[:3,:3]
    # invtransform = torch.linalg.inv(transform)
    # pcd = np.hstack((pcd,np.ones((pcd.shape[0],1))))
    # # pcd = np.dot(invtransform,pcd)
    # pcd = invtransform@pcd.T
    # pcd = np.asarray(transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])@np.asarray(pcd)

    # pcd_maxs = pcd.max(axis=1)
    # pcd_mins = pcd.min(axis=1)
    # pcd_ranges = pcd_maxs - pcd_mins
    # print(pcd_ranges)

    # obpcd = pts["object_pcd_points"]
    # obpcd = obpcd.T / dataparser_scale
    # obpcd = np.vstack((obpcd, np.ones((1, obpcd.shape[1]))))

    # obpcd = invtransform @ obpcd
    # obpcd = np.asarray(transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(obpcd)
    # obpcd_centroid = np.mean(obpcd[:3, :].T, axis=0)
    # print(obpcd_centroid)

    # # obpcd_clus = o3d.geometry.PointCloud()
    # # obpcd_clus.points=o3d.utility.Vector3dVector(obpcd[:3, :].T)
    # # obpcd_clus.colors=o3d.utility.Vector3dVector(object_pcd_colors)
    # # target_maxs = target_look_at_point.max(axis=1)
    # # target_mins = target_look_at_point.min(axis=1)
    # # target_ranges = target_maxs - target_mins
    # # print(target_ranges)