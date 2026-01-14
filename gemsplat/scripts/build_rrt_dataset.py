import numpy as np
import json
import os
from pathlib import Path
from typing import Dict,Union,Tuple,List,Literal
from tqdm.notebook import trange

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from controller.vr_mpc import VehicleRateMPC
from controller.pilot import Pilot
import dynamics.quadcopter_config as qc
import dynamics.quadcopter_simulate as qs
from sympy import O
import synthesize.trajectory_helper as th
import synthesize.nerf_utils as nf
import visualize.plot_synthesize as ps
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from PIL import Image
from io import BytesIO

from gemsplat.scripts.utils.nerf_utils import NeRF
from gemsplat.scripts.utils.scene_editing_utils import get_centroid
import gemsplat.scripts.getSemanticPoints as gsp
import gemsplat.scripts.genHemisphere as gh
import open3d as o3d
from open3d.visualization import O3DVisualizer

from synthesize.solvers import (
    min_snap as ms,
)

from rrt_datagen_v9 import *

#%% First load the gemsplat and filter for the objective(s)
def get_objectives(nerf:nf.NeRF, objectives):
    viz=False
    transform = torch.eye(4)
    dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
    transform[:3,:3] = dataparser_transform[:3,:3]
    invtransform = torch.linalg.inv(transform)

    epcds, epcds_arr, epcds_bounds, pcd, pcd_mask, pcd_attr = gsp.rescale_point_cloud(nerf)

    # Translate to gemsplat syntax
    positives = objectives
    threshold = 0.6 #Currently may be overloaded within get_points()
    filter_radius = 0.075 #Currently may be overloaded within get_points()
    filter_radius = [filter_radius] * len(objectives)

    threshold_obj = [threshold] * len(objectives)
    # print(pcd.keys())
    # env_points = np.asarray(pcd["env_pcd"].points).T
    
    obj_targets = []

    for idx, obj in enumerate(objectives):

        print('*' * 50)
        print(f'Processing Object: {obj}')
        print('*' * 50)

        # source location
        src_centroid, src_z_bounds, scene_pcd, similarity_mask, other_attr = get_centroid(nerf=nerf,
                                                                                        env_pcd=pcd,
                                                                                        pcd_attr=pcd_attr,
                                                                                        positives=objectives[idx],
                                                                                        negatives='window,wall,floor,ceiling,object, things, stuff, texture',
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

        if viz:
            o3d.visualization.draw_plotly([pcd_clus])
        # # remove outliers
        # pcd_clus, _ = pcd_clus.remove_radius_outlier(nb_points=30, radius=0.03) # r=0.03 maybe nb_points=5

        obj_targets.append(src_centroid)
        print(f"src_centroid: {src_centroid}")

    #NOTE: This is used for camera sphere
    # hemisphere_radius = [0.1, 0.1, 0.1]
    # theta_intervals = [-np.pi / 2, -np.pi, -3*np.pi/2, 0.00] #TODO (azimuth) get this dynamically from the drone?
    # phi_intervals = [80*np.pi/180] #elevation
    # exclusion_radius = 0.03 #Determines how close the camera can get to any point in the environment
    # pose_targets = gh.get_hemisphere(nerf, np.asarray(pts["env_pcd"].points).T, pts["src_centroid"], hemisphere_radius, theta_intervals, phi_intervals, exclusion_radius)
    # print(pose_targets)

    for i in range(len(obj_targets)):
        obj_targets[i] = obj_targets[i].reshape(3, -1)
        # print(f"obj_targets shape: {obj_targets[i].shape}")

        obj_targets[i] = obj_targets[i] / dataparser_scale
        obj_targets[i] = np.vstack((obj_targets[i], np.ones((1, obj_targets[i].shape[1]))))
        # print(f"obj_targets shape: {obj_targets[i].shape}")

        obj_targets[i] = invtransform @ obj_targets[i]
        # print(f"obj_targets shape: {obj_targets[i].shape}")

        obj_targets[i] = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(obj_targets[i])

        obj_targets[i] = obj_targets[i][:3, :].T

    return obj_targets, epcds_bounds, epcds, epcds_arr

#%% Second generate RRT* paths through the environment

def generate_rrt_paths(config_file, pcd, pcd_arr, objectives:List[str], obj_targets, env_bounds, viz=True):

    def get_config_option(option_name, prompt, valid_options=None, default=None):
        if option_name in config:
            value = config[option_name]
            print(f"{option_name} set to {value} from config.yml")
            if valid_options and value not in valid_options:
                print(f"Invalid value for {option_name} in config.yml. Using default or prompting.")
                value = None
        else:
            value = None

        if value is None:
            value = input(prompt).strip()
            if valid_options:
                while value not in valid_options:
                    print(f"Invalid input. Valid options are: {', '.join(valid_options)}")
                    value = input(prompt).strip()
            if default and not value:
                value = default
        return value

    config = {}
    # Check if the configuration file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) or {}
        print("Configuration loaded from config.yml")
    else:
        print("Configuration file not found. Proceeding with interactive inputs.")

    # Ask the user to select the algorithm
    algorithm_input = get_config_option(
        'algorithm',
        "Select algorithm (RRT/RRT*): ",
        valid_options=['RRT', 'RRT*'],
        default='RRT'
    ).upper()

    # Select dimension: 1, 2, or 3
    dimension = get_config_option(
        'dimension',
        "Enter the dimension (2 or 3): ",
        valid_options=['2', '3'],
        default='2'
    )
    dimension = int(dimension)

    # Choose whether to prevent edge overlap
    prevent_edge_overlap_input = get_config_option(
        'prevent_edge_overlap',
        "Prevent edge overlap? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    prevent_edge_overlap = prevent_edge_overlap_input.lower() == 'y'

    # Choose whether to use exact edge lengths
    exact_step_input = get_config_option(
        'exact_step',
        "Exact edge lengths? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    exact_step = exact_step_input.lower() == 'y'

    bounded_step = False
    use_branch_pruning = False
    if not exact_step:
        # Choose whether to bound edge lengths
        bounded_step_input = get_config_option(
            'bounded_step',
            "Bound edge lengths? (y/n): ",
            valid_options=['y', 'n'],
            default='y'
        )
        bounded_step = bounded_step_input.lower() == 'y'

    # Initialize RRT
    #NOTE Currently only does 2D, have hardcoded the envbounds
    # ebounds = (env_bounds["minbound"][:2], env_bounds["maxbound"][:2])
    # ebounds = [tuple(env_bounds["minbound"][:2]), tuple(env_bounds["maxbound"][:2])]
    ebounds = [(env_bounds["minbound"][0], env_bounds["maxbound"][0]), (env_bounds["minbound"][1], env_bounds["maxbound"][1])]
    print(f"env_bounds: {ebounds}")

    trajset = {}
    for target, pose in zip(objectives, obj_targets):
        pose = pose.flatten()
        # print(f"shape of pose: {pose.shape}")
        print(f"target: {target}")
        print(f"pose: {pose}")
        # print(f"shape of start: {start.shape}")
        rrt = RRT(
            env_arr=pcd_arr,
            env_pts=pcd,
            start=pose[:2],
            bounds=ebounds,
            altitude=pose[2],
            dimension=dimension,
            step_size=1.0,
            collision_check_resolution=0.1,
            max_iter=2000,
            exact_step=exact_step,
            bounded_step=bounded_step,
            algorithm=algorithm_input,  # Pass the selected algorithm
            prevent_edge_overlap=prevent_edge_overlap
        )

        # Build RRT
        rrt.build_rrt()

        rrt.visualize(show_sampled_points=True)
        
        # Get all leaf nodes in the tree
        leaf_nodes = [node for node in rrt.nodes if not node.children]

        # Extract paths from each leaf node
        paths = []
        for leaf_node in leaf_nodes:
            path = rrt.get_path_from_leaf_to_root(leaf_node)
            paths.append(path)
        
        trajset[target] = paths
        
        # if viz:
        #     # Generate colors using a colormap
        #     num_trajectories = len(paths)
        #     cmap = plt.get_cmap("tab20")  # Choose a colormap with many distinct colors

        #     # Generate a color for each trajectory
        #     colors = [cmap(i % cmap.N)[:3] for i in range(num_trajectories)]  # RGB tuples

        #     # Extract x and y coordinates
        #     linsets = []
        #     for idx, traj in enumerate(paths):
        #         color = colors[idx]
                
        #         line_set = create_line_set(traj, color=color)
        #         linsets.append(line_set)
            
        #         # x_coords = [position[0] for position in path]
        #         # y_coords = [position[1] for position in path]
        #         # z_coords = pose[2] * np.ones_like(x_coords)
        #         # xyz_coords = np.vstack((x_coords, y_coords, z_coords)).T

        #         # lines = [[i, i + 1] for i in range(len(xyz_coords) - 1)]
        #         # colors = [[1, 0, 0] for _ in lines]  # Red color

        #         # trajectory_line_set = o3d.geometry.LineSet(
        #         #     points=o3d.utility.Vector3dVector(xyz_coords),
        #         #     lines=o3d.utility.Vector2iVector(lines)
        #         # )
        #         # trajectory_line_set.colors = o3d.utility.Vector3dVector(colors)

        #     # Combine all geometries
        #     geometries = [pcd] + linsets

        #     # Visualize
        #     o3d.visualization.draw_geometries(
        #         geometries,
        #         window_name="Point Cloud with RRT* Trajectories",
        #         width=800,
        #         height=600
        #     )

    return trajset

def create_line_set(trajectory, color=[1, 0, 0]):
        """
        Creates an Open3D LineSet from a trajectory.

        Args:
            trajectory (np.ndarray): Array of shape (N, 3).
            color (list): RGB color for the trajectory.

        Returns:
            o3d.geometry.LineSet: The LineSet object.
        """
        # Create lines connecting consecutive points
        lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
        colors = [color for _ in lines]  # Assign the same color to all lines in the trajectory

        # Create the LineSet object
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

#%% Fourth find 3D waypoints

#%% Fifth generate the rollouts - this function is kinda like "main" for this part of the pipeline
def generate_rollout_data(cohort:str,objectives:List[str],drone:str,method:str,
                          nerf:nf.NeRF,
                          Nro_tp:int,
                          Nro_sv:int=100,
                          Ntp_sc:int=10):
    """ This is designed to work like generate_rollout_data() from the SFTI program
    Args:
            cohort:         Name of the cohort.
            objectives:        Names of courses. NOTE: THIS WILL BE DIFFERENT HERE BECAUSE THERE now only OBJECTIVES
            drone:          Name of the drone.
            method:         Name of the method.
            nerf:           NeRF model.
            Nro_tp:         Number of rollouts per time point.
            Nro_sv:         Number of rollouts per save.
            Ntp_sc:         Number of time points per second.
            batch_size:     Number of rollouts to generate in a single batch.
            plot_sample_size:  Number of sample flights to plot per course.

        Returns:
            None:           (flight data saved to cohort directory)
        """
    # Some useful path(s)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Load drone and method configs
    cohort_path = os.path.join(workspace_path,"cohorts",cohort)
    drone_path  = os.path.join(workspace_path,"configs","drones",drone+".json")
    method_path = os.path.join(workspace_path,"configs","methods",method+".json")

    with open(drone_path) as json_file:
        drone_config = json.load(json_file)
    with open(method_path) as json_file:
        method_config = json.load(json_file)
        sample_set_config = method_config["sample_set"]
        trajectory_set_config = method_config["trajectory_set"]
        drone_set_config = method_config["drone_set"]
        
    # Create cohort directory (if it does not exist)
    if not os.path.exists(cohort_path):
        os.makedirs(cohort_path)

    # Generate base drone configuration
    base_drone_config = qc.generate_preset_config(drone_config)

    # Print some useful information
    print("==========================================================================")
    print("Cohort :",cohort)
    print("Method :",method)
    print("Drone  :",drone)
    print("Courses:",objectives)

    # Generate rollouts for each objective 
    #NOTE: This is going to look for an RRT* tree of waypoints to solve with ACADOS
    for objective in objectives:
        # Load objectives_config
        objective_path = os.path.join(workspace_path,"configs","objective",objective+".json")

        with open(objective_path) as json_file:
            course_config = json.load(json_file)

        # Generate ideal trajectory
        #NOTE We're going to solve this with ACADOS
        # Tpi,CPi = ms.solve(course_config)

        # Generate Sample Set Batches
        Ntp = Ntp_sc*int(Tpi[-1])                                       # Number of time points per trajectory
        Nsp = Nro_tp*Ntp                                                # Number of sample points (total)
        
        Tsp = np.tile(np.linspace(Tpi[0],Tpi[-1],Ntp+1)[:-1],Nro_tp)    # Entire sample points array
        Tsp += np.random.uniform(-1/Ntp_sc,1/Ntp_sc,Nsp)                # Add some noise to the sample points array
        Tsp = np.clip(Tsp,Tpi[0],Tpi[-1])                               # Clip the sample points array
        np.random.shuffle(Tsp)                                          # Shuffle the sample points array

        TTsp = np.split(Tsp,np.arange(Nro_sv,Nsp,Nro_sv))               # Split the sample points array into their batches
        
        # Print some diagnostics
        Ndc = int(sample_set_config["rollout_duration"]*sample_set_config["simulation"]["hz_ctl"])

        print("--------------------------------------------------------------------------")
        print("Course Name :",course)
        print("Rollout Reps:",Nro_tp,"(per time point)")
        print("Rollout Rate:",Ntp_sc,"(per second)")
        print("Rollout Data:",Ndc,"(per sample)")
        print("Sample Size :",Nsp,"(total)")
        print("Batch Sizes :", len(TTsp)-1, "x", Nro_sv,"+ 1 x", len(TTsp[-1]),"(samples)")

        # Generate Sample Set Batches
        Ndata = 0
        for idx in trange(len(TTsp)):
            # Get the current batch
            Tsp = TTsp[idx]
            
            # Generate sample drones
            Drones = generate_drones(len(Tsp),drone_set_config,base_drone_config)

            # Generate sample perturbations
            Perturbations  = generate_perturbations(Tsp,trajectory_set_config,Tpi,CPi)

            # Generate rollout data
            Trajectories,Images = generate_rollouts(course_config,sample_set_config,nerf,Drones,Perturbations)

            # Save the rollout data
            save_rollouts(cohort,course,Trajectories,Images,idx)

            # Update the data count
            Ndata += sum([trajectory["Ndata"] for trajectory in Trajectories])

        # Print some diagnostics
        print("--------------------------------------------------------------------------")
        print("Generated ",Ndata," points of data.")
        print("--------------------------------------------------------------------------")

def generate_rollouts(
        course_config:Dict[str,Dict[str,Union[float,np.ndarray]]],
        sample_set_config:Dict[str,Union[int,bool]],
        nerf:nf.NeRF,
        Drones:Dict[str,Union[np.ndarray,str,int,float]],
        Perturbations:Dict[str,Union[float,np.ndarray]]
        ) -> Tuple[List[Dict[str,Union[np.ndarray,np.ndarray,np.ndarray]]],List[torch.Tensor]]:
    """
    Generates rollout data for the quadcopter given a list of drones and initial states (perturbations).
    The rollout comprises trajectory data and image data. The trajectory data is generated by running
    the MPC controller on the quadcopter for a fixed number of steps. The trajectory data consists of
    time, states [p,v,q], body rate inputs [fn,w], objective state, data count, solver timings, advisor
    data, rollout id, and course name. The image data is generated by rendering the quadcopter at each
    state in the trajectory data. The image data consists of the image data and the data count.

    Args:
        course_config:          Course config dictionary.
        sample_set_config:      Sample set config dictionary.
        nerf:                   NeRF model.
        Drones:          List of drone configurations.
        Perturbations:          List of perturbed initial states.

    Returns:
        Trajectories:           List of trajectory rollouts.
        Images:                 List of image rollouts.
    """

    # Unpack sample set config
    mu_md = np.array(sample_set_config["model_noise"]["mean"])
    std_md = np.array(sample_set_config["model_noise"]["std"])
    mu_sn = np.array(sample_set_config["sensor_noise"]["mean"])
    std_sn = np.array(sample_set_config["sensor_noise"]["std"])
    hz_ctl = sample_set_config["simulation"]["hz_ctl"]
    hz_sim = sample_set_config["simulation"]["hz_sim"]
    t_dly = sample_set_config["simulation"]["delay"]
    dt_ro = sample_set_config["rollout_duration"]
    
    # Unpack the trajectory
    Tpi,CPi = ms.solve(course_config)
    
    # Initialize rollout variables
    Trajectories,Images = [],[]

    # Rollout the trajectories
    for idx,(drone,perturbation) in enumerate(zip(Drones,Perturbations)):
        # Unpack rollout variables
        t0,x0 = perturbation["t0"],perturbation["x0"]
        obj = th.ts_to_obj(Tpi,CPi)
        tf = t0 + dt_ro

        # Some useful intermediate variables
        policy = VehicleRateMPC(course_config,drone,hz_ctl)
        simulator = policy.generate_simulator(hz_sim)

        # Simulate the flight
        Tro,Xro,Uro,Imgs,Tsol,Adv = qs.simulate_flight(policy,simulator,
                                                       t0,tf,x0,obj,nerf,hz_sim,
                                                       mu_md=mu_md,std_md=std_md,
                                                       mu_sn=mu_sn,std_sn=std_sn,
                                                       t_dly=t_dly)
        
        # TODO: either make Xid from Tpi,CPi or maybe it's ok to leave it blank?
        trajectory = {
            "Tro":Tro,"Xro":Xro,"Uro":Uro,
            "Xid":None,"obj":obj,"Ndata":Uro.shape[1],"Tsol":Tsol,"Adv":Adv,
            "rollout_id":str(idx).zfill(5),
            "course":course_config["name"],
            "drone":drone}

        images = {
            "images":Imgs,
            "rollout_id":str(idx).zfill(5),"course":course_config["name"]
        }

        # Store rollout data
        Trajectories.append(trajectory)
        Images.append(images)

        # Delete the generated code
        policy.clear_generated_code()

    return Trajectories,Images

def parameterize_trajectories(branches, constant_velocity, sampling_frequency):
    """
    Generates trajectories for multiple branches, each with constant velocity and specified sampling frequency.

    Args:
        branches (list of lists): A list where each element is a branch represented as a list of xyz positions.
        constant_velocity (float): The constant speed to travel along each branch.
        sampling_frequency (float): The sampling frequency in samples per second.

    Returns:
        trajectories (list of numpy.ndarray): A list where each element is a trajectory array for a branch.
                                              Each trajectory array has the shape (18, num_samples), where each
                                              column vector specifies: 
                                              [dt, x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz, u1, u2, u3, u4]
    """
    def plot_trajectories_xy(trajectories):
        plt.figure(figsize=(8, 6))
        
        for i, trajectory in enumerate(trajectories):
            x = trajectory[1, :]  # X positions
            y = trajectory[2, :]  # Y positions
            plt.plot(x, y, label=f'Branch {i}')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectories in XY Plane')
        # plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Ensure equal scaling on both axes
        plt.show()

    viz=True

    trajectories = []

    for branch_index, positions in enumerate(branches):
        # Ensure positions are in numpy array format
        positions = np.array(positions)
        
        # Skip empty branches
        if positions.shape[0] == 0:
            print(f"Branch {branch_index} is empty. Skipping.")
            continue
        
        # Check if positions are in 2D; if not, pad with zeros for z-coordinate
        if positions.shape[1] == 2:
            positions = np.hstack((positions, np.zeros((positions.shape[0], 1))))
        elif positions.shape[1] != 3:
            raise ValueError(f"Positions in branch {branch_index} must have shape (n, 2) or (n, 3).")
        
        # Skip branches with less than two positions
        if positions.shape[0] < 2:
            print(f"Branch {branch_index} has less than two nodes. Skipping.")
            continue

        # Compute differences between consecutive positions
        diffs = np.diff(positions, axis=0)  # Shape: (n-1, 3)
        
        # Compute the distances between consecutive positions
        segment_lengths = np.linalg.norm(diffs, axis=1)  # Shape: (n-1,)
        
        # Compute cumulative distances along the path
        cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)  # Shape: (n,)
        
        # Total distance
        total_distance = cumulative_distances[-1]
        
        # Handle zero-length branches
        if total_distance == 0:
            print(f"Branch {branch_index} has zero total length. Skipping.")
            continue
        
        # Total time to traverse the path at constant speed
        total_time = total_distance / constant_velocity
        
        # Compute cumulative times at each position along the path
        cumulative_times = cumulative_distances / constant_velocity  # Shape: (n,)
        
        # Total number of samples
        num_samples = int(np.ceil(total_time * sampling_frequency))
        
        # Generate array of sample times
        sample_times = np.linspace(0, total_time, num_samples)  # Shape: (num_samples,)
        
        # Set up interpolation functions for x, y, z positions
        interp_func_x = interp1d(cumulative_times, positions[:, 0], kind='linear')
        interp_func_y = interp1d(cumulative_times, positions[:, 1], kind='linear')
        interp_func_z = interp1d(cumulative_times, positions[:, 2], kind='linear')
        
        # Interpolate positions at sample times
        x_samples = interp_func_x(sample_times)
        y_samples = interp_func_y(sample_times)
        z_samples = interp_func_z(sample_times)
        
        # Assemble positions
        positions_samples = np.vstack((x_samples, y_samples, z_samples)).T  # Shape: (num_samples, 3)
        
        # Compute velocities
        dt = 1.0 / sampling_frequency  # Time difference between samples
        positions_diff = np.diff(positions_samples, axis=0)  # Shape: (num_samples - 1, 3)
        segment_lengths_samples = np.linalg.norm(positions_diff, axis=1)  # Shape: (num_samples - 1,)
        
        # Avoid division by zero in case of zero-length segments
        segment_lengths_samples[segment_lengths_samples == 0] = 1e-8
        
        # Compute unit direction vectors
        directions = positions_diff / segment_lengths_samples[:, np.newaxis]  # Shape: (num_samples - 1, 3)
        
        # Multiply by constant speed to get velocities
        velocities = directions * constant_velocity  # Shape: (num_samples - 1, 3)
        
        # Append the last velocity to match the number of samples
        last_velocity = velocities[-1]
        velocities = np.vstack((velocities, last_velocity[np.newaxis, :]))  # Shape: (num_samples, 3)
        
        # Compute orientation (quaternion) based on velocity direction
        quaternions = []
        for velocity in velocities:
            if np.linalg.norm(velocity) == 0:
                # If the velocity vector is zero, set a default orientation
                quaternion = R.from_euler('z', 0).as_quat()
            else:
                # Align quaternion with velocity vector direction
                quaternion = R.from_rotvec(np.arctan2(velocity[1], velocity[0]) * np.array([0, 0, 1])).as_quat()
            quaternions.append(quaternion)
        quaternions = np.array(quaternions)  # Shape: (num_samples, 4)

        # Set angular rates and motor commands to zero
        angular_rates = np.zeros((num_samples, 3))  # wx, wy, wz
        motor_commands = np.zeros((num_samples, 4))  # u1, u2, u3, u4
        
        # Assemble the trajectory array
        trajectory = np.vstack((
            np.full(num_samples, dt),             # dt
            x_samples, y_samples, z_samples,      # x, y, z
            velocities[:, 0], velocities[:, 1], velocities[:, 2],  # vx, vy, vz
            quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3],  # qx, qy, qz, qw
            angular_rates[:, 0], angular_rates[:, 1], angular_rates[:, 2],  # wx, wy, wz
            motor_commands[:, 0], motor_commands[:, 1], motor_commands[:, 2], motor_commands[:, 3]  # u1, u2, u3, u4
        ))  # Shape: (18, num_samples)
        
        trajectories.append(trajectory)
    
    if viz is True:
        plot_trajectories_xy(trajectories)
    
    return trajectories

# def parameterize_trajectories(branches, constant_velocity, sampling_frequency):
#     """
#     Generates trajectories for multiple branches, each with constant velocity and specified sampling frequency.

#     Args:
#         branches (list of lists): A list where each element is a branch represented as a list of xyz positions.
#         constant_velocity (float): The constant speed to travel along each branch.
#         sampling_frequency (float): The sampling frequency in samples per second.

#     Returns:
#         trajectories (list of numpy.ndarray): A list where each element is a trajectory array for a branch.
#                                               Each trajectory array has the shape (8, num_samples), where each
#                                               column vector specifies: [t, x, y, z, vx, vy, vz, yaw]
#     """
#     # Assuming 'trajectories' is the list of trajectory arrays returned by parameterize_trajectories
#     def plot_trajectories_xy(trajectories):
#         plt.figure(figsize=(8, 6))
        
#         for i, trajectory in enumerate(trajectories):
#             x = trajectory[1, :]  # X positions
#             y = trajectory[2, :]  # Y positions
#             plt.plot(x, y, label=f'Branch {i}')
        
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.title('Trajectories in XY Plane')
#         # plt.legend()
#         plt.grid(True)
#         plt.axis('equal')  # Ensure equal scaling on both axes
#         plt.show()

#     viz=True
#     trajectories = []

#     for branch_index, positions in enumerate(branches):
#         # Ensure positions are in numpy array format
#         positions = np.array(positions)
        
#         # Skip empty branches
#         if positions.shape[0] == 0:
#             print(f"Branch {branch_index} is empty. Skipping.")
#             continue
        
#         # Check if positions are in 2D; if not, pad with zeros for z-coordinate
#         if positions.shape[1] == 2:
#             positions = np.hstack((positions, np.zeros((positions.shape[0], 1))))
#         elif positions.shape[1] != 3:
#             raise ValueError(f"Positions in branch {branch_index} must have shape (n, 2) or (n, 3).")
        
#         # Skip branches with less than two positions
#         if positions.shape[0] < 2:
#             print(f"Branch {branch_index} has less than two nodes. Skipping.")
#             continue

#         # Compute differences between consecutive positions
#         diffs = np.diff(positions, axis=0)  # Shape: (n-1, 3)
        
#         # Compute the distances between consecutive positions
#         segment_lengths = np.linalg.norm(diffs, axis=1)  # Shape: (n-1,)
        
#         # Compute cumulative distances along the path
#         cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)  # Shape: (n,)
        
#         # Total distance
#         total_distance = cumulative_distances[-1]
        
#         # Handle zero-length branches
#         if total_distance == 0:
#             print(f"Branch {branch_index} has zero total length. Skipping.")
#             continue
        
#         # Total time to traverse the path at constant speed
#         total_time = total_distance / constant_velocity
        
#         # Compute cumulative times at each position along the path
#         cumulative_times = cumulative_distances / constant_velocity  # Shape: (n,)
        
#         # Total number of samples
#         num_samples = int(np.ceil(total_time * sampling_frequency))
        
#         # Generate array of sample times
#         sample_times = np.linspace(0, total_time, num_samples)  # Shape: (num_samples,)
        
#         # Set up interpolation functions for x, y, z positions
#         interp_func_x = interp1d(cumulative_times, positions[:, 0], kind='linear')
#         interp_func_y = interp1d(cumulative_times, positions[:, 1], kind='linear')
#         interp_func_z = interp1d(cumulative_times, positions[:, 2], kind='linear')
        
#         # Interpolate positions at sample times
#         x_samples = interp_func_x(sample_times)
#         y_samples = interp_func_y(sample_times)
#         z_samples = interp_func_z(sample_times)
        
#         # Assemble positions
#         positions_samples = np.vstack((x_samples, y_samples, z_samples)).T  # Shape: (num_samples, 3)
        
#         # Compute velocities
#         dt = 1.0 / sampling_frequency  # Time difference between samples
#         positions_diff = np.diff(positions_samples, axis=0)  # Shape: (num_samples - 1, 3)
#         segment_lengths_samples = np.linalg.norm(positions_diff, axis=1)  # Shape: (num_samples - 1,)
        
#         # Avoid division by zero in case of zero-length segments
#         segment_lengths_samples[segment_lengths_samples == 0] = 1e-8
        
#         # Compute unit direction vectors
#         directions = positions_diff / segment_lengths_samples[:, np.newaxis]  # Shape: (num_samples - 1, 3)
        
#         # Multiply by constant speed to get velocities
#         velocities = directions * constant_velocity  # Shape: (num_samples - 1, 3)
        
#         # Append the last velocity to match the number of samples
#         last_velocity = velocities[-1]
#         velocities = np.vstack((velocities, last_velocity[np.newaxis, :]))  # Shape: (num_samples, 3)
        
#         # Compute yaw angles
#         vel_xy = velocities[:, :2]  # Shape: (num_samples, 2)
#         yaw_angles = np.arctan2(vel_xy[:, 1], vel_xy[:, 0])  # Shape: (num_samples,)
        
#         # Assemble the trajectory array
#         trajectory = np.vstack((
#             sample_times,
#             positions_samples.T,
#             velocities.T,
#             yaw_angles
#         ))  # Shape: (8, num_samples)
        
#         trajectories.append(trajectory)

#     if viz is True:
#         plot_trajectories_xy(trajectories)
    
#     return trajectories

#%%
if __name__ == '__main__':
    #%%
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    # modelpth = Path("/home/admin/StanfordMSL/SFTI-Program/nerf_data/outputs/sv_1018_2_gemsplat/gemsplat/2024-10-19_204236")
    # modelpth = Path("/home/admin/StanfordMSL/SFTI-Program/nerf_data/outputs/sv_1007_gemsplat/gemsplat/2024-10-09_113200")
    modelpth = Path("/home/admin/StanfordMSL/SFTI-Program/nerf_data/outputs/sv_917_3_left_gemsplat/gemsplat/2024-09-26_114823")

    # Initialize an empty dictionary for config
    config_file = '/home/admin/StanfordMSL/gemsplat/gemsplat/scripts/rrt3config.yml'
    
    # initialize NeRF
    nerf = NeRF(config_path=modelpth / "config.yml",
                test_mode="inference", #"inference", "eval"
                dataset_mode="test",
                device=device)

    # camera intrinsics
    H, W, K = nerf.get_camera_intrinsics()
    K = K.to(device)

    # Set objectives and find their coordinates
    objectives = ["microwave"]

    obj_targets, env_bounds, epcds, epcds_arr = get_objectives(nerf, objectives)
    print(epcds_arr.shape)
    print(obj_targets)

    #%% Generate RRT* Paths
    # start = np.array([-4.0, 2.0])
    trajset = generate_rrt_paths(config_file, epcds, epcds_arr, objectives, obj_targets, env_bounds)
    print(trajset.keys())

    #NOTE: trajset['microwave'][X] is a specific trajectory paramterized by its nodes

    #%%
    trajectories = parameterize_trajectories(trajset['microwave'], 1.2, 4)
    # print(traj1.shape)

    #%%
    viz = True
    j = 0
    if viz:
        for key, obj_trajs in trajset.items():

            num_trajectories = len(obj_trajs)
            cmap = plt.get_cmap("tab20")  # Choose a colormap with many distinct colors

            # Generate a color for each trajectory
            colors = [cmap(i % cmap.N)[:3] for i in range(num_trajectories)]  # RGB tuples

            # print(f"obj_targets: {obj_targets[:1][0].flatten()}")
            objt = obj_targets[:][0].flatten()
            # print(f"objt: {objt[2]}")

            linsets = []
            for idx, traj in enumerate(obj_trajs):

                x_coords = [position[0] for position in traj]
                y_coords = [position[1] for position in traj]
                z_coords = objt[2] * np.ones_like(x_coords)
                xyz_coords = np.vstack((x_coords, y_coords, z_coords)).T

                color = colors[idx]
                    
                line_set = create_line_set(xyz_coords, color=color)
                linsets.append(line_set)

            # Combine all geometries
            geometries = [epcds] + linsets

            # vis = O3DVisualizer("Open3D Visualizer - Point Cloud with Trajectories", width=1200, height=800)
            # vis.show()

            # # Add the point cloud
            # vis.add_geometry("PointCloud", epcds)

            # # Add all trajectories
            # for idx, line_set in enumerate(linsets):
            #     vis.add_geometry(f"Trajectory_{idx}", line_set)
            # Visualize
            # o3d.visualization.draw_geometries(geometries)
            o3d.visualization.draw_plotly(geometries)
                # window_name="Point Cloud with RRT* Trajectories",
                # width=800,
                # height=600)

# %%
