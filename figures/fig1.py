'''
Code to make Figure 1 distances travelled, and the velocity variances as in Supplementary Information. 
'''

import os 
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from matplotlib.colors import LogNorm


# Set directory paths in the file 
SIM_OUTPUT_PATH = "output/" #simulation output path containing folders for varying-sigma and varying-tau
PLOT_SAVE_PATH = "plots/" #path to save plots in, should also contain folders for varying-sigma and varying-tau


# Estimating coastline based on the domain lon/lat
coastline = {
        'x1': -124.5, 'y1': 40,
        'x2': -120, 'y2': 34,
    }
coastline['m'] = (coastline['y2'] - coastline['y1']) / (coastline['x2'] - coastline['x1'])
m = coastline['m']
b = coastline['y1'] - coastline['m'] * coastline['x1'] #coastline intercept
coastline['normal_vector'] = np.array([coastline['m'], -1])
coastline['normal_vector'] /= np.linalg.norm(coastline['normal_vector'])
normal_u = coastline['normal_vector'][0]
normal_v = coastline['normal_vector'][1]
R = 6371 # radius of the Earth in km


def parse_args():
    parser = argparse.ArgumentParser(description="Set run mode and figure type.")
    parser.add_argument('--varying', choices=['sig', 'tau', 'all'], required=True, 
                        help='Choose to plot the varying ratio via sigma, tau, or all of the runs.')
    parser.add_argument('--plot', choices=['distance', 'variance'], required=True,)
    return parser.parse_args()


def set_paths(varying):
    traj_path = SIM_OUTPUT_PATH
    plot_path = PLOT_SAVE_PATH
    if varying == 'sig':
        plot_path += 'varying-sigma/'
    elif varying == 'tau':
        plot_path += 'varying-tau/'                
 
    return traj_path, plot_path


def calculate_distance_travelled(longitudes, latitudes):
    '''
    For a given trajectory, gives (all in km): 
    - total distance travelled/length of the trajectory by adding up the displacements at each time step
    - the total alongshore distance, by getting the portion of each displacement in the alongshore direction (positive or negative)
    - the total offshore distance, in the same way (not just the displacement between first and last trajectory points) 
    '''
    lat_rad = np.radians(latitudes) # convert lat/lon from degrees to radians 
    lon_rad = np.radians(longitudes)
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    displacement_ns = dlat * R  # meridional displacement (north-south)
    displacement_ew = dlon * R * np.cos(lat_rad[:-1]) #zonal displacement (east-west) accounts for latitude
    displacements = np.vstack((displacement_ew, displacement_ns)).T     # combine displacements into array of vectors

    normal_vector = np.array([m, -1]) / np.sqrt(m**2 + 1)  # calculate unit normal
    parallel_vector = np.array([1, m]) / np.sqrt(m**2 + 1) # unit vector parallel to coast
    
    along_shore_distances = np.abs(np.dot(displacements, parallel_vector)) #along shore portion of displacements
    off_shore_distances = np.abs(np.dot(displacements, normal_vector)) #offshore portion of displacements
    intercept_adjustment = np.abs(b / np.sqrt(1 + m**2))  # need to adjust the offshore distances due to the y-intercept

    total_trajectory_distance = np.sum(np.sqrt(displacement_ew**2 + displacement_ns**2)) # total trajectory distance calculated as the hypotenuse of a triangle made by n-s and e-w (not the alongshore and offshore!)

    total_along_shore_distance = np.sum(along_shore_distances)
    total_off_shore_distance = np.sum(off_shore_distances) + intercept_adjustment
    
    return total_trajectory_distance, total_along_shore_distance, total_off_shore_distance


def calculate_distances(traj_dict):
    '''
    Uses the calculate_distance_travelled function, but for the whole dictionary of trajectories, and then finds the (domain) average.
    '''
    total_distances = []
    along_shore_distances = []
    off_shore_distances = []

    for key, traj in traj_dict.items():
        total_distance, along_shore_distance, off_shore_distance = calculate_distance_travelled(traj[0], traj[1])
        total_distances.append(total_distance)
        along_shore_distances.append(along_shore_distance)
        off_shore_distances.append(off_shore_distance)

    # Calculate the average distances
    avg_offshore_distance = np.mean(np.array(off_shore_distances))
    avg_alongshore_distance = np.mean(np.array(along_shore_distances))
    avg_total_distance = np.mean(np.array(total_distances))

    return avg_total_distance, avg_offshore_distance, avg_alongshore_distance


def calculate_ratios(params):
    sig_u = params[0]
    sig_v = params[1]
    tau_u= params[2]
    tau_v = params[3]
    ratio_u = sig_u * math.sqrt(tau_u)
    ratio_v = sig_v * math.sqrt(tau_v)
    return ratio_u, ratio_v


def plot_ratios_vs_field(ratio, field, drifter_ratio, drifter_field, title, plot_path):
    plt.figure(figsize=(10, 5), tight_layout = True)
    scatter = plt.scatter(ratio, field, c='dimgray', s=200)    
    plt.xlabel(r'$\sigma^u \sqrt{\tau^u}$', fontsize=24)
    plt.ylabel("Distance (km)", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(title, fontsize=30)
    plt.savefig(plot_path+title + '.png')
    plt.close()


def calculate_velocity_variances(u_vel_dict, v_vel_dict):
    u_variances = []
    v_variances = []
    for key, u_vel in u_vel_dict.items():
        u_vel = u_vel_dict[key]
        u_variance = np.var(u_vel)
        u_variances.append(u_variance)
    for key, v_vel in v_vel_dict.items():
        v_vel = v_vel_dict[key]
        v_variance = np.var(v_vel)
        v_variances.append(v_variance)
    avg_u_variance = np.mean(np.array(u_variances))
    avg_v_variance = np.mean(np.array(v_variances))
    return avg_u_variance, avg_v_variance


def plot_distances_together(ratio_u, total_distances, offshore_distances, alongshore_distances, plot_path):
    plt.figure(figsize=(10, 10), tight_layout=True)
    plt.plot(ratio_u, total_distances, color='rebeccapurple', label='NSWE', linewidth = 3)
    plt.plot(ratio_u, offshore_distances, color='steelblue', label=r'$\perp$ to Coast', linewidth=3)
    plt.plot(ratio_u, alongshore_distances, color='firebrick', label=r'$\parallel$ to Coast', linewidth=3)
    plt.xlabel(r'Zonal Disp. Param. $\sigma^u \sqrt{\tau^u}$', fontsize=32)
    plt.ylabel('Total Displacement (km)', fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(np.arange(0, 1001, 200), fontsize=28)  
    plt.ylim(0, 1050)  
    plt.legend(fontsize=28, loc='upper left')
    plt.savefig(plot_path + 'Distances_vs_Ratio.png')
    plt.close()


def main():
    args = parse_args()
    traj_path, plot_path = set_paths(args.varying)
    print(f"Trajectory path: {traj_path}")
    print(f"Path to save the plots in: {plot_path}")
    if args.plot == 'distance':
        filename_prefix = ''
        filename_suffix = '-trajs.pkl' 
        if args.varying == 'sig':
            filename_suffix = '0.15-0.702-0.691-trajs.pkl'  # only the simulations that were varying in u 
        elif args.varying == 'tau':
            filename_prefix = '0.171-0.165' 
            filename_suffix = '-0.7-trajs.pkl'
        ratio_u = []
        ratio_v = []
        offshore_distances = []
        alongshore_distances = []
        total_distances = []
        for filename in os.listdir(traj_path):
            if filename.endswith(filename_suffix) and filename.startswith(filename_prefix):
                file_path = os.path.join(traj_path, filename)
                with open(file_path, 'rb') as file:
                    traj_dict = pickle.load(file)
                
                filename = os.path.splitext(filename)[0] # extract the filename without the extension
                params = filename.split('-')[:-1] # extract the parameters from the filename
                params = [float(i) for i in params]
                print(f"Processing: {params}")

                u_ratio, v_ratio = calculate_ratios(params)
                ratio_u.append(u_ratio)
                ratio_v.append(v_ratio)
                total_distance, offshore_distance, alongshore_distance = calculate_distances(traj_dict)
                total_distances.append(total_distance)
                offshore_distances.append(offshore_distance)
                alongshore_distances.append(alongshore_distance)
        drifter_filename = '0.171-0.165-0.702-0.691-trajs.pkl'
        drifter_params = drifter_filename.split('-')[:-1]
        drifter_params = [float(i) for i in drifter_params]
        drifter_ratio_u, drifter_ratio_v = calculate_ratios(drifter_params)
        drifter_total_distance, drifter_offshore_distance, drifter_alongshore_distance = calculate_distances(traj_dict)

        # Plotting against u because it is the more significant direction in the flux calculation (but this line can be changed to plot v)
        plot_distances_together(ratio_u, total_distances, offshore_distances, alongshore_distances, plot_path)

    elif args.plot == 'variance':
        filename_prefix = ''
        filename_suffix_u = '-u-vel.pkl'
        filename_suffix_v = '-v-vel.pkl'

        if args.varying == 'sig':
            filename_suffix_u = '0.15-0.702-0.691-u-vel.pkl'  # only varying in u
            filename_suffix_v = '0.15-0.702-0.691-v-vel.pkl'  # matching v
        elif args.varying == 'tau':
            filename_prefix = '0.171-0.165-0.7-' 
            filename_suffix_u = '-u-vel.pkl'
            filename_suffix_v = '-v-vel.pkl'

        u_vel_variance_data = []
        v_vel_variance_data = []
        ratios_u = []
        ratios_v = []
        counter = 0

        for filename in os.listdir(traj_path):
            if filename.endswith(filename_suffix_u) and filename.startswith(filename_prefix):
                file_path_u = os.path.join(traj_path, filename)
                file_path_v = file_path_u.replace('-u-vel.pkl', '-v-vel.pkl')
                try:
                    with open(file_path_u, 'rb') as file_u, open(file_path_v, 'rb') as file_v:
                        u_vel_dict = pickle.load(file_u)
                        v_vel_dict = pickle.load(file_v)
                    filename_base = os.path.splitext(filename)[0]
                    params = filename_base.split('-')[:-2]
                    params = [float(i) for i in params]
                    ratio_u, ratio_v = calculate_ratios(params)
                    ratios_u.append(ratio_u)
                    ratios_v.append(ratio_v)
                    print(f"Processing: {params}")
                    u_vel_variance, v_vel_variance = calculate_velocity_variances(u_vel_dict, v_vel_dict)
                    u_vel_variance_data.append(u_vel_variance)
                    v_vel_variance_data.append(v_vel_variance)

                except FileNotFoundError as e:
                    print(f"Error: {e}. No velocity file for: {filename}")

        drifter_filename = '0.171-0.165-0.702-0.691-u-vel.pkl'
        drifter_file_path_u = os.path.join(traj_path, drifter_filename)
        drifter_file_path_v = drifter_file_path_u.replace('-u-vel.pkl', '-v-vel.pkl')
        with open(drifter_file_path_u, 'rb') as file_u, open(drifter_file_path_v, 'rb') as file_v:
            drifter_u_vel_dict = pickle.load(file_u)
            drifter_v_vel_dict = pickle.load(file_v)
        drifter_params = drifter_filename.split('-')[:-2]
        drifter_params = [float(i) for i in drifter_params]
        drifter_ratio_u, drifter_ratio_v = calculate_ratios(drifter_params)
        drifter_u_vel_variance, drifter_v_vel_variance = calculate_velocity_variances(drifter_u_vel_dict, drifter_v_vel_dict)

        plot_ratios_vs_field(ratios_u, u_vel_variance_data, drifter_ratio_u, drifter_u_vel_variance, "Avg. U Velocity Variance", plot_path)

if __name__ == '__main__':
    main()