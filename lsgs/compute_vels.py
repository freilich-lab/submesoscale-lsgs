"""
Computes the velocities along the trajectories. 
Mostly uses functions in utils. 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from scipy.interpolate import interp1d
import argparse
import sys
import xarray as xr
from utils import clean_dict, get_velocities

delta_t = 6*60*60  # six hours in seconds
save_path_lsgs = 'lsgs-runs/' #where the trajectories are saved (assuming the velocity should also be saved here)
save_path_adv = 'adv-runs/' #where the trajectories for the unmodified run are saved

def main(file_ext):    
    with open('/users/elseitz/data/lulabel/output/lsgs-runs/'+file_ext+'-trajs.pkl', 'rb') as f:
        lsgs_traj = pickle.load(f) #load the trajectory
    lsgs_traj = {k: lsgs_traj[k] for k in lsgs_traj if k in adv_traj}
    u_vel_lsgs = {} #initialize empty dicts for the velocities to be computed (they will be identified with the start point of the trajectory)
    v_vel_lsgs = {}
    for key, val in tqdm(lsgs_traj.items()):
        lon = val[0]
        lat = val[1]
        u_vel, v_vel = get_velocities(lon, lat, delta_t)
        u_vel_lsgs[key] = u_vel
        v_vel_lsgs[key] = v_vel

    # Save results 
    run_name_traj = file_ext+'-trajs.pkl'
    run_name_u_vel = file_ext+'-u-vel.pkl'
    run_name_v_vel = file_ext+'-v-vel.pkl'
    with open(save_path_lsgs + run_name_traj, 'wb') as pkl_file:
        pickle.dump(lsgs_traj, pkl_file)
    with open(save_path_lsgs + run_name_u_vel, 'wb') as pkl_file:
        pickle.dump(u_vel_lsgs, pkl_file)
    with open(save_path_lsgs + run_name_v_vel, 'wb') as pkl_file:
        pickle.dump(v_vel_lsgs, pkl_file)

    # #Uncomment to do this for the advected trajectories instead
    # #This is just put separately in case the advected trajectories are stored in a different file or under a different naming convention
    # with open('adv-trajs.pkl', 'rb') as f: 
    #      adv_traj = pickle.load(f)
    # adv_traj = {k: adv_traj[k] for k in adv_traj if k in lsgs_traj}
    #u_vel_adv = {}
    #v_vel_adv = {}
    # for key, val in tqdm(adv_traj.items()):
    #     lon = val[0]
    #     lat = val[1]
    #     u_vel, v_vel = get_velocities(lon, lat, delta_t)
    #     u_vel_adv[key] = u_vel
    #     v_vel_adv[key] = v_vel
    # with open(save_path_adv + run_name_traj, 'wb') as pkl_file:
    #     pickle.dump(adv_traj, pkl_file)
    # with open(save_path_adv + run_name_u_vel, 'wb') as pkl_file:
    #     pickle.dump(u_vel_adv, pkl_file)
    # with open(save_path_adv + run_name_v_vel, 'wb') as pkl_file:
    #     pickle.dump(v_vel_adv, pkl_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute LSGS trajectories')
    parser.add_argument('sig_u', type=float, help='Realistic/true zonal turbulent velocity fluctuation')
    parser.add_argument('sig_v', type=float, help='Realistic/true meridional turbulent velocity fluctuation')
    parser.add_argument('tau_u', type=float, help='Realistic/true zonal decorrelation timescale')
    parser.add_argument('tau_v', type=float, help='Realistic/true meridional decorrelation timescale')
    args = parser.parse_args()
    args.sig_u = round(args.sig_u, 3)
    args.sig_v = round(args.sig_v, 3)
    args.tau_u = round(args.tau_u, 3)
    args.tau_v = round(args.tau_v, 3)
    file_ext = str(args.sig_u) + '-' + str(args.sig_v) + '-' + str(args.tau_u) + '-' + str(args.tau_v)
    main(file_ext)

