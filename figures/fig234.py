'''
Plotting code for: 
Figure 2: distribution of the fluxes as a function of dispersion parameter, as each lambda 
Figure 3: lines of best fit 
Figure 4: flux as a function of the uptake rate lambda, for different dispersion parameter values 

To run: use command line arguments, e.g. 
python3 plot-dist-each-lambda.py --mode original --figure figure2 --varying all
'''

import os 
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import argparse

# Set data and save paths in file, below 
SIM_VELOCITY_PATH = "simulations_velocity_path/"
SIM_FLUX_PATH = "simulations_flux_path/"
DRIFTER_SIM_PATH = "drifter_params_simulations_path/" # path to the trajectories / simulations run using the exact drifter parameters
SIM_FLUX_REV_PATH = "simulations_flux_path_reversed/" # path to the flux data for the simulations with the reversed nutrient gradient
DRIFTER_SIM_REV_PATH = "drifter_params_simulations_path_reversed/" # path to the trajectories / simulations run using the exact drifter parameters with the reversed nutrient gradient

SAVE_PATH_DISTRIBUTIONS = "distributions_of_flux_across_domains/" # path to save the plots for the distribution of fluxes across all trajectories (as in supplementary information)
SAVE_PATH_FIG_23 = "figure2-3/" # path to save the plots for figure 2 and 3, should contain subfolders for varying-sigma and varying-tau
SAVE_PATH_FIG_4 = "figure4/" # path to save the plots for figure 4, should contain subfolders for varying-sigma and varying-tau
SAVE_PATH_FIG_23_REV = "figure2-3-reversed/" # path to save the plots for versions of figure 2 and 3 but with reversed nutrient gradient, should contain subfolders for varying-sigma and varying-tau
SAVE_PATH_FIG_4_REV = "figure4-reversed/" # path to save the plots for figure 4 but with reversed nutrient gradient, should contain subfolders for varying-sigma and varying-tau

TIME_CONVERSION = 24  # conversion factor for lambda values from hour^-1 to day^-1

# Coastline approximated based on lat/lon in domain
coastline = {
        'x1': -124.5, 'y1': 40,
        'x2': -120, 'y2': 34,
    }
coastline['m'] = (coastline['y2'] - coastline['y1']) / (coastline['x2'] - coastline['x1'])
coastline['b'] = coastline['y1'] - coastline['m'] * coastline['x1']
coastline['normal_vector'] = np.array([coastline['m'], -1])
coastline['normal_vector'] /= np.linalg.norm(coastline['normal_vector'])
normal_u = coastline['normal_vector'][0]
normal_v = coastline['normal_vector'][1]


def parse_args():
    parser = argparse.ArgumentParser(description="Set run mode and figure type.")
    parser.add_argument('--mode', choices=['original', 'reversed'], required=True, 
                        help='Choose between original or reversed mode.')
    parser.add_argument('--figure', choices=['figure2', 'figure4'], required=True, 
                        help='Choose figure 2 (also runs figure 3!) or figure 4.')
    parser.add_argument('--varying', choices=['sig', 'tau', 'all'], required=True, 
                        help='Choose to plot the varying ratio via sigma, tau, or all of the runs. Figure 2/3 goes with sigma or tau')
    return parser.parse_args()


def set_paths(mode, figure, varying):
    drifter_filename = 'sigu-sigv-tauu-tauv-flux.pkl' # this script expects this naming convention, where sig_u, etc. are floats
    adv_filename = 'sigu-sigv-tauu-tauv-flux.pkl' 
    dist_path_non_params = SAVE_PATH_DISTRIBUTIONS
    vel_path = SIM_VELOCITY_PATH
    if mode == 'original':
        directory_path = SIM_FLUX_PATH
        real_param_dir = DRIFTER_SIM_PATH
        if figure == 'figure2':
            dist_path = SAVE_PATH_FIG_23              
        elif figure == 'figure4':
            dist_path = SAVE_PATH_FIG_4
    elif mode == 'reversed':
        directory_path = SIM_FLUX_REV_PATH
        real_param_dir = DRIFTER_SIM_REV_PATH
        if figure == 'figure2':
            dist_path = SAVE_PATH_FIG_23_REV
        elif figure == 'figure4':
            dist_path = SAVE_PATH_FIG_4_REV
    if figure == 'figure2':
        if varying == 'sig':
            dist_path += 'varying-sigma/'
        elif varying == 'tau':
            dist_path += 'varying-tau/' 
    return vel_path, directory_path, real_param_dir, drifter_filename, adv_filename, dist_path, dist_path_non_params


def calculate_ratios(params):
    sig_u = params[0]
    sig_v = params[1]
    tau_u= params[2]
    tau_v = params[3]
    ratio_u = sig_u*math.sqrt(tau_u)
    ratio_v = sig_v*math.sqrt(tau_v)
    return ratio_u, ratio_v


def calculate_lambda_flux_averages(flux_dict):
    lambda_flux_aggregate = {}
    for _, lambda_dict in flux_dict.items():
        for lambda_value, flux_timeseries in lambda_dict.items():
            # Calculate the average flux over time for this lambda
            avg_flux = np.mean(flux_timeseries)
            if lambda_value not in lambda_flux_aggregate:
                lambda_flux_aggregate[lambda_value] = [avg_flux]
            else:
                lambda_flux_aggregate[lambda_value].append(avg_flux)
    return lambda_flux_aggregate


def plot_lambda_flux_dist(lambda_flux_dict, param_title, dist_path_non_params):
    plt.figure(figsize=(10, 10), tight_layout = True)
    colormap = plt.cm.viridis
    lambda_values = np.array(list(lambda_flux_dict.keys()))*TIME_CONVERSION
    norm = LogNorm(vmin=lambda_values.min(), vmax=lambda_values.max())
    # Loop through each lambda, plotting the KDE for its distribution of flux averages
    for lambda_value, flux_averages in lambda_flux_dict.items():
        color = colormap(norm(lambda_value*TIME_CONVERSION))
        sns.kdeplot(flux_averages color=color, bw_adjust=0.5)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label=r'$\lambda$ Value (in day$^{-1}$)')
    plt.title('Distribution of Time-Integrated Flux, '+param_title)
    plt.xlabel('Average Flux')
    plt.ylabel('Kernel Density')
    plt.savefig(dist_path_non_params + str(param_title)+'.png')
    plt.close()


def plot_ratios_vs_field(ratio, color_arr, field, drifter_flux, drifter_ratio, adv_flux, adv_ratio, xlabel, ylabel, title, dist_path, colorbar_label, disp_param, mode):
    #Plots the scatter as shown in Fig. 2
    #The scatter will be colored according to other_val, i.e. the parameter in the other direction
    plt.figure(figsize=(10, 8), tight_layout = True)
    if disp_param == 'sig':
        norm = plt.Normalize(0, 0.45) # based on the range of parameters simulated
    else:
        norm = plt.Normalize(0.11, 0.18) # based on the range of tau simulated and corresponding parameter values
    cmap = plt.cm.cividis
    scatter = plt.scatter(ratio, field, c=color_arr, cmap=cmap, norm=norm)    
    plt.scatter(drifter_ratio, drifter_flux, c='crimson', marker="*", s=500)
    # Can also add the only advection one here if wanted, but it's outside of the range from the rest 
    cbar = plt.colorbar(scatter, label=colorbar_label)
    cbar.set_label(colorbar_label, fontsize=32)
    if disp_param == 'sig':
            plt.xlim([0, 0.45])
            xticks = [0, 0.1, 0.2, 0.3, 0.4]
            plt.xticks(xticks)
    else:
        if mode == 'original':
            plt.xlim([0.11, 0.18])
            plt.yticks([3, 3.5, 4, 4.5, 5])
            plt.xticks([0.12, 0.14, 0.16, 0.18])
        else:
            plt.xlim([0.15, 0.25])
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    cbar.ax.tick_params(labelsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.savefig(dist_path+title + '.png')
    plt.close()



def plot_best_fit(lambda_values, agg_data, ratio, xlabel, title, dist_path, dist_param, mode):
    #Make a plot of the best fit line for each lambda value, as in Fig. 3
    norm = LogNorm(lambda_values.min(), lambda_values.max())
    sm = plt.cm.ScalarMappable(cmap="GnBu", norm=norm)
    sm.set_array([])
    fig = plt.figure(figsize=(10, 8), tight_layout = True)
    ax = fig.add_subplot(1, 1, 1)
    for lambda_val, fluxes in agg_data.items():
        z = np.polyfit(ratio, fluxes, 1)
        p = np.poly1d(z)
        line = p(ratio)
        shadow_upper = line + np.std(fluxes)
        shadow_lower = line - np.std(fluxes)
        color = sm.to_rgba(lambda_val)
        ax.plot(ratio, line, color=color)
        ax.fill_between(ratio, shadow_lower, shadow_upper, color=color, alpha=0.01)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Uptake Rate, day$^{-1}$', fontsize=32)
    cbar.ax.tick_params(labelsize=24)

    if mode=='original':
        if dist_param == 'tau':
            plt.ylim([0, 8]) #based on the actual data, to get the best visualization (these should be adjusted based on the run)
            plt.xticks(np.linspace(0.12, 0.28, 5)) 
        else:
            plt.ylim([-5, 20])
    else:
        if dist_param == 'tau':
            plt.ylim([-2.5, 2])
        else:
            plt.ylim([-8, 8])

    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel('$\mu$mol m$^{-2}$ day$^{-1}$', fontsize=32)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    plt.savefig(dist_path+title + '.png')
    plt.close()

def plot_flux_vs_lambda(aggregated_data, ratio_u, ratio_v, dist_path):
    # Plots the flux with lambda on the x-axis, as in Fig. 4
    # In aggregated_data, for each lambda, the corresponding fluxes are added in the same order as the ratio lists
    # For plotting flux vs. lambda for each ratio, need to go through all of the keys in the dictionary, plot those on the x axis, and 
    # plot the same index element in each value list on the y-axis. Then color by the corresponding ratio value 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), tight_layout=True)
    lambdas = list(aggregated_data.keys())

    norm = plt.Normalize(0, 0.45) #based on the range of parameters simulated, change based on runs being plotted
    cmap = plt.cm.PuBu

    # Plotting for the first subplot (ax1) using ratio_u
    for i, ratio in enumerate(ratio_u):
        fluxes = [aggregated_data[lam][i] for lam in lambdas]
        ax1.plot(lambdas, fluxes, color=cmap(norm(ratio)))

    sm_u = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_u.set_array([])
    cbar = fig.colorbar(sm_u, ax=ax1, label = r'Zonal Disp. Param. $\sigma^u\sqrt{\tau^u}$')
    ax1.set_xscale('log')
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(r'Zonal Disp. Param. $\sigma^u\sqrt{\tau^u}$', fontsize=32)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    ax1.set_xlabel('Uptake Rate, day$^{-1}$', fontsize=32)
    ax1.set_ylabel('$\mu$mol m$^{-2}$ day$^{-1}$', fontsize=32)

    # Plotting for the second subplot (ax2) using ratio_v 
    for i, ratio in enumerate(ratio_v):
        fluxes = [aggregated_data[lam][i] for lam in lambdas]
        ax2.plot(lambdas, fluxes, color=cmap(norm(ratio)))

    sm_v = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_v.set_array([])
    colorbar2 = fig.colorbar(sm_v, ax=ax2, label=r'Merid. Disp. Param. $\sigma^v\sqrt{\tau^v}$')
    colorbar2.ax.tick_params(labelsize=24)
    colorbar2.set_label(r'Merid. Disp. Param. $\sigma^v\sqrt{\tau^v}$', fontsize=32)
    ax2.tick_params(axis='x', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24)
    ax2.set_xscale('log')
    ax2.set_xlabel('Uptake Rate, day$^{-1}$', fontsize=32)
    ax2.set_ylabel('$\mu$mol m$^{-2}$ day$^{-1}$', fontsize=32)

    title = 'Average Flux vs. Uptake Rate'
    plt.savefig(dist_path+title + '.png')
    plt.close()


def main():
    args = parse_args()
    vel_path, directory_path, real_param_dir, drifter_filename, adv_filename, dist_path, dist_path_non_params = set_paths(args.mode, args.figure, args.varying)
    print(f"Velocity path: {vel_path}")
    print(f"Directory path: {directory_path}")
    print(f"Path to save the plots in: {dist_path}")
    ratio_u = []
    ratio_v =[]
    list_of_params_lists = []
    lambda_values = np.logspace(np.log10(0.005), np.log10(2), 20) # lambda values in the simulations 
    aggregated_data = {np.round(lambda_value, 4): [] for lambda_value in lambda_values}

    # The drifter one is separate because it has 4 set parameters, we don't want to vary any 
    drifter_data = {np.round(lambda_value, 4): [] for lambda_value in lambda_values}
    print(drifter_filename)
    with open(real_param_dir + drifter_filename, 'rb') as file:
        drifter_flux = pickle.load(file)
    drifter_flux_dict = calculate_lambda_flux_averages(drifter_flux)
    for lambda_val, fluxes in drifter_flux_dict.items():
        lamda_adj = np.round(lambda_val*TIME_CONVERSION, 4)
        drifter_data[lamda_adj].append(np.mean(fluxes)) 
    drifter_filename = os.path.splitext(drifter_filename)[0]
    drift_params = drifter_filename.split('-')[:-1]
    drift_params = [float(i) for i in drift_params]
    ratio_drifter_u, ratio_drifter_v = calculate_ratios(drift_params)

    # The advection one is separate for the same reason 
    adv_data = {np.round(lambda_value, 4): [] for lambda_value in lambda_values}
    with open(real_param_dir + adv_filename, 'rb') as file:
        adv_flux = pickle.load(file)
    adv_flux_dict = calculate_lambda_flux_averages(adv_flux)
    for lambda_val, fluxes in adv_flux_dict.items():
        lamda_adj = np.round(lambda_val*TIME_CONVERSION, 4)
        adv_data[lamda_adj].append(np.mean(fluxes))
    adv_filename = os.path.splitext(adv_filename)[0]
    adv_params = adv_filename.split('-')[:-1]
    adv_params = [float(i) for i in adv_params]
    ratio_adv_u, ratio_adv_v = calculate_ratios(adv_params)
    
    # Looping through the rest of the files of interest
    filename_prefix = ''
    filename_suffix = '-flux.pkl'  
    if args.varying == 'sig':
        filename_suffix = 'tauu-tauv-flux.pkl' # expects this file naming convention, where tauu and tauv are the drifter parameters as floats
    elif args.varying == 'tau':
        filename_prefix = 'sigu-sigv' # expects this file naming convention, where sigu and sigv are the drifter parameters as floats

    # Process files based on the selected mode and varying parameter
    for filename in os.listdir(directory_path):
        if filename.endswith(filename_suffix) and filename.startswith(filename_prefix):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'rb') as file:
                    flux_dict = pickle.load(file)
                filename = os.path.splitext(filename)[0] # extract the filename without the extension
                params = filename.split('-')[:-1] # extract the parameters from the filename
                params = [float(i) for i in params]
                print(f"Processing: {params}")
                list_of_params_lists.append(params)
                lambda_flux_dict = calculate_lambda_flux_averages(flux_dict)
                param_title = '-'.join([str(i) for i in params])
                #To plot the distributions of fluxes across the domain, as in Supplementary Information
                #plot_lambda_flux_dist(lambda_flux_dict, param_title, dist_path_non_params)
                for lambda_val, fluxes in lambda_flux_dict.items():
                    lamda_adj = np.round(lambda_val, 4) 
                    aggregated_data[lamda_adj].append(np.mean(fluxes)) 
                u_ratio, v_ratio = calculate_ratios(params)
                ratio_u.append(u_ratio)
                ratio_v.append(v_ratio)
    
    
    if args.figure == 'figure2':
        plot_best_fit(lambda_values, aggregated_data, ratio_u, r'Zonal Disp. Param. $\sigma^u\sqrt{\tau^u}$', r'Best Fit, Avg. Flux vs. $\sigma^u\sqrt{\tau^u}$', dist_path, args.varying, args.mode)
        plot_best_fit(lambda_values, aggregated_data, ratio_v, r'Merid. Disp. Param. $\sigma^v\sqrt{\tau^v}$', r'Best Fit, Avg. Flux vs. $\sigma^v\sqrt{\tau^v}$', dist_path, args.varying, args.mode)
        for lambda_val, flux in aggregated_data.items():
            drifter_flux = drifter_data[lambda_val]
            adv_flux = adv_data[lambda_val]
            plot_ratios_vs_field(ratio_u, ratio_v, flux, drifter_flux, ratio_drifter_u, adv_flux, ratio_adv_u, r'Zonal Disp. Param. $\sigma^u\sqrt{\tau^u}$', '$\\mu$mol m$^{-2}$ day$^{-1}$', r'Avg. Flux vs. $\sigma^u\sqrt{\tau^u}$, $\lambda=$'+str(lambda_val), dist_path, r'Merid. Disp. Param. $\sigma^v\sqrt{\tau^v}$', args.varying, args.mode)
            plot_ratios_vs_field(ratio_v, ratio_u, flux, drifter_flux, ratio_drifter_v, adv_flux, ratio_adv_v, r'Merid. Disp. Param. $\sigma^v\sqrt{\tau^v}$', '$\\mu$mol m$^{-2}$ day$^{-1}$', r'Avg. Flux vs. $\sigma^v\sqrt{\tau^v}$, $\lambda=$'+str(lambda_val), dist_path, r'Zonal Disp. Param. $\sigma^u\sqrt{\tau^u}$', args.varying, args.mode)
    if args.figure=='figure4':
        plot_flux_vs_lambda(aggregated_data, ratio_u, ratio_v, dist_path)

if __name__ == '__main__':
    main()
