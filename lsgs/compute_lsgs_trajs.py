# Implements the LSGS model from Haza et al. 2007 and 2012. 
# Also has an option to make unmodified trajectories using the coarse resolution velocity field.

########## IMPORTS ##########
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
import xarray as xr
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  
import sys
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from utils import calculate_acceleration, convert_velocity, find_average_trajectory


# Load datasets
velocity_ds = xr.open_dataset('aviso_data.nc') # dataset that contains the coarse resolution velocity data
calculated_model_params = pd.read_csv('calculated_model_params_2023.csv') # pre-calculated parameters for the model data
save_path_lsgs = 'lsgs-runs/' # path to save the LSGS trajectories

# Constants 
AVG_LATITUDE = 35  # picking a latitude for the conversion of the turbulent velocity fluctuations into the correct units (should not matter that much because it itself is an average)

# Time range 
START_DATE = np.datetime64('2022-05-12T23:07:27.000000000')
END_DATE = np.datetime64('2023-05-12T23:07:27.000000000')
# Corresponding to what the start date of the trajectories should actually be (for purposes of running the trajectories over just a season)
START_TIME = np.datetime64('2023-04-11T00:00:00.000000000')

def preprocess_velocity_data(vel_ds):
    return vel_ds.sel(time=slice(START_DATE, END_DATE), longitude=slice(-140, -120), latitude=slice(26,42), depth=0)

def extract_calculated_model_params(calculated_model_params_df):
    return calculated_model_params_df.loc[0, ['sigma_u', 'sigma_v', 'tau_u', 'tau_v']].astype(float).to_list()

def get_times(vel_ds):
    time = vel_ds['time']
    start_time_idx = np.where(time.values == START_TIME)[0][0]
    end_time = time.max().values
    end_time_hrs = (end_time - START_TIME) / np.timedelta64(1, 'D')  
    t_span = [0, end_time_hrs]
    t_eval = (time.values[:-1] - START_TIME) / np.timedelta64(1, 'D')
    return t_span, t_eval, start_time_idx

def calculate_eta_var(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, sigma_m_u, sigma_m_v, tau_m_u, tau_m_v):
    # Calculate the variance for eta which will be used for the initial conditions 
    eta_variance_u = ((sigma_r_u * np.sqrt(tau_m_u) - sigma_m_u * np.sqrt(tau_r_u))**2 +
                    (sigma_r_u * np.sqrt(tau_r_u) - sigma_m_u * np.sqrt(tau_m_u))**2) / (tau_r_u + tau_m_u)
    eta_variance_v = ((sigma_r_v * np.sqrt(tau_m_v) - sigma_m_v * np.sqrt(tau_r_v))**2 +
                    (sigma_r_v * np.sqrt(tau_r_v) - sigma_m_v * np.sqrt(tau_m_v))**2) / (tau_r_v + tau_m_v)
    return eta_variance_u, eta_variance_v

def calculate_lsgs_params(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, velocity_ds, lon_arr, lat_arr):
    sigma_m_u, sigma_m_v, tau_m_u, tau_m_v = extract_calculated_model_params(calculated_model_params)
    sigma_r_u, sigma_r_v = convert_velocity([sigma_r_u, sigma_r_v], AVG_LATITUDE)
    sigma_m_u, sigma_m_v = convert_velocity([sigma_m_u, sigma_m_v], AVG_LATITUDE)
    a_u = (sigma_r_u * np.sqrt(tau_m_u)) / (sigma_m_u * np.sqrt(tau_r_u)) - 1
    b_u = (sigma_r_u / (sigma_m_u * np.sqrt(tau_m_u * tau_r_u))) - (1 / tau_r_u)
    c_u = -1 / tau_r_u
    a_v = (sigma_r_v * np.sqrt(tau_m_v)) / (sigma_m_v * np.sqrt(tau_r_v)) - 1
    b_v = (sigma_r_v / (sigma_m_v * np.sqrt(tau_m_v * tau_r_v))) - (1 / tau_r_v)
    c_v = -1 / tau_r_v
    t_span, t_eval, start_t = get_times(velocity_ds)
    eta_variance_u, eta_variance_v = calculate_eta_var(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, sigma_m_u, sigma_m_v, tau_m_u, tau_m_v)
    lsgs_params = {'a_u': a_u, 'b_u': b_u, 'c_u': c_u, 'a_v': a_v, 'b_v': b_v, 'c_v': c_v, 'lat_arr': lat_arr, 'lon_arr': lon_arr, 
    't_span': t_span, 't_eval': t_eval, 'start_t': start_t, 'eta_variance_u': eta_variance_u, 'eta_variance_v': eta_variance_v}
    return lsgs_params

def extract_params(params):
    """
    Extracts parameters from the dictionary and returns them in a tuple.
    """
    keys = ['a_u', 'b_u', 'c_u', 'a_v', 'b_v', 'c_v', 'u_m_whole_interp', 'v_m_whole_interp', 'u_m_interp', 'v_m_interp', 'u_double_prime_interp', 'v_double_prime_interp']
    return tuple(params[key] for key in keys)


def prepare_interpolators(u_m, v_m, t_grid, lat_arr, lon_arr, start_t):
    """
    Prepare the interpolator given the data and the grid (because the numerical integration will need continuous data).
    """
    u_m_new = np.nan_to_num(u_m[start_t:])
    v_m_new = np.nan_to_num(v_m[start_t:])
    u_m_interp = RegularGridInterpolator((t_grid, lat_arr, lon_arr), u_m_new, bounds_error=True)
    v_m_interp = RegularGridInterpolator((t_grid, lat_arr, lon_arr), v_m_new, bounds_error=True)
    return u_m_interp, v_m_interp

def ode_system(t, y, params):
    a_u, b_u, c_u, a_v, b_v, c_v, u_m_whole_interp, v_m_whole_interp, u_m_interp, v_m_interp, u_double_prime_interp, v_double_prime_interp = extract_params(params) #Extract from the dictionary
    x0, y0, eta_u, eta_v = y # from the previous iteration

    # Get turbulent velocity at the current position
    u_m = u_m_interp([t, y0, x0])[0]
    v_m = v_m_interp([t, y0, x0])[0]
    
    # The mean velocity is over the whole year, but at the current position 
    cap_u_m = u_m_whole_interp([y0, x0])[0]
    cap_v_m = v_m_whole_interp([y0, x0])[0]

    #Subtract the mean from the velocities
    u_m_prime = u_m - cap_u_m
    v_m_prime = v_m - cap_v_m

    #Calculate the acceleration based on u_m
    u_double_prime = u_double_prime_interp([t, y0, x0])[0]
    v_double_prime = v_double_prime_interp([t, y0, x0])[0]

    #Differential equations 
    d_eta_u = a_u * u_double_prime + b_u * u_m_prime + c_u * eta_u
    d_eta_v = a_v * v_double_prime + b_v * v_m_prime + c_v * eta_v
    dx_c_dt_x = u_m + eta_u
    dx_c_dt_y = v_m + eta_v

    #Assemble
    d_xc_dt = [dx_c_dt_x, dx_c_dt_y] 
    d_eta_dt = [d_eta_u, d_eta_v]

    return d_xc_dt+d_eta_dt

def solve_ode_sys(space_init, t_grid, eta_variance_u, eta_variance_v, params_for_ivp, num_trajectories=5):
    """
    Simulate multiple trajectories and return their average positions.
    """
    traj_x_vals, traj_y_vals = [], []    
    space_init = list(space_init)
    for _ in range(num_trajectories):
        eta_init = [np.random.normal(0, np.sqrt(eta_variance_u)), np.random.normal(0, np.sqrt(eta_variance_v))]
        y_init = space_init + eta_init # since a system of equations, concatenate them together 
        sol_xc = solve_ivp(ode_system, [t_grid[0], t_grid[-1]], y_init, t_eval=t_grid, args=(params_for_ivp,), method="RK45")
        traj_x_vals.append(sol_xc.y[0])
        traj_y_vals.append(sol_xc.y[1])
        avg_traj_x, avg_traj_y = find_average_trajectory(traj_x_vals, traj_y_vals)
    return avg_traj_x, avg_traj_y

def get_params_for_ivp_update(u_m, v_m, t_grid, params):
        lon_arr = params['lon_arr']
        lat_arr = params['lat_arr']
        start_t = params['start_t']

        params_for_ivp = {k: v for k, v in params.items() if k.startswith(('a', 'b', 'c'))} 

        # Calculate the mean velocity over the whole year, for each location
        u_mean = np.mean(u_m, axis=0) 
        v_mean = np.mean(v_m, axis=0)
        u_m_whole_interp = RegularGridInterpolator((lat_arr, lon_arr), u_mean, bounds_error=False, fill_value=None)
        v_m_whole_interp = RegularGridInterpolator((lat_arr, lon_arr), v_mean, bounds_error=False, fill_value=None)
        u_m_interp, v_m_interp = prepare_interpolators(u_m, v_m, t_grid, lat_arr, lon_arr, start_t)

        # Accelerations
        # Derivative of u'=u-avg should be the same as derivative of u since avg is constant
        u_double_prime = calculate_acceleration(t_grid, u_m[start_t:])
        v_double_prime = calculate_acceleration(t_grid, v_m[start_t:])
        #Fill in the missing values of u_double_prime and v_double_prime with 0
        u_double_prime_interp, v_double_prime_interp = prepare_interpolators(u_double_prime, v_double_prime, t_grid, lat_arr, lon_arr, 0)
        params_for_ivp.update({
            'u_m_whole_interp': u_m_whole_interp, 'v_m_whole_interp': v_m_whole_interp, 'u_m_interp': u_m_interp, 'v_m_interp': v_m_interp,
            'u_double_prime_interp': u_double_prime_interp, 'v_double_prime_interp': v_double_prime_interp
        })
        return params_for_ivp

def calc_traj(space_init, t_grid, eta_variance_u, eta_variance_v, params_for_ivp):
    try:
        traj_x, traj_y = solve_ode_sys(space_init, t_grid, eta_variance_u, eta_variance_v, params_for_ivp)
        return space_init, traj_x, traj_y
    except ValueError as e:
        return None

def adv_sys(t, y, u_adv_interp, v_adv_interp):
    x0, y0 = y
    dx_dt = u_adv_interp([t, y0, x0])[0]
    dy_dt = v_adv_interp([t, y0, x0])[0]
    return (dx_dt, dy_dt)

def calc_adv_traj(adv_params, space_init):
    try:
        t_grid = adv_params['t_grid']
        u_adv_interp = adv_params['u_adv_interp']
        v_adv_interp = adv_params['v_adv_interp']
        sol_adv = solve_ivp(adv_sys, [t_grid[0], t_grid[-1]], space_init, t_eval=t_grid, args=(u_adv_interp, v_adv_interp), method="RK45")
        adv_x = sol_adv.y[0]
        adv_y = sol_adv.y[1]
        return space_init, adv_x, adv_y
    except ValueError as e:
        return None

def main(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v):
    # Load geostrophic velocity data
    velocity_ds_processed = preprocess_velocity_data(velocity_ds)
    lon_arr = velocity_ds_processed.longitude.values
    lat_arr = velocity_ds_processed.latitude.values

    # Set up parameters, time grid, space grid 
    lsgs_params = calculate_lsgs_params(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, velocity_ds_processed, lon_arr, lat_arr)
    start_t = lsgs_params['start_t']
    eta_variance_u = lsgs_params['eta_variance_u']
    eta_variance_v = lsgs_params['eta_variance_v']
    t_grid = velocity_ds_processed['time'][start_t:].values
    t_grid = (t_grid - t_grid[0]) / np.timedelta64(1, 'D')
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    #Only get space points where there is data
    space_grid = [(lon, lat) for lon in lon_arr for lat in lat_arr if not np.isnan(velocity_ds_processed['uo'].sel(latitude=lat, longitude=lon, method='nearest').values[0])]
    #subdivide space_grid 
    # Convert velocity from units of m/s to degrees/day in order to align with everything else (note we are just approximating the earth as a sphere)
    u_m = velocity_ds_processed['uo'].values
    v_m = velocity_ds_processed['vo'].values
    u_m = np.nan_to_num(u_m)
    v_m = np.nan_to_num(v_m)
    u_m_conv, v_m_conv = convert_velocity((u_m, v_m), lat_grid)

    u_adv_interp, v_adv_interp = prepare_interpolators(u_m_conv, v_m_conv, t_grid, lat_arr, lon_arr, start_t)
    adv_params = {'lat_arr': lat_arr, 'lon_arr': lon_arr, 't_grid': t_grid, 'start_t': start_t, 'u_adv_interp': u_adv_interp, 'v_adv_interp': v_adv_interp}

    params_for_lsgs_ivp = get_params_for_ivp_update(u_m_conv, v_m_conv, t_grid, lsgs_params)

    run_name = '{}-{}-{}-{}-trajs.pkl'.format(round(sigma_r_u, 3), round(sigma_r_v, 3), round(tau_r_u,3), round(tau_r_v,3))

    lsgs_trajs_dict = {}
    adv_trajs_dict = {}

    #CALCULATE LSGS TRAJECTORIES
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_traj, space_init, t_grid, eta_variance_u, eta_variance_v, params_for_lsgs_ivp) for space_init in space_grid]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                           desc="Computing LSGS Trajectories", file = sys.stdout, position = 0, leave = True):
            if future.result() is not None:
                result = future.result()
                space_init, sol_x, sol_y = result
                lsgs_trajs_dict[space_init] = (sol_x, sol_y)
    
    with open(save_path_lsgs+run_name, 'wb') as pkl_file:
        pickle.dump(lsgs_trajs_dict, pkl_file)


    # #CALCULATE NON-LSGS TRAJECTORIES WITH THE SAME SPACE, TIME GRIDS 
    # save_path_adv = 'adv-runs/'
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(calc_adv_traj, adv_params, space_init) for space_init in space_grid]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
    #                        desc="Computing Geostrophic Adv. Trajectories", file = sys.stdout, position = 0, leave = True):
    #         if future.result() is not None:
    #             result = future.result()
    #             space_init, sol_x, sol_y = result
    #             adv_trajs_dict[space_init] = (sol_x, sol_y)
    
    # with open(save_path_adv+run_name, 'wb') as pkl_file:
    #     pickle.dump(adv_trajs_dict, pkl_file)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute LSGS trajectories')
    parser.add_argument('sig_u', type=float, help='Realistic/true zonal turbulent velocity fluctuations')
    parser.add_argument('sig_v', type=float, help='Realistic/true meridional turbulent velocity fluctuations')
    parser.add_argument('tau_u', type=float, help='Realistic/true zonal Lagrangian decorrelation timescale')
    parser.add_argument('tau_v', type=float, help='Realistic/true meridional Lagrangian decorrelation timescale')
    args = parser.parse_args()
    file_ext = str(args.sig_u) + '-' + str(args.sig_v) + '-' + str(args.tau_u) + '-' + str(args.tau_v)
    main(args.sig_u, args.sig_v, args.tau_u, args.tau_v)
