'''
Generates the comparison of trajectories, as in Figure 1. 
'''

########## IMPORTS ##########
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
import xarray as xr
from scipy.integrate import solve_ivp
from utils import update_position, calculate_acceleration
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  
import sys
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from utils import find_average_trajectory
plt.rcParams['font.family'] = 'serif'


# Load datasets
velocity_ds = xr.open_dataset('data/dataset-uv-nrt-hourly_1695588955720.nc') # load the AVISO data
calculated_model_params = pd.read_csv('data/calculated_model_params_2023.csv') # load the calculated parameters from the drifter data
drifter_ds = xr.open_dataset('data/SMODE_IOP2_surface_drifter_0-4695268.nc') # load the drifter data

# Constants
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds
meters_per_degree = 111320
AVG_LATITUDE = 35  # picking a latitude for the conversion of the turbulent velocity fluctuations into the correct units (does not matter that much because it itself is an average)


def define_constants():
    # Defining the coastline 
    coastline = {
        'x1': -124.5, 'y1': 40,
        'x2': -120, 'y2': 34,
    }
    coastline['m'] = (coastline['y2'] - coastline['y1']) / (coastline['x2'] - coastline['x1'])
    coastline['b'] = coastline['y1'] - coastline['m'] * coastline['x1']
    coastline['normal_vector'] = np.array([coastline['m'], -1])
    coastline['normal_vector'] /= np.linalg.norm(coastline['normal_vector'])
    coastline['midpoint'] = {
        'x': (coastline['x1'] + coastline['x2']) / 2,
        'y': (coastline['y1'] + coastline['y2']) / 2
    }
    return coastline


def Nzero(x, y, m, b, k=5.5, l=0.25):
    """
    Nutrient gradient, as in the nutrients.py script. 
    """
    A = m
    B = -1
    C = b
    distance = abs(A*x + B*y + C) / np.sqrt(A**2 + B**2)
    grad = k - l* distance
    grad = grad * 1025 #unit conversion to get in mumol/m^3
    return grad


def preprocess_velocity_data(vel_ds, start_date, end_date):
    return vel_ds.sel(time=slice(start_date, end_date), depth=0)


def extract_calculated_model_params(calculated_model_params_df):
    return calculated_model_params_df.loc[0, ['sigma_u', 'sigma_v', 'tau_u', 'tau_v']].astype(float).to_list()


def get_times(vel_ds, start_time):
    time = vel_ds['time']
    start_time_idx = np.where(time.values == start_time)[0][0]
    end_time = time.max().values
    end_time_hrs = (end_time - start_time) / np.timedelta64(1, 'D')  
    t_span = [0, end_time_hrs]
    t_eval = (time.values[:-1] - start_time) / np.timedelta64(1, 'D')
    return t_span, t_eval, start_time_idx

def convert_velocity(velocity_m_s, latitude):
    adjusted_factor = np.cos(np.radians(latitude)) # only the u velocity should be adjusted for latitude 
    u = (velocity_m_s[0] / (meters_per_degree * adjusted_factor)) * SECONDS_PER_DAY 
    v = (velocity_m_s[1] / meters_per_degree) * SECONDS_PER_DAY
    return (u,v)

def calculate_eta_var(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, sigma_m_u, sigma_m_v, tau_m_u, tau_m_v):
    """
    Calculate the variance for eta which will be used for the initial conditions 
    Using the formula in Haza et al. paper
    """
    eta_variance_u = ((sigma_r_u * np.sqrt(tau_m_u) - sigma_m_u * np.sqrt(tau_r_u))**2 +
                    (sigma_r_u * np.sqrt(tau_r_u) - sigma_m_u * np.sqrt(tau_m_u))**2) / (tau_r_u + tau_m_u)
    eta_variance_v = ((sigma_r_v * np.sqrt(tau_m_v) - sigma_m_v * np.sqrt(tau_r_v))**2 +
                    (sigma_r_v * np.sqrt(tau_r_v) - sigma_m_v * np.sqrt(tau_m_v))**2) / (tau_r_v + tau_m_v)
    return eta_variance_u, eta_variance_v


def calculate_lsgs_params(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, velocity_ds, lon_arr, lat_arr, start_time):
    """
    Calculate the parameter values, as in the paper 
    """
    sigma_m_u, sigma_m_v, tau_m_u, tau_m_v = extract_calculated_model_params(calculated_model_params)
    print(sigma_m_u, sigma_m_v, tau_m_u, tau_m_v)
    sigma_r_u, sigma_r_v = convert_velocity([sigma_r_u, sigma_r_v], AVG_LATITUDE)
    sigma_m_u, sigma_m_v = convert_velocity([sigma_m_u, sigma_m_v], AVG_LATITUDE)
    a_u = (sigma_r_u * np.sqrt(tau_m_u)) / (sigma_m_u * np.sqrt(tau_r_u)) - 1
    b_u = (sigma_r_u / (sigma_m_u * np.sqrt(tau_m_u * tau_r_u))) - (1 / tau_r_u)
    c_u = -1 / tau_r_u
    a_v = (sigma_r_v * np.sqrt(tau_m_v)) / (sigma_m_v * np.sqrt(tau_r_v)) - 1
    b_v = (sigma_r_v / (sigma_m_v * np.sqrt(tau_m_v * tau_r_v))) - (1 / tau_r_v)
    c_v = -1 / tau_r_v
    t_span, t_eval, start_t = get_times(velocity_ds, start_time)
    eta_variance_u, eta_variance_v = calculate_eta_var(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, sigma_m_u, sigma_m_v, tau_m_u, tau_m_v)
    lsgs_params = {'a_u': a_u, 'b_u': b_u, 'c_u': c_u, 'a_v': a_v, 'b_v': b_v, 'c_v': c_v, 'lat_arr': lat_arr, 'lon_arr': lon_arr, 
    't_span': t_span, 't_eval': t_eval, 'start_t': start_t, 'eta_variance_u': eta_variance_u, 'eta_variance_v': eta_variance_v}
    return lsgs_params

def extract_params(params):
    """
    Extracts parameters from the dictionary and returns them as a tuple containing all extracted parameters.
    """
    keys = ['a_u', 'b_u', 'c_u', 'a_v', 'b_v', 'c_v', 'u_m_whole_interp', 'v_m_whole_interp', 'u_m_interp', 'v_m_interp', 'u_double_prime_interp', 'v_double_prime_interp']
    return tuple(params[key] for key in keys)


def prepare_interpolators(u_m, v_m, t_grid, lat_arr, lon_arr, start_t):
    """
    Prepare the interpolator given the data and the grid.
    """
    u_m_new = np.nan_to_num(u_m[start_t:])
    v_m_new = np.nan_to_num(v_m[start_t:])
    u_m_interp = RegularGridInterpolator((t_grid, lat_arr, lon_arr), u_m_new, bounds_error=False, fill_value=None)
    v_m_interp = RegularGridInterpolator((t_grid, lat_arr, lon_arr), v_m_new, bounds_error=False, fill_value=None)
    return u_m_interp, v_m_interp


def ode_system(t, y, params):
    """
    The ODE system associated with the LSGS model, as in the paper. 
    """
    #Extract params
    a_u, b_u, c_u, a_v, b_v, c_v, u_m_whole_interp, v_m_whole_interp, u_m_interp, v_m_interp, u_double_prime_interp, v_double_prime_interp = extract_params(params)
    x0, y0, eta_u, eta_v = y
    u_m = u_m_interp([t, y0, x0])[0]
    v_m = v_m_interp([t, y0, x0])[0]
    
    cap_u_m = u_m_whole_interp([y0, x0])[0]
    cap_v_m = v_m_whole_interp([y0, x0])[0]

    #Subtract the mean from the velocities
    u_m_prime = u_m - cap_u_m 
    v_m_prime = v_m - cap_v_m

    #Calculate the acceleration based on u_m
    u_double_prime = u_double_prime_interp([t, y0, x0])[0]
    v_double_prime = v_double_prime_interp([t, y0, x0])[0]

    #Differential equations, as in the paper 
    d_eta_u = a_u * u_double_prime + b_u * u_m_prime + c_u * eta_u
    d_eta_v = a_v * v_double_prime + b_v * v_m_prime + c_v * eta_v
    dx_c_dt_x = u_m + eta_u
    dx_c_dt_y = v_m + eta_v
    d_xc_dt = [dx_c_dt_x, dx_c_dt_y] 
    d_eta_dt = [d_eta_u, d_eta_v]

    return d_xc_dt+d_eta_dt

def solve_ode_sys(space_init, t_grid, eta_variance_u, eta_variance_v, params_for_ivp, num_trajectories=10):
    """
    Simulate multiple trajectories.
    """
    traj_x_vals, traj_y_vals = [], []
    for _ in range(num_trajectories):
        eta_init = [np.random.normal(0, np.sqrt(eta_variance_u)), np.random.normal(0, np.sqrt(eta_variance_v))]
        y_init = space_init + eta_init
        sol_xc = solve_ivp(ode_system, [t_grid[0], t_grid[-1]], y_init, t_eval=t_grid, args=(params_for_ivp,), method="RK45")
        traj_x_vals.append(sol_xc.y[0])
        traj_y_vals.append(sol_xc.y[1])
        print("Iteration "+str(_)+" computed")
    return traj_x_vals, traj_y_vals 


def calc_traj(u_m, v_m, space_init, t_grid, params):
    try:
        eta_variance_u = params['eta_variance_u']
        eta_variance_v = params['eta_variance_v']
        lon_arr = params['lon_arr']
        lat_arr = params['lat_arr']
        start_t = params['start_t']
        print(drifter_ds.time.values)
        params_for_ivp = {k: v for k, v in params.items() if k.startswith(('a', 'b', 'c'))} 
        u_m = np.nan_to_num(u_m)
        v_m = np.nan_to_num(v_m)
        u_mean = np.mean(u_m, axis=0) 
        v_mean = np.mean(v_m, axis=0)     
        u_m_whole_interp = RegularGridInterpolator((lat_arr, lon_arr), u_mean, bounds_error=False, fill_value=None)
        v_m_whole_interp = RegularGridInterpolator((lat_arr, lon_arr), v_mean, bounds_error=False, fill_value=None)
        u_m_interp, v_m_interp = prepare_interpolators(u_m, v_m, t_grid, lat_arr, lon_arr, start_t)
        # Derivative of u'=u-avg should be the same as derivative of u since avg is constant
        u_double_prime = calculate_acceleration(t_grid, u_m[start_t:])
        v_double_prime = calculate_acceleration(t_grid, v_m[start_t:])
        #Fill in the missing values of u_double_prime and v_double_prime with 0
        u_double_prime_interp, v_double_prime_interp = prepare_interpolators(u_double_prime, v_double_prime, t_grid, lat_arr, lon_arr, 0)
        print(u_m_interp([0, 35, -125]), v_m_interp([0, 35, -125]), u_double_prime_interp([0, 35, -125]), v_double_prime_interp([0, 35, -125]), )
        params_for_ivp.update({
            'u_m_whole_interp': u_m_whole_interp, 'v_m_whole_interp': v_m_whole_interp, 'u_m_interp': u_m_interp, 'v_m_interp': v_m_interp,
            'u_double_prime_interp': u_double_prime_interp, 'v_double_prime_interp': v_double_prime_interp
        })

        traj_x, traj_y = solve_ode_sys(space_init, t_grid, eta_variance_u, eta_variance_v, params_for_ivp)
        return traj_x, traj_y
    except ValueError as e:
        print("Error in trajectory", e)
        return None

def adv_sys(t, y, u_adv_interp, v_adv_interp):
    x0, y0 = y
    dx_dt = u_adv_interp([t, y0, x0])[0]
    dy_dt = v_adv_interp([t, y0, x0])[0]
    return [dx_dt, dy_dt]

def calculate_adv_traj(u_m, v_m, space_init, t_grid, lat_arr, lon_arr, start_t):
    u_adv_interp, v_adv_interp = prepare_interpolators(u_m, v_m, t_grid, lat_arr, lon_arr, start_t)
    sol_adv = solve_ivp(adv_sys, [t_grid[0], t_grid[-1]], space_init, t_eval=t_grid, args=(u_adv_interp, v_adv_interp), method="RK45")
    adv_x = sol_adv.y[0]
    adv_y = sol_adv.y[1]
    return adv_x, adv_y

def main(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v):
    start_date = np.datetime64('2022-05-12T23:07:27.000000000')
    end_date = np.datetime64('2023-05-12T23:07:27.000000000')
    start_time = np.datetime64('2023-04-23T00:00:00.000000000') #chosen based on the drifter data
    velocity_ds_processed = preprocess_velocity_data(velocity_ds, start_date, end_date)
    lon_arr = velocity_ds_processed.longitude.values
    lat_arr = velocity_ds_processed.latitude.values
    lsgs_params = calculate_lsgs_params(sigma_r_u, sigma_r_v, tau_r_u, tau_r_v, velocity_ds_processed, lon_arr, lat_arr, start_time)
    start_t = lsgs_params['start_t']
    t_grid = velocity_ds_processed['time'][start_t:].values
    t_grid = (t_grid - t_grid[0]) / np.timedelta64(1, 'D')
    u_m = velocity_ds_processed['uo'].values
    v_m = velocity_ds_processed['vo'].values
    
    drifter_lon = drifter_ds['longitude'].values
    drifter_lat = drifter_ds['latitude'].values

    start_lon = drifter_lon[0]
    start_lat = drifter_lat[0]
    
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    #convert using the convert_velocity function and lat_arr
    u_m_conv, v_m_conv = convert_velocity((u_m, v_m), lat_grid)
    adv_sol_x, adv_sol_y = calculate_adv_traj(u_m_conv, v_m_conv, [start_lon, start_lat], t_grid, lat_arr, lon_arr, start_t)
    x_sol, y_sol = calc_traj(u_m_conv, v_m_conv, [start_lon, start_lat], t_grid, lsgs_params)
    # Plot the trajectories
    coastline = define_constants()
    # Custom colormap for shading the background nutrient gradient in a more visible way 
    colors = [(0.98, 0.98, 0.98),  # Very light gray
          (0.95, 0.95, 0.95),  # Slightly darker
          (0.9, 0.9, 0.9),  # Light gray
          (0.85, 0.85, 0.85),     # Medium gray
          (0.7, 0.7, 0.7),     # Dark gray
          (0.5, 0.5, 0.5)]     # Very dark gray
    custom_cmap = LinearSegmentedColormap.from_list("custom_grey", colors)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-130, -120, 30, 40], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='snow', zorder=150)
    gridlines = ax.gridlines(draw_labels=True)
    gridlines.xlabel_style = {'size': 14}  
    gridlines.ylabel_style = {'size': 14}
    nzero = Nzero(lon_grid, lat_grid, coastline['m'], coastline['b'])
    # Plot the coastline
    ax.plot([coastline['x1'], coastline['x2']], [coastline['y1'], coastline['y2']], transform=ccrs.PlateCarree(), color="coral", linestyle='dashed', zorder=200)
    # Plot normal vector 
    ax.quiver(coastline['midpoint']['x'], coastline['midpoint']['y'], 
          coastline['normal_vector'][0], coastline['normal_vector'][1], 
          transform=ccrs.PlateCarree(), color="coral", zorder=200)
    c = ax.pcolormesh(lon_grid, lat_grid, nzero, transform=ccrs.PlateCarree(), vmin=0, vmax=5, cmap=custom_cmap)
    cbar = plt.colorbar(c, ax=ax, label='$\\mu $ mol m$^{-3}$', fraction=0.046, pad=0.1)
    ax.plot(drifter_lon, drifter_lat, transform=ccrs.PlateCarree(), color="deepskyblue", zorder=100, label="Drifter")
    ax.plot(adv_sol_x, adv_sol_y, transform=ccrs.PlateCarree(), color="gold", zorder=100, label="Satellite")
    for i in range(len(x_sol)):
        ax.plot(x_sol[i], y_sol[i], transform=ccrs.PlateCarree(), color="firebrick", zorder=100, linestyle='dotted')
    # Get the average of the different initializations (10 random intializations)
    traj_lons = np.asarray(x_sol)
    traj_lats = np.asarray(y_sol)
    avg_lons, avg_lats = find_average_trajectory(traj_lons, traj_lats)
    ax.plot(avg_lons, avg_lats, transform=ccrs.PlateCarree(), color="red", zorder=100)
    ax.plot([], [], color="red",label="LSGS (and avg.)")
    ax.legend(loc = 'lower left')
    plt.savefig('TestTrajectoriesAgainstNutrients.png')


if __name__ == "__main__":
    main(0.21750026018038376,0.27096586723921556,0.7722222222222223,0.6981481481481482)