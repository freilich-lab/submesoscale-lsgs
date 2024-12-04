"""
Computes the flux along generated trajectories using the logistic model, for the whole sweep of lambda values (a different value in a dictionary for each lambda) and saves the results.
"""
### IMPORTS ###
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
import pickle
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import warnings

# Set matplotlib font
plt.rcParams["font.family"] = "serif"

### PARAMETERS ###
delta_t = 6*60*60  # six hours in seconds
start = 0.005  # start of lambda range in day^{-1} 
end = 2  # end of lambda range in day^{-1}
num_values = 20 # number of lambda values
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600

# Time range: overall year of interest (for purposes of calculating turbulent velocity)
START_DATE = np.datetime64('2022-05-12T23:07:27.000000000')
END_DATE = np.datetime64('2023-05-12T23:07:27.000000000')
# Corresponding to what the start date of the trajectories should actually be (for purposes of running the trajectories over just a season)
START_TIME = np.datetime64('2023-04-11T00:00:00.000000000')


velocity_ds = xr.open_dataset('aviso_data.nc')
lsgs_data_path = 'lsgs-runs/'
adv_data_path = 'adv-runs/'
save_path_lsgs = 'lsgs-runs-flux/'
save_path_adv = 'adv-runs-flux/'

# DEFINE ORIGINAL NUTRIENT GRADIENT 
def Nzero(x, y, m, b, k=5.5, l=0.25):
    """
    Calculate the gradient value at a point (x, y) based on its distance from a line defined by y = mx + b.
    The gradient decreases linearly with the perpendicular distance from the line.
    - m and b are the slope and intercept, respectively
    - k is the maximum value (which is on the line)
    - l is the decay rate of the gradient with distance
    """
    A = m
    B = -1
    C = b
    distance = abs(A*x + B*y + C) / np.sqrt(A**2 + B**2)
    grad = k - l* distance
    # This is in micromol/kg, convert to micromol/m^3 by multiplying by 1025 kg/m^3
    grad = grad * 1025
    return grad

# Comment out the above and uncomment out the below function in order to have the reversed nutrient gradient instead 
# DEFINE ORIGINAL NUTRIENT GRADIENT (REVERSED VERSION -- DECAY IN Y)
# def Nzero(x, y, m, b, k=5.5, l=0.3):
#     """
#     Reversed version.
#     """
#     adj_slope = m*2.5
#     A = -1/adj_slope
#     B = -1
#     C = 42-124.5*(1/adj_slope)
#     distance = abs(A*x + B*y + C) / np.sqrt(A**2 + B**2)
#     grad = k - l* distance
#     #This is in micromol/kg, convert to micromol/m^3 by multiplying by 1025 kg/m^3
#     grad = grad * 1025
#     return grad

def preprocess_velocity_data(vel_ds):
    return vel_ds.sel(time=slice(START_DATE, END_DATE), longitude=slice(-136, -120), latitude=slice(32,44), depth=0)

def reaction_term(t, P, lambda_param, x_t, y_t, coastline):
    m = coastline['m']
    b = coastline['b']
    x_val = x_t(t)
    y_val = y_t(t)
    # Calculate the baseline nutrient concentration / carrying capacity based on the current position
    nzero = Nzero(x_val, y_val, m, b)
    dPdt = lambda_param * P*(1 - (P / nzero))
    return dPdt

def define_constants():
    # Defining the coastline 
    coastline = {'x1': -124.5, 'y1': 40,'x2': -120, 'y2': 34}
    coastline['m'] = (coastline['y2'] - coastline['y1']) / (coastline['x2'] - coastline['x1'])
    coastline['b'] = coastline['y1'] - coastline['m'] * coastline['x1']
    coastline['normal_vector'] = np.array([coastline['m'], -1])
    coastline['normal_vector'] /= np.linalg.norm(coastline['normal_vector'])
    lambda_day = np.logspace(np.log10(start), np.log10(end), num_values)
    lamdas = lambda_day / HOURS_PER_DAY
    return coastline, lamdas

def get_mean_interps():
    velocity_ds_processed = preprocess_velocity_data(velocity_ds)
    u_m = velocity_ds_processed['uo'].values
    v_m = velocity_ds_processed['vo'].values
    lat_arr = velocity_ds_processed['latitude'].values
    lon_arr = velocity_ds_processed['longitude'].values
    u_m = np.nan_to_num(u_m)
    v_m = np.nan_to_num(v_m)
    u_mean = np.mean(u_m, axis=0) 
    v_mean = np.mean(v_m, axis=0)
    u_m_whole_interp = RegularGridInterpolator((lat_arr, lon_arr), u_mean, bounds_error=False, fill_value=None)
    v_m_whole_interp = RegularGridInterpolator((lat_arr, lon_arr), v_mean, bounds_error=False, fill_value=None)
    return u_m_whole_interp, v_m_whole_interp

def calc_along_trajectory(key, val, u_vel, v_vel, u_mean, v_mean, coastline, lamdas):
    '''
    Calculates the flux along a trajectory, which will then be identified in the dictionary by the starting point of the trajectory.
    A flux value (this is calculated with respect to the anomalies as in the paper) is calculated for each lambda value.
    '''
    try:
        lon = val[0]
        lat = val[1]
        u_vels = u_vel[key]
        v_vels = v_vel[key]
        lon_init = key[0]
        lat_init = key[1]
        times = np.linspace(0, delta_t * len(lon), len(lon))
        x_t = interp1d(times, lon, kind='linear', fill_value="extrapolate")
        y_t = interp1d(times, lat, kind='linear', fill_value="extrapolate")
        anomalies = {}
        flux = {}
        for lamda in lamdas:
            lamda_s = lamda / SECONDS_PER_HOUR
            Nzero_start =Nzero(lon_init, lat_init, coastline['m'], coastline['b']) 
            sol = solve_ivp(reaction_term, [0, delta_t * len(lon)], [Nzero_start],
                            args=(lamda_s, x_t, y_t, coastline), method="RK45", t_eval=times)
            N_t = sol.y[0]
            anom = N_t - Nzero(lon, lat, coastline['m'], coastline['b'])
            anomalies[lamda] = anom
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            u_mean_values = u_mean(np.array([lat_flat, lon_flat]).T)
            v_mean_values = v_mean(np.array([lat_flat, lon_flat]).T)
            uflux = anom[1:] * (u_vels[:len(N_t) - 1] - u_mean_values[:len(N_t) - 1])
            vflux = anom[1:] * (v_vels[:len(N_t) - 1] - v_mean_values[:len(N_t) - 1])
            flux[lamda] = uflux * coastline['normal_vector'][0] + vflux * coastline['normal_vector'][1]
        return (lon_init, lat_init), anomalies, flux
    except Exception as e:
        return None


def main(file_ext):
    with open(lsgs_data_path + file_ext + '-trajs.pkl', 'rb') as f: # load trajectories to compute flux along
        lsgs_traj = pickle.load(f)
    with open(lsgs_data_path + file_ext + '-u-vel.pkl', 'rb') as f: # load the corresponding velocity data
        u_vel_lsgs = pickle.load(f)
    with open(lsgs_data_path + file_ext + '-v-vel.pkl', 'rb') as f:
        v_vel_lsgs = pickle.load(f)
    
    anom_lsgs = {}
    flux_lsgs = {}

    coastline, lamdas= define_constants()
    u_mean_interp, v_mean_interp = get_mean_interps()

    #bad_trajs = 0 #to count the number of bad trajectories ie ones that leave the domain
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_along_trajectory, key, val, u_vel_lsgs, v_vel_lsgs, u_mean_interp, v_mean_interp, coastline, lamdas) for key, val in lsgs_traj.items()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                           desc="Computing Flux along Trajectories", file = sys.stdout, position = 0, leave = True):
            if future.result() is not None:
                result = future.result()
                space_init, anom, flux = result
                anom_lsgs[space_init] = anom
                flux_lsgs[space_init] = flux
            #else:
                #bad_trajs += 1
    #print(f"LSGS trajectories with errors: {bad_trajs}")
    
    with open(save_path_lsgs + file_ext + '-anom.pkl', 'wb') as pkl_file:
        pickle.dump(anom_lsgs, pkl_file)
    with open(save_path_lsgs + file_ext + '-flux.pkl', 'wb') as pkl_file:
        pickle.dump(flux_lsgs, pkl_file)
    print("LSGS results saved")

    # Uncomment the below to find the fluxes for the advected (non-LSGS) trajectories instead
    # with open(adv_data_path + file_ext + '-trajs.pkl', 'rb') as f:
    #     adv_traj = pickle.load(f)
    # with open(adv_data_path + file_ext + '-u-vel.pkl', 'rb') as f:
    #     u_vel_adv = pickle.load(f)
    # with open(adv_data_path + file_ext + '-v-vel.pkl', 'rb') as f:
    #     v_vel_adv = pickle.load(f)

    # anom_adv = {}
    # flux_adv = {}
    
    #calc_along_trajectory((-135.625, 43.375), adv_traj[(-135.625, 43.375)], u_vel_adv, v_vel_adv, u_mean_interp, v_mean_interp, coastline, lamdas)
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(calc_along_trajectory, key, val, u_vel_adv, v_vel_adv, u_mean_interp, v_mean_interp, coastline, lamdas) for key, val in adv_traj.items()]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
    #                        desc="Computing Adv Trajectories", file = sys.stdout, position = 0, leave = True):
    #         if future.result() is not None:
    #             result = future.result()
    #             space_init, anom, flux = result
    #             anom_adv[space_init] = anom
    #             flux_adv[space_init] = flux
    
    # with open(save_path_adv + file_ext + '-anom.pkl', 'wb') as pkl_file:
    #     pickle.dump(anom_adv, pkl_file)
    # with open(save_path_adv + file_ext + '-flux.pkl', 'wb') as pkl_file:
    #     pickle.dump(flux_adv, pkl_file)
    # print("Adv results saved")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute LSGS trajectories')
    parser.add_argument('sig_u', type=float, help='Realistic/true zonal turbulent velocity fluctuations')
    parser.add_argument('sig_v', type=float, help='Realistic/true meridional turbulent velocity fluctuations')
    parser.add_argument('tau_u', type=float, help='Realistic/true zonal decorrelation timescale')
    parser.add_argument('tau_v', type=float, help='Realistic/true meridional decorrelation timescale)
    args = parser.parse_args()
    args.sig_u = round(args.sig_u, 3)
    args.sig_v = round(args.sig_v, 3)
    args.tau_u = round(args.tau_u, 3)
    args.tau_v = round(args.tau_v, 3)
    file_ext = str(args.sig_u) + '-' + str(args.sig_v) + '-' + str(args.tau_u) + '-' + str(args.tau_v)
    main(file_ext)