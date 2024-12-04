'''
Calculates the parameters to run the LSGS procedure. 
We assume a multiple timescale decomposition: u=Um+um'+u', where the AVISO is Um+um' and the drifter is u'. 
We interpret the decorrelation timescale to be associated with the full velocity field and the turbulent velocity fluctuation / rms to be specifically associated with the turbulent velocity. 
Then we calculate the drifter sigmas and taus using u' (there is no mean to subtract when considering sigma). 
We calculate the satellite sigmas using um' and the satellite taus using Um+um'. 

'''
import os
import csv
import math
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


DRIFTER_PATH = 'drifter-data/2023' # directory containing all the drifter files 
MIN_TO_SEC = 60
HOUR_TO_SEC = 60 * MIN_TO_SEC
HOUR_PER_DAY = 24
R = 6371000 # Earth's radius in meters approximately 
SECONDS_PER_DAY = HOUR_PER_DAY * HOUR_TO_SEC
DRIFTER_INTERVAL = 5 # in minutes, approx. time between measurements 
SAT_INTERVAL = 6 # in hours, approx. time between satellite measurements
DRIFT_SUBSAMPLE = 72 # subsample the drifter data to match the satellite data, this is given 5 minute drifter and 6 hour satellite for example
SAT_FILENAME = 'drifter-data/calculated_model_params_2023.csv' # name of file to save calculated sat. params in 
DRIFT_FILENAME = 'drifter-data/calculated_drifter_params_2023.csv' # name of file to save calculated drifter params in
VEL_DS_FILENAME = 'dataset-uv-nrt-hourly_1695588955720.nc' # name of the velocity dataset file (including path to it)


def create_velocity_interpolators(satellite_ds, depth_level=0):
    """
    Create time-dependent interpolators for velocity fields.
    depth_level is the index of depth level to use (e.g. 0 for surface, 1 for 15m)

    Returns a tuple(u_interpolator, v_interpolator) for velocity components
    """
    times = satellite_ds.time.values.astype('datetime64[s]').astype('float64')
    lats = satellite_ds.latitude.values
    lons = satellite_ds.longitude.values
    u_data = satellite_ds.uo.isel(depth=depth_level).values
    v_data = satellite_ds.vo.isel(depth=depth_level).values
    #based on the formatting of the AVISO data, depending on the satellite data used, may need to change the order
    u_interpolator = RegularGridInterpolator((times, lats, lons), u_data, bounds_error=False, fill_value=np.nan)
    v_interpolator = RegularGridInterpolator((times, lats, lons), v_data, bounds_error=False, fill_value=np.nan)
    return u_interpolator, v_interpolator


def advection_system(t, y):
    lon, lat = y
    current_time = t_start + t
    u = u_interp([current_time, lat, lon])[0]
    v = v_interp([current_time, lat, lon])[0]
    if np.isnan(u) or np.isnan(v):
        return [0, 0]
    return [u/SECONDS_PER_DAY, v/SECONDS_PER_DAY]  # convert to degrees/second


def calculate_trajectory(drifter_ds, satellite_ds, u_interp, v_interp):
    """
    Calculate satellite trajectory using matched 6-hour timesteps.
    """
    # Get drifter times at 6-hour intervals (subsample drifter data given it's finer resolution than this) in order to have an accurate comparison
    drifter_start = drifter_ds.time.values[0]
    drifter_times = drifter_ds.time.values[::DRIFT_SUBSAMPLE] 
    print(f"Number of 6-hour drifter timesteps: {len(drifter_times)}")
    # Get matching satellite times
    sat_times = satellite_ds.time.values
    mask = (sat_times >= drifter_start) & (sat_times <= drifter_times[-1])
    t_eval = sat_times[mask]
    # If there is a mismatch in the available data (which there probably is) use the shorter amount of the calculation (likely this is the drifter data)
    if len(t_eval) != len(drifter_times):
        min_length = min(len(t_eval), len(drifter_times))
        t_eval = t_eval[:min_length]
        drifter_times = drifter_times[:min_length]
    print(f"Using {len(t_eval)} timesteps for trajectory calculation") # in order to know how much data the parameters are being calculated over
    initial_pos = [float(drifter_ds.longitude.isel(time=0)),float(drifter_ds.latitude.isel(time=0))] # get initial position from drifter
    t_start = t_eval[0].astype('datetime64[s]').astype('float64')
    t_seconds = np.array([t.astype('datetime64[s]').astype('float64') - t_start for t in t_eval])
    sol = solve_ivp(advection_system, [0, t_seconds[-1]], initial_pos, t_eval=t_seconds, method="RK45", rtol=1e-6, atol=1e-6)
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None, None, None 
    return sol.y[0], sol.y[1], t_eval


def compare_drifter_satellite_trajectories(drifter_ds, satellite_ds, depth_level=0):
    """
    Compare trajectories using exact satellite timesteps.
    """
    u_interp, v_interp = create_velocity_interpolators(satellite_ds, depth_level)
    sat_lon, sat_lat, sat_times = calculate_trajectory(drifter_ds, satellite_ds, u_interp, v_interp)
    if sat_lon is None:
        return None
    return {'times': sat_times, 'drifter_lon': drifter_ds.longitude.values, 'drifter_lat': drifter_ds.latitude.values, 'satellite_lon': sat_lon, 'satellite_lat': sat_lat}


def calculate_trajectory_velocities(results):
    """
    Calculate velocities of each (satellite, drifter) trajectories using the full time resolution available / corresponding to the calculated trajectories. 
    """
    sat_vel = np.zeros((len(results['satellite_lon'])-1, 2))
    drifter_vel = np.zeros((len(results['drifter_lon'])-1, 2))
    
    for i in range(len(results['drifter_lon'])-1):
        dlon = results['drifter_lon'][i+1] - results['drifter_lon'][i]
        dlat = results['drifter_lat'][i+1] - results['drifter_lat'][i]
        avg_lat = (results['drifter_lat'][i+1] + results['drifter_lat'][i]) / 2 # to go into the dx which involves latitude
        dx = R * np.cos(np.radians(avg_lat)) * np.radians(dlon)
        dy = R * np.radians(dlat)
        drifter_vel[i,0] = dx / (DRIFTER_INTERVAL*MIN_TO_SEC)
        drifter_vel[i,1] = dy / (DRIFTER_INTERVAL*MIN_TO_SEC)
  
    for i in range(len(results['satellite_lon'])-1):
        dlon = results['satellite_lon'][i+1] - results['satellite_lon'][i]
        dlat = results['satellite_lat'][i+1] - results['satellite_lat'][i]
        avg_lat = (results['satellite_lat'][i+1] + results['satellite_lat'][i]) / 2
        dx = R * np.cos(np.radians(avg_lat)) * np.radians(dlon)
        dy = R * np.radians(dlat)
        sat_vel[i,0] = dx / (SAT_INTERVAL * HOUR_TO_SEC)
        sat_vel[i,1] = dy / (SAT_INTERVAL * HOUR_TO_SEC)
    
    return {
        'drifter_times': results['times'][:-1],
        'satellite_times': results['times'][:-1],  # satellite times are 6-hourly / the original interval
        'drifter_vel': drifter_vel,
        'satellite_vel': sat_vel
    }


def estimate_rms(vel_1, vel_2):
    """
    Estimate the RMS for each component of the velocity. 
    Note that it doesn't matter which order you put the components in because each is computed the same way, 
    just that the corresponding calculations are returned in the same order you put them in.  
    """
    # Calculate rms (root mean square) of velocity fluctuations
    rms_1 = np.sqrt(np.nanmean(vel_1 ** 2))
    rms_2 = np.sqrt(np.nanmean(vel_2 ** 2))
    return rms_1, rms_2


def autocorr(signal):
    """
    Using the unbiased estimator version.
    """
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)
    signal = signal - mean
    # Calculate full correlation
    r = np.correlate(signal, signal, mode='full')
    # Get the central n lags (including zero lag), zero lag is at index n-1 and we want lags 0 to n-1
    r = r[n-1:2*n-1]
    autocorr = r / (var * np.arange(n, 0, -1))
    return autocorr


def find_decorrelation_time(autocorr, dt, threshold=0.367879):
    """
    Find the decorrelation time scale as the time lag where autocorrelation drops below a threshold, default is 1/e (approx.).
    """
    decorrelation_time = np.argmax(autocorr < threshold) # this assumes there is enough data that the autocorr does drop below the threshold
    decorrelation_time_in_days = decorrelation_time*dt / HOUR_PER_DAY
    return decorrelation_time_in_days


def analyze_velocities(all_results, satellite_ds):
    """
    Get the velocity statistics. For the rms, use the best available resolution for each in order to capture that higher resolution should enable more variance. 
    For the decorrelation timescale, need to subsample the drifter data otherwise the measurements will not be comparable, because the drifter one is available at such a finer timescale. 
    The drifter data allows a much more granular calculation of the decorrelation timescale (e.g. in 5 minute increments), which, if used, would not be comparable to that for the satellite data which is only possible in 
    (e.g.) 6 hour increments. The finer resolution is incorporated into the calculation of the velocity field which we then subsample; 
    so subsampling still captures effects of higher resolution without causing the measurements to be incomparable.
    """
    for i, results in enumerate(all_results):
        if results is None:
            continue
            
        print(f"\nAnalyzing trajectory {i+1}")
        
        # Get original velocities
        vel_results = calculate_trajectory_velocities(results)
    
        # Find the mean velocities, which we approximate as the mean over one year from start
        traj_start = results['times'][0]
        one_year_forward = traj_start + np.timedelta64(365, 'D')
        sat_times = satellite_ds.time.values
        year_mask = (sat_times >= traj_start) & (sat_times <= one_year_forward)
        u_mean = np.mean(satellite_ds.uo.isel(depth=0).values[year_mask], axis=0)
        v_mean = np.mean(satellite_ds.vo.isel(depth=0).values[year_mask], axis=0)
        # Make mean velocity interpolators, i.e. only with respect to space and not time
        lat_arr = satellite_ds.latitude.values
        lon_arr = satellite_ds.longitude.values
        u_mean_interp = RegularGridInterpolator((lat_arr, lon_arr), u_mean, bounds_error=False, fill_value=None)
        v_mean_interp = RegularGridInterpolator((lat_arr, lon_arr), v_mean, bounds_error=False, fill_value=None)
        
        sat_lons = results['satellite_lon'][:-1] # get positions for mean removal
        sat_lats = results['satellite_lat'][:-1]
        
        satellite_u = vel_results['satellite_vel'][:,0].copy()  
        satellite_v = vel_results['satellite_vel'][:,1].copy()
        sat_decorr_z = autocorr(satellite_u)
        sat_decorr_m = autocorr(satellite_v)

        # Remove mean from satellite velocities before calculating RMS
        for i in range(len(sat_lons)):
            mean_u = u_mean_interp((sat_lats[i], sat_lons[i]))
            mean_v = v_mean_interp((sat_lats[i], sat_lons[i]))
            satellite_u[i] -= mean_u
            satellite_v[i] -= mean_v

        drifter_rms_z, drifter_rms_m = estimate_rms(drifter_u, drifter_v)
        drifter_u = vel_results['drifter_vel'][::DRIFT_SUBSAMPLE,0] # otherwise the decorrelation timescale can be computed at a much finer scale so wouldn't be comparable
        drifter_v = vel_results['drifter_vel'][::DRIFT_SUBSAMPLE,1]
        drifter_decorr_z = autocorr(drifter_u)
        drifter_decorr_time_z = find_decorrelation_time(drifter_decorr_z, SAT_INTERVAL) # need to use the same sampling interval to get anything comparable
        drifter_decorr_m = autocorr(drifter_v)
        drifter_decorr_time_m = find_decorrelation_time(drifter_decorr_m, SAT_INTERVAL)
        sat_rms_z, sat_rms_m = estimate_rms(satellite_u, satellite_v)
        sat_decorr_time_z = find_decorrelation_time(sat_decorr_z, SAT_INTERVAL)
        sat_decorr_time_m = find_decorrelation_time(sat_decorr_m, SAT_INTERVAL)
   
    return drifter_rms_z, drifter_rms_m, drifter_decorr_time_z, drifter_decorr_time_m, sat_rms_z, sat_rms_m, sat_decorr_time_z, sat_decorr_time_m


def main():
    file_paths = [os.path.join(DRIFTER_PATH, f) for f in os.listdir(DRIFTER_PATH) if os.path.isfile(os.path.join(DRIFTER_PATH, f))]
    var_names = ['sigma_u', 'sigma_v', 'tau_u', 'tau_v']
    velocity_ds = xr.open_dataset(VEL_DS_FILENAME)
    # Calculate the parameters for the drifters
    drift_rms_z_list, drift_rms_m_list, drift_decorr_z_list, drift_decorr_m_list, sat_rms_z_list, sat_rms_m_list, sat_decorr_z_list, sat_decorr_m_list = ([] for _ in range(8))

    for i in tqdm(range(len(file_paths))): 
        drift_ds_val= xr.open_dataset(file_paths[i])
        drift_ds = drift_ds_val.where(drift_ds_val['position_QCflag']==1, drop=True)
        results = compare_drifter_satellite_trajectories(drift_ds, velocity_ds)
        vel_results = calculate_trajectory_velocities(results)
        drift_rms_z, drift_rms_m, drift_decorr_time_z, drift_decorr_time_m, sat_rms_z, sat_rms_m, sat_decorr_time_z, sat_decorr_time_m = analyze_velocities([results], velocity_ds)
        drift_rms_z_list.append(drift_rms_z)
        drift_rms_m_list.append(drift_rms_m)
        drift_decorr_z_list.append(drift_decorr_time_z)
        drift_decorr_m_list.append(drift_decorr_time_m)
        sat_rms_z_list.append(sat_rms_z)
        sat_rms_m_list.append(sat_rms_m)
        sat_decorr_z_list.append(sat_decorr_time_z)
        sat_decorr_m_list.append(sat_decorr_time_m)

    avg_drift_rms_z, avg_drift_rms_m, avg_drift_decorr_z, avg_drift_decorr_m, avg_sat_rms_z, avg_sat_rms_m, avg_sat_decorr_z, avg_sat_decorr_m = map(np.nanmean, [drift_rms_z_list, drift_rms_m_list, drift_decorr_z_list, drift_decorr_m_list, sat_rms_z_list, sat_rms_m_list, sat_decorr_z_list, sat_decorr_m_list])
    print("DRIFTER PARAMS:",avg_drift_rms_z, avg_drift_rms_m, avg_drift_decorr_z, avg_drift_decorr_m)
    print("SAT PARAMS:", avg_sat_rms_z, avg_sat_rms_m, avg_sat_decorr_z, avg_sat_decorr_m)

    values = [avg_sat_rms_z, avg_sat_rms_m, avg_sat_decorr_z, avg_sat_decorr_m]
    with open(SAT_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(var_names)  
        writer.writerow(values)
    values = [avg_drift_rms_z, avg_drift_rms_m, avg_drift_decorr_z, avg_drift_decorr_m]
    with open(DRIFT_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(var_names)  
        writer.writerow(values)
    
    
if __name__ == "__main__":
    main()