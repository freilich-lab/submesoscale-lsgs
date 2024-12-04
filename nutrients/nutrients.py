'''
Calculates and plots the background nutrient, as in Fig. 1. Can also be used to compare to the WOA data.
'''
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

filepath = "plots/" # change this to the path where the plot should be saved in 
NUTRIENT_DATA_FILE = "data/woa23_all_n04_01.nc" # change this to the path of the nutrient data file
plt.rcParams["font.family"] = "serif"

def define_constants():
    # Defining the coastline based on the geographic domain of interest
    coastline = {
        'x1': -124.5, 'y1': 40,
        'x2': -120, 'y2': 34,
    }
    coastline['m'] = (coastline['y2'] - coastline['y1']) / (coastline['x2'] - coastline['x1'])
    coastline['b'] = coastline['y1'] - coastline['m'] * coastline['x1']
    coastline['normal_vector'] = np.array([coastline['m'], -1])
    coastline['normal_vector'] /= np.linalg.norm(coastline['normal_vector'])
    return coastline

def Nzero(x, y, m, b, k=5.5, l=0.25):
    """
    Calculate the gradient value at a point (x, y) based on its distance from a line defined by y = mx + b.
    The gradient decreases linearly with the perpendicular distance from the line.
    - m and b are the slope and intercept of the line 
    - k is the maximum nutrient value (which is on the line)
    - l is the decay rate of the gradient with distance
    k and l were chosen to approximately match the WOA data, but avoid any negative values and to represent a linear decay (the WOA data is not quite linear, but nearly)
    """
    # Calculate the perpendicular distance from the point to the line
    A = m
    B = -1
    C = b
    distance = abs(A*x + B*y + C) / np.sqrt(A**2 + B**2)
    grad = k - l* distance
    # This is in micromol/kg, convert to micromol/m^3 by multiplying by 1025 kg/m^3
    grad = grad * 1025
    return grad

    
def main():
    nutrient_ds = xr.open_dataset(NUTRIENT_DATA_FILE, decode_times=False)
    n = nutrient_ds['n_an'][0, 0, :, :]
    extent = [-140, -120, 28, 40] # domain trajectories will be simulated in
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True)
    lon = nutrient_ds['lon']
    lat = nutrient_ds['lat']
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    c = ax.pcolormesh(lon_grid, lat_grid, n, transform=ccrs.PlateCarree(), vmax=5)
    plt.colorbar(c, orientation='horizontal', label='micromol/kg')
    ax.legend()
    plt.title("Nitrate Objectively Analyzed Climatology", fontsize=20)
    plt.savefig(filepath+'nutrients-woa.png')

    coastline = define_constants()  # get the approximate coastline 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True)
    lon_region = lon.sel(lon=slice(-140, -120))
    lat_region = lat.sel(lat=slice(26, 42))
    lon_region_grid, lat_region_grid = np.meshgrid(lon_region, lat_region)
    nzero = Nzero(lon_region_grid, lat_region_grid, coastline['m'], coastline['b'])
    c = ax.pcolormesh(lon_region_grid, lat_region_grid, nzero, transform=ccrs.PlateCarree(), vmin=0, vmax=5)
    plt.colorbar(c, orientation='horizontal', label='$\\mu $ mol m$^{-3}$')
    ax.legend()
    plt.title("Initial Nutrient Gradient", fontsize=20)
    print(np.nanmin(nzero)) # to verify no negative values 
    plt.savefig(filepath+'initial_nutrient_gradient.png')

if __name__ == "__main__":
    main()