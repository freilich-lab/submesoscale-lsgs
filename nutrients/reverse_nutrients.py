'''
Calculates and plots the reversed background nutrient, as in Supplementary Information. Can also be used to compare to the WOA data.
'''
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

filepath = "plots/" # Change this to the path where the plot should be saved in 
plt.rcParams["font.family"] = "serif"
NUTRIENT_DATA_FILE = "data/woa23_all_n04_01.nc" # change this to the path of the nutrient data file

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
    return coastline

def Nzero(x, y, m, b, k=5.5, l=0.3):
    """
    Calculate the gradient value at a point (x, y) based on its distance from a line defined by y = mx + b.
    The gradient decreases linearly with the perpendicular distance from the line.
    - m and b are the slope and intercept of the line 
    - k is the maximum nutrient value (which is on the line)
    - l is the decay rate of the gradient with distance
    k and l were chosen to approximately match the WOA data, but avoid any negative values and to represent a linear decay (the WOA data is not quite linear, but nearly)
    """
    # Calculate the perpendicular distance from the point to the line
    adj_slope = m*2.5
    A = -1/adj_slope
    B = -1
    C = 42-124.5*(1/adj_slope) #approximate, based on the domain
    distance = abs(A*x + B*y + C) / np.sqrt(A**2 + B**2)
    grad = k - l* distance
    # This is in micromol/kg, convert to micromol/m^3 by multiplying by 1025 kg/m^3
    grad = grad * 1025
    return grad

    
def main():
    nutrient_ds = xr.open_dataset(NUTRIENT_DATA_FILE, decode_times=False)
    n = np.nanmean(nutrient_ds['n_an'][0, 0, :, :].values)
    print(n)
    extent = [-140, -120, 28, 40]
    lon = nutrient_ds['lon']
    lat = nutrient_ds['lat']
    coastline = define_constants()
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
    plt.title("Reversed Initial Nutrient Gradient", fontsize=20)
    print(np.nanmin(nzero)) # Check that the minimum value is at least 0
    plt.savefig(filepath+'reversed_initial_nutrient_gradient.png')

    
if __name__ == "__main__":
    main()