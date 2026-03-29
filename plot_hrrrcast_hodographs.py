import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import Hodograph
import numpy as np
import datetime
from datetime import timedelta
import requests
import os
import traceback
import glob
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
REGION = [-83.5, -75.5, 32.5, 37.5]    
OUTPUT_DIR = "images"
GRID_SPACING = 25              
BOX_SIZE = 100000              
REQUESTED_LEVELS = [1000, 925, 850, 700, 500, 250]

# --- SPC HREF STYLE CAPE CONFIGURATION (0-100 White) ---
CAPE_LEVELS = [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 9000]

CAPE_COLORS = [
    '#ffffff', # 0-100: White
    '#e1e1e1', '#c0c0c0', '#808080', '#626262', # Grays
    '#9dc2ff', '#4169e1', '#0000cd', # Blues
    '#00ff00', '#008000', # Greens
    '#ffff00', # Yellow
    '#ff8c00', # Orange
    '#ff0000', # Red
    '#ff00ff', # Magenta
    '#800080'  # Purple
]

CAPE_CMAP = mcolors.ListedColormap(CAPE_COLORS)
CAPE_NORM = mcolors.BoundaryNorm(CAPE_LEVELS, CAPE_CMAP.N)

def get_latest_valid_run():
    now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    # Check up to 24 hours back. We'll just look for a run that has at least f03 
    # so we don't accidentally grab a completely useless 1-hour run.
    for hours_back in range(2, 25):
        check_time = now - datetime.timedelta(hours=hours_back)
        run = check_time.strftime('%H')
        date_str = check_time.strftime('%Y%m%d')
        
        base_url = f"https://noaa-gsl-experimental-pds.s3.amazonaws.com/HRRRCast/{date_str}/{run}"
        test_file = f"hrrrcast.avg.t{run}z.pgrb2.f03"
        test_url = f"{base_url}/{test_file}"
        
        try:
            r = requests.get(test_url, stream=True, timeout=10)
            if r.status_code == 200:
                r.close()
                print(f"Found run with at least 3 hours of data: {date_str} {run}Z (Lag: {hours_back} hours)")
                return date_str, run, check_time
            r.close()
        except requests.RequestException:
            pass
            
    print("Could not find ANY decent runs in the last 24 hours. Chaos reigns.")
    fallback = now - datetime.timedelta(hours=5)
    return fallback.strftime('%Y%m%d'), fallback.strftime('%H'), fallback

def download_file(date_str, run, fhr):
    base_url = f"https://noaa-gsl-experimental-pds.s3.amazonaws.com/HRRRCast/{date_str}/{run}"
    filename = f"hrrrcast.avg.t{run}z.pgrb2.f{fhr:02d}" 
    url = f"{base_url}/{filename}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    if os.path.exists(filename):
        try: os.remove(filename)
        except: pass

    try:
        print(f"  -> Attempting download of f{fhr:02d}...")
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404: 
                print(f"     [MISSING] f{fhr:02d} does not exist on S3.")
                return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return filename
    except Exception as e:
        print(f"     [ERROR] Download failed for f{fhr:02d}: {e}")
        return None

def get_segment_color(p_start, p_end):
    avg_p = (p_start + p_end) / 2.0
    if avg_p >= 850: return 'magenta'
    elif 700 <= avg_p < 850: return 'red'
    elif 500 <= avg_p < 700: return 'green'
    else: return 'gold'

def plot_colored_hodograph(ax, u, v, levels):
    safe_len = min(len(u), len(levels))
    for k in range(safe_len - 1):
        color = get_segment_color(levels[k], levels[k+1])
        ax.plot([u[k], u[k+1]], [v[k], v[k+1]], color=color, linewidth=2.5)

def cleanup_old_runs(current_date, current_run):
    prefix = f"hrrrcast_hodo_cape_{current_date}_{current_run}z"
    for f in glob.glob(os.path.join(OUTPUT_DIR, "hrrrcast_hodo_cape_*.png")):
        if not os.path.basename(f).startswith(prefix):
            try: os.remove(f)
            except: pass

def process_forecast_hour(date_obj, date_str, run, fhr):
    grib_file = download_file(date_str, run, fhr)
    
    if not grib_file: 
        return False # Return False so we don't log this as a successful hour

    try:
        ds_u = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
        ds_v = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
        ds_wind = xr.merge([ds_u, ds_v])
        ds_cape = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'cape', 'typeOfLevel': 'surface'}})

        fig = plt.figure(figsize=(16, 12), facecolor='white')
        fig.subplots_adjust(bottom=0.18, top=0.93)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)

        # CAPE
        if ds_cape is not None:
            cape_vals = np.nan_to_num(ds_cape['cape'].values.squeeze(), nan=0.0)
            cape_vals[cape_vals < 100] = 0
            
            lats = ds_cape.latitude.values
            lons = ds_cape.longitude.values
            lons = (lons + 180) % 360 - 180 

            cape_plot = ax.pcolormesh(lons, lats, cape_vals, 
                                      cmap=CAPE_CMAP, norm=CAPE_NORM,
                                      shading='auto', transform=ccrs.PlateCarree(), alpha=0.6)
            
            ax_cbar_cape = fig.add_axes([0.15, 0.10, 0.7, 0.02]) 
            cb_cape = plt.colorbar(cape_plot, cax=ax_cbar_cape, orientation='horizontal')
            
            spc_ticks = [100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 9000]
            cb_cape.set_ticks(spc_ticks)
            cb_cape.ax.set_xticklabels([str(t) for t in spc_ticks], fontsize=10)
            cb_cape.set_label('Surface-based CAPE (J/kg)', fontsize=12, weight='bold')

        # HODOGRAPHS
        legend_elements = [
            mlines.Line2D([], [], color='magenta', lw=3, label='0-1.5 km'),
            mlines.Line2D([], [], color='red', lw=3, label='1.5-3 km'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km'),
            mlines.Line2D([], [], color='black', lw=0.5, alpha=0.5, label='Rings: 20 kts') 
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Hodograph Layers", framealpha=0.9).set_zorder(100)

        ds_wind_filtered = ds_wind.sel(isobaricInhPa=REQUESTED_LEVELS, method='nearest')
        
        u_kts = ds_wind_filtered['u'].metpy.convert_units('kts').values.squeeze()
        v_kts = ds_wind_filtered['v'].metpy.convert_units('kts').values.squeeze()
        
        lons_wind = ds_wind_filtered.longitude.values
        lats_wind = ds_wind_filtered.latitude.values
        
        for i in range(0, lons_wind.shape[0], GRID_SPACING):
            for j in range(0, lons_wind.shape[1], GRID_SPACING):
                
                if np.isnan(u_kts[:, i, j]).any(): continue
                
                lon_val = lons_wind[i, j]
                lon_val = lon_val - 360 if lon_val > 180 else lon_val
                
                if not (REGION[0] < lon_val < REGION[1] and REGION[2] < lats_wind[i, j] < REGION[3]): continue
                
                proj_pnt = ax.projection.transform_point(lons_wind[i, j], lats_wind[i, j], ccrs.PlateCarree())
                sub_ax = ax.inset_axes([proj_pnt[0]-BOX_SIZE/2, proj_pnt[1]-BOX_SIZE/2, BOX_SIZE, BOX_SIZE], transform=ax.transData, zorder=20)
                h = Hodograph(sub_ax, component_range=80)
                h.add_grid(increment=20, color='black', alpha=0.3, linewidth=0.5)
                
                plot_colored_hodograph(h.ax, u_kts[:, i, j], v_kts[:, i, j], REQUESTED_LEVELS)
                sub_ax.axis('off')

        valid_time = date_obj + timedelta(hours=fhr)
        valid_str = valid_time.strftime("%a %H:%00Z")
        plt.suptitle(f"HRRRCast CAPE + Hodographs | Run: {date_str} {run}Z | Valid: {valid_str} (f{fhr:02d})", 
                     fontsize=20, weight='bold', y=0.98)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename_png = f"{OUTPUT_DIR}/hrrrcast_hodo_cape_{date_str}_{run}z_f{fhr:02d}.png"
        plt.savefig(filename_png, bbox_inches='tight', dpi=100) 
        print(f"     [SUCCESS] Saved image for f{fhr:02d}")
        plt.close(fig)
        return True # Successfully processed!

    except Exception: 
        print(f"     [ERROR] Failed to process F{fhr}:")
        traceback.print_exc()
        return False
    finally:
        if grib_file and os.path.exists(grib_file): 
            try: os.remove(grib_file)
            except: pass

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_valid_run()
    print(f"\n=========================================")
    print(f"STARTING PROCESSING FOR RUN: {date_str} {run}Z")
    print(f"=========================================\n")
    
    # Array to track which hours actually survived the gauntlet
    successful_fhrs = []
    
    for fhr in range(1, 49): 
        success = process_forecast_hour(date_obj, date_str, run, fhr)
        if success:
            successful_fhrs.append(fhr)
    
    cleanup_old_runs(date_str, run)
    
    # Write dynamic configuration file for HTML
    print("\nWriting config.js for web front-end...")
    with open("config.js", "w") as f:
        f.write(f'const RUN_DATE = "{date_str}";\n')
        f.write(f'const RUN_HOUR = "{run}";\n')
        # This writes something like: const AVAILABLE_HOURS = [1, 2, 3, 4...];
        f.write(f'const AVAILABLE_HOURS = {successful_fhrs};\n')

    print(f"Done! Successfully plotted {len(successful_fhrs)} frames out of 48.")
