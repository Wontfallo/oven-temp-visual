import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For 2D plots and some colormap utilities
from scipy.interpolate import griddata
from mayavi import mlab # The main Mayavi module for scripting
mlab.options.backend ='pyside2'


# --- Configuration ---
csv_filename = "oven_data.csv"
ANIMATION_INTERVAL_MS = 250  # Increased for potentially heavier rendering
MAX_FRAMES_FOR_DEMO = None    # For faster testing, set to None for all frames

# --- Sensor Coordinate Definitions ---
x_coords_map = {'left': 2, 'center': 7, 'right': 12}
y_coords_map = {'front': 2, 'center': 7, 'back': 12}
z_coords_map = {'bottom': 2, 'middle': 7, 'top': 12}

sensor_positions_map = {
    '1': (x_coords_map['left'], y_coords_map['front'], z_coords_map['bottom']),
    '2': (x_coords_map['left'], y_coords_map['back'], z_coords_map['bottom']),
    '3': (x_coords_map['right'], y_coords_map['back'], z_coords_map['bottom']),
    '4': (x_coords_map['right'], y_coords_map['front'], z_coords_map['bottom']),
    '5': (x_coords_map['center'], y_coords_map['center'], z_coords_map['bottom']),
    '6': (x_coords_map['left'], y_coords_map['front'], z_coords_map['middle']),
    '7': (x_coords_map['left'], y_coords_map['back'], z_coords_map['middle']),
    '8': (x_coords_map['right'], y_coords_map['back'], z_coords_map['middle']),
    '9': (x_coords_map['right'], y_coords_map['front'], z_coords_map['middle']),
    '10': (x_coords_map['center'], y_coords_map['center'], z_coords_map['middle']),
    '11': (x_coords_map['left'], y_coords_map['front'], z_coords_map['top']),
    '12': (x_coords_map['left'], y_coords_map['back'], z_coords_map['top']),
    '13': (x_coords_map['right'], y_coords_map['back'], z_coords_map['top']),
    '14': (x_coords_map['right'], y_coords_map['front'], z_coords_map['top']),
    '15': (x_coords_map['center'], y_coords_map['center'], z_coords_map['top'])
}
sensor_column_names = [str(i) for i in range(1, 16)]
sensor_points_xyz_np = np.array([sensor_positions_map[s] for s in sensor_column_names])
sensor_x = sensor_points_xyz_np[:, 0]
sensor_y = sensor_points_xyz_np[:, 1]
sensor_z = sensor_points_xyz_np[:, 2]

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

for col in sensor_column_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df_cleaned = df.dropna(subset=sensor_column_names).reset_index(drop=True)

if df_cleaned.empty:
    print("Error: No valid numeric temperature data found after cleaning.")
    exit()

all_sensor_temps = df_cleaned[sensor_column_names].values.flatten()
min_temp_global = np.nanmin(all_sensor_temps)
max_temp_global = np.nanmax(all_sensor_temps)
print(f"Global temperature range for sensors: {min_temp_global:.2f}°C to {max_temp_global:.2f}°C")

# --- Grid for Interpolation ---
grid_res_x_pts, grid_res_y_pts, grid_res_z_pts = 30, 30, 30
xi = np.linspace(0, 14, grid_res_x_pts)
yi = np.linspace(0, 14, grid_res_y_pts)
zi = np.linspace(0, 14, grid_res_z_pts)
grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')


# --- Mayavi 3D Animated Volumetric Plot with Isosurfaces ---
def plot_animated_mayavi_volume_iso(data_df):
    num_frames = len(data_df) if MAX_FRAMES_FOR_DEMO is None else min(len(data_df), MAX_FRAMES_FOR_DEMO)
    
    # Create the Mayavi scene
    fig_mlab = mlab.figure(bgcolor=(0.15, 0.15, 0.15), size=(1000, 800)) # Darker grey
    fig_mlab.scene.anti_aliasing_frames = 8 # For smoother visuals

    # Initial data interpolation for the first frame
    temps_at_time_0 = data_df.loc[0, sensor_column_names].values.astype(float)
    volumetric_temp_data = griddata(sensor_points_xyz_np, temps_at_time_0,
                                    (grid_x, grid_y, grid_z), method='linear', 
                                    fill_value=min_temp_global) # Fill with a value

    src = mlab.pipeline.scalar_field(grid_x, grid_y, grid_z, volumetric_temp_data)
    
    # --- Volume Rendering ---
    vol = mlab.pipeline.volume(src, vmin=min_temp_global, vmax=max_temp_global)
    # Adjust volume colormap and opacity transfer function (OTF)
    # This is CRITICAL for good visuals.
    # A common approach is to make lower values more transparent.
    from tvtk.util import traits
    vol.volume_property.set_color(traits.ctf.कूलवॉर्म()) # Using a Mayavi CTF
    
    # Opacity: make it somewhat visible across the range, more opaque at higher values
    # Points are (scalar_value, opacity_value)
    # Normalize scalar_value to 0-1 range if your temperatures are not already scaled.
    # For simplicity, assuming temps are somewhat spread.
    # This creates a ramp: more opaque for higher temperatures.
    # This is a common point of failure if not set correctly for your data's range.
    ctf = vol.module_manager.scalar_lut_manager.ctf
    ctf.alpha = [0.0, 0.0, 0.05, 0.1, 0.3, 0.5, 0.8] # Opacity points
    # You might need to manually adjust these based on your data range via the GUI first
    # and then hardcode good values. Or normalize your data from 0-1.
    # For now, let's try a simple ramp:
    otf = vol.volume_property.get_scalar_opacity()
    otf.remove_all_points()
    otf.add_point(min_temp_global, 0.0)
    otf.add_point(min_temp_global + (max_temp_global - min_temp_global) * 0.2, 0.01)
    otf.add_point(min_temp_global + (max_temp_global - min_temp_global) * 0.5, 0.05)
    otf.add_point(min_temp_global + (max_temp_global - min_temp_global) * 0.8, 0.2)
    otf.add_point(max_temp_global, 0.6)


    # --- Isosurfaces ---
    # Example: three isosurfaces
    iso_contour_values = np.linspace(min_temp_global + (max_temp_global-min_temp_global)*0.25, 
                                     max_temp_global - (max_temp_global-min_temp_global)*0.25, 3)
    iso_surfaces = []
    for val in iso_contour_values:
        iso = mlab.pipeline.iso_surface(src, contours=[val], opacity=0.3, colormap='viridis')
        iso_surfaces.append(iso)

    # --- Sensor points ---
    sensor_plot = mlab.points3d(sensor_x, sensor_y, sensor_z, 
                                scale_factor=0.5, color=(0.8, 0.8, 0.2)) # Yellowish
    sensor_labels = []
    for i, name in enumerate(sensor_column_names):
        label = mlab.text3d(sensor_x[i]+0.2, sensor_y[i]+0.2, sensor_z[i]+0.2, 
                            str(name), scale=0.3, color=(1,1,1))
        sensor_labels.append(label)


    mlab.scalarbar(vol, title='Volume Temp (°C)', orientation='vertical', nb_labels=5)
    # mlab.colorbar(iso_surfaces[0], title='Isosurface Temp (°C)', orientation='horizontal') # Colorbar for one iso as example
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', ranges=[0,14,0,14,0,14], color=(1,1,1))
    mlab.outline(color=(0.7,0.7,0.7))
    title_obj = mlab.title("Animated Volumetric Temperature Profile", height=0.95, color=(1,1,1), size=0.5)
    time_text_obj = mlab.text(0.02, 0.9, "Time: 0.0s", width=0.35, color=(1,1,1))
    oven_text_obj = mlab.text(0.02, 0.85, "Oven: 0.0°C", width=0.35, color=(1,1,1))

    @mlab.animate(delay=ANIMATION_INTERVAL_MS)
    def anim_loop():
        for frame_idx in range(num_frames):
            temps_at_time = data_df.loc[frame_idx, sensor_column_names].values.astype(float)
            new_volumetric_data = griddata(sensor_points_xyz_np, temps_at_time,
                                           (grid_x, grid_y, grid_z), method='linear', fill_value=min_temp_global)
            
            src.mlab_source.scalars = new_volumetric_data # Update the data source
            
            # Update text annotations
            time_s = data_df.loc[frame_idx, 'Time_(s)']
            oven_mon_temp = data_df.loc[frame_idx, 'Oven_Mon']
            time_text_obj.text = f"Time: {time_s:.1f}s"
            oven_text_obj.text = f"Oven: {oven_mon_temp:.2f}°C"
            
            fig_mlab.scene.render()
            yield

    ani_instance = anim_loop()
    mlab.show()
    return ani_instance


# --- Mayavi Static Multiple Isosurfaces Plot (Refined) ---
def plot_mayavi_static_isosurfaces_topographic(data_df, time_index, num_isosurfaces=5):
    mlab.figure(bgcolor=(0.95, 0.95, 0.95), fgcolor=(0,0,0), size=(900, 750)) # Lighter background
    mlab.clf()

    temps_at_time = data_df.loc[time_index, sensor_column_names].values.astype(float)
    volumetric_temp_data = griddata(sensor_points_xyz_np, temps_at_time,
                                     (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan)

    min_t_step = np.nanmin(volumetric_temp_data)
    max_t_step = np.nanmax(volumetric_temp_data)

    if np.isnan(min_t_step) or np.isnan(max_t_step) or min_t_step >= max_t_step - 1: # Ensure a valid range
        print(f"Cannot generate useful isosurfaces for time index {time_index}.")
        print(f"Interpolated temp range: {min_t_step} to {max_t_step}")
        return

    iso_levels = np.linspace(min_t_step + (max_t_step - min_t_step) * 0.1, # Avoid extreme edges
                             max_t_step - (max_t_step - min_t_step) * 0.1,
                             num_isosurfaces)
    iso_levels = np.round(iso_levels, 1)
    print(f"Plotting isosurfaces at temperatures: {iso_levels}°C")
    
    src = mlab.pipeline.scalar_field(grid_x, grid_y, grid_z, volumetric_temp_data)
    
    # Use a sequential colormap like 'YlOrRd' or 'Reds' for temperature
    cmap = plt.cm.get_cmap('YlOrRd', num_isosurfaces + 2) # +2 for better color spread
    
    for i, level in enumerate(iso_levels):
        iso = mlab.pipeline.iso_surface(src, contours=[level], opacity=0.3 + i * (0.5/num_isosurfaces))
        # Color the surface based on its value (or use a distinct color per level)
        iso.module_manager.scalar_lut_manager.lut_mode = 'coolwarm' # or 'hot', 'jet'
        iso.module_manager.scalar_lut_manager.data_range = [min_temp_global, max_temp_global]
        # Or, to give each surface a distinct color from a colormap:
        # actor = iso.actor.property
        # color_val = cmap(float(i+1) / (num_isosurfaces+1))[:3] # Get RGB from colormap
        # actor.color = color_val

    # Plot sensor points as spheres, colored by their actual temperature
    s_plot = mlab.points3d(sensor_x, sensor_y, sensor_z, temps_at_time,
                           scale_mode='none', scale_factor=0.6,
                           colormap='coolwarm', vmin=min_temp_global, vmax=max_temp_global)
    mlab.colorbar(s_plot, title='Sensor Temp (°C)', orientation='vertical', nb_labels=5)

    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', ranges=[0,14,0,14,0,14])
    mlab.outline()
    time_s = data_df.loc[time_index, 'Time_(s)']
    mlab.title(f"Isosurfaces at Time: {time_s:.1f}s", height=0.95, size=0.5)
    mlab.view(azimuth=45, elevation=60, distance='auto', focalpoint='auto')
    mlab.show()


# --- Matplotlib 2D Plots ---
# (Keep the plot_2d_analysis function from the previous response, it's good for analysis)
def plot_2d_analysis(data_df, time_index):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"2D Analysis (Selected Time for Heatmap: {data_df.loc[time_index, 'Time_(s)']:.1f}s)", fontsize=16, y=0.99)

    center_sensors = ['5', '10', '15']
    axs[0,0].plot(data_df['Time_(s)'], data_df['Oven_Mon'], label='Oven Monitor', color='k', linestyle='--')
    for sensor in center_sensors:
        axs[0,0].plot(data_df['Time_(s)'], data_df[sensor], label=f'Sensor {sensor} (Center)')
    axs[0,0].set_xlabel('Time (s)'); axs[0,0].set_ylabel('Temperature (°C)')
    axs[0,0].set_title('Center Sensor & Oven Temperatures Over Time'); axs[0,0].legend(); axs[0,0].grid(True)

    layers = {
        'Bottom (Z=2")': ['1', '2', '3', '4', '5'],
        'Middle (Z=7")': ['6', '7', '8', '9', '10'],
        'Top (Z=12")': ['11', '12', '13', '14', '15']
    }
    for layer_name, sensors_in_layer in layers.items():
        avg_temp = data_df[sensors_in_layer].mean(axis=1)
        axs[0,1].plot(data_df['Time_(s)'], avg_temp, label=layer_name)
    axs[0,1].set_xlabel('Time (s)'); axs[0,1].set_ylabel('Avg Layer Temp (°C)')
    axs[0,1].set_title('Average Temperature per Layer Over Time'); axs[0,1].legend(); axs[0,1].grid(True)

    middle_layer_sensors = ['6', '7', '8', '9', '10']
    middle_coords_x = np.array([sensor_positions_map[s][0] for s in middle_layer_sensors])
    middle_coords_y = np.array([sensor_positions_map[s][1] for s in middle_layer_sensors])
    middle_temps = data_df.loc[time_index, middle_layer_sensors].values.astype(float)

    grid_x2d_fine, grid_y2d_fine = np.mgrid[0:14:100j, 0:14:100j]
    try:
        grid_temps2d = griddata(np.column_stack((middle_coords_x, middle_coords_y)), middle_temps,
                                (grid_x2d_fine, grid_y2d_fine), method='cubic', fill_value=np.nanmin(middle_temps))
    except ValueError:
        grid_temps2d = griddata(np.column_stack((middle_coords_x, middle_coords_y)), middle_temps,
                                (grid_x2d_fine, grid_y2d_fine), method='linear', fill_value=np.nanmin(middle_temps))

    im = axs[1,0].imshow(grid_temps2d.T, extent=(0,14,0,14), origin='lower', cmap='coolwarm',
                        aspect='auto', vmin=min_temp_global, vmax=max_temp_global)
    axs[1,0].scatter(middle_coords_x, middle_coords_y, c=middle_temps, cmap='coolwarm',
                     edgecolors='k', s=80, vmin=min_temp_global, vmax=max_temp_global)
    for i, txt in enumerate(middle_layer_sensors):
        axs[1,0].annotate(txt, (middle_coords_x[i], middle_coords_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    axs[1,0].set_xlabel('X-axis (inch)'); axs[1,0].set_ylabel('Y-axis (inch)')
    axs[1,0].set_title(f'Heatmap of Middle Layer (Time {data_df.loc[time_index, "Time_(s)"]:.1f}s)')
    fig.colorbar(im, ax=axs[1,0], label='Temp (°C)', fraction=0.046, pad=0.04)

    axs[1,1].text(0.05, 0.95, "Thermodynamic & Fluid Mechanics Insights:", transform=axs[1,1].transAxes, fontsize=12, va='top', fontweight='bold')
    axs[1,1].text(0.05, 0.85,
              ("- Heating patterns: Observe from animations/isosurfaces.\n"
               "- Thermal stratification: See Avg Layer Temp plot. Calculate (Top - Bottom).\n"
               "- Uniformity: Std. dev. of 15 sensors or within layers.\n"
               "- Heating Rates: dT/dt from time series.\n"
               "- Convection hints: Swirling patterns in volume, rapid local changes."),
              transform=axs[1,1].transAxes, fontsize=9, va='top', wrap=True)
    axs[1,1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    print("Choose visualization type:")
    print("1: Animated 3D Volumetric Plot with Isosurfaces (Mayavi)")
    print("2: Static 3D Multiple Isosurfaces Plot (Mayavi)")
    print("3: 2D Analysis Plots (Matplotlib)")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        print("Starting Mayavi animation. This may take a moment...")
        print("Close the Mayavi window when done.")
        plot_animated_mayavi_volume_iso(df_cleaned)

    elif choice == '2':
        target_time_s = 5832.0 
        try:
            time_input_str = input(f"Enter target time in seconds for isosurface plot (e.g., {target_time_s}, press Enter for default): ")
            time_input = float(time_input_str) if time_input_str else target_time_s
        except ValueError:
            print(f"Invalid time input, using default {target_time_s}s.")
            time_input = target_time_s
        
        time_index = (df_cleaned['Time_(s)'] - time_input).abs().idxmin()
        print(f"Selected time step for isosurface: Index {time_index}, Time {df_cleaned.loc[time_index, 'Time_(s)']:.1f}s")
        
        num_iso = 5
        try:
            num_iso_str = input(f"Enter number of isosurfaces to display (e.g., {num_iso}, press Enter for default): ")
            num_iso = int(num_iso_str) if num_iso_str else num_iso
        except ValueError:
            print(f"Invalid number, using default {num_iso} isosurfaces.")

        plot_mayavi_static_isosurfaces_topographic(df_cleaned, time_index, num_isosurfaces=num_iso)
        print("Close the Mayavi window when done.")

    elif choice == '3':
        target_time_s = 5832.0 
        try:
            time_input_2d_str = input(f"Enter target time in seconds for heatmap (e.g., {target_time_s}, press Enter for default): ")
            time_input_2d = float(time_input_2d_str) if time_input_2d_str else target_time_s
        except ValueError:
            print(f"Invalid time input, using default {target_time_s}s.")
            time_input_2d = target_time_s
            
        time_index_2d = (df_cleaned['Time_(s)'] - time_input_2d).abs().idxmin()
        plot_2d_analysis(df_cleaned, time_index_2d)
    else:
        print("Invalid choice.")