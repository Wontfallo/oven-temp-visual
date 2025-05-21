import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For 2D plots and colormaps
from scipy.interpolate import griddata
from mayavi import mlab # The main Mayavi module for scripting

# --- Configuration ---
csv_filename = "oven_data.csv"
ANIMATION_INTERVAL_MS = 200  # Milliseconds between frames (slower for Mayavi rendering)
MAX_FRAMES_FOR_DEMO = None # Set to a number (e.g., 50-100) for faster testing, or None for all

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
sensor_points_xyz = np.array([sensor_positions_map[s] for s in sensor_column_names])

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
grid_res_x_pts, grid_res_y_pts, grid_res_z_pts = 30, 30, 30 # Number of points for interpolation grid

xi = np.linspace(0, 14, grid_res_x_pts)
yi = np.linspace(0, 14, grid_res_y_pts)
zi = np.linspace(0, 14, grid_res_z_pts)
grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')


# --- Mayavi 3D Animated Volumetric Plot ---
def plot_animated_mayavi_volume(data_df):
    num_frames = len(data_df) if MAX_FRAMES_FOR_DEMO is None else min(len(data_df), MAX_FRAMES_FOR_DEMO)

    @mlab.animate(delay=ANIMATION_INTERVAL_MS)
    def anim():
        # Create the Mayavi scene
        fig_mlab = mlab.figure(bgcolor=(0.1, 0.1, 0.1), size=(800, 700))
        fig_mlab.scene.anti_aliasing_frames = 8 # For smoother visuals

        # Initial data interpolation for the first frame
        temps_at_time_0 = data_df.loc[0, sensor_column_names].values.astype(float)
        volumetric_temp_data = griddata(sensor_points_xyz, temps_at_time_0,
                                        (grid_x, grid_y, grid_z), method='linear', fill_value=min_temp_global)

        # Create a scalar field source
        src = mlab.pipeline.scalar_field(grid_x, grid_y, grid_z, volumetric_temp_data)
        
        # Volume rendering
        # Adjust opacity transfer function for better visuals:
        # This creates a ramp: more opaque for higher temperatures.
        vol = mlab.pipeline.volume(src, vmin=min_temp_global, vmax=max_temp_global)
        otf = vol.module_manager.scalar_lut_manager.lut.alpha
        otf.initialize(256) # Number of points in the transfer function
        # Example: make low temps transparent, mid temps semi-transparent, high temps more opaque
        otf.set_table([0, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0] * (256 // 9 +1) )[:256]
        # You might need to experiment a lot with `otf.set_table([...])` for the best visual
        vol.module_manager.scalar_lut_manager.lut_mode = 'coolwarm'


        # Optional: Add one or two key isosurfaces
        # Example: an isosurface at a mid-range temperature
        mid_iso_temp = (min_temp_global + max_temp_global) / 2.5
        iso1 = mlab.pipeline.iso_surface(src, contours=[mid_iso_temp], opacity=0.2, colormap='winter')

        # Add a color bar for the volume rendering
        mlab.scalarbar(vol, title='Temperature (°C)', orientation='vertical')
        mlab.axes(xlabel='X (inch)', ylabel='Y (inch)', zlabel='Z (inch)', ranges=[0,14,0,14,0,14])
        mlab.outline(color=(0.7,0.7,0.7))
        mlab.title("Animated 3D Volumetric Temperature Profile", height=0.95, color=(1,1,1))
        time_text = mlab.text(0.02, 0.9, "Time: 0.0s", width=0.3, color=(1,1,1))
        oven_text = mlab.text(0.02, 0.85, "Oven: 0.0°C", width=0.3, color=(1,1,1))

        for frame_idx in range(num_frames):
            temps_at_time = data_df.loc[frame_idx, sensor_column_names].values.astype(float)
            new_volumetric_data = griddata(sensor_points_xyz, temps_at_time,
                                           (grid_x, grid_y, grid_z), method='linear', fill_value=min_temp_global)
            
            # Update the scalar field data
            src.mlab_source.scalars = new_volumetric_data
            
            # Update text annotations
            time_s = data_df.loc[frame_idx, 'Time_(s)']
            oven_mon_temp = data_df.loc[frame_idx, 'Oven_Mon']
            time_text.text = f"Time: {time_s:.1f}s"
            oven_text.text = f"Oven: {oven_mon_temp:.2f}°C"

            fig_mlab.scene.render() # Force redraw
            yield

    ani_instance = anim() # This starts the animation in a Mayavi window
    mlab.show()
    return ani_instance


# --- Mayavi Static Multiple Isosurfaces Plot ---
def plot_mayavi_static_isosurfaces(data_df, time_index, num_isosurfaces=5):
    mlab.figure(bgcolor=(0.05, 0.05, 0.05), size=(900, 750))
    mlab.clf() # Clear the current figure

    temps_at_time = data_df.loc[time_index, sensor_column_names].values.astype(float)
    volumetric_temp_data = griddata(sensor_points_xyz, temps_at_time,
                                     (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan)

    # Define isosurface levels dynamically based on the current data range
    min_t_step = np.nanmin(volumetric_temp_data)
    max_t_step = np.nanmax(volumetric_temp_data)
    
    if np.isnan(min_t_step) or np.isnan(max_t_step) or min_t_step == max_t_step:
        print(f"Cannot generate isosurfaces for time index {time_index}: data range is too small or all NaN.")
        print(f"Sensor temps at this step: {temps_at_time}")
        return

    iso_levels = np.linspace(min_t_step + (max_t_step - min_t_step) * 0.1,
                             max_t_step - (max_t_step - min_t_step) * 0.1,
                             num_isosurfaces)
    iso_levels = np.round(iso_levels, 1)
    print(f"Plotting isosurfaces at temperatures: {iso_levels}°C")

    src = mlab.pipeline.scalar_field(grid_x, grid_y, grid_z, volumetric_temp_data)

    # Plot multiple isosurfaces with varying opacity and a good colormap
    opacities = np.linspace(0.1, 0.6, num_isosurfaces) # More opaque for hotter/inner surfaces
    
    for i, level in enumerate(iso_levels):
        mlab.pipeline.iso_surface(src, contours=[level], opacity=opacities[i], colormap='coolwarm',
                                  vmin=min_temp_global, vmax=max_temp_global)

    mlab.scalarbar(title='Temperature (°C)', orientation='vertical', nb_labels=num_isosurfaces)
    mlab.axes(xlabel='X (inch)', ylabel='Y (inch)', zlabel='Z (inch)', ranges=[0,14,0,14,0,14])
    mlab.outline(color=(0.7,0.7,0.7))
    time_s = data_df.loc[time_index, 'Time_(s)']
    oven_mon_temp = data_df.loc[time_index, 'Oven_Mon']
    mlab.title(f"3D Isosurface Temperature Profile at Time: {time_s:.1f}s (Oven: {oven_mon_temp:.2f}°C)",
               height=0.95, color=(1,1,1))
    mlab.view(azimuth=60, elevation=65, distance='auto', focalpoint='auto')
    mlab.show()


# --- Matplotlib 2D Plots (from previous, slightly adapted) ---
def plot_2d_analysis(data_df, time_index):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"2D Analysis (Selected Time for Heatmap: {data_df.loc[time_index, 'Time_(s)']:.1f}s)", fontsize=16, y=0.99)

    # 1. Time series for center sensors
    center_sensors = ['5', '10', '15']
    axs[0,0].plot(data_df['Time_(s)'], data_df['Oven_Mon'], label='Oven Monitor', color='k', linestyle='--')
    for sensor in center_sensors:
        axs[0,0].plot(data_df['Time_(s)'], data_df[sensor], label=f'Sensor {sensor} (Center)')
    axs[0,0].set_xlabel('Time (s)'); axs[0,0].set_ylabel('Temperature (°C)')
    axs[0,0].set_title('Center Sensor & Oven Temperatures Over Time'); axs[0,0].legend(); axs[0,0].grid(True)

    # 2. Average temperature per layer
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

    # 3. Heatmap of Middle Layer at specified time_index
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

    # 4. Analysis Text Placeholder
    axs[1,1].text(0.05, 0.95, "Thermodynamic & Fluid Mechanics Insights:", transform=axs[1,1].transAxes, fontsize=12, va='top', fontweight='bold')
    axs[1,1].text(0.05, 0.85,
              ("- Observe heating patterns: Which areas heat up first/last?\n"
               "- Assess thermal stratification: Significant Z-axis temperature differences?\n"
               "  (e.g., (Top_Avg - Bottom_Avg) over time)\n"
               "- Uniformity: How does std. dev. of all 15 sensors change?\n"
               "  A lower std. dev. indicates better uniformity.\n"
               "- Heating Rates: Calculate dT/dt for individual sensors or averages.\n"
               "- Qualitative Convection: Rapid, uneven temp changes might suggest\n"
               "  convective currents. Smoother changes suggest conduction dominance.\n"
               "- Relate to Oven_Mon: How well do internal temps track the oven setpoint/monitor?"),
              transform=axs[1,1].transAxes, fontsize=9, va='top', wrap=True)
    axs[1,1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    print("Choose visualization type:")
    print("1: Animated 3D Volumetric Plot (Mayavi)")
    print("2: Static 3D Multi-Isosurface Plot (Mayavi)")
    print("3: 2D Analysis Plots (Matplotlib)")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        print("Starting Mayavi animation. This may take a moment to initialize...")
        print("Close the Mayavi window when done.")
        # Note: The anim() function itself will run mlab.show() when done.
        plot_animated_mayavi_volume(df_cleaned)

    elif choice == '2':
        target_time_s = 5832.0 # Example default, as in your image
        try:
            time_input = float(input(f"Enter target time in seconds for isosurface plot (e.g., {target_time_s}): ") or target_time_s)
        except ValueError:
            time_input = target_time_s
        
        time_index = (df_cleaned['Time_(s)'] - time_input).abs().idxmin()
        print(f"Selected time step for isosurface: Index {time_index}, Time {df_cleaned.loc[time_index, 'Time_(s)']:.1f}s")
        
        plot_mayavi_static_isosurfaces(df_cleaned, time_index, num_isosurfaces=4)
        print("Close the Mayavi window when done.")

    elif choice == '3':
        target_time_s = 5832.0 # Example default for heatmap
        try:
            time_input_2d = float(input(f"Enter target time in seconds for heatmap (e.g., {target_time_s}): ") or target_time_s)
        except ValueError:
            time_input_2d = target_time_s

        time_index_2d = (df_cleaned['Time_(s)'] - time_input_2d).abs().idxmin()
        plot_2d_analysis(df_cleaned, time_index_2d)
    else:
        print("Invalid choice.")