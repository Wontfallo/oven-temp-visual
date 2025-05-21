import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import griddata
from skimage.measure import marching_cubes
from matplotlib import cm # For colormaps

# --- Configuration ---
csv_filename = "oven_data.csv"
ANIMATION_INTERVAL_MS = 150
MAX_FRAMES_FOR_DEMO = 200 # Set to None for all frames, or a number for testing

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

# --- Grid for Interpolation ---
grid_res_x, grid_res_y, grid_res_z = 30j, 30j, 30j # Increased resolution for smoother visuals
xi = np.linspace(0, 14, int(grid_res_x.imag))
yi = np.linspace(0, 14, int(grid_res_y.imag))
zi = np.linspace(0, 14, int(grid_res_z.imag))
grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')


# --- Plotting Functions ---

def plot_animated_interpolated_scatter(data_df):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Animated 3D Interpolated Temperature Profile (Scatter)", fontsize=16, y=0.98)

    temps_initial = data_df.loc[0, sensor_column_names].values.astype(float)
    grid_temps_initial = griddata(sensor_points_xyz, temps_initial,
                                  (grid_x, grid_y, grid_z), method='linear', fill_value=min_temp_global)

    x_flat, y_flat, z_flat = grid_x.ravel(), grid_y.ravel(), grid_z.ravel()
    temps_flat_initial = grid_temps_initial.ravel()

    scat = ax.scatter(x_flat, y_flat, z_flat, c=temps_flat_initial, cmap='coolwarm',
                      s=25, vmin=min_temp_global, vmax=max_temp_global, alpha=0.5, edgecolor='none')

    cbar = fig.colorbar(scat, ax=ax, pad=0.15, fraction=0.03, label='Temperature (°C)')
    ax.set_xlabel('X (inch)'); ax.set_ylabel('Y (inch)'); ax.set_zlabel('Z (inch)')
    ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_zlim(0, 14)
    ax.view_init(elev=30, azim=-60)
    title_artist = ax.set_title("", fontsize=10)

    def update(frame):
        temps_at_time = data_df.loc[frame, sensor_column_names].values.astype(float)
        grid_temps = griddata(sensor_points_xyz, temps_at_time,
                              (grid_x, grid_y, grid_z), method='linear', fill_value=min_temp_global)
        temps_flat = grid_temps.ravel()
        scat.set_array(temps_flat)
        
        time_s = data_df.loc[frame, 'Time_(s)']
        oven_mon_temp = data_df.loc[frame, 'Oven_Mon']
        rt1_temp = data_df.loc[frame, 'RT1']
        rt2_temp = data_df.loc[frame, 'RT2']
        title_text = (f'Time: {time_s:.1f}s; Oven: {oven_mon_temp:.2f}°C\n'
                      f'RT1: {rt1_temp:.2f}°C; RT2: {rt2_temp:.2f}°C')
        title_artist.set_text(title_text)
        return scat, title_artist

    num_frames = len(data_df) if MAX_FRAMES_FOR_DEMO is None else min(len(data_df), MAX_FRAMES_FOR_DEMO)
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=ANIMATION_INTERVAL_MS, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return ani

def plot_static_isosurfaces(data_df, time_index, iso_temps, alphas, colors):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    temps_at_time = data_df.loc[time_index, sensor_column_names].values.astype(float)
    volumetric_temp_data = griddata(sensor_points_xyz, temps_at_time,
                                     (grid_x, grid_y, grid_z), method='linear', fill_value=np.nan) # Use NaN for fill

    ax.set_xlabel('X (inch)'); ax.set_ylabel('Y (inch)'); ax.set_zlabel('Z (inch)')
    ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_zlim(0, 14)
    ax.view_init(elev=25, azim=120)
    
    plotted_something = False
    for i, iso_temp in enumerate(iso_temps):
        try:
            # Handle NaN values that marching_cubes doesn't like by masking
            masked_volume = np.ma.masked_invalid(volumetric_temp_data)
            if np.all(masked_volume.mask): # All values are NaN
                print(f"Skipping isosurface for {iso_temp}°C: all interpolated data is NaN.")
                continue

            verts, faces, normals, values = marching_cubes(
                masked_volume.filled(fill_value=np.nanmin(masked_volume)), # Fill NaNs with a value outside typical range
                level=iso_temp,
                spacing=(xi[1]-xi[0], yi[1]-yi[0], zi[1]-zi[0]) # dx, dy, dz
            )
            if verts.size > 0:
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                color=colors[i % len(colors)], lw=0.1, alpha=alphas[i % len(alphas)], antialiased=True)
                plotted_something = True
            else:
                print(f"No isosurface found at {iso_temp}°C for the selected time step.")

        except Exception as e:
            print(f"Error during marching_cubes for isosurface temp {iso_temp}°C: {e}")

    if not plotted_something:
        print("No isosurfaces were plotted.")
        plt.close(fig)
        return

    time_s = data_df.loc[time_index, 'Time_(s)']
    oven_mon_temp = data_df.loc[time_index, 'Oven_Mon']
    fig.suptitle(f"3D Isosurfaces at Time: {time_s:.1f}s (Oven: {oven_mon_temp:.2f}°C)", fontsize=16, y=0.99)
    ax.set_title(f"Isosurfaces at: " + ", ".join([f"{t}°C" for t in iso_temps]), fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def plot_2d_heatmaps_and_timeseries(data_df, time_index):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"2D Analysis at Time: {data_df.loc[time_index, 'Time_(s)']:.1f}s", fontsize=16, y=0.98)

    # 1. Time series for center sensors
    center_sensors = ['5', '10', '15'] # Bottom, Middle, Top center
    for sensor in center_sensors:
        axs[0,0].plot(data_df['Time_(s)'], data_df[sensor], label=f'Sensor {sensor} (Center)')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Temperature (°C)')
    axs[0,0].set_title('Center Sensor Temperatures Over Time')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # 2. Average temperature per layer
    layers = {
        'Bottom (Z=2")': ['1', '2', '3', '4', '5'],
        'Middle (Z=7")': ['6', '7', '8', '9', '10'],
        'Top (Z=12")': ['11', '12', '13', '14', '15']
    }
    for layer_name, sensors_in_layer in layers.items():
        avg_temp = data_df[sensors_in_layer].mean(axis=1)
        axs[0,1].plot(data_df['Time_(s)'], avg_temp, label=layer_name)
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Average Temperature (°C)')
    axs[0,1].set_title('Average Temperature per Layer Over Time')
    axs[0,1].legend()
    axs[0,1].grid(True)

    # 3. Heatmap of Middle Layer (Sensors 6-10) at specified time_index
    middle_layer_sensors = ['6', '7', '8', '9', '10']
    middle_coords_x = np.array([sensor_positions_map[s][0] for s in middle_layer_sensors])
    middle_coords_y = np.array([sensor_positions_map[s][1] for s in middle_layer_sensors])
    middle_temps = data_df.loc[time_index, middle_layer_sensors].values.astype(float)

    # Create a grid for 2D interpolation
    grid_x2d, grid_y2d = np.mgrid[min(middle_coords_x):max(middle_coords_x):100j,
                                 min(middle_coords_y):max(middle_coords_y):100j]
    
    try:
        grid_temps2d = griddata(np.column_stack((middle_coords_x, middle_coords_y)), middle_temps,
                                (grid_x2d, grid_y2d), method='cubic', fill_value=np.nanmin(middle_temps))
        # Use 'cubic' for 2D as it's supported and gives smoother results
    except ValueError as e_2d_cubic:
        print(f"Cubic interpolation failed for 2D heatmap, trying linear: {e_2d_cubic}")
        grid_temps2d = griddata(np.column_stack((middle_coords_x, middle_coords_y)), middle_temps,
                                (grid_x2d, grid_y2d), method='linear', fill_value=np.nanmin(middle_temps))


    im = axs[1,0].imshow(grid_temps2d.T, extent=(min(middle_coords_x), max(middle_coords_x),
                                             min(middle_coords_y), max(middle_coords_y)),
                        origin='lower', cmap='coolwarm', aspect='auto',
                        vmin=min_temp_global, vmax=max_temp_global)
    axs[1,0].scatter(middle_coords_x, middle_coords_y, c=middle_temps, cmap='coolwarm',
                     edgecolors='k', s=80, vmin=min_temp_global, vmax=max_temp_global)
    for i, txt in enumerate(middle_layer_sensors):
        axs[1,0].annotate(txt, (middle_coords_x[i], middle_coords_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    axs[1,0].set_xlabel('X-axis (inch)')
    axs[1,0].set_ylabel('Y-axis (inch)')
    axs[1,0].set_title(f'Heatmap of Middle Layer (Sensors 6-10) at Time {data_df.loc[time_index, "Time_(s)"]:.1f}s')
    fig.colorbar(im, ax=axs[1,0], label='Temperature (°C)')

    # 4. Temperature Gradients (example between two points at specified time_index)
    # Example: Gradient between sensor 5 (center bottom) and sensor 10 (center middle)
    temp5 = data_df.loc[time_index, '5']
    temp10 = data_df.loc[time_index, '10']
    pos5 = np.array(sensor_positions_map['5'])
    pos10 = np.array(sensor_positions_map['10'])
    delta_z = pos10[2] - pos5[2]
    if delta_z != 0:
        gradient_z_5_10 = (temp10 - temp5) / delta_z
    else:
        gradient_z_5_10 = np.nan
        
    axs[1,1].text(0.05, 0.9, "Qualitative Analysis & Gradients:", transform=axs[1,1].transAxes, fontsize=12, fontweight='bold')
    axs[1,1].text(0.05, 0.8, f"Time: {data_df.loc[time_index, 'Time_(s)']:.1f}s", transform=axs[1,1].transAxes)
    axs[1,1].text(0.05, 0.7, f"Approx. Z-gradient (S5-S10): {gradient_z_5_10:.2f} °C/inch", transform=axs[1,1].transAxes)
    axs[1,1].text(0.05, 0.6, "Further analysis can explore uniformity (std dev),\n "
                           "heating rates (temp change / time interval),\n "
                           "and thermal stratification trends.", transform=axs[1,1].transAxes, va='top')
    axs[1,1].axis('off')
    axs[1,1].set_title("Basic Gradient Example & Analysis Notes")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("Choose visualization type:")
    print("1: Animated 3D Interpolated Scatter Plot")
    print("2: Static 3D Multi-Isosurface Plot (for a specific time)")
    print("3: 2D Analysis Plots (Time Series, Layer Averages, Heatmap)")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        print("Generating animated interpolated scatter plot...")
        ani_scatter = plot_animated_interpolated_scatter(df_cleaned)
    elif choice == '2':
        target_time_s = 5832.0 # Example: The time step you mentioned
        time_index = (df_cleaned['Time_(s)'] - target_time_s).abs().idxmin()
        print(f"Selected time step for isosurface: Index {time_index}, Time {df_cleaned.loc[time_index, 'Time_(s)']:.1f}s")

        temps_at_selected_time = df_cleaned.loc[time_index, sensor_column_names].values.astype(float)
        min_t_step = np.min(temps_at_selected_time)
        max_t_step = np.max(temps_at_selected_time)
        print(f"Temperature range at this time step for sensors: {min_t_step:.2f}°C to {max_t_step:.2f}°C")

        # Define a few isosurface temperatures based on the data range
        iso_temps_to_plot = np.linspace(min_t_step + (max_t_step - min_t_step)*0.2,
                                      max_t_step - (max_t_step - min_t_step)*0.2,
                                      3) # Plot 3 isosurfaces
        iso_temps_to_plot = np.round(iso_temps_to_plot, 1)
        
        print(f"Plotting isosurfaces at approximately: {iso_temps_to_plot} °C")
        
        # Define colors and transparencies for each isosurface
        # Example: hotter surfaces are more opaque and reddish, cooler are more transparent and bluish
        colors = ['blue', 'green', 'red', 'purple', 'orange'] 
        alphas = [0.2, 0.3, 0.4, 0.5, 0.6] # Increasing opacity for hotter surfaces

        plot_static_isosurfaces(df_cleaned, time_index, iso_temps_to_plot, alphas, colors)
    elif choice == '3':
        target_time_s = 5832.0 # Example time for heatmap
        time_index_2d = (df_cleaned['Time_(s)'] - target_time_s).abs().idxmin()
        print(f"Generating 2D analysis plots. Heatmap is for time step: Index {time_index_2d}, Time {df_cleaned.loc[time_index_2d, 'Time_(s)']:.1f}s")
        plot_2d_heatmaps_and_timeseries(df_cleaned, time_index_2d)
    else:
        print("Invalid choice.")

    print(f"\nGlobal temperature range across all sensors/time: {min_temp_global:.2f}°C to {max_temp_global:.2f}°C")