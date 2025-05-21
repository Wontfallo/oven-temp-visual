import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import griddata
from skimage.measure import marching_cubes # For isosurface

# --- Configuration ---
csv_filename = "oven_data.csv"
ANIMATION_INTERVAL_MS = 150  # Milliseconds between frames
MAX_FRAMES_FOR_DEMO = None   # Set to a number (e.g., 100) for faster testing, or None for all frames

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
# Define the resolution of the interpolation grid. Higher values = denser grid = smoother but slower.
grid_res_x, grid_res_y, grid_res_z = 20j, 20j, 20j # 20 points along each axis

grid_x, grid_y, grid_z = np.mgrid[0:14:grid_res_x,  # X-range of your cube
                                  0:14:grid_res_y,  # Y-range
                                  0:14:grid_res_z]  # Z-range


# --- Function for Animated 3D Interpolated Scatter Plot ---
def plot_animated_interpolated_scatter(data_df):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Animated 3D Interpolated Temperature Profile", fontsize=16, y=0.98)

    # Initial scatter plot (will be updated)
    # We plot the grid points, colored by interpolated temperature
    # For the first frame:
    temps_initial = data_df.loc[0, sensor_column_names].values.astype(float)
    grid_temps_initial = griddata(sensor_points_xyz, temps_initial,
                                  (grid_x, grid_y, grid_z), method='cubic', fill_value=min_temp_global) # Use cubic for smoother results

    # Flatten for scatter plot
    x_flat, y_flat, z_flat = grid_x.ravel(), grid_y.ravel(), grid_z.ravel()
    temps_flat_initial = grid_temps_initial.ravel()
    
    # Filter out points far from actual sensors if desired, or use alpha for transparency
    # For this demo, we plot all interpolated grid points.
    scat = ax.scatter(x_flat, y_flat, z_flat, c=temps_flat_initial, cmap='coolwarm',
                      s=15, vmin=min_temp_global, vmax=max_temp_global, alpha=0.6, edgecolor='none')

    cbar = fig.colorbar(scat, ax=ax, pad=0.15, fraction=0.03, label='Temperature (°C)')

    ax.set_xlabel('X (inch) - Left to Right')
    ax.set_ylabel('Y (inch) - Front to Back')
    ax.set_zlabel('Z (inch) - Bottom to Top')
    ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_zlim(0, 14)
    ax.view_init(elev=25, azim=45)
    title_artist = ax.set_title("", fontsize=10) # Placeholder for dynamic title

    def update(frame):
        temps_at_time = data_df.loc[frame, sensor_column_names].values.astype(float)
        grid_temps = griddata(sensor_points_xyz, temps_at_time,
                              (grid_x, grid_y, grid_z), method='cubic', fill_value=min_temp_global)
        temps_flat = grid_temps.ravel()
        
        scat._offsets3d = (x_flat, y_flat, z_flat) # Re-set points if necessary (though they are static)
        scat.set_array(temps_flat) # Update colors

        time_s = data_df.loc[frame, 'Time_(s)']
        oven_mon_temp = data_df.loc[frame, 'Oven_Mon']
        rt1_temp = data_df.loc[frame, 'RT1']
        rt2_temp = data_df.loc[frame, 'RT2']
        title_text = (f'Time: {time_s:.1f}s; OvenMon: {oven_mon_temp:.2f}°C\n'
                      f'RT1: {rt1_temp:.2f}°C; RT2: {rt2_temp:.2f}°C (Interpolated Grid)')
        title_artist.set_text(title_text)
        return scat, title_artist

    num_frames = len(data_df) if MAX_FRAMES_FOR_DEMO is None else min(len(data_df), MAX_FRAMES_FOR_DEMO)
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=ANIMATION_INTERVAL_MS, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return ani # Keep a reference

# --- Function for Static 3D Isosurface Plot ---
def plot_static_isosurface(data_df, time_index, isosurface_temp):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    temps_at_time = data_df.loc[time_index, sensor_column_names].values.astype(float)
    
    # Interpolate temperatures onto the grid
    # Using 'linear' for griddata as 'cubic' can sometimes produce artifacts outside convex hull
    # that marching_cubes might struggle with. 'linear' is generally safer for isosurfacing.
    volumetric_temp_data = griddata(sensor_points_xyz, temps_at_time,
                                     (grid_x, grid_y, grid_z), method='linear', fill_value=min_temp_global)


    try:
        # Marching cubes to find the surface
        verts, faces, normals, values = marching_cubes(
            volume=volumetric_temp_data,
            level=isosurface_temp,
            spacing=( (grid_x[1,0,0]-grid_x[0,0,0]), # dx
                      (grid_y[0,1,0]-grid_y[0,0,0]), # dy
                      (grid_z[0,0,1]-grid_z[0,0,0])  # dz
                    )
        )
        # Offset vertices to match grid origin if mgrid starts from non-zero (here it's 0)
        # verts[:, 0] += grid_x.min()
        # verts[:, 1] += grid_y.min()
        # verts[:, 2] += grid_z.min()

    except Exception as e:
        print(f"Error during marching_cubes for isosurface temp {isosurface_temp}°C: {e}")
        print("This can happen if the isosurface level is outside the data range or the data is too uniform.")
        plt.close(fig)
        return

    if verts.size == 0:
        print(f"No isosurface found at {isosurface_temp}°C for the selected time step.")
        print(f"Temperature range at this step: {np.nanmin(temps_at_time):.2f}°C to {np.nanmax(temps_at_time):.2f}°C")
        print(f"Interpolated grid range: {np.nanmin(volumetric_temp_data):.2f}°C to {np.nanmax(volumetric_temp_data):.2f}°C")
        plt.close(fig)
        return

    # Plot the isosurface
    # You can color the surface by its z-value, or a fixed color, etc.
    # Here, we use a colormap based on Z values of the vertices for some shading.
    surf_cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=np.min(verts[:, 2]), vmax=np.max(verts[:, 2]))
    face_colors = surf_cmap(norm(values)) # Or use values from marching_cubes if desired

    mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           cmap=surf_cmap, lw=0.1, edgecolor='gray', alpha=0.7, antialiased=True)
                           # You can add facecolors=face_colors if you calculated them

    # Add a color bar representing the Z-values of the surface (or the isosurface value itself)
    # If coloring by Z:
    # cbar = fig.colorbar(mesh, ax=ax, pad=0.1, fraction=0.03, label='Z-height of Isosurface (inch)')
    # Or, for a fixed isosurface temperature, a colorbar might not be directly mapping the surface colors
    # but can indicate the chosen isosurface level. For simplicity, we'll note it in the title.
    
    ax.set_xlabel('X (inch) - Left to Right')
    ax.set_ylabel('Y (inch) - Front to Back')
    ax.set_zlabel('Z (inch) - Bottom to Top')
    ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_zlim(0, 14)
    ax.view_init(elev=25, azim=120) # Different view for isosurface

    time_s = data_df.loc[time_index, 'Time_(s)']
    oven_mon_temp = data_df.loc[time_index, 'Oven_Mon']
    rt1_temp = data_df.loc[time_index, 'RT1']
    rt2_temp = data_df.loc[time_index, 'RT2']

    fig.suptitle(f"3D Isosurface at T = {isosurface_temp}°C", fontsize=16, y=0.99)
    ax.set_title((f'Time: {time_s:.1f}s; OvenMon: {oven_mon_temp:.2f}°C\n'
                  f'RT1: {rt1_temp:.2f}°C; RT2: {rt2_temp:.2f}°C'), fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("Choose visualization type:")
    print("1: Animated 3D Interpolated Scatter Plot")
    print("2: Static 3D Isosurface Plot (for a specific time and temperature)")
    
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        print("Generating animated interpolated scatter plot...")
        # Keep a reference to ani so it doesn't get garbage collected
        # if running in some environments (like a plain script).
        ani_scatter = plot_animated_interpolated_scatter(df_cleaned)
        print("Animation window should be open. Close it to end the script.")
    elif choice == '2':
        print("\nFor the static isosurface plot:")
        # Select a time step (e.g., the last one, or one you know is interesting)
        # For example, the time step around 5832s which you showed an image for.
        # Find the closest index:
        target_time_s = 5832.0 
        time_index = (df_cleaned['Time_(s)'] - target_time_s).abs().idxmin()
        print(f"Selected time step: Index {time_index}, Time {df_cleaned.loc[time_index, 'Time_(s)']:.1f}s")

        # Temperatures at this specific time step for the 15 sensors
        temps_at_selected_time = df_cleaned.loc[time_index, sensor_column_names].values.astype(float)
        min_t_step = np.min(temps_at_selected_time)
        max_t_step = np.max(temps_at_selected_time)
        print(f"Temperature range at this time step: {min_t_step:.2f}°C to {max_t_step:.2f}°C")

        # Choose an isosurface temperature (must be within the range of your data at that time)
        # Let's pick a temperature in the middle of the range for this example, or a specific high temp
        
        default_iso_temp = (min_t_step + max_t_step) / 2 
        if max_temp_global > 100: # If there are high temps, pick one
             default_iso_temp = max(min_t_step + 10, min(max_t_step -10, 100.0)) # Try 100C if in range
        
        try:
            iso_temp_input = float(input(f"Enter the temperature for the isosurface (e.g., {default_iso_temp:.1f}): "))
        except ValueError:
            print(f"Invalid input, using default: {default_iso_temp:.1f}°C")
            iso_temp_input = default_iso_temp
            
        print(f"Generating static isosurface plot for T = {iso_temp_input}°C...")
        plot_static_isosurface(df_cleaned, time_index, iso_temp_input)
    else:
        print("Invalid choice.")

    print(f"\nGlobal temperature range across all sensors/time: {min_temp_global:.2f}°C to {max_temp_global:.2f}°C")