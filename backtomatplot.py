import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from skimage import measure
import time
import os
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
csv_filename = "oven_data.csv"
OUTPUT_DIR = "oven_visualization"
FPS = 5
DURATION_SECONDS = 20  # How many seconds of data to visualize

# --- Sensor Coordinate Definitions ---
x_coords_map = {'left': 2, 'center': 7, 'right': 12}
y_coords_map = {'front': 2, 'center': 7, 'back': 12}
z_coords_map = {'bottom': 2, 'middle': 7, 'top': 12}
sensor_positions_map = {
    '1':(x_coords_map['left'],y_coords_map['front'],z_coords_map['bottom']),
    '2':(x_coords_map['left'],y_coords_map['back'],z_coords_map['bottom']),
    '3':(x_coords_map['right'],y_coords_map['back'],z_coords_map['bottom']),
    '4':(x_coords_map['right'],y_coords_map['front'],z_coords_map['bottom']),
    '5':(x_coords_map['center'],y_coords_map['center'],z_coords_map['bottom']),
    '6':(x_coords_map['left'],y_coords_map['front'],z_coords_map['middle']),
    '7':(x_coords_map['left'],y_coords_map['back'],z_coords_map['middle']),
    '8':(x_coords_map['right'],y_coords_map['back'],z_coords_map['middle']),
    '9':(x_coords_map['right'],y_coords_map['front'],z_coords_map['middle']),
    '10':(x_coords_map['center'],y_coords_map['center'],z_coords_map['middle']),
    '11':(x_coords_map['left'],y_coords_map['front'],z_coords_map['top']),
    '12':(x_coords_map['left'],y_coords_map['back'],z_coords_map['top']),
    '13':(x_coords_map['right'],y_coords_map['back'],z_coords_map['top']),
    '14':(x_coords_map['right'],y_coords_map['front'],z_coords_map['top']),
    '15':(x_coords_map['center'],y_coords_map['center'],z_coords_map['top'])
}
sensor_column_names = [str(i) for i in range(1, 16)]

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

df_cleaned = df.dropna(subset=sensor_column_names).reset_index(drop=True)
for col in sensor_column_names:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=sensor_column_names).reset_index(drop=True)

if df_cleaned.empty:
    print("Error: No valid numeric temperature data found after cleaning.")
    exit()

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Determine global temperature range ---
all_sensor_temps = df_cleaned[sensor_column_names].values.flatten()
min_temp_global = np.nanmin(all_sensor_temps)
max_temp_global = np.nanmax(all_sensor_temps)
print(f"Global temperature range: {min_temp_global:.2f}°C to {max_temp_global:.2f}°C")

# --- Create 3D visualization frames ---
def create_frame(frame_idx, save_path):
    # Get data for this frame
    temps_at_time = df_cleaned.loc[frame_idx, sensor_column_names].values.astype(float)
    time_s = df_cleaned.loc[frame_idx, 'Time_(s)']
    oven_temp = df_cleaned.loc[frame_idx, 'Oven_Mon']
    
    # Create figure with large size for better visibility
    fig = plt.figure(figsize=(16, 12), dpi=100, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Extract sensor coordinates
    sensor_x = np.array([sensor_positions_map[s][0] for s in sensor_column_names])
    sensor_y = np.array([sensor_positions_map[s][1] for s in sensor_column_names])
    sensor_z = np.array([sensor_positions_map[s][2] for s in sensor_column_names])
    
    # Create a grid for interpolation
    grid_x, grid_y, grid_z = np.mgrid[0:14:20j, 0:14:20j, 0:14:20j]
    
    # Interpolate temperature data
    points = np.column_stack((sensor_x, sensor_y, sensor_z))
    grid_temps = griddata(points, temps_at_time, (grid_x, grid_y, grid_z), method='linear')
    
    # Fill NaN values with nearest neighbor interpolation
    mask = np.isnan(grid_temps)
    if np.any(mask):
        grid_temps[mask] = griddata(points, temps_at_time, 
                                   (grid_x[mask], grid_y[mask], grid_z[mask]), 
                                   method='nearest', fill_value=np.min(temps_at_time))
    
    # Ensure no NaN values remain
    grid_temps = np.nan_to_num(grid_temps, nan=np.min(temps_at_time))
    
    # Create a mask for temperatures within certain ranges for visualization
    min_temp_frame = np.min(temps_at_time)
    max_temp_frame = np.max(temps_at_time)
    
    # Calculate levels based on the current frame's temperature range
    # This ensures we always have valid isosurfaces
    levels = np.linspace(min_temp_frame, max_temp_frame, 6)
    
    # Plot isosurfaces with different colors and transparency
    colors = ['blue', 'cyan', 'green', 'yellow', 'red']
    alphas = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    for i in range(len(levels)-1):
        try:
            # Only attempt to create isosurface if the level is within the data range
            if levels[i] < np.max(grid_temps) and levels[i] > np.min(grid_temps):
                verts, faces, _, _ = measure.marching_cubes(grid_temps, levels[i])
                # Scale vertices to match our coordinate system
                verts = verts * np.array([14/20, 14/20, 14/20])
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                               triangles=faces, color=colors[i], alpha=alphas[i])
        except Exception as e:
            print(f"Warning: Could not create isosurface at level {levels[i]}: {e}")
            continue
    
    # Plot sensor points with temperature color mapping
    scatter = ax.scatter(sensor_x, sensor_y, sensor_z, 
                        c=temps_at_time, cmap='jet', 
                        s=100, edgecolor='black', 
                        vmin=min_temp_global, vmax=max_temp_global)
    
    # Add sensor labels
    for i, txt in enumerate(sensor_column_names):
        ax.text(sensor_x[i], sensor_y[i], sensor_z[i], txt, size=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Temperature (°C)', size=14, color='white')
    cbar.ax.tick_params(labelsize=12, colors='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Set axis labels with clear directions based on oven orientation
    ax.set_xlabel('X-axis (inch) - Left to Right', fontsize=14, color='white')
    ax.set_ylabel('Y-axis (inch) - Front to Back', fontsize=14, color='white')
    ax.set_zlabel('Z-axis (inch) - Bottom to Top', fontsize=14, color='white')
    
    # Set axis limits
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.set_zlim(0, 14)
    
    # Set tick colors to white for visibility on black background
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Add grid lines
    ax.grid(True, color='white', alpha=0.3)
    
    # Add title and info text - positioned to avoid overlap
    # Position time and temperature info at the top left corner
    ax.text2D(0.05, 0.95, f"Time: {time_s:.1f}s", transform=ax.transAxes, 
             fontsize=16, color='white', verticalalignment='top')
    ax.text2D(0.05, 0.90, f"Oven: {oven_temp:.1f}°C", transform=ax.transAxes, 
             fontsize=16, color='white', verticalalignment='top')
    
    # Add title at the top center
    ax.text2D(0.5, 0.95, "3D Oven Temperature Visualization", transform=ax.transAxes,
             fontsize=18, color='white', verticalalignment='top', horizontalalignment='center')
    
    # Add a wireframe to show the oven boundaries
    xx, yy = np.meshgrid([0, 14], [0, 14])
    ax.plot_wireframe(xx, yy, np.zeros_like(xx), color='white', alpha=0.3)  # Bottom
    ax.plot_wireframe(xx, yy, np.ones_like(xx)*14, color='white', alpha=0.3)  # Top
    ax.plot_wireframe(xx, np.zeros_like(xx), np.array([[0, 14], [0, 14]]), color='white', alpha=0.3)  # Front
    ax.plot_wireframe(xx, np.ones_like(xx)*14, np.array([[0, 14], [0, 14]]), color='white', alpha=0.3)  # Back
    ax.plot_wireframe(np.zeros_like(xx), xx, np.array([[0, 14], [0, 14]]), color='white', alpha=0.3)  # Left
    ax.plot_wireframe(np.ones_like(xx)*14, xx, np.array([[0, 14], [0, 14]]), color='white', alpha=0.3)  # Right
    
    # Add door indication on the front face (y=2)
    # Door hinges on the left (x=2), handle on the right (x=12)
    door_x = [2, 12, 12, 2, 2]
    door_y = [2, 2, 2, 2, 2]
    door_z = [0, 0, 14, 14, 0]
    ax.plot(door_x, door_y, door_z, 'r-', linewidth=3, alpha=0.7)  # Door outline in red
    
    # Add handle indication on the right side of the door
    handle_x = [11, 12]
    handle_y = [2, 2]
    handle_z = [7, 7]
    ax.plot(handle_x, handle_y, handle_z, 'y-', linewidth=4)  # Handle in yellow
    
    # Add text to indicate front of oven
    ax.text(7, 1, 7, "FRONT (DOOR)", color='red', fontsize=12, 
            horizontalalignment='center', verticalalignment='center')
    
    # Set view angle to show the front of the oven (where the door is)
    # This view shows the front face with door clearly visible
    ax.view_init(elev=20, azim=-60)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', facecolor='black')
    plt.close(fig)

# --- Create animation frames ---
print("Creating animation frames...")
total_frames = min(int(DURATION_SECONDS * FPS), len(df_cleaned))
frame_indices = np.linspace(0, len(df_cleaned)-1, total_frames, dtype=int)

for i, frame_idx in enumerate(frame_indices):
    print(f"Processing frame {i+1}/{total_frames}...")
    save_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
    create_frame(frame_idx, save_path)

print(f"Animation frames saved to {OUTPUT_DIR}")
print("To create a video, you can use ffmpeg:")
print(f"ffmpeg -framerate {FPS} -i {OUTPUT_DIR}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p oven_animation.mp4")
