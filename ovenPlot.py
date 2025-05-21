import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- Configuration ---
csv_filename = "oven_data.csv"
ANIMATION_INTERVAL_MS = 100  # Milliseconds between frames (e.g., 100ms for 10 FPS)
MAX_FRAMES_FOR_DEMO = None # Set to a number (e.g., 200) for faster testing, or None for all frames

# --- Sensor Coordinate Definitions ---
# Cube dimensions: 14" x 14" x 14". Origin (0,0,0) to (14,14,14).
# X-axis: Left (smaller X) to Right (larger X)
# Y-axis: Front (smaller Y) to Back (larger Y) - as per image
# Z-axis: Bottom (smaller Z) to Top (larger Z)

x_coords_map = {'left': 2, 'center': 7, 'right': 12}  # Inches from left wall
y_coords_map = {'front': 2, 'center': 7, 'back': 12} # Inches from front wall
z_coords_map = {'bottom': 2, 'middle': 7, 'top': 12} # Inches from bottom

# Map sensor numbers (CSV column names '1' to '15') to (x,y,z) coordinates
sensor_positions_map = {
    # Layer 1 (Bottom, Z=z_coords_map['bottom']) - Sensors 1-5
    '1': (x_coords_map['left'], y_coords_map['front'], z_coords_map['bottom']),    # Front-Left (as per image interpretation)
    '2': (x_coords_map['left'], y_coords_map['back'], z_coords_map['bottom']),     # Back-Left
    '3': (x_coords_map['right'], y_coords_map['back'], z_coords_map['bottom']),    # Back-Right
    '4': (x_coords_map['right'], y_coords_map['front'], z_coords_map['bottom']),   # Front-Right
    '5': (x_coords_map['center'], y_coords_map['center'], z_coords_map['bottom']), # Center

    # Layer 2 (Middle, Z=z_coords_map['middle']) - Sensors 6-10
    '6': (x_coords_map['left'], y_coords_map['front'], z_coords_map['middle']),
    '7': (x_coords_map['left'], y_coords_map['back'], z_coords_map['middle']),
    '8': (x_coords_map['right'], y_coords_map['back'], z_coords_map['middle']),
    '9': (x_coords_map['right'], y_coords_map['front'], z_coords_map['middle']),
    '10': (x_coords_map['center'], y_coords_map['center'], z_coords_map['middle']),

    # Layer 3 (Top, Z=z_coords_map['top']) - Sensors 11-15
    '11': (x_coords_map['left'], y_coords_map['front'], z_coords_map['top']),
    '12': (x_coords_map['left'], y_coords_map['back'], z_coords_map['top']),
    '13': (x_coords_map['right'], y_coords_map['back'], z_coords_map['top']),
    '14': (x_coords_map['right'], y_coords_map['front'], z_coords_map['top']),
    '15': (x_coords_map['center'], y_coords_map['center'], z_coords_map['top'])
}

sensor_column_names = [str(i) for i in range(1, 16)]

# Extract coordinates
xs = np.array([sensor_positions_map[s][0] for s in sensor_column_names])
ys = np.array([sensor_positions_map[s][1] for s in sensor_column_names])
zs = np.array([sensor_positions_map[s][2] for s in sensor_column_names])

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script, or provide the full path.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Ensure sensor columns are numeric and drop rows with any NaN in sensor data
for col in sensor_column_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=sensor_column_names).reset_index(drop=True)

if df.empty:
    print("Error: No valid numeric temperature data found for sensors 1-15 after cleaning.")
    exit()

# Determine global min and max temperatures for consistent color scaling
all_sensor_temps = df[sensor_column_names].values.flatten()
min_temp = np.nanmin(all_sensor_temps)
max_temp = np.nanmax(all_sensor_temps)

# --- Set up the 3D Plot ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Initial scatter plot (colors will be updated in animation)
initial_temps_for_plot = df.loc[0, sensor_column_names].values.astype(float)
scat = ax.scatter(xs, ys, zs, c=initial_temps_for_plot, cmap='coolwarm', 
                  s=100, vmin=min_temp, vmax=max_temp, depthshade=True, edgecolor='black')

# Add a color bar
cbar = fig.colorbar(scat, ax=ax, pad=0.15, fraction=0.03) # Adjusted padding and fraction
cbar.set_label('Temperature (°C)')

# Set plot labels and limits
ax.set_xlabel('X-axis (inch) - Left to Right')
ax.set_ylabel('Y-axis (inch) - Front to Back')
ax.set_zlabel('Z-axis (inch) - Bottom to Top')
ax.set_xlim(0, 14)
ax.set_ylim(0, 14)
ax.set_zlim(0, 14)
ax.view_init(elev=25, azim=45) # Adjust view angle (elevation, azimuth)

# Add text annotations for sensor numbers
for i, name in enumerate(sensor_column_names):
    ax.text(xs[i] + 0.3, ys[i] + 0.3, zs[i] + 0.3, f'{name}', color='black', fontsize=7)

# Initialize title artist (will be updated in animation)
current_time_s = df.loc[0, 'Time_(s)']
current_oven_mon = df.loc[0, 'Oven_Mon']
current_rt1 = df.loc[0, 'RT1']
current_rt2 = df.loc[0, 'RT2']
initial_title_text = (f'Time: {current_time_s:.1f}s; OvenMon: {current_oven_mon:.2f}°C\n'
                      f'RT1: {current_rt1:.2f}°C; RT2: {current_rt2:.2f}°C')
title_artist = ax.set_title(initial_title_text, fontsize=10)


# --- Animation Function ---
def update_plot(frame_number, data_df, sensor_cols, scatter_artist, title_art):
    temps_at_time = data_df.loc[frame_number, sensor_cols].values.astype(float)
    
    # Update scatter plot colors
    scatter_artist.set_array(temps_at_time) # More direct way to update colors
    
    # Update title with current time and Oven_Mon temperature
    time_s = data_df.loc[frame_number, 'Time_(s)']
    oven_mon_temp = data_df.loc[frame_number, 'Oven_Mon']
    rt1_temp = data_df.loc[frame_number, 'RT1']
    rt2_temp = data_df.loc[frame_number, 'RT2']
    
    title_text = (f'Time: {time_s:.1f}s; OvenMon: {oven_mon_temp:.2f}°C\n'
                  f'RT1: {rt1_temp:.2f}°C; RT2: {rt2_temp:.2f}°C')
    title_art.set_text(title_text)
    
    return scatter_artist, title_art

# --- Create and Show Animation ---
num_total_frames = len(df)
frames_to_render = num_total_frames
if MAX_FRAMES_FOR_DEMO is not None:
    frames_to_render = min(num_total_frames, MAX_FRAMES_FOR_DEMO)

ani = animation.FuncAnimation(fig, update_plot, frames=frames_to_render,
                              fargs=(df, sensor_column_names, scat, title_artist),
                              interval=ANIMATION_INTERVAL_MS,
                              blit=True, repeat=False)

fig.suptitle("3D Oven Chamber Temperature Profile", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.93]) # Adjust rect to make space for suptitle and bottom elements

# --- Optional: Save the animation ---
# Make sure you have ffmpeg (for mp4) or pillow (for gif) installed
output_filename_mp4 = "3d_temp_profile.mp4"
output_filename_gif = "3d_temp_profile.gif"
try:
    print(f"Attempting to save animation to {output_filename_mp4}...")
    ani.save(output_filename_mp4, writer='ffmpeg', fps=max(1, 1000 // ANIMATION_INTERVAL_MS), dpi=150)
    print(f"Animation saved as {output_filename_mp4}")
except Exception as e:
     print(f"Could not save as MP4 (ffmpeg might be missing or an error occurred): {e}")
try:
    print(f"Attempting to save animation to {output_filename_gif}...")
    ani.save(output_filename_gif, writer='pillow', fps=max(1, 1000//ANIMATION_INTERVAL_MS))
    print(f"Animation saved as {output_filename_gif}")
except Exception as e_gif:
        print(f"Could not save as GIF (pillow might be missing or an error occurred): {e_gif}")

plt.show()

print(f"Displaying animation for {frames_to_render} of {num_total_frames} available time steps.")
print(f"Global temperature range for sensors {', '.join(sensor_column_names)}: {min_temp:.2f}°C to {max_temp:.2f}°C")