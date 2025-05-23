import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

# --- Style ---
plt.style.use('dark_background')

# --- Constants ---
OVEN_SIZE = 14.0
# NEW MARGIN for sensors to be 0.5" from the wall (13" cube in 14" oven)
SENSOR_MARGIN_FROM_WALL = 0.1

NUM_PLANES = 3
SENSORS_PER_PLANE = 5
TOTAL_SENSORS = NUM_PLANES * SENSORS_PER_PLANE
GRID_DENSITY_3D_VOLUME = 40
GRID_DENSITY_SLICE_HEATMAP = 40

# --- Configuration ---
USER_DATA_FILE = "ovendata.csv"
CSV_TIME_COLUMN = "Time_(s)"
CSV_SENSOR_COLUMNS = [str(i) for i in range(1, 16)]
OVEN_MON_COLUMN = "Oven_Mon"
CSV_EXTRA_COLUMNS_FOR_3DTITLE = ["RT1", "RT2"]

# --- Sensor Setup ---
def get_sensor_coordinates_and_labels():
    coords = []; display_labels = []
    # Use the new SENSOR_MARGIN_FROM_WALL
    margin = SENSOR_MARGIN_FROM_WALL
    center_xy = OVEN_SIZE / 2.0 # Center remains the same

    # Define coordinates based on the margin
    front_y = margin
    back_y = OVEN_SIZE - margin
    left_x = margin
    right_x = OVEN_SIZE - margin

    # Sensor layout per plane (FL, BL, BR, FR, Center)
    xy_pattern_per_plane = [
        (left_x, front_y),       # 1: Front-Left
        (left_x, back_y),        # 2: Back-Left
        (right_x, back_y),       # 3: Back-Right
        (right_x, front_y),      # 4: Front-Right
        (center_xy, center_xy)   # 5: Center
    ]
    z_spacing = OVEN_SIZE / (NUM_PLANES + 1)
    # Z-levels can also be adjusted if the "13-inch cube" implies Z extent too
    # For now, assume Z is still spaced within the full 14" height.
    # If Z also needs to be within a 13" Z-height, then z_levels would need adjustment.
    # Let's assume Z is spaced relative to SENSOR_MARGIN_FROM_WALL for bottom/top planes.
    # Effective height for sensor placement: OVEN_SIZE - 2 * SENSOR_MARGIN_FROM_WALL
    z_start = SENSOR_MARGIN_FROM_WALL
    z_end = OVEN_SIZE - SENSOR_MARGIN_FROM_WALL
    if NUM_PLANES > 1:
        z_levels = np.linspace(z_start, z_end, NUM_PLANES)
    else:
        z_levels = [OVEN_SIZE / 2.0] # Single plane in center
    # Sticking to original Z spacing logic for now, but making them relative to full oven height.
    #z_levels = [z_spacing * (i + 1) for i in range(NUM_PLANES)]


    sensor_num_offset = 0
    for plane_idx, z_coord in enumerate(z_levels):
        for pattern_idx, (x_coord, y_coord) in enumerate(xy_pattern_per_plane):
            coords.append((x_coord, y_coord, z_coord)); display_labels.append(str(sensor_num_offset + pattern_idx + 1))
        sensor_num_offset += SENSORS_PER_PLANE
    return np.array(coords), display_labels, z_levels

PHYSICAL_SENSOR_COORDS, PHYSICAL_SENSOR_LABELS, Z_LEVELS_ACTUAL = get_sensor_coordinates_and_labels()

# Identify sensors for the NEW XZ Slice Heatmap (Front Face Y = SENSOR_MARGIN_FROM_WALL)
SLICE_Y_VALUE = SENSOR_MARGIN_FROM_WALL # Front face
SENSORS_ON_XZ_SLICE_INDICES = [
    i for i, coord in enumerate(PHYSICAL_SENSOR_COORDS)
    if np.isclose(coord[1], SLICE_Y_VALUE) # Check if Y-coordinate is at the front
]
SENSORS_ON_XZ_SLICE_LABELS = [PHYSICAL_SENSOR_LABELS[i] for i in SENSORS_ON_XZ_SLICE_INDICES]
SENSORS_ON_XZ_SLICE_XZ_COORDS = PHYSICAL_SENSOR_COORDS[SENSORS_ON_XZ_SLICE_INDICES, ::2] # Get X, Z coords (index 0 and 2)

# --- Load Data & Temp Range Calc (Same as before) ---
try:
    temp_df = pd.read_csv(USER_DATA_FILE)
    print(f"Successfully loaded data from: {USER_DATA_FILE}")
    if OVEN_MON_COLUMN not in temp_df.columns:
        print(f"Warning: Column '{OVEN_MON_COLUMN}' not found. Creating dummy column."); temp_df[OVEN_MON_COLUMN] = np.nan
except FileNotFoundError: print(f"ERROR: Data file '{USER_DATA_FILE}' not found."); exit()
except Exception as e: print(f"Error loading data: {e}"); exit()
temp_df['Average_Temp'] = temp_df[CSV_SENSOR_COLUMNS].mean(axis=1)
all_times = temp_df[CSV_TIME_COLUMN].values
all_temp_series = [temp_df[col].values.astype(float) for col in CSV_SENSOR_COLUMNS]
if OVEN_MON_COLUMN in temp_df.columns and not temp_df[OVEN_MON_COLUMN].isnull().all():
    all_temp_series.append(temp_df[OVEN_MON_COLUMN].values.astype(float))
all_valid_temps = np.concatenate([series[~np.isnan(series)] for series in all_temp_series])
TEMP_MIN_ALL = np.min(all_valid_temps) if len(all_valid_temps) > 0 else 0
TEMP_MAX_ALL = np.max(all_valid_temps) if len(all_valid_temps) > 0 else 100
TEMP_RANGE_ALL = TEMP_MAX_ALL - TEMP_MIN_ALL
if TEMP_RANGE_ALL == 0: TEMP_RANGE_ALL = 1
print(f"Overall Temperature Range: Min={TEMP_MIN_ALL:.2f}°C, Max={TEMP_MAX_ALL:.2f}°C")


# --- Visualization Setup ---
fig = plt.figure(figsize=(20, 11))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.9, 1.1], height_ratios=[1.6, 1],
                       hspace=0.35, wspace=0.22, left=0.08, right=0.96, top=0.93, bottom=0.07)
ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
ax_line_plots = fig.add_subplot(gs[0, 1])
ax_slice_heatmap = fig.add_subplot(gs[1, 0])
ax_qual_analysis = fig.add_subplot(gs[1, 1]); ax_qual_analysis.axis('off')

# Interpolation grids
xi_3d_vol = np.linspace(0, OVEN_SIZE, GRID_DENSITY_3D_VOLUME); yi_3d_vol = np.linspace(0, OVEN_SIZE, GRID_DENSITY_3D_VOLUME); zi_3d_vol = np.linspace(0, OVEN_SIZE, GRID_DENSITY_3D_VOLUME)
xg_3d_vol, yg_3d_vol, zg_3d_vol = np.meshgrid(xi_3d_vol, yi_3d_vol, zi_3d_vol, indexing='ij')

# For XZ Slice Heatmap (at Y = SLICE_Y_VALUE)
xs_slice = np.linspace(0, OVEN_SIZE, GRID_DENSITY_SLICE_HEATMAP)
zs_slice = np.linspace(0, OVEN_SIZE, GRID_DENSITY_SLICE_HEATMAP)
xg_slice_xz, zg_slice_xz = np.meshgrid(xs_slice, zs_slice, indexing='ij') # X varies along rows, Z along columns
yg_slice_xz = np.full_like(xg_slice_xz, SLICE_Y_VALUE) # Y is constant
# Query points for griddata will be (N, 3) for the slice
slice_query_points_xz = np.stack((xg_slice_xz.ravel(), yg_slice_xz.ravel(), zg_slice_xz.ravel()), axis=-1)

cmap_3d_volume = plt.get_cmap('inferno'); norm_3d_volume = Normalize(vmin=TEMP_MIN_ALL, vmax=TEMP_MAX_ALL)
cmap_slice_heatmap = plt.get_cmap('coolwarm'); norm_slice_heatmap = Normalize(vmin=TEMP_MIN_ALL, vmax=TEMP_MAX_ALL)
# ... (Placeholders - same as before) ...
volume_scatter = None; sensor_scatters_list = []; sensor_annotations_list = []
avg_chamber_temp_line = None; current_avg_chamber_marker = None
oven_mon_line = None; current_oven_mon_marker = None
slice_heatmap_img = None; fig_cbar_3d_obj = None; fig_cbar_slice_obj = None; analysis_text_obj = None


def draw_oven_outline(ax_obj): # ... (same as before) ...
    verts = [(0,0,0),(OVEN_SIZE,0,0),(OVEN_SIZE,OVEN_SIZE,0),(0,OVEN_SIZE,0),(0,0,OVEN_SIZE),(OVEN_SIZE,0,OVEN_SIZE),(OVEN_SIZE,OVEN_SIZE,OVEN_SIZE),(0,OVEN_SIZE,OVEN_SIZE)]
    edges = [[verts[0],verts[1]],[verts[1],verts[2]],[verts[2],verts[3]],[verts[3],verts[0]],[verts[4],verts[5]],[verts[5],verts[6]],[verts[6],verts[7]],[verts[7],verts[4]],[verts[0],verts[4]],[verts[1],verts[5]],[verts[2],verts[6]],[verts[3],verts[7]]]
    for edge in edges: ax_obj.plot(*zip(*edge),color='lightgray',linestyle='--',alpha=0.6)

# --- Animation Function ---
def update_plot(frame_num):
    global volume_scatter, sensor_scatters_list, sensor_annotations_list, avg_chamber_temp_line, current_avg_chamber_marker, oven_mon_line, current_oven_mon_marker, slice_heatmap_img, fig_cbar_3d_obj, fig_cbar_slice_obj, analysis_text_obj
    ax_3d.cla(); ax_line_plots.cla(); ax_slice_heatmap.cla(); ax_qual_analysis.cla(); ax_qual_analysis.axis('off')
    sensor_scatters_list.clear(); sensor_annotations_list.clear()
    current_time_val = all_times[frame_num]
    current_temps_all_sensors = temp_df.iloc[frame_num][CSV_SENSOR_COLUMNS].values.astype(float)
    current_avg_chamber_temp = temp_df['Average_Temp'].iloc[frame_num]
    current_oven_mon_temp = temp_df[OVEN_MON_COLUMN].iloc[frame_num] if OVEN_MON_COLUMN in temp_df.columns and not pd.isna(temp_df[OVEN_MON_COLUMN].iloc[frame_num]) else None

    # === 3D Plot Update (Volumetric Scatter) ===
    # ... (Same as before, using xg_3d_vol, yg_3d_vol, zg_3d_vol) ...
    query_points_3d_vol = np.stack((xg_3d_vol.ravel(), yg_3d_vol.ravel(), zg_3d_vol.ravel()), axis=-1)
    fill_val_3d = TEMP_MIN_ALL - (TEMP_RANGE_ALL*0.1) if TEMP_RANGE_ALL>0 else TEMP_MIN_ALL-1
    try: interpolated_temps_flat_3d_vol = griddata(PHYSICAL_SENSOR_COORDS, current_temps_all_sensors, query_points_3d_vol, method='linear', fill_value=fill_val_3d)
    except Exception: interpolated_temps_flat_3d_vol = griddata(PHYSICAL_SENSOR_COORDS, current_temps_all_sensors, query_points_3d_vol, method='nearest')
    interpolated_temps_flat_3d_vol = np.nan_to_num(interpolated_temps_flat_3d_vol, nan=fill_val_3d)
    alpha_values_flat_3d = np.clip((interpolated_temps_flat_3d_vol - TEMP_MIN_ALL) / TEMP_RANGE_ALL if TEMP_RANGE_ALL > 0 else 0, 0.0, 1.0)**1.5
    alpha_values_flat_3d = np.clip(alpha_values_flat_3d, 0.01, 0.5)
    colors_flat_3d = cmap_3d_volume(norm_3d_volume(interpolated_temps_flat_3d_vol)); colors_flat_3d[:, 3] = alpha_values_flat_3d
    volume_scatter = ax_3d.scatter(xg_3d_vol.ravel(), yg_3d_vol.ravel(), zg_3d_vol.ravel(), c=colors_flat_3d, s=50, marker='o', depthshade=True, edgecolors='none')
    sensor_colors_3d = cmap_3d_volume(norm_3d_volume(current_temps_all_sensors))
    sensor_sizes = np.clip(30 + 120 * (np.clip((current_temps_all_sensors - TEMP_MIN_ALL) / TEMP_RANGE_ALL if TEMP_RANGE_ALL > 0 else 0, 0, 1)), 30, 150)
    for i in range(TOTAL_SENSORS):
        sc = ax_3d.scatter(PHYSICAL_SENSOR_COORDS[i,0], PHYSICAL_SENSOR_COORDS[i,1], PHYSICAL_SENSOR_COORDS[i,2], c=[sensor_colors_3d[i]], s=sensor_sizes[i], marker='X', edgecolor='lightgray', linewidth=0.7, depthshade=True)
        sensor_scatters_list.append(sc)
        an = ax_3d.text(PHYSICAL_SENSOR_COORDS[i,0]+0.3, PHYSICAL_SENSOR_COORDS[i,1]+0.3, PHYSICAL_SENSOR_COORDS[i,2], f"{PHYSICAL_SENSOR_LABELS[i]}\n{current_temps_all_sensors[i]:.1f}°C", fontsize=7, color='lightgray', ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.1", fc="dimgray", ec="none", alpha=0.7))
        sensor_annotations_list.append(an)
    ax_3d.set_xlim(0,OVEN_SIZE); ax_3d.set_ylim(0,OVEN_SIZE); ax_3d.set_zlim(0,OVEN_SIZE)
    ax_3d.set_xlabel("X (width)",labelpad=10,color='lightgray'); ax_3d.set_ylabel("Y (depth)",labelpad=10,color='lightgray'); ax_3d.set_zlabel("Z (height)",labelpad=10,color='lightgray')
    title_3d_extras = " | ".join([f"{col}: {temp_df[col].iloc[frame_num]:.1f}°C" for col in CSV_EXTRA_COLUMNS_FOR_3DTITLE if col in temp_df.columns])
    ax_3d.set_title(f"3D Gradient @ Time: {current_time_val:.1f}s\n{title_3d_extras}", fontsize=10, color='lightgray', y=1.02)
    draw_oven_outline(ax_3d); ax_3d.view_init(elev=25, azim=-120)

    # === Line Plots ===
    # ... (Same as before) ...
    avg_chamber_temp_line, = ax_line_plots.plot(all_times[:frame_num+1], temp_df['Average_Temp'].iloc[:frame_num+1], color='deepskyblue', lw=2, label=f'Avg Chamber ({current_avg_chamber_temp:.1f}°C)')
    current_avg_chamber_marker, = ax_line_plots.plot(current_time_val, current_avg_chamber_temp, 'o', color='red', markersize=7)
    if current_oven_mon_temp is not None:
        oven_mon_line, = ax_line_plots.plot(all_times[:frame_num+1], temp_df[OVEN_MON_COLUMN].iloc[:frame_num+1].dropna(), color='springgreen', lw=2, label=f'{OVEN_MON_COLUMN} ({current_oven_mon_temp:.1f}°C)')
        current_oven_mon_marker, = ax_line_plots.plot(current_time_val, current_oven_mon_temp, 'o', color='yellow', markersize=7)
    ax_line_plots.set_xlim(all_times[0],all_times[-1]); ax_line_plots.set_ylim(TEMP_MIN_ALL-5, TEMP_MAX_ALL+10)
    ax_line_plots.set_xlabel("Time (s)",color='lightgray'); ax_line_plots.set_ylabel("Temperature (°C)",color='lightgray')
    ax_line_plots.set_title("Chamber Temperatures Over Time", fontsize=10, color='lightgray')
    ax_line_plots.legend(loc='upper left', fontsize=8, facecolor='dimgray', edgecolor='lightgray', labelcolor='lightgray'); ax_line_plots.grid(True, linestyle=':', alpha=0.5, color='gray')

    # === XZ SLICE Heatmap (Front Face at Y = SENSOR_MARGIN_FROM_WALL) ===
    try:
        interpolated_temps_slice_flat_xz = griddata(
            PHYSICAL_SENSOR_COORDS, current_temps_all_sensors,
            slice_query_points_xz, method='linear', fill_value=TEMP_MIN_ALL
        )
    except Exception:
        interpolated_temps_slice_flat_xz = griddata(
            PHYSICAL_SENSOR_COORDS, current_temps_all_sensors,
            slice_query_points_xz, method='nearest'
        )
    interpolated_temps_slice_grid_xz = np.nan_to_num(interpolated_temps_slice_flat_xz, nan=TEMP_MIN_ALL).reshape(xg_slice_xz.shape)
    
    # xg_slice_xz has X varying along rows (dim 0), zg_slice_xz has Z varying along columns (dim 1)
    # For imshow to have X on x-axis and Z on y-axis, we plot the transpose.
    slice_heatmap_img = ax_slice_heatmap.imshow(interpolated_temps_slice_grid_xz.T, origin='lower', aspect='auto',
                                               extent=[0, OVEN_SIZE, 0, OVEN_SIZE], # X-range, Z-range
                                               cmap=cmap_slice_heatmap, norm=norm_slice_heatmap)
    ax_slice_heatmap.set_title(f"Front Face (X-Z) Heatmap at Y={SLICE_Y_VALUE:.1f}\" @ {current_time_val:.1f}s", fontsize=9, color='lightgray')
    ax_slice_heatmap.set_xlabel("X-axis (width, inch)", color='lightgray')
    ax_slice_heatmap.set_ylabel("Z-axis (height, inch)", color='lightgray')

    # Plot markers for sensors that are ON this front XZ slice
    sensors_on_slice_temps_xz = current_temps_all_sensors[SENSORS_ON_XZ_SLICE_INDICES]
    sensors_on_slice_marker_colors_xz = cmap_slice_heatmap(norm_slice_heatmap(sensors_on_slice_temps_xz))
    for i, global_idx in enumerate(SENSORS_ON_XZ_SLICE_INDICES):
        x_coord, z_coord = SENSORS_ON_XZ_SLICE_XZ_COORDS[i]
        actual_temp = sensors_on_slice_temps_xz[i]
        ax_slice_heatmap.plot(x_coord, z_coord, 'o', ms=10, mec='white', mfc=sensors_on_slice_marker_colors_xz[i], mew=1.5)
        ax_slice_heatmap.text(x_coord + 0.3, z_coord + 0.3, f"{PHYSICAL_SENSOR_LABELS[global_idx]}\n{actual_temp:.1f}°C",
                               color='white', fontsize=7, ha='left', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.1", fc="dimgray", ec="none", alpha=0.7))

    # === Qualitative Analysis Text ===
    # Using center sensors T5, T10, T15 for Z-gradient as before
    center_column_indices = [i for i, coord in enumerate(PHYSICAL_SENSOR_COORDS) if np.isclose(coord[0], OVEN_SIZE/2) and np.isclose(coord[1], OVEN_SIZE/2)]
    if len(center_column_indices) >=2 : # Need at least two points for gradient
        center_bottom_temp = current_temps_all_sensors[center_column_indices[0]]
        center_top_temp = current_temps_all_sensors[center_column_indices[-1]]
        delta_z_center = PHYSICAL_SENSOR_COORDS[center_column_indices[-1], 2] - PHYSICAL_SENSOR_COORDS[center_column_indices[0], 2]
        z_gradient_center = (center_top_temp - center_bottom_temp) / delta_z_center if delta_z_center else 0
    else:
        z_gradient_center = np.nan
    std_dev_all_sensors = np.std(current_temps_all_sensors)
    analysis_text = (f"Qualitative Analysis & Gradients:\n\n"
                     f"Time: {current_time_val:.1f}s\n"
                     f"Center Z-Gradient (Top-Bottom): {z_gradient_center:.2f} °C/inch\n"
                     f"Temp. Uniformity (StdDev): {std_dev_all_sensors:.2f} °C\n\n"
                     f"Notes:\n- Heatmap shows Front Face (XZ plane at Y={SLICE_Y_VALUE:.1f}\").")
    analysis_text_obj = ax_qual_analysis.text(0.05,0.95,analysis_text,transform=ax_qual_analysis.transAxes,fontsize=9,color='lightgray',va='top',ha='left',bbox=dict(boxstyle="round,pad=0.5",fc="dimgray",ec="none",alpha=0.5))

    # === Colorbars (once) ===
    if frame_num == 0: # ... (Colorbar logic same as before, using ax_slice_heatmap for the second cbar) ...
        if fig_cbar_3d_obj is None:
            cbar_3d_ax = fig.add_axes([ax_3d.get_position().x1 + 0.01, ax_3d.get_position().y0, 0.015, ax_3d.get_position().height])
            fig_cbar_3d_obj = fig.colorbar(cm.ScalarMappable(norm=norm_3d_volume, cmap=cmap_3d_volume), cax=cbar_3d_ax)
            fig_cbar_3d_obj.set_label('Temp (°C)',color='lightgray',labelpad=5); fig_cbar_3d_obj.ax.tick_params(labelsize=8,colors='lightgray')
        if fig_cbar_slice_obj is None:
            cbar_slice_ax = fig.add_axes([ax_slice_heatmap.get_position().x1 + 0.01, ax_slice_heatmap.get_position().y0, 0.015, ax_slice_heatmap.get_position().height])
            fig_cbar_slice_obj = fig.colorbar(cm.ScalarMappable(norm=norm_slice_heatmap, cmap=cmap_slice_heatmap), cax=cbar_slice_ax)
            fig_cbar_slice_obj.set_label('Temp (°C)',color='lightgray',labelpad=5); fig_cbar_slice_obj.ax.tick_params(labelsize=8,colors='lightgray')
    
    returned_artists = ([volume_scatter] + sensor_scatters_list + sensor_annotations_list +
                        [avg_chamber_temp_line, current_avg_chamber_marker, slice_heatmap_img, analysis_text_obj] +
                        list(ax_slice_heatmap.get_children()))
    if current_oven_mon_temp is not None and oven_mon_line: returned_artists.extend([oven_mon_line, current_oven_mon_marker])
    return returned_artists

# --- Create and Run Animation ---
num_frames = len(temp_df) if not temp_df.empty else 0
if num_frames == 0: print("No data to animate. Exiting."); exit()
ani = animation.FuncAnimation(fig, update_plot, frames=num_frames, interval=50, blit=False, repeat=False)

# Main title (big font, bold, lightgray) using fig.text
fig.text(0.19, 0.995, "Asteroid Ovens - Thermal Analysis Dashboard",
         ha='center', va='top', fontsize=20, fontweight='bold', color='darkblue')

# Author name (smaller font, below main title, darkorange) using fig.text
# Adjust the vertical position (y) to place it below the main title
fig.text(0.041, 0.96, "By: Gerald Jackson",
         ha='center', va='top', fontsize=11, color='darkorange') # Adjust y as needed

print("Saving animation...")
ani.save('oven_analysis_layout_v3.mp4', writer='ffmpeg', fps=10, dpi=150, progress_callback=lambda i, n: print(f'Saving frame {i+1} of {n}'))
print("Animation saved.")

plt.show()