import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For 2D plots
from scipy.interpolate import griddata
from mayavi import mlab # The main Mayavi module for scripting

# --- Configuration ---
csv_filename = "oven_data.csv"
ANIMATION_INTERVAL_MS = 200  # Milliseconds between frames
MAX_FRAMES_FOR_DEMO = 100    # Set to None for all frames, or a number for testing

# --- Sensor Coordinate Definitions --- (Same as your previous correct setup)
x_coords_map = {'left': 2, 'center': 7, 'right': 12}
y_coords_map = {'front': 2, 'center': 7, 'back': 12}
z_coords_map = {'bottom': 2, 'middle': 7, 'top': 12}
sensor_positions_map = {
    '1':(x_coords_map['left'],y_coords_map['front'],z_coords_map['bottom']),'2':(x_coords_map['left'],y_coords_map['back'],z_coords_map['bottom']),
    '3':(x_coords_map['right'],y_coords_map['back'],z_coords_map['bottom']),'4':(x_coords_map['right'],y_coords_map['front'],z_coords_map['bottom']),
    '5':(x_coords_map['center'],y_coords_map['center'],z_coords_map['bottom']),'6':(x_coords_map['left'],y_coords_map['front'],z_coords_map['middle']),
    '7':(x_coords_map['left'],y_coords_map['back'],z_coords_map['middle']),'8':(x_coords_map['right'],y_coords_map['back'],z_coords_map['middle']),
    '9':(x_coords_map['right'],y_coords_map['front'],z_coords_map['middle']),'10':(x_coords_map['center'],y_coords_map['center'],z_coords_map['middle']),
    '11':(x_coords_map['left'],y_coords_map['front'],z_coords_map['top']),'12':(x_coords_map['left'],y_coords_map['back'],z_coords_map['top']),
    '13':(x_coords_map['right'],y_coords_map['back'],z_coords_map['top']),'14':(x_coords_map['right'],y_coords_map['front'],z_coords_map['top']),
    '15':(x_coords_map['center'],y_coords_map['center'],z_coords_map['top'])}
sensor_column_names = [str(i) for i in range(1, 16)]
sensor_points_xyz_np = np.array([sensor_positions_map[s] for s in sensor_column_names])
sensor_x, sensor_y, sensor_z = sensor_points_xyz_np[:,0], sensor_points_xyz_np[:,1], sensor_points_xyz_np[:,2]

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()
df_cleaned = df.dropna(subset=sensor_column_names).reset_index(drop=True)
for col in sensor_column_names: # Ensure numeric after potential NA drop
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=sensor_column_names).reset_index(drop=True)

if df_cleaned.empty:
    print("Error: No valid numeric temperature data found after cleaning.")
    exit()

all_sensor_temps = df_cleaned[sensor_column_names].values.flatten()
min_temp_global = np.nanmin(all_sensor_temps)
max_temp_global = np.nanmax(all_sensor_temps)
print(f"Global temperature range for sensors: {min_temp_global:.2f}°C to {max_temp_global:.2f}°C")

# --- Grid for Interpolation ---
grid_res_pts = 30j # Use complex number for mgrid to specify number of points
grid_x_mg, grid_y_mg, grid_z_mg = np.mgrid[
    0:14:grid_res_pts,
    0:14:grid_res_pts,
    0:14:grid_res_pts
]

# --- Mayavi 3D Animated Volumetric Plot with Isosurfaces ---
def plot_animated_mayavi_volume_iso(data_df):
    num_frames = len(data_df) if MAX_FRAMES_FOR_DEMO is None else min(len(data_df), MAX_FRAMES_FOR_DEMO)
    
    # Create figure with dark background
    fig_mlab = mlab.figure(bgcolor=(0.05, 0.05, 0.1), size=(1200, 900))
    fig_mlab.scene.anti_aliasing_frames = 8
    
    # --- Initial Frame Data ---
    temps_at_time_0 = data_df.loc[0, sensor_column_names].values.astype(float)
    
    # Better interpolation with both linear and nearest neighbor methods
    volumetric_temp_data_0 = griddata(sensor_points_xyz_np, temps_at_time_0,
                                     (grid_x_mg, grid_y_mg, grid_z_mg), method='linear')
    
    # Fill NaN values with nearest neighbor interpolation
    mask = np.isnan(volumetric_temp_data_0)
    if np.any(mask):
        volumetric_temp_data_0[mask] = griddata(sensor_points_xyz_np, temps_at_time_0,
                                              (grid_x_mg[mask], grid_y_mg[mask], grid_z_mg[mask]), 
                                              method='nearest')

    min_temp_frame0 = np.nanmin(volumetric_temp_data_0)
    max_temp_frame0 = np.nanmax(volumetric_temp_data_0)
    
    src = mlab.pipeline.scalar_field(grid_x_mg, grid_y_mg, grid_z_mg, volumetric_temp_data_0)
    
    # --- Volume Rendering with MUCH higher opacity ---
    vol = mlab.pipeline.volume(src, vmin=min_temp_global, vmax=max_temp_global)
    volume_property = vol.volume_property
    otf = volume_property.get_scalar_opacity()
    otf.remove_all_points()
    
    # Much more aggressive opacity settings to show the "fluid gas"
    otf.add_point(min_temp_global, 0.1)  # Start with some opacity
    otf.add_point(min_temp_global + (max_temp_global - min_temp_global) * 0.25, 0.2)
    otf.add_point(min_temp_global + (max_temp_global - min_temp_global) * 0.5, 0.4)
    otf.add_point(min_temp_global + (max_temp_global - min_temp_global) * 0.75, 0.6)
    otf.add_point(max_temp_global, 0.8)  # High opacity for maximum temperature
    
    # Use a vibrant colormap
    vol.module_manager.scalar_lut_manager.lut_mode = 'jet'
    
    # --- Sensor visualization ---
    # Larger, more visible sensor points
    sensor_viz_points = mlab.points3d(
        sensor_x, sensor_y, sensor_z,
        scale_mode='none', 
        scale_factor=0.5,  # Larger points
        color=(1, 1, 0),  # Bright yellow for visibility
        resolution=16  # Smoother spheres
    )
    
    # Add clear sensor labels
    for i, name in enumerate(sensor_column_names):
        mlab.text3d(
            sensor_x[i] + 0.2, 
            sensor_y[i] + 0.2, 
            sensor_z[i] + 0.2, 
            str(name), 
            scale=0.5,  # Larger text
            color=(1, 1, 1)  # White text
        )
    
    # --- UI elements with proper positioning ---
    # Colorbar with larger font
    cb = mlab.scalarbar(vol, title='Temperature (°C)', orientation='vertical', nb_labels=8)
    cb.scalar_bar.unconstrained_font_size = True
    cb.label_text_property.font_size = 16  # Much larger font
    cb.label_text_property.bold = True
    cb.title_text_property.font_size = 18  # Much larger title
    cb.title_text_property.bold = True
    cb.scalar_bar_representation.position = [0.88, 0.1]  # Position on right
    cb.scalar_bar_representation.position2 = [0.1, 0.8]  # Size
    
    # Axes with clear labels
    axes = mlab.axes(
        xlabel='X-axis: Left to Right', 
        ylabel='Y-axis: Front to Back', 
        zlabel='Z-axis: Bottom to Top',
        ranges=[0, 14, 0, 14, 0, 14],
        color=(1, 1, 1),  # White axes
        line_width=2.0,  # Thicker lines
        nb_labels=8  # More labels
    )
    
    # Improve axes appearance
    axes.axes.font_factor = 1.5  # Much larger font
    axes.axes.label_format = '%.0f'  # Integer format
    axes.label_text_property.font_size = 14
    axes.label_text_property.bold = True
    axes.title_text_property.font_size = 16
    axes.title_text_property.bold = True
    
    # Add outline
    outline = mlab.outline(color=(0.8, 0.8, 0.8), line_width=2.0)
    
    # --- Position text information in separate locations to avoid overlap ---
    # Title at top center
    title_obj = mlab.title(
        "3D Oven Temperature Visualization", 
        height=0.95,
        color=(1, 1, 1),
        size=0.5
    )
    
    # Time and oven text at top left with NO overlap
    time_text_obj = mlab.text(0.01, 0.95, "Time: 0.0s", width=0.3, color=(1, 1, 1), figure=fig_mlab)
    oven_text_obj = mlab.text(0.01, 0.90, "Oven: 0.0°C", width=0.3, color=(1, 1, 1), figure=fig_mlab)
    
    # Set initial view
    mlab.view(azimuth=45, elevation=50, distance=35, focalpoint=(7, 7, 7))
    
    @mlab.animate(delay=ANIMATION_INTERVAL_MS, ui=True)
    def anim_loop_func():
        for frame_idx in range(num_frames):
            if frame_idx % 10 == 0:
                print(f"Rendering frame {frame_idx}/{num_frames}...")
                
            temps_at_time = data_df.loc[frame_idx, sensor_column_names].values.astype(float)
            
            # Better interpolation for visualization
            new_volumetric_data = griddata(
                sensor_points_xyz_np, 
                temps_at_time,
                (grid_x_mg, grid_y_mg, grid_z_mg), 
                method='linear'
            )
            
            # Fill NaN values with nearest neighbor interpolation
            mask = np.isnan(new_volumetric_data)
            if np.any(mask):
                new_volumetric_data[mask] = griddata(
                    sensor_points_xyz_np, 
                    temps_at_time,
                    (grid_x_mg[mask], grid_y_mg[mask], grid_z_mg[mask]), 
                    method='nearest'
                )
            
            # Update the scalar field
            src.mlab_source.scalars = new_volumetric_data
            
            # Update text displays with clear positioning
            time_s = data_df.loc[frame_idx, 'Time_(s)']
            oven_mon_temp = data_df.loc[frame_idx, 'Oven_Mon']
            time_text_obj.text = f"Time: {time_s:.1f}s"
            oven_text_obj.text = f"Oven: {oven_mon_temp:.1f}°C"
            
            yield
    
    ani_instance = anim_loop_func()
    mlab.show()
    return ani_instance

# --- Static Isosurface Plot (Mayavi - already robust, but check fill_value) ---
def plot_mayavi_static_isosurfaces(data_df, time_index, num_isosurfaces=5):
    mlab.figure(bgcolor=(0.9, 0.9, 0.9), fgcolor=(0.1,0.1,0.1), size=(1000, 800))
    mlab.clf()

    temps_at_time = data_df.loc[time_index, sensor_column_names].values.astype(float)
    # For marching cubes, it's often better to fill with NaN and let it handle missing regions
    # OR fill with a value guaranteed to be outside the isosurface levels you pick.
    # If you fill with np.nan, ensure marching_cubes handles it or mask the array.
    volumetric_temp_data = griddata(sensor_points_xyz_np, temps_at_time,
                                     (grid_x_mg, grid_y_mg, grid_z_mg), method='linear',
                                     fill_value=np.nan) # CHANGED to NaN for marching_cubes

    min_t_step = np.nanmin(volumetric_temp_data)
    max_t_step = np.nanmax(volumetric_temp_data)

    if np.isnan(min_t_step) or np.isnan(max_t_step) or abs(max_t_step - min_t_step) < 0.1: # Ensure a valid range
        print(f"Cannot generate useful isosurfaces for time index {time_index}. Interpolated data range too small or all NaN.")
        print(f"Interpolated temp range: {min_t_step} to {max_t_step}")
        mlab.close(all=True) # Close the empty figure
        return
    
    iso_levels_raw = np.linspace(min_t_step + (max_t_step - min_t_step) * 0.05, # Start closer to actual min
                             max_t_step - (max_t_step - min_t_step) * 0.05, # End closer to actual max
                             num_isosurfaces)
    iso_levels = np.round(np.unique(iso_levels_raw), 1) # Ensure unique and rounded
    
    if len(iso_levels) == 0:
        print(f"Could not define distinct isosurface levels for time index {time_index}.")
        mlab.close(all=True)
        return
        
    print(f"Plotting {len(iso_levels)} isosurfaces at temperatures: {iso_levels}°C")
    
    # Mask NaN values for marching_cubes
    masked_volume = np.ma.masked_invalid(volumetric_temp_data)
    # Fill masked values with something outside the expected contour range, or handle error
    filled_volume = masked_volume.filled(fill_value=min_t_step - 1) 


    src = mlab.pipeline.scalar_field(grid_x_mg, grid_y_mg, grid_z_mg, filled_volume) # Use filled volume
    
    colormap = plt.cm.get_cmap('plasma', len(iso_levels) + 2) 
    opacities = np.linspace(0.1, 0.4, len(iso_levels)) 

    for i, level in enumerate(iso_levels):
        try:
            color_tuple = colormap(float(i+1) / (len(iso_levels)+1))[:3] 
            mlab.pipeline.iso_surface(src, contours=[level], opacity=opacities[i], color=color_tuple)
        except Exception as e_iso: # Catch errors if a specific contour fails
            print(f"Could not generate isosurface for level {level}°C: {e_iso}")


    sensor_plot = mlab.points3d(sensor_x, sensor_y, sensor_z, temps_at_time,
                                scale_mode='none', scale_factor=0.5,
                                colormap='coolwarm', vmin=min_temp_global, vmax=max_temp_global)
    mlab.colorbar(sensor_plot, title='Sensor Temp (°C)', orientation='vertical', nb_labels=5)
    
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', ranges=[0,14,0,14,0,14])
    mlab.outline()
    time_s = data_df.loc[time_index, 'Time_(s)']
    mlab.title(f"3D Isosurfaces at Time: {time_s:.1f}s", height=0.95, size=0.4)
    mlab.view(azimuth=135, elevation=70, distance='auto')
    mlab.show()


# --- 2D Matplotlib Plots (keep as is or adapt as needed) ---
def plot_2d_analysis(data_df, time_index):
    # (This function can be the same as the one in the previous response)
    # ... (pasting it here for completeness) ...
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

    grid_x2d_fine_mp, grid_y2d_fine_mp = np.mgrid[0:14:100j, 0:14:100j]
    try:
        grid_temps2d = griddata(np.column_stack((middle_coords_x, middle_coords_y)), middle_temps,
                                (grid_x2d_fine_mp, grid_y2d_fine_mp), method='cubic', fill_value=np.nanmin(middle_temps))
    except ValueError:
        grid_temps2d = griddata(np.column_stack((middle_coords_x, middle_coords_y)), middle_temps,
                                (grid_x2d_fine_mp, grid_y2d_fine_mp), method='linear', fill_value=np.nanmin(middle_temps))

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
    print("Ensure Mayavi and its dependencies (VTK, PyQt5/PySide2) are correctly installed.")
    print("If Mayavi windows don't appear or are blank, try running from a standard terminal.")
    print("-" * 30)
    print("Choose visualization type:")
    print("1: Animated 3D Volumetric Plot with Isosurfaces (Mayavi)")
    print("2: Static 3D Multiple Isosurfaces Plot (Mayavi)")
    print("3: 2D Analysis Plots (Matplotlib)")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        print("Starting Mayavi animation. This may take a moment...")
        print("You can interact with the Mayavi window (rotate, zoom).")
        print("Close the Mayavi window when done with the animation or to stop.")
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
        
        num_iso = 4
        try:
            num_iso_str = input(f"Enter number of isosurfaces to display (e.g., {num_iso}, press Enter for default): ")
            num_iso = int(num_iso_str) if num_iso_str else num_iso
        except ValueError:
            print(f"Invalid number, using default {num_iso} isosurfaces.")

        plot_mayavi_static_isosurfaces(df_cleaned, time_index, num_isosurfaces=num_iso)
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