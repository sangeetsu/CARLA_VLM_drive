import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
import matplotlib.patches as patches
import matplotlib.cm as cm

## SET YOUR PARTICIPANT ID HERE
#participant_num = 'BJ7377'
#participant_num = 'AR4924'
#participant_num = 'AM5287'
participant_num = 'CE3693'

def calculate_velocity(df):
    df['dt'] = df['t'].diff()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['velocity_ms'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']
    df['velocity_ms'].fillna(0, inplace=True)
    return df['velocity_ms']

def smooth_velocity(velocity, window_size=50):
    return velocity.rolling(window=window_size, min_periods=1, center=True).mean()

def is_within_zone(x, y, start, end):
    return (start[0] <= x <= end[0] or start[0] >= x >= end[0]) and (start[1] <= y <= end[1] or start[1] >= y >= end[1])

def calculate_r_squared_nearest_with_zones(participant_id, simulation_csv_path, human_trajectory_csv_path, viz_dir, zone_df, participant_num):
    ref_df = pd.read_csv("baseline_trunc.csv")
    sim_df = pd.read_csv(simulation_csv_path)
    human_df = pd.read_csv(human_trajectory_csv_path, skiprows=1, usecols=[0, 1, 2], names=['t', 'x', 'y'])
    sim_df = sim_df.sort_values(by='TimeStep')
    ref_df = ref_df.sort_values(by='time')
    human_df = human_df.apply(pd.to_numeric, errors='coerce')
    human_df = human_df[(human_df[['x', 'y']] != 0).all(axis=1)]

    #Open the Related JSON for the ERROR VALUES
    jsonfile = 'JSONBASE/'+participant_num+'_GAgains.json'
    with open(jsonfile, 'r') as theFile:
        gains = json.load(theFile)
        velError = gains['fitness_values']['Velocity_Error']
        posError = gains['fitness_values']['Position_Error']

    # Begin Converstion to Reference Point (I.E. everything the human does to sim, have both do to ref)
   # Get the last position from the simulation data
    last_sim_pos = sim_df[['PosX', 'PosY']].iloc[-1].values
    last_ref_pos = ref_df[['x','y']].iloc[-1].values

    # Calculate the Euclidean distance of each human/sim trajectory point from the last ref point
    human_distances = np.sqrt((human_df['x'] - last_ref_pos[0])**2 + (human_df['y'] - last_ref_pos[1])**2)
    sim_distances = np.sqrt((sim_df['PosX'] - last_ref_pos[0])**2 + (sim_df['PosY'] - last_ref_pos[1])**2)

    # Truncate human_df/sim_df where the distance starts increasing significantly beyond the reference's last point
    cutoff_idxH = human_distances.idxmin()
    cutoff_idxS = sim_distances.idxmin()

    # Trim the human trajectory to this cutoff index
    human_df = human_df.iloc[:cutoff_idxH + 1].reset_index(drop=True)
    sim_df = sim_df.iloc[:cutoff_idxS + 1].reset_index(drop=True)


    human_df = human_df.iloc[1:]
    sim_df = sim_df.iloc[1:]
    ref_df = ref_df.iloc[1:]

    # MAY HAVE TO MAKE A CHANGE.... HERE?
    human_df['velocity_ms'] = calculate_velocity(human_df)
    human_df['velocity_ms'] = smooth_velocity(human_df['velocity_ms'])
    human_positions = human_df[['x', 'y']].values
    human_tree = cKDTree(human_positions)
    sim_velocities = np.sqrt(sim_df['VelX']**2 + sim_df['VelY']**2)
    sim_positions = sim_df[['PosX', 'PosY']].values
    sim_tree = cKDTree(sim_positions)
    ref_positions = ref_df[['x','y']].values
    distancesH, indicesH = human_tree.query(ref_positions)
    distancesS, indicesS = sim_tree.query(ref_positions)
    nearest_human_velocities = human_df.iloc[indicesH]['velocity_ms'].values
    nearest_sim_velocities = sim_velocities[indicesS].values
    #r2_position = r2_score(human_positions[indicesH], sim_positions)
    # For this do a second run that matches up human to sim to get ACCURATE R^2... versus finding only the indices that line up to the ref...? Maybe I'll just do the conglom error for both? 
    distancesH2,indicesH2 = human_tree.query(sim_positions)
    nearest_human_velocities2 = human_df.iloc[indicesH2]['velocity_ms'].values
    r2_velocity = r2_score(nearest_human_velocities2, sim_velocities)
    #********************************** NOTE TO SELF, THIS IS WHERE THE CHANGE FOR R^2 HAPPENS *****************

    plt.figure(figsize=(18,6))
    plt.suptitle(participant_id)
    plt.figtext(0.1, 0.01, f"R^2 (velocity): {r2_velocity}", wrap=True, horizontalalignment='center', fontsize=12)
    #plt.figtext(0.9, 0.01, f"R^2 (position): {r2_position}", wrap=True, horizontalalignment='center', fontsize=12)
    plt.figtext(0.9, 0.01, f"Positional Error: {posError}", wrap=True, horizontalalignment='center', fontsize=12)
    landmarks_df = pd.read_csv('assets/plot_landmarks.csv')

    # Uncomment if you want velocities in mph
    nearest_sim_velocities = nearest_sim_velocities * 2.23694
    nearest_human_velocities = nearest_human_velocities * 2.23694

    # Calculate percentage of track completed
    track_percentage = np.linspace(0, 100, len(ref_df))

    # Find the range of track_percentage corresponding to zones
    start_zone = zone_df.iloc[0]  # First zone (Zone A)
    end_zone = zone_df.iloc[-1]   # Last zone

    start_zone_indices = ref_df.apply(lambda row: is_within_zone(row['x'], row['y'], (start_zone['x1'], start_zone['y1']), (start_zone['x2'], start_zone['y2'])), axis=1)
    end_zone_indices = ref_df.apply(lambda row: is_within_zone(row['x'], row['y'], (end_zone['x1'], end_zone['y1']), (end_zone['x2'], end_zone['y2'])), axis=1)

    # Calculate the start and end percentage
    first_zone_idx = np.where(start_zone_indices)[0][0]

    # Check if any points were found in the end_zone
    end_zone_idx_array = np.where(end_zone_indices)[0]

    # If end_zone has points, use the last one; otherwise, set a default (e.g., the last point of sim_df)
    if len(end_zone_idx_array) > 0:
        last_zone_idx = end_zone_idx_array[-1]
    else:
        last_zone_idx = len(sim_df) - 1  # Set to the last point in sim_df if no points in end_zone

    start_percentage = track_percentage[first_zone_idx]
    end_percentage = track_percentage[last_zone_idx]

    plt.subplot(1, 2, 1)
    plt.plot(track_percentage, nearest_sim_velocities, 'b-', label='Simulation', linewidth=1)
    plt.plot(track_percentage, nearest_human_velocities, 'r--', label='Human', linewidth=1)

    # Fix y-axis range and add alphabet labels at a fixed y-position
    plt.ylim(0, 120)
    plt.xlim(start_percentage, end_percentage)  # Limit x-axis to start and end zone

    zone_labels = list("ABCDEFGH")  # Labels for zones

    zone_label_y = 110  # Fixed y-position for the zone labels

    for i, (_, zone) in enumerate(zone_df.iterrows()):
        color = 'grey' if i % 2 == 0 else 'white'  # Alternate grey and white for zones
        zone_indices = ref_df.apply(lambda row: is_within_zone(row['x'], row['y'], (zone['x1'], zone['y1']), (zone['x2'], zone['y2'])), axis=1)
        in_zone_indices = [idx for idx, in_zone in enumerate(zone_indices) if in_zone]

        if in_zone_indices:
            plt.axvspan(track_percentage[in_zone_indices[0]], track_percentage[in_zone_indices[-1]], color=color, alpha=0.3)
            mid_idx = in_zone_indices[len(in_zone_indices) // 2]
            plt.text(track_percentage[mid_idx], zone_label_y, zone_labels[i], fontsize=14, color='black', ha='center')


    # Convert sim_velocities and track_percentage to numpy arrays for safer indexing
    nearest_sim_velocities = np.array(nearest_sim_velocities)
    nearest_human_velocities = np.array(nearest_human_velocities)
    track_percentage = np.array(track_percentage)

    # Calculate interval for placing 10 triangles
    interval = max(1, len(sim_df) // 10)  # Ensure interval is at least 1 to avoid division by zero

    # Get 10 evenly spaced triangle indices within the bounds of the data
    triangle_indices = np.linspace(0, len(ref_df) - 1, num=10, dtype=int)

    # Add triangles to the velocity plots
    plt.plot(track_percentage[triangle_indices], nearest_sim_velocities[triangle_indices], 'v', color='blue', label='Simulation Markers', markersize=8)
    plt.plot(track_percentage[triangle_indices], nearest_human_velocities[triangle_indices], 'v', color='red', label='Human Markers', markersize=8)

    plt.xlabel('Track Completed (%)')
    plt.ylabel('Velocity (mph)')
    plt.title('Velocity Comparison')
    plt.legend(loc='lower right')  # Move the legend to the bottom right

    plt.subplot(1, 2, 2)
    plt.plot(-sim_df['PosX'], sim_df['PosY'], 'b-', markersize=3, linewidth=1, alpha=0.7)
    plt.plot(-human_df['x'], human_df['y'], 'r--', markersize=3, linewidth=1, alpha=0.7)
    plt.scatter(-0.01*landmarks_df['x'], 0.01*landmarks_df['y'], color='red', marker='^', label='Landmarks')
    labels_assets = landmarks_df['property']
    for i, label in enumerate(labels_assets):
        plt.text(-0.01*landmarks_df['x'][i], 0.01*landmarks_df['y'][i], label, fontsize=9, color='black')

    for i, (_, zone) in enumerate(zone_df.iterrows()):
        color = 'grey' if i % 2 == 0 else 'white'  # Match the shading to velocity comparison
        rect = patches.Rectangle((-zone['x1'], zone['y1']),
                                -(zone['x2'] - zone['x1']), (zone['y2'] - zone['y1']),
                                linewidth=1, edgecolor='black', facecolor=color, alpha=0.3)
        plt.gca().add_patch(rect)
        plt.text(-0.5 * (zone['x1'] + zone['x2']), 0.5 * (zone['y1'] + zone['y2']), f"Zone {zone_labels[i]}",
                fontsize=10, color='black', verticalalignment='center', horizontalalignment='center')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Position Comparison')
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, participant_id + 'viz_with_zones.png'))
    
    r2_position = 0
    return r2_position, r2_velocity

# Data files and their paths
participant_id = participant_num + 'final'
simulation_csv_path = 'simulation_log_' + participant_id + '.csv'
human_trajectory_csv_path = 'BestPID/' + participant_id + '.csv'
viz_dir = 'controller_comparison'

# Load the zone list CSV
zone_list_path = 'assets/zone_list.csv'
zone_df = pd.read_csv(zone_list_path)

r2_pos, r2_vel = calculate_r_squared_nearest_with_zones(participant_id, simulation_csv_path, human_trajectory_csv_path, viz_dir, zone_df,participant_num)
print(f"Participant ID: {participant_id}")
print(f"R squared for position: {r2_pos}")
print(f"R squared for velocity: {r2_vel}")
