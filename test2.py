import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
import matplotlib.patches as patches
import matplotlib.cm as cm

## SET YOUR PARTICIPANT ID HERE
# participant_num = 'BJ7377'
participant_num = 'AR4924'
# participant_num = 'AM5287'

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

def calculate_r_squared_nearest_with_zones(participant_id, simulation_csv_path, human_trajectory_csv_path, viz_dir, zone_df):
    sim_df = pd.read_csv(simulation_csv_path)
    human_df = pd.read_csv(human_trajectory_csv_path, skiprows=1, usecols=[0, 1, 2], names=['t', 'x', 'y'])
    sim_df = sim_df.sort_values(by='TimeStep')
    human_df = human_df.apply(pd.to_numeric, errors='coerce')
    human_df = human_df[(human_df[['x', 'y']] != 0).all(axis=1)]
    human_df = human_df.iloc[1:]
    sim_df = sim_df.iloc[1:]
    
    human_df['velocity_ms'] = calculate_velocity(human_df)
    human_df['velocity_ms'] = smooth_velocity(human_df['velocity_ms'])
    sim_velocities = np.sqrt(sim_df['VelX']**2 + sim_df['VelY']**2)
    human_positions = human_df[['x', 'y']].values
    human_tree = cKDTree(human_positions)
    sim_positions = sim_df[['PosX', 'PosY']].values
    distances, indices = human_tree.query(sim_positions)
    nearest_human_velocities = human_df.iloc[indices]['velocity_ms'].values
    r2_position = r2_score(human_positions[indices], sim_positions)
    r2_velocity = r2_score(nearest_human_velocities, sim_velocities)

    plt.figure(figsize=(18,6))
    plt.suptitle(participant_id)
    plt.figtext(0.1, 0.01, f"R^2 (velocity): {r2_velocity}", wrap=True, horizontalalignment='center', fontsize=12)
    plt.figtext(0.9, 0.01, f"R^2 (position): {r2_position}", wrap=True, horizontalalignment='center', fontsize=12)
    landmarks_df = pd.read_csv('assets/plot_landmarks.csv')
    school_zone_start = (0.01 * landmarks_df[landmarks_df['property'] == 'SZ']['x'].values[0], -60)
    school_zone_end = (0.01 * landmarks_df[landmarks_df['property'] == 'SZ']['x'].values[1], -40)
    school_zone_indices = sim_df.apply(lambda row: is_within_zone(row['PosX'], row['PosY'], school_zone_start, school_zone_end), axis=1)

    #Uncomment if you want velocities in mph
    sim_velocities = sim_velocities * 2.23694
    nearest_human_velocities = nearest_human_velocities * 2.23694

    # Calculate percentage of track completed
    track_percentage = np.linspace(0, 100, len(sim_df))

    plt.subplot(1, 2, 1)
    plt.plot(track_percentage, sim_velocities, 'b-', label='Simulation', linewidth=1)
    plt.plot(track_percentage, nearest_human_velocities, 'r--', label='Human', linewidth=1)

    # Alternate grey and white highlights for zones, with Zone 3 highlighted in yellow
    for i, (_, zone) in enumerate(zone_df.iterrows()):
        if zone['zone'] == 3:
            color = 'yellow'  # Highlight Zone 3 in yellow
        else:
            color = 'lightgrey' if i % 2 == 0 else 'white'  # Alternate grey and white for other zones
        
        zone_indices = sim_df.apply(lambda row: is_within_zone(row['PosX'], row['PosY'], (zone['x1'], zone['y1']), (zone['x2'], zone['y2'])), axis=1)
        for idx, in_zone in enumerate(zone_indices):
            if in_zone:
                if idx < len(track_percentage) - 1:
                    plt.axvspan(track_percentage[idx], track_percentage[idx+1], color=color, alpha=0.3)
                else:
                    plt.axvspan(track_percentage[idx], track_percentage[idx], color=color, alpha=0.3)

    plt.xlabel('Track Completed (%)')
    plt.ylabel('Velocity (mph)')
    plt.title('Velocity Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(-sim_df['PosX'], sim_df['PosY'], 'b-', markersize=3, linewidth=1, alpha=0.7)
    plt.plot(-human_df['x'], human_df['y'], 'r--', markersize=3, linewidth=1, alpha=0.7)
    plt.scatter(-0.01*landmarks_df['x'], 0.01*landmarks_df['y'], color='red', marker='^', label='Landmarks')
    labels_assets = landmarks_df['property']
    for i, label in enumerate(labels_assets):
        plt.text(-0.01*landmarks_df['x'][i], 0.01*landmarks_df['y'][i], label, fontsize=9, color='black')

    # Define a list of colors or use a colormap
    colors = cm.get_cmap('tab10', len(zone_df))
    # colors = 'lightgrey' if i % 2 == 0 else 'white'

    for i, (_, zone) in enumerate(zone_df.iterrows()):
        color = colors(i)
        rect = patches.Rectangle((-zone['x1'], zone['y1']),
                                -(zone['x2'] - zone['x1']), (zone['y2'] - zone['y1']),
                                linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(-0.5 * (zone['x1'] + zone['x2']), 0.5 * (zone['y1'] + zone['y2']), "School Zone (3.0)" if zone['zone']==3 else f"Zone {zone['zone']}",
                fontsize=10, color=color, verticalalignment='center', horizontalalignment='center')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Position Comparison')
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, participant_id + 'viz_with_zones.png'))
    
    return r2_position, r2_velocity

# Data files and their paths
participant_id = participant_num + 'final'
simulation_csv_path = 'simulation_log_' + participant_id + '.csv'
human_trajectory_csv_path = 'BestPID/' + participant_id + '.csv'
viz_dir = 'controller_comparison'

# Load the zone list CSV
zone_list_path = 'assets/zone_list.csv'
zone_df = pd.read_csv(zone_list_path)

r2_pos, r2_vel = calculate_r_squared_nearest_with_zones(participant_id, simulation_csv_path, human_trajectory_csv_path, viz_dir, zone_df)
print(f"Participant ID: {participant_id}")
print(f"R squared for position: {r2_pos}")
print(f"R squared for velocity: {r2_vel}")
