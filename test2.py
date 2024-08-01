import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree

## SET YOUR PARTICIPANT ID HERE
# participant_num = 'BJ7377'
# participant_num = 'AR4924'
participant_num = 'AM5287'

def calculate_velocity(df):
    # Calculate differences in time, x, and y
    df['dt'] = df['t'].diff()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()

    # Calculate velocity in m/s
    df['velocity_ms'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']

    # Handle the first row (NaN due to diff) and any potential division by zero
    df['velocity_ms'].fillna(0, inplace=True)

    return df['velocity_ms']

def smooth_velocity(velocity, window_size=50):
    # Apply a moving average to smooth the velocity
    return velocity.rolling(window=window_size, min_periods=1, center=True).mean()

def calculate_r_squared_nearest(participant_id, simulation_csv_path, human_trajectory_csv_path, viz_dir):
    # Load simulation data
    sim_df = pd.read_csv(simulation_csv_path)
    
    # Load human trajectory data, ignoring the header and using the first, second, and third columns for t, x, y
    human_df = pd.read_csv(human_trajectory_csv_path, skiprows=1, usecols=[0, 1, 2], names=['t', 'x', 'y'])
    
    # Ensure the data is sorted by time
    sim_df = sim_df.sort_values(by='TimeStep')
    
    # Ensure all values are numeric
    human_df = human_df.apply(pd.to_numeric, errors='coerce')

    # Truncate all entries at the end of the trajectory where position is zero
    human_df = human_df[(human_df[['x', 'y']] != 0).all(axis=1)]

    # Truncate first row of human data
    human_df = human_df.iloc[1:]
    sim_df = sim_df.iloc[1:]
    
    # Calculate velocity for human trajectory data
    human_df['velocity_ms'] = calculate_velocity(human_df)

    # Smooth the human velocity values
    human_df['velocity_ms'] = smooth_velocity(human_df['velocity_ms'])

    # Use the absolute value of simulation velocities
    sim_velocities = np.sqrt(sim_df['VelX']**2 + sim_df['VelY']**2)  # Assuming VelX and VelY are already in m/s
    
    # Build a KD-tree for fast spatial lookup of nearest neighbors
    human_positions = human_df[['x', 'y']].values
    human_tree = cKDTree(human_positions)
    
    # Find the nearest human position for each simulation position
    sim_positions = sim_df[['PosX', 'PosY']].values
    distances, indices = human_tree.query(sim_positions)
    
    # Extract the corresponding human velocity data
    nearest_human_velocities = human_df.iloc[indices]['velocity_ms'].values
    
    # Calculate R squared values
    r2_position = r2_score(human_positions[indices], sim_positions)
    r2_velocity = r2_score(nearest_human_velocities, sim_velocities)  # Compare velocity magnitudes
    
    # Visualization: Plot velocities
    plt.figure(figsize=(16, 10))
    # Figure title as participant ID
    plt.suptitle(participant_id)
    
    # Add r2_position and r2_velocity below the figures as text
    plt.figtext(0.25, 0.01, f"R^2 for velocity: {r2_velocity}", wrap=True, horizontalalignment='center', fontsize=12)
    plt.figtext(0.75, 0.01, f"R^2 for position: {r2_position}", wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.subplot(2, 2, 1)
    plt.plot(range(len(sim_df)), sim_velocities, 'b-', label='Simulation', linewidth=1)
    plt.plot(range(len(nearest_human_velocities)), nearest_human_velocities, 'r--', label='Human', linewidth=1)
    plt.xlabel('Index')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Comparison')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(sim_df['PosX'], -sim_df['PosY'], 'b-', markersize=3, linewidth=1, alpha=0.7)
    plt.plot(human_df['x'], -human_df['y'], 'r--', markersize=3, linewidth=1, alpha=0.7)
    plt.xlabel('PosX')
    plt.ylabel('PosY')
    plt.title('Position Comparison')
    plt.legend()
    
    # Add subplot for position comparison (x and y together)
    plt.subplot(2, 2, 3)
    plt.scatter(human_df['x'].iloc[indices], sim_df['PosX'], c='blue', s=2, linewidth=1, alpha=0.5)#, label='PosX')
    plt.scatter(human_df['y'].iloc[indices], sim_df['PosY'], c='blue', s=2, linewidth=1, alpha=0.5)#, label='PosY')
    plt.plot(human_df['x'], human_df['x'], 'r--', linewidth=1)  # Diagonal line for perfect correlation
    plt.plot(human_df['y'], human_df['y'], 'r--', linewidth=1)  # Diagonal line for perfect correlation
    plt.xlabel('Human Position')
    plt.ylabel('Sim Position')
    plt.title('Position Comparison')
    plt.legend()
    
    # Add subplot for velocity comparison
    plt.subplot(2, 2, 4)
    plt.scatter(nearest_human_velocities, sim_velocities, c='blue', s=2, alpha=0.5)
    plt.plot(nearest_human_velocities, nearest_human_velocities, 'r--', linewidth=1)  # Diagonal line for perfect correlation
    plt.xlabel('Human Velocity')
    plt.ylabel('Sim Velocity')
    plt.title('Velocity Comparison')
    
    plt.savefig(os.path.join(viz_dir, participant_id+'viz.png'))
    plt.show()
    
    return r2_position, r2_velocity

# Data files and their paths
participant_id = participant_num + 'final'
simulation_csv_path = 'simulation_log_' + participant_id + '.csv'
human_trajectory_csv_path = '/home/sangeetsu/carla_packaged/Virtuous_Vehicle_Tuner/BestPID/' + participant_id + '.csv'
viz_dir = '/home/sangeetsu/carla_packaged/Virtuous_Vehicle_Tuner/controller_comparison/'

r2_pos, r2_vel = calculate_r_squared_nearest(participant_id, simulation_csv_path, human_trajectory_csv_path, viz_dir)
print(f"Participant ID: {participant_id}")
print(f"R squared for position: {r2_pos}")
print(f"R squared for velocity: {r2_vel}")
