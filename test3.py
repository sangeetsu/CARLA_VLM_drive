import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree

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

def calculate_r_squared_nearest(simulation_csv_path, human_trajectory_csv_path):
    # Load simulation data
    sim_df = pd.read_csv(simulation_csv_path)
    
    # Load human trajectory data, ignoring the header and using the first, second, and third columns for t, x, y
    human_df = pd.read_csv(human_trajectory_csv_path, skiprows=1, usecols=[0, 1, 2], names=['t', 'x', 'y'])
    
    # Ensure the data is sorted by time
    sim_df = sim_df.sort_values(by='TimeStep')
    
    # Ensure all values are numeric
    human_df = human_df.apply(pd.to_numeric, errors='coerce')
    
    # Calculate velocity for human trajectory data
    human_df['velocity_ms'] = calculate_velocity(human_df)
    
    # Convert simulation velocity to m/s if needed (example conversion, modify based on your data's units)
    # Assuming simulation velocities are in some unit that needs conversion
    # Convert simulation velocity from km/h to m/s (if needed, modify as necessary)
    # sim_df['VelX'] = sim_df['VelX'] * 1000 / 3600  # Example conversion if sim_df['VelX'] is in km/h
    
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
    
    # Debugging: Print some matched velocities from the middle of the simulation
    middle_index = len(sim_positions) // 2
    print("\nSample matched velocities (simulation -> human) at middle index:")
    for i in range(middle_index-2, middle_index+3):
        print(f"Sim Pos: {sim_positions[i]} Sim Vel: {sim_velocities.iloc[i]} -> Human Vel: {nearest_human_velocities[i]}")
    
    # Visualization: Plot velocities
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(sim_df)), sim_velocities, 'b-', label='Simulation VelX')
    plt.plot(range(len(nearest_human_velocities)), nearest_human_velocities, 'r--', label='Human Vel')
    plt.xlabel('Index')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(sim_df['PosX'], sim_df['PosY'], 'bo-', label='Simulation Position')
    plt.plot(human_df['x'], human_df['y'], 'ro-', label='Human Position')
    plt.xlabel('PosX')
    plt.ylabel('PosY')
    plt.title('Position Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return r2_position, r2_velocity

# Example usage after the simulation loop
simulation_csv_path = 'simulation_log_DP8189old.csv'
human_trajectory_csv_path = '/home/sangeetsu/carla_packaged/Virtuous_Vehicle_Tuner/participant_data/DP8189final.csv'

r2_pos, r2_vel = calculate_r_squared_nearest(simulation_csv_path, human_trajectory_csv_path)
print(f"R squared for position: {r2_pos}")
print(f"R squared for velocity: {r2_vel}")
