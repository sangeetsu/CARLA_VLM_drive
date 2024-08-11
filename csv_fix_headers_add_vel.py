import pandas as pd
import numpy as np
import os

# Your calculate_velocity function here
def calculate_velocity(t, x, y):
    velocity_ms = np.zeros(len(t))#in m/s
    velocity_mph = np.zeros(len(t))#in mph
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dx_m = dx# * 0.1643
        dy_m = dy# * 0.1643
        if dt != 0:
            velocity_ms[i] = np.sqrt(dx_m**2 + dy_m**2) / dt# in m/s
            velocity_mph[i] = velocity_ms[i] * 2.23694# in mph
    return velocity_ms


# Function to process each CSV file
def process_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # print(df.columns)
    # Rename 'Unnamed: 0' to 't'
    df.rename(columns={'Unnamed: 0': 't'}, inplace=True)
    
    # Calculate velocities
    t = df['t'].values
    x = df['x'].values
    y = df['y'].values
    velocity_mph = calculate_velocity(t, x, y)
    
    # Insert the 'velocity' column before the 'throttle' column
    throttle_index = df.columns.get_loc('throttle')
    df.insert(throttle_index, 'velocity', velocity_mph)

    # Save the modified DataFrame to the 'combined_scrapes' directory
    df.to_csv(os.path.join(combined_dir, os.path.basename(file_path)), index=False, header=True)
    print(f'Processed file {file_path}')

dir = os.getcwd()
parent_dir = os.path.dirname(dir)
print('parent_dir', parent_dir)
finals_mac = ['/home/mommymythra/Carla/tuner/Virtuous_Vehicle_Tuner/BestPID']
csv_paths = finals_mac
combined_dir = os.path.join(parent_dir, 'combined_scrapes')
if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)
print(combined_dir)
# Process all CSV files
for path in csv_paths:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_csv(file_path)
