import pandas as pd
import numpy as np
import os

# Your calculate_velocity function here
#def calculate_velocity(t, x, y):
#    velocity_ms = np.zeros(len(t))#in m/s
#    velocity_mph = np.zeros(len(t))#in mph
#    for i in range(1, len(t)):
#        dt = t[i] - t[i-1]
#        dx = x[i] - x[i-1]
#        dy = y[i] - y[i-1]
#        dx_m = dx# * 0.1643
#        dy_m = dy# * 0.1643
#        if dt != 0:
#            velocity_ms[i] = np.sqrt(dx_m**2 + dy_m**2) / dt# in m/s
#            velocity_mph[i] = velocity_ms[i] * 2.23694# in mph
#    return velocity_ms



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


# Function to process each CSV file
def process_csv(file_path):
    # Read the CSV file into a DataFrame
    #df = pd.read_csv(file_path)
    # print(df.columns)
    # Rename 'Unnamed: 0' to 't'
    #df.rename(columns={'Unnamed: 0': 't'}, inplace=True)
    
    # Calculate velocities
    #t = df['t'].values
    #x = df['x'].values
    #y = df['y'].values
    #velocity_mph = calculate_velocity(t, x, y)
    # Calculate velocity for human trajectory data


     # Load human trajectory data, ignoring the header and using the first, second, and third columns for t, x, y
    human_df = pd.read_csv(file_path, skiprows=1, usecols=[0, 1, 2, 3], names=['t', 'x', 'y', 'z'])
    
    # Ensure all values are numeric
    human_df = human_df.apply(pd.to_numeric, errors='coerce')

    # Truncate all entries at the end of the trajectory where position is zero
    human_df = human_df[(human_df[['x', 'y']] != 0).all(axis=1)]

    #truncate first row of human data
    human_df = human_df.iloc[1:]
    
    # Calculate velocity for human trajectory data
    human_df['velocity_ms'] = calculate_velocity(human_df)

    # Smooth the human velocity values
    human_df['velocity_ms'] = smooth_velocity(human_df['velocity_ms'])
    print(human_df)


    
    # Insert the 'velocity' column before the 'throttle' column
    #throttle_index = df.columns.get_loc('throttle')
    #df.insert(throttle_index, 'velocity', velocity_mph)

    # Save the modified DataFrame to the 'combined_scrapes' directory
    print(combined_dir)
    human_df.to_csv(os.path.join(combined_dir, os.path.basename(file_path)), index=False, header=True)
    print(f'Processed file {file_path}')

dir = os.getcwd()
parent_dir = os.path.dirname(dir)
print('parent_dir', parent_dir)
finals_mac = ['/home/mommymythra/Carla/tuner/Virtuous_Vehicle_Tuner/BestPID']
csv_paths = finals_mac
combined_dir = os.path.join(parent_dir, 'Virtuous_Vehicle_Tuner/combined_scrapes')
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
