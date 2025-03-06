import carla
import numpy as np
import pandas as pd
import os
import sys
import time
import argparse
import torch
from PIL import Image
import pickle
from blipper import BLIPEmbedder
import math
import cv2

sys.path.append('../')
from agents.navigation.global_route_planner import GlobalRoutePlanner

# Global variables
actor_list = []
frame_buffer = []
frame_timestamps = []
capture_frames = True
embedding_save_path = "embeddings"
blip_embedder = None
save_reference_frames = False
current_participant_id = None

def traj_loader(participant_csv):
    """Load participant trajectory data from CSV"""
    df = pd.read_csv(participant_csv, header=0)
    df.columns = ['t', 'x', 'y', 'z', 'throttle', 'steering', 'brake', 'hand_brake', 'reverse', 'manual_gear_shift', 'gear']
    df['vel_mph'] = calculate_velocity(df)
    df['vel_mps'] = df['vel_mph'] / 2.23694
    return df

def calculate_velocity(df):
    """Calculate velocity from position data"""
    df['dt'] = df['t'].diff()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['velocity_ms'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']
    df['velocity_mph'] = df['velocity_ms'] * 2.23694
    df['velocity_ms'].fillna(0, inplace=True)
    df['velocity_mph'].fillna(0, inplace=True)
    return df['velocity_mph']

def calculate_yaw(df):
    """Calculate yaw angle based on trajectory direction"""
    # Calculate direction vectors between consecutive points
    dx = df['x'].diff().fillna(0)
    dy = df['y'].diff().fillna(0)
    
    # Calculate yaw angle in degrees (0 is east/right, 90 is north/up)
    yaw = np.degrees(np.arctan2(dy, dx))
    
    # Fill first value using second point direction
    if len(yaw) > 1:
        yaw.iloc[0] = yaw.iloc[1]
    
    return yaw

def process_rgb_frame(image, world, vehicle):
    """Process RGB frames captured by the camera"""
    if not capture_frames:
        return
        
    # Convert CARLA raw image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Remove alpha channel
    
    # Store the frame with its timestamp
    frame_buffer.append(array)
    timestamp = world.get_snapshot().timestamp.elapsed_seconds
    frame_timestamps.append(timestamp)
    
    # If we're saving reference frames, save each image to disk
    if save_reference_frames:
        try:
            # Create directory with absolute path to ensure it exists
            frame_dir = os.path.abspath(f"reference_frames/{current_participant_id}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # Format timestamp with precision to avoid duplicates
            frame_number = len(frame_buffer) - 1
            frame_path = f"{frame_dir}/frame_{frame_number:05d}_{timestamp:.3f}.jpg"
            
            # Save the image
            cv2.imwrite(frame_path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            
            # Print confirmation for the first few frames
            if frame_number < 5 or frame_number % 50 == 0:
                print(f"Saved frame {frame_number} to {frame_path}")
                
        except Exception as e:
            print(f"Error saving frame: {e}")

def attach_rgb_camera(world, vehicle):
    """Attach an RGB camera to the vehicle with the same position as manual_control.py third-person view"""
    # Get vehicle bounding box to properly scale camera position
    bound_x = 0.5 + vehicle.bounding_box.extent.x
    bound_y = 0.5 + vehicle.bounding_box.extent.y
    bound_z = 0.5 + vehicle.bounding_box.extent.z
    
    # Use the third-person view position from manual_control.py (index 0):
    # (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0))
    transform = carla.Transform(
        carla.Location(x=-2.0*bound_x, y=0.0*bound_y, z=2.0*bound_z),
        carla.Rotation(pitch=8.0)  # 8 degree pitch as in the original camera
    )
    
    # Set up camera with resolution suitable for BLIP2 processing
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '512')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '90')  # Match the default FOV
    
    # Create and attach the camera
    rgb_camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    actor_list.append(rgb_camera)
    
    # Set up the callback to capture frames
    rgb_camera.listen(lambda image: process_rgb_frame(image, world, vehicle))
    
    print(f"Attached third-person camera at position: x={-2.0*bound_x:.2f}, y={0.0*bound_y:.2f}, z={2.0*bound_z:.2f} with pitch=8.0")
    return rgb_camera

def generate_and_save_embeddings(participant_id):
    """Generate BLIP2 embeddings from captured frames and save them"""
    global blip_embedder, frame_buffer, frame_timestamps
    
    if not frame_buffer:
        print("No frames captured for embedding generation")
        return {}
    
    print(f"Generating embeddings for {len(frame_buffer)} frames")
    
    # Initialize BLIP embedder if not already done
    if blip_embedder is None:
        blip_embedder = BLIPEmbedder()
    
    try:
        # Generate embeddings in batches
        embeddings_dict = blip_embedder.generate_embeddings(frame_buffer, frame_timestamps)
        
        # Save embeddings
        os.makedirs(embedding_save_path, exist_ok=True)
        save_path = f"{embedding_save_path}/{participant_id}_blip_embeddings.pkl"
        
        # Save as pickle
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        
        print(f"Embeddings saved to {save_path}")
        
        # Clear buffers
        frame_buffer.clear()
        frame_timestamps.clear()
        
        return embeddings_dict
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return {}

def waypoint_gen(world, amap):
    """
    Returns a list of waypoints from the start to the end.
    This is the same function used in GA_PID.py to ensure consistency.
    """
    # Initialize the global route planner
    grp = GlobalRoutePlanner(amap, 2)
    
    # Define your custom starting locations
    custom_starting_locations = []
    for x in range (124, -82, -2):
        k = x + 0.1
        custom_starting_locations.append(carla.Location(k, -2.1, 2.1538))

    # Convert custom starting locations to waypoints using the map
    custom_starting_waypoints = [amap.get_waypoint(location) for location in custom_starting_locations]

    # Generate the route using the Global Route Planner
    spawn_points = world.get_map().get_spawn_points()
    start = carla.Location(spawn_points[0].location)
    end = carla.Location(spawn_points[13].location)
    route = grp.trace_route(start, end)
    
    # Extract locations from the route's waypoints
    route_waypoints = [wp_tuple[0] for wp_tuple in route]
    # Get rid of the last 30 points in route_waypoints
    route_waypoints = route_waypoints[:-30]

    # Prepend the custom starting locations to the route locations
    final_route = custom_starting_waypoints + route_waypoints

    return final_route

def test_reference_embeddings(participant_id, debug=False, force=False, save_frames=True):
    """
    Generate reference embeddings by capturing frames at the exact waypoints
    that will be used in the simulation
    """
    global blip_embedder, frame_buffer, frame_timestamps, capture_frames
    global save_reference_frames, current_participant_id
    
    # Set up frame saving
    save_reference_frames = save_frames
    current_participant_id = participant_id
    
    # Create reference frames directory if saving frames
    if save_frames:
        frame_dir = os.path.abspath(f"reference_frames/{participant_id}")
        os.makedirs(frame_dir, exist_ok=True)
        print(f"Will save frames to {frame_dir}/")
        # Test write permissions by creating a test file
        try:
            test_file = f"{frame_dir}/test_write.txt"
            with open(test_file, 'w') as f:
                f.write("Testing write permissions")
            os.remove(test_file)
            print("Write permissions confirmed for frames directory")
        except Exception as e:
            print(f"WARNING: Could not write to frames directory: {e}")
    
    # Check if reference embeddings already exist
    reference_embeddings_path = f"{embedding_save_path}/reference_{participant_id}_blip_embeddings.pkl"
    if os.path.exists(reference_embeddings_path) and not force:
        print(f"Reference embeddings already exist at {reference_embeddings_path}")
        print("Use --force flag to regenerate them")
        return
    
    print(f"Loading trajectory data for participant {participant_id}")
    participant_path = f'participant_data/{participant_id}final.csv'
    track_data = traj_loader(participant_path)
    
    # Add yaw calculation to track_data
    track_data['yaw'] = calculate_yaw(track_data)
    
    print("Connecting to CARLA server...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Clean up any existing actors
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.'):
            actor.destroy()
    
    # Configure synchronous mode for deterministic playback
    original_settings = world.get_settings()
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = 0.1
    world.apply_settings(new_settings)
    
    try:
        print("Spawning vehicle...")
        # Use specific blueprint model
        blueprint = world.get_blueprint_library().filter('vehicle.*model3*')[0]
        
        # Get starting position from trajectory data
        start_data = track_data.iloc[0]
        
        spawn_transform = carla.Transform(
            carla.Location(x=float(start_data['x']), y=float(start_data['y']), z=float(start_data['z'])),
            carla.Rotation(yaw=float(start_data['yaw']))
        )
        
        vehicle = world.spawn_actor(blueprint, spawn_transform)
        actor_list.append(vehicle)
        
        # Disable physics since we'll manually position the vehicle
        vehicle.set_simulate_physics(False)
        
        # Add spectator camera to follow the vehicle
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=-10, z=10), carla.Rotation(-45, 0, 0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        
        # Add RGB camera for frame capture (first-person view)
        rgb_camera = attach_rgb_camera(world, vehicle)
        
        # Setup for capturing frames
        capture_frames = True
        frame_buffer.clear()
        frame_timestamps.clear()
        
        # Get spectator to follow the vehicle
        spectator = world.get_spectator()
        
        # Generate waypoints exactly as done in GA_PID.py
        my_custom_waypoints = waypoint_gen(world, world.get_map())
        
        print(f"Generated {len(my_custom_waypoints)} waypoints for reference embedding capture")
        
        # For debugging, draw the waypoints
        if debug:
            print("Drawing waypoints...")
            for i, waypoint in enumerate(my_custom_waypoints):
                if i % 5 == 0:  # Draw every 5th point to reduce clutter
                    world.debug.draw_point(
                        waypoint.transform.location,
                        size=0.1, life_time=0, color=carla.Color(0, 255, 0)
                    )
        
        # Position the vehicle at each waypoint and capture frames
        print("Starting trajectory-based frame capture...")
        for i, waypoint in enumerate(my_custom_waypoints):
            if i % 10 == 0 or debug:
                print(f"Waypoint {i}/{len(my_custom_waypoints)}: Finding closest trajectory point")
                
            # Find the closest point in the participant trajectory to this waypoint
            waypoint_loc = np.array([waypoint.transform.location.x, waypoint.transform.location.y])
            trajectory_points = track_data[['x', 'y']].to_numpy()
            
            # Calculate distances to all trajectory points
            distances = np.sqrt(np.sum((trajectory_points - waypoint_loc)**2, axis=1))
            
            # Find index of closest point
            closest_idx = np.argmin(distances)
            closest_point = track_data.iloc[closest_idx]
            
            # Position vehicle at the closest trajectory point with participant's original orientation
            transform = carla.Transform(
                carla.Location(x=float(closest_point['x']), y=float(closest_point['y']), z=float(closest_point['z'])),
                carla.Rotation(yaw=float(closest_point['yaw']))
            )
            
            vehicle.set_transform(transform)
            
            # Draw current position
            if debug:
                world.debug.draw_point(
                    vehicle.get_location(),
                    size=0.1, life_time=0.5, 
                    color=carla.Color(r=255, g=0, b=0)
                )
                # Also show the original waypoint in a different color for comparison
                world.debug.draw_point(
                    waypoint.transform.location,
                    size=0.1, life_time=0.5,
                    color=carla.Color(r=0, g=0, b=255)
                )
            
            # Update spectator view
            spectator.set_transform(camera.get_transform())
            
            # Tick the world to advance simulation and capture frame
            world.tick()
            
            # Short sleep for debugging
            if debug:
                time.sleep(0.05)
        
        print(f"Waypoint-based capture complete, captured {len(frame_buffer)} frames")
        
        # Generate embeddings and save
        if len(frame_buffer) > 0:
            embeddings = generate_and_save_embeddings(f"reference_{participant_id}")
            print(f"Reference embeddings saved for {participant_id}")
        else:
            print("No frames were captured during waypoint traversal!")
        
    except Exception as e:
        print(f"Error during reference embedding generation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original settings
        world.apply_settings(original_settings)
        
        # Cleanup
        capture_frames = False
        print("Cleaning up actors...")
        for actor in actor_list:
            try:
                actor.destroy()
            except Exception as e:
                print(f"Error destroying actor: {e}")
        
        actor_list.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate reference BLIP embeddings from participant trajectory')
    parser.add_argument('-i', '--id', type=str, required=True, help='Participant ID')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode with visualization')
    parser.add_argument('-f', '--force', action='store_true', help='Force regenerate embeddings even if they exist')
    parser.add_argument('--no_save_frames', action='store_true', 
                        help='Disable saving individual frames to reference_frames directory')
    
    args = parser.parse_args()
    
    # Make frame saving enabled by default
    save_frames = not args.no_save_frames
    
    print("Starting reference embedding generation script")
    print(f"Frame saving is {'disabled' if args.no_save_frames else 'enabled'}")
    test_reference_embeddings(args.id, args.debug, args.force, save_frames)
    print("Done!")