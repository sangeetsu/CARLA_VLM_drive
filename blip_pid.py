import carla
import random
import math
import numpy as np
import time
import json
import os
import pandas as pd
import pygad as pg
import threading
import argparse
# import transforms3d
#import sklearn
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import controller as myPID
sys.path.append('../')
from agents.navigation.global_route_planner import GlobalRoutePlanner
import torch
from PIL import Image
from blipper import BLIPEmbedder
import os

#Numpy precision setup
np.set_printoptions(precision=25, suppress=True)

actor_list = [] 
waypoint = carla.Location()
angle = 0
lock = threading.Lock()

#Collision flag
collFlag = False
# Blip2 golbal variables
blip_embedder = None
capture_frames = True
frame_buffer = []
frame_timestamps = []
embedding_save_path = "embeddings"

def initialize_blip_embedder():
    global blip_embedder
    if (blip_embedder is None):
        print("Initializing BLIP embedder...")
        blip_embedder = BLIPEmbedder()
        print("BLIP embedder initialized successfully")

# Function is just a waypoint check because if my waypoint is at the end, I want to do a low error: they made it
# Param:
#   waypoint - a pre-selected waypoint from the carla map waypoint list
#   waylist - the full list of waypoints
# Return:
#   Boolean - If waypoint is in the last 15, True, else False
def waypoint_last_15_check(waypoint,waylist):
    wayselect = waylist[-15:]
    for wp in wayselect:
        if waypoint is wp:
            return True
    return False

# This function is the fitness function of the GA
# It takes in the GA instance and the solution set (6 PID values)
# And calls the run_simulator function, which takes in 6 PID values and returns
# a list of errors
# These errors are input into findMeanError to get values used for calculating fitness
# Param:
#   ga_instance - an instance of the genetic algorithm
#   solution - 6 PID values, 1 safety buffer value, and 1 static speed_adherance value in a list
#   solution_idx - id of the solution set in the population
# Return:
#   fitness1 - 1/MSE of trajectory error
#   fitness2 - 1/MSE of velocity error
def fitness_func(ga_instance, solution, solution_idx):
    global participant_id
    # print("Current Solution: ", solution)
    error = run_simulator(solution)
    #initialize blip embedder
    initialize_blip_embedder()
    
    # Generate BLIP embeddings from captured frames
    embeddings_dict = generate_and_save_embeddings(participant_id)
    
    # Calculate standard error metrics
    for actor in actor_list:
        try:
            actor.destroy()
        except Exception as e:
            print(f"Error occurred while destroying actor: {e}")
    actor_list.clear()
    trajE, velE = findMeanError(error)
    
    # Load reference embeddings if available
    reference_embeddings_path = f"{embedding_save_path}/reference_{participant_id.replace('final', '')}_blip_embeddings.pkl"
    blip_similarity = 1.0  # Default value if we can't calculate similarity
    
    # Debug information
    print(f"Reference embeddings path: {reference_embeddings_path}")
    print(f"File exists: {os.path.exists(reference_embeddings_path)}")
    print(f"BLIP embedder initialized: {blip_embedder is not None}")

    # If we have reference embeddings, calculate similarity
    if os.path.exists(reference_embeddings_path) and blip_embedder is not None:
        try:
            reference_dict = blip_embedder.load_embeddings(reference_embeddings_path)
            
            # Debug info about embeddings
            print(f"Reference dict type: {type(reference_dict)}")
            print(f"Reference dict keys: {list(reference_dict.keys())[:5]}")
            print(f"Reference embedding type: {type(list(reference_dict.values())[0])}")
            
            print(f"Current embeddings dict type: {type(embeddings_dict)}")
            print(f"Current embeddings keys: {list(embeddings_dict.keys())[:5]}")
            print(f"Current embedding type: {type(list(embeddings_dict.values())[0])}")
            
            # Extract embeddings from dictionaries - handle different possible structures
            if isinstance(list(reference_dict.values())[0], np.ndarray):
                # Direct embedding values
                current_embeddings = list(embeddings_dict.values())
                reference_embeddings = list(reference_dict.values())
            else:
                # Nested structure (e.g., timestamp -> {embedding, metadata})
                print("Detected nested embedding structure, extracting...")
                current_embeddings = [v['embedding'] if isinstance(v, dict) and 'embedding' in v 
                                     else v for v in embeddings_dict.values()]
                reference_embeddings = [v['embedding'] if isinstance(v, dict) and 'embedding' in v 
                                       else v for v in reference_dict.values()]
            
            # Calculate similarity using DTW
            blip_similarity = blip_embedder.get_fitness_component(reference_embeddings, current_embeddings)
            print(f"BLIP2 Similarity score: {blip_similarity}")
        except Exception as e:
            print(f"Error calculating BLIP similarity: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping BLIP similarity calculation - missing requirements")
    
    # Apply BLIP similarity as a weighting factor (70% trajectory, 30% visual)
    visualWeight = 0.3
    trajWeight = 0.7
    finalFit = (1/trajE) * (trajWeight + visualWeight * blip_similarity) 
    finalFit2 = 1/velE
    print("Trajectory Error: ", trajE, " Velocity Error: ", velE, " BLIP2 Similarity: ", blip_similarity)
    return [finalFit, finalFit2]

# This function acts as a total error calculation from the array that is passed in
# sums all of trajectory reward and velocity reward, which is a pre-squared error value
# from the simulation. The function with square every error value, since desired value will be 0
# It then finishes the MSE calculation by dividing by the length of the array. 
# Param:
#   error - list of lists containing each timestep of trajectory or velocity reward (error)
# Return:
#   trajE - MSE of trajectory error
#   velE - MSE of velocity error
def findMeanError(error):
    trajE = 0
    velE = 0
    for e in error:
        trajE += (e[0] ** 2)
        velE += (e[1] ** 2)
    trajE = trajE / len(error)
    velE = velE / len(error)
    return trajE, velE

# This function runs the simulator for the fitness function
# Param:
#   PIDInput: list of 6 PID values in ORDER of KPt, KIt, KDt, KPv, KIs,KDs
# Return:
#   rewardsArr - array of "reward" values that act as error in the fitness func
def run_simulator(PIDInput):
    print("STARTING CARLA CLIENT SIMULATION")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    # global world
    world = client.get_world()
    spectator = world.get_spectator()

    rewardsArr = []
    counter = 0
    global capture_frames, frame_buffer, frame_timestamps
    capture_frames = True
    frame_buffer.clear()
    frame_timestamps.clear()

    # This section is for controlling the timestep. 
    # Old timestep for 60 fps = 0.0167
    new_settings = world.get_settings()
    # These enable some substepping 
    # new_settings.substepping = True
    # new_settings.max_substep_delta_time = 0.02
    # new_settings.max_substeps=16
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = .1
    world.apply_settings(new_settings) 

    spawn_actor(world)
    my_custom_waypoints = waypoint_gen(world, world.get_map())

    # Vehicle properties setup
    # print(actor_list)
    # global actor_list, vehicle, camera
    camera = actor_list[1]
    vehicle = actor_list[0]
    
    # Add RGB camera for frame capture
    rgb_camera = attach_rgb_camera(world, vehicle)
    
    # Rest of setup code
    physics_control = vehicle.get_physics_control()
    max_steer = physics_control.wheels[0].max_steer_angle
    rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
    offset = rear_axle_center - vehicle.get_location()
    wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
    vehicle.set_simulate_physics(True)
    throttle_brake_pid = myPID.PIDLongitudinalController(vehicle,PIDInput[0], PIDInput[1], PIDInput[2],world.get_settings().fixed_delta_seconds)
    steering_pid = myPID.PIDLateralController(vehicle,PIDInput[3], PIDInput[4], PIDInput[5],world.get_settings().fixed_delta_seconds)
    
    # This allows for pure pursuit weighting should the PID controller be insufficient. 
    # Default setting is pid is weighteed to be 1.
    pp_weight = 0
    pid_weight = 1 - pp_weight

    snap = world.get_snapshot()
    # Grab the velocity adherance value
    velAdh = PIDInput[7]
    # initial waypoint grab. Functionally just initializes a variable.  
    waypoint = get_next_waypoint(world, vehicle, my_custom_waypoints, PIDInput[6])
    while counter < 480:
        # Get simulation time
        frame = world.get_snapshot().frame
        # Update the camera view
        spectator.set_transform(camera.get_transform())

        # Get next waypoint
        waypoint = get_next_waypoint(world, vehicle, my_custom_waypoints, PIDInput[6])

        # Control structure accounts for grabbing an faulty waypoint
        # If grabbed, give a full set of high reward (error)
        # Else if the waypoint is in the last 15, give full reward (0 error)
        # Else run the loop
        if waypoint is None:
            print("faulty_waypoint")
            ranger = int(counter * 10)
            for count in range(ranger,7500,1):
                rewardsArr.append([100,100])
            counter = 750 
        elif waypoint_last_15_check(waypoint, my_custom_waypoints):
            ranger = int(counter * 10)
            for count in range(ranger,7500,1):
                rewardsArr.append([0,0])
            counter = 750
        else:
            world.debug.draw_point(waypoint.transform.location, life_time=5)

            # Get the current transform of the ego vehicle
            transform = vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            current_x = location.x
            current_y = location.y
            current_z = location.z
            current_throttle = vehicle.get_control().throttle
            current_brake = vehicle.get_control().brake
            current_steering = vehicle.get_control().steer
            current_velocity_3D = vehicle.get_velocity()
            current_velocity = np.sqrt(current_velocity_3D.x**2 + current_velocity_3D.y**2 + current_velocity_3D.z**2)
            current_heading = rotation.yaw

            # Target velocity calculation
            # Completed by grabbing the reference data and finding the closest X,Y position on the map as the 
            # waypoint and comparing the reference speed limit to current speed.

            waypoint_location = waypoint.transform.location
            closest_idx = np.argmin(np.sum((base_traj_data[['x', 'y']].values - np.array([waypoint_location.x, waypoint_location.y]))**2, axis=1))
            closest_data = base_traj_data.iloc[closest_idx]
            # add velAdh to the velocity target data in order to change how much a driver adheres to the speed limit
            target_velocity = closest_data['speed_limit']
            # This will set the target to be a non-zero value as long as the speed limit or expectation isn't 0
            adhere = target_velocity + velAdh
            if target_velocity <= 0:
                adhere = 0
            elif target_velocity > 0 and adhere <= 0:
                adhere = target_velocity
            target_velocity = adhere 
            target_heading = calculate_heading(closest_idx, track_data, waypoint.transform.location)
            # Control vehicle's throttle and steering
            
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location

            # Calculate the reward values (error)
            velocity_reward, trajectory_reward = calculate_reward(current_x, current_y, current_throttle, current_brake, current_steering, current_velocity, track_data)
            rewardsArr.append([velocity_reward,trajectory_reward])

            # Calculate target throttle and steer values
            # Throttle may be negative. If target throttle is < 0, then we are braking.
            # Negate throttle and apply to brake. Else apply to throttle.
            target_throttle, pid_target_steer = get_target_values(throttle_brake_pid, steering_pid, current_velocity,target_velocity, current_heading, target_heading, world.get_settings().fixed_delta_seconds, waypoint, vehicle)
            if target_throttle > 0:
                target_brake = 0
            elif target_throttle < 0:
                target_brake = -target_throttle
                target_throttle = 0
            else:
                target_brake = 0
            
            # This is a relic shard. It will calculate the pure pursuit steering, and combined steer
            # Is the pid_target_steer and weight combined with pure pursuit and weight. 
            # Since default pure pursuit weight is 0, PID is the typical only steer value.
            pp_steer = control_pure_pursuit(vehicle_transform, waypoint.transform, max_steer, wheelbase)
            combined_steer = pp_weight * pp_steer + pid_weight * pid_target_steer

            control = carla.VehicleControl(target_throttle, combined_steer, target_brake)
            vehicle.apply_control(control)
            
            

            # wait_for_tick is async. Tick is sync.
            # Since we are synchronous, we are just sending a tick.
            world.tick()

            # If a collision is detected, apply worst reward and end
            global collFlag
            if collFlag is True:
                ranger = int(counter * 10)
                for count in range(ranger,7500,1):
                    rewardsArr.append([999,999])
                counter = 750
                collFlag = False
            else:
                snap = world.get_snapshot()
                counter = round(counter + 0.1,2)
    
    # After simulation loop is complete
    capture_frames = False
    return rewardsArr



# Returns only the waypoints in one lane
# Param:
#   waypoint_list - A list of all the waypoints on the map
#   lane - lane we are searching for
# Return:
#   waypoints - waypoints in one lane
def single_lane(waypoint_list, lane):
    waypoints = []
    for i in range(len(waypoint_list) - 1):
        if waypoint_list[i].lane_id == lane:
            waypoints.append(waypoint_list[i])
    return waypoints
    
# Returns only the waypoints that are not along the straights
# Param:
#   waypoints - current list of waypoints
# Return:
#   curvy_waypoints - found waypoints
def get_curvy_waypoints(waypoints):
    curvy_waypoints = []
    for i in range(len(waypoints) - 1):
        x1 = waypoints[i].transform.location.x
        y1 = waypoints[i].transform.location.y
        x2 = waypoints[i+1].transform.location.x
        y2 = waypoints[i+1].transform.location.y
        if (abs(x1 - x2) > 1) and (abs(y1 - y2) > 1):
            print("x1: " + str(x1) + "  x2: " + str(x2))
            print(abs(x1 - x2))
            print("y1: " + str(y1) + "  y2: " + str(y2))
            print(abs(y1 - y2))
            curvy_waypoints.append(waypoints[i])
      
    # To make the path reconnect to the starting location
    curvy_waypoints.append(curvy_waypoints[0])

    return curvy_waypoints

# Gets the bearing of the vehicle in degrees
# Params:
#   vehicle_location - current x,y position of the car
#   Waypoint_location - Current x,y position of the waypoint
# Return:
#   the angle between the vehicle and the waypoint
def get_bearing(vehicle_location, waypoint_location):
    """Calculate the bearing from the vehicle to the waypoint in degrees."""
    delta_x = waypoint_location.x - vehicle_location.x
    delta_y = waypoint_location.y - vehicle_location.y
    return math.degrees(math.atan2(delta_y, delta_x))

# Gets the yaw of the vehicle in degrees
# Params:
#   vehicle - the car
# Return:
#   yaw of the vehicle
def get_vehicle_yaw(vehicle):
    """Get the vehicle's yaw (heading) in degrees."""
    vehicle_transform = vehicle.get_transform()
    vehicle_rotation = vehicle_transform.rotation
    return vehicle_rotation.yaw

# Gets the next waypoitn from the list of waypoints while using a safety buffer calculation
# The safety buffer acts to simulate driver look-ahead when figuring out where they need to be
# And when
# Params:
#   world - the current CARLA instance
#   vehicle - the CARLA car
#   waypoints - the list of waypoints on the map
#   safety_buffer - a scaling value applied to a simple calculation for distance
# Return:
#   next_waypoint - the isolated waypoint to be referenced
def get_next_waypoint(world, vehicle, waypoints, safety_buffer):
    vehicle_location = vehicle.get_location()
    vehicle_yaw = get_vehicle_yaw(vehicle)
    min_distance = 1000
    next_waypoint = None

    # Distance calculation (simple) using safety buffer value and current velocity
    # Find the max between the Threshold and 10 (minimum distance to next waypoint)
    current_velocity_3D = vehicle.get_velocity()
    current_velocity = np.sqrt(current_velocity_3D.x**2 + current_velocity_3D.y**2 + current_velocity_3D.z**2)
    THRESH = safety_buffer*(current_velocity**2)/(2*9.8)
    maxi = max(THRESH, 10)
    for waypoint in waypoints:
        waypoint_location = waypoint.transform.location
        bearing = get_bearing(vehicle_location, waypoint_location)

        # Calculate the difference in angle, taking care to normalize it between -180 and 180
        angle_diff = (bearing - vehicle_yaw + 180) % 360 - 180

        # Check if the waypoint is within a certain angle in front of the vehicle
        if -100 < angle_diff < 100:
            distance = vehicle_location.distance(waypoint_location)

            # Find the closest waypoint that is not too close to the vehicle
            if maxi < distance < min_distance:
                min_distance = distance
                next_waypoint = waypoint
    return next_waypoint

# Calculates a target steering value for pure pursuit
# Param:
#   vehicle_tr - the CARLA vehicle
#   waypoint_tr - the chosen waypoint
#   max_steer - the maximum steer value allowed
#   wheelbase - the wheelbase of the car
# Return:
#   final target steer value
def control_pure_pursuit(vehicle_tr, waypoint_tr, max_steer, wheelbase):
    wp_loc_rel = relative_location(vehicle_tr, waypoint_tr.location) + carla.Vector3D(wheelbase, 0, 0)
    wp_ar = [wp_loc_rel.x, wp_loc_rel.y]
    d2 = wp_ar[0]**2 + wp_ar[1]**2
    steer_rad = math.atan(2 * wheelbase * wp_loc_rel.y / d2)
    steer_deg = math.degrees(steer_rad)
    steer_deg = np.clip(steer_deg, -max_steer, max_steer)
    return steer_deg / max_steer

# Calculates the relative location between a provided object and a provided location
# Param:
#   frame - an object in CARLA
#   location - an x,y,z position in space
# Return:
#   The relative location of the frame in comparison to the space 
def relative_location(frame, location):
  origin = frame.location
  forward = frame.get_forward_vector()
  right = frame.get_right_vector()
  up = frame.get_up_vector()
  disp = location - origin
  x = np.dot([disp.x, disp.y, disp.z], [forward.x, forward.y, forward.z])
  y = np.dot([disp.x, disp.y, disp.z], [right.x, right.y, right.z])
  z = np.dot([disp.x, disp.y, disp.z], [up.x, up.y, up.z])
  return carla.Vector3D(x, y, z)

# This function passes in the vehicle and applies a simple control schematic 
# It was useful for earlier testing, but is no longer used
# Param:
#   vehicle - the CARLA vehicle
# Return:
#   No Return
def control_vehicle(vehicle):
    global angle
    physics_control = vehicle.get_physics_control()
    max_steer = physics_control.wheels[0].max_steer_angle
    rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
    offset = rear_axle_center - vehicle.get_location()
    wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
    throttle = 0.5
    vehicle_transform = vehicle.get_transform()
    steer = degrees_to_steering_percentage(angle)
    control = carla.VehicleControl(throttle, steer)
    vehicle.apply_control(control)

#Returns a list of waypoints from the start to the end
# Params:
#   world - the CARLA instance
#   amap - map data
# Returns:
#   final_route - the list of waypoints from the map
def waypoint_gen(world, amap):
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
    #get rid of the last 10 points in route_waypoints
    route_waypoints = route_waypoints[:-30]

    # Prepend the custom starting locations to the route locations
    final_route = custom_starting_waypoints + route_waypoints

    return final_route


# Places the spawn point, spawns the car, and attaches all the sensors the car
# Params:
#   world - instance of CARLA
# Returns:
#   vehicle - the spawned vehicle
#   camera - the spawned camera 
def spawn_actor(world):
    blueprint = world.get_blueprint_library().filter('vehicle.*model3*')[0]
    my_spawn_point = carla.Transform(carla.Location(x=127.1, y=-2.1, z=2.1538), carla.Rotation(yaw=180))
    vehicle = world.spawn_actor(blueprint, my_spawn_point)#carla.Transform(location, rotation))
    actor_list.append(vehicle)
    vehicle.set_simulate_physics(True)
    transform = carla.Transform(carla.Location(x=0.8, z=1.7))

    #Add spectator camera
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-10,z=10), carla.Rotation(-45,0,0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera) #Add to actor_list at [1]

    # attach_lidar(world, vehicle, transform)
    attach_collision_sensor(world, vehicle, transform)

    return vehicle, camera

def attach_rgb_camera(world, vehicle, transform=None):
    """
    Attach an RGB camera to the vehicle for capturing frames that will be processed by BLIP2
    
    Args:
        world: CARLA world object
        vehicle: Vehicle to attach camera to
        transform: Transform for camera placement
    
    Returns:
        rgb_camera: The attached camera sensor
    """
    if transform is None:
        # Default camera position - forward-facing
        transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
    
    # Set up camera with resolution suitable for BLIP2 processing
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '512')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '100')
    
    # Create and attach the camera
    rgb_camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    actor_list.append(rgb_camera)
    
    # Set up the callback to capture frames
    rgb_camera.listen(lambda image: process_rgb_frame(image, world, vehicle))
    return rgb_camera

def process_rgb_frame(image, world, vehicle):
    """
    Process RGB frames captured by the camera, storing them for BLIP2 embedding generation
    
    Args:
        image: CARLA image from RGB camera
        world: CARLA world object
        vehicle: The vehicle object
    """
    if not capture_frames:
        return
        
    # Convert CARLA raw image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Remove alpha channel
    
    # Only save frames when we're at a waypoint of interest
    # This reduces redundant frames and focuses on key trajectory points
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    
    # Store the frame with its timestamp
    frame_buffer.append(array)
    frame_timestamps.append(world.get_snapshot().timestamp.elapsed_seconds)

def generate_and_save_embeddings(participant_id):
    """
    Generate BLIP2 embeddings from captured frames and save them
    
    Args:
        participant_id: ID of the current participant for file naming
    """
    global blip_embedder, frame_buffer, frame_timestamps
    
    if not frame_buffer:
        print("No frames captured for embedding generation")
        return {}
    
    print(f"Generating embeddings for {len(frame_buffer)} frames")
    
    # Initialize BLIP embedder if not already done
    if blip_embedder is None:
        blip_embedder = BLIPEmbedder()
    
    # Generate embeddings in batches
    embeddings_dict = blip_embedder.generate_embeddings(frame_buffer, frame_timestamps)
    
    # Save embeddings
    os.makedirs(embedding_save_path, exist_ok=True)
    save_path = f"{embedding_save_path}/{participant_id}_blip_embeddings.pkl"
    blip_embedder.save_embeddings(embeddings_dict, save_path)
    
    # Clear buffers
    frame_buffer.clear()
    frame_timestamps.clear()
    
    return embeddings_dict

def generate_reference_embeddings(participant_id):
    """
    Generate reference BLIP embeddings from ground truth trajectory
    Uses the participant's actual recorded trajectory to create reference embeddings
    
    Args:
        participant_id: ID of the current participant
    """
    global blip_embedder
    
    # Clean the participant_id by removing 'final' suffix if present
    clean_id = participant_id.replace('final', '')
    
    # Import the necessary functions from test_ref_embed.py
    import test_ref_embed
    
    # Check if reference embeddings already exist
    reference_embeddings_path = f"{embedding_save_path}/reference_{clean_id}_blip_embeddings.pkl"
    if os.path.exists(reference_embeddings_path):
        print(f"Reference embeddings already exist at {reference_embeddings_path}. Loading...")
        # Initialize embedder if not done already, so it can be used later
        if blip_embedder is None:
            blip_embedder = BLIPEmbedder()
        # Return the existing embeddings
        return blip_embedder.load_embeddings(reference_embeddings_path)
    
    # Call the test_reference_embeddings function from test_ref_embed.py
    print(f"Generating reference embeddings for participant {clean_id} using test_ref_embed approach...")
    
    # Pass the clean ID to test_ref_embed
    test_ref_embed.test_reference_embeddings(clean_id, debug=False, force=True, save_frames=True)
    
    # After generating embeddings, load and return them
    if os.path.exists(reference_embeddings_path):
        if blip_embedder is None:
            blip_embedder = BLIPEmbedder()
        embeddings = blip_embedder.load_embeddings(reference_embeddings_path)
        print(f"Successfully loaded reference embeddings for {clean_id}")
        return embeddings
    else:
        print(f"Warning: Failed to generate reference embeddings for {clean_id}")
        return {}

# Constructs and attaches a lidar sensor to the car
# Params:
#   world - the instance of CARLA
#   vehicle - the CARLA vehicle
#   transform - a transformation object
# Return:
#   lidar_sensor - the constructed sensor object
def attach_lidar(world, vehicle, transform):
    #configure LIDAR sensor to only output 2d
    # Find the blueprint of the sensor.
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    # Set the time in seconds between sensor captures
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_bp.set_attribute('channels', '1')
    lidar_bp.set_attribute('upper_fov', '0')
    lidar_bp.set_attribute('lower_fov', '0')
    lidar_bp.set_attribute('range', '30') #10 is default
    lidar_bp.set_attribute('points_per_second', '500')
    #With 2 channels, and 100 points per second, here are 250 points per scan

    lidar_sensor = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    actor_list.append(lidar_sensor) #Add at actor_list[2]

    #Commented out to decrease processing
    lidar_sensor.listen(lambda data: save_lidar_image(data, world, vehicle))
    return lidar_sensor


# Constructs and attaches a collision sensor to the car
# Params:
#   world - the CARLA instance
#   vehicle - the CARLA car
#   transform - a transformation object
# Returns:
#   collision_sensor - constructed sensor object
def attach_collision_sensor(world, vehicle, transform):
    #Configure collision sensor
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, transform, attach_to=vehicle)
    actor_list.append(collision_sensor) #Add at actor_list[3]

    collision_sensor.listen(lambda data: collision_event(data, world, vehicle))

    return collision_sensor

# When a collision is detected elsewhere, this function is called
# It sets the collision flag to True
# Params:
#   data - data
#   world - the CARLA instance
#   vehicle - the CARLA car
# Return: 
#   No return
def collision_event(data, world, vehicle):
    print("COLLISION")
    global collFlag
    collFlag = True

# Converts polar coordinates to cartesian coordinates
# Params:
#   polar_coordinates - a set of polar coordinates
# Returns:
#   cartesian_coordinates - a set of cartesian coordinates

def polar_to_cartesian(polar_coordinates):
    cartesian_coordinates = []
    for point in polar_coordinates:
        r = point[0] 
        theta = point[1]
        x = r * math.cos(theta * math.pi/180.0)
        y = r * math.sin(theta * math.pi/180.0)
        cartesian_point = [x, y, point[2]]
        cartesian_coordinates.append(cartesian_point)
    return cartesian_coordinates

# Converts cartesian coordinates to polar coordinates
# Param:
#   cartesian_coordinates - a set of cartesian coordinates
# Return:
#   polar_coordinates - a set of polar coordinates
def cartesian_to_polar(cartesian_coordinates):
    #Parameter: an array of 3d coordinate triplets
    # [[X1, Y1, Z1], [X2, Y2, Z2], ...]
    polar_coordinates = []
    
    for point in cartesian_coordinates:
        x = point[0] 
        y = point[1]
        radius = np.sqrt(x * x + y * y)
        theta = np.arctan2(y,x)
        theta = 180 * theta / math.pi #Convert from radians to degrees
        polar_point = [radius, theta, point[2]]
        polar_coordinates.append(polar_point)
    return polar_coordinates

# Plots polar coordinates on a graph
# Param:
#   polar_coordinates - a set of polar coordinates
# Return:
#   No return
def graph_polars(polar_coordinates):
    time.sleep(1)
    polar_coordinates.sort(key = lambda point: point[1])
    w = 4
    h = 3
    d = 70
    plt.figure(figsize=(w, h), dpi=d)
    thetas = np.array(polar_coordinates)[:,1] / 180 * math.pi
    r = np.array(polar_coordinates)[:,0]
    ax = plt.subplot(projection='polar')
    ax.plot(thetas, r, 'o', color='black')
    plt.savefig("mygraph.png")

# Saves the image taken from a lidar sensor
# Params:
#   image - image from a lidar sensor
#   world - instance of CARLA
#   vehicle - the CARLA car
# Return:
#   No Return
def save_lidar_image(image, world, vehicle):
    global angle
    global lock
    if not lock.acquire(False):
        return
    #Convert raw data to coordinates (x,y,z)
    points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    polars = cartesian_to_polar(points)
    graph_polars(polars)
    points = polar_to_cartesian(polars)
    polars.sort(key = lambda point: point[1])
    find_disparity(polars)
    lock.release()

# Finds the disparity in polar coordinates
# Param:
#   polar_coordinates - a set of polar coordinates
# Return:
#   No Return value 
def find_disparity(polar_coordinates):
    global angle
    max_distance = 0
    max_disparity_pair = [[],[]]
    polar_coordinates = [i for i in polar_coordinates if i[1] > 0]
    for i in range(len(polar_coordinates) - 1):
        r1 = polar_coordinates[i][0]
        theta1 =  polar_coordinates[i][1] / 180 * math.pi
        r2 = polar_coordinates[i+1][0]
        theta2 =  polar_coordinates[i+1][1] / 180 * math.pi
        distance = math.sqrt(abs(r1*r1 + r2*r2 - 2 *r1 *r2 * np.cos(theta1 - theta2)))
        if distance > max_distance:
            max_distance = distance
            max_disparity_pair[0] = polar_coordinates[i]
            max_disparity_pair[1] = polar_coordinates[i+1]
    angle = (max_disparity_pair[1][1] + max_disparity_pair[0][1]) /2 
    return 

# Returns a steering percentage value as described below
# Params:
#   degrees - a desired degree value
# Return:
#   returns a steering percentage
def degrees_to_steering_percentage(degrees):
    """ Returns a steering "percentage" value between 0.0 (left) and 1.0
    (right) that is as close as possible to the requested degrees. The car's
    wheels can't turn more than max_angle in either direction. """
    degrees = -(degrees - 90)
    max_angle = 45
    if degrees < -max_angle:
        return 1.0
    if degrees > max_angle:
        return -1.0
    if abs(degrees) < 5:
        return 0
    return - (degrees / max_angle)

# Selects the nth rightt lane waypoint
# Param:
#   waypoint - a chosen waypoint
#   n - the nth waypoint?
# Return:
#   out_waypoint - the corrected waypoint in the correct lane 
def get_right_lane_nth(waypoint, n):
    out_waypoint = waypoint
    for i in range(n):
        out_waypoint = out_waypoint.get_right_lane()
    return out_waypoint

# Selects the nth left lane waypoint
# Params:
#   waypoint - chosen waypoint
#   n - nth waypoint?
# Return:
#   out_waypoint - the corrected waypoint in the correct lane
def get_left_lane_nth(waypoint, n):
    out_waypoint = waypoint
    for i in range(n):
        out_waypoint = out_waypoint.get_left_lane()
    return out_waypoint

# Returns a waypoint to help change lanes
# Params:
#   waypoint - a chosen waypoint
#   n - nth waypoint?
# Return:
#   corrected lane waypoint
def change_lane(waypoint, n):
    if (n > 0):
        return get_left_lane_nth(waypoint, n)
    else:
        return get_right_lane_nth(waypoint, n)

# Gets the transform object of the Vehicle
# Param:
#   vehicle_location - location of the CARLA car
#   angle - passed in angle in degrees
#   d - a set value
# Return:
#   Returns a CARLA Transform object from the vehicle
def get_transform(vehicle_location, angle, d=6.4):
        a = math.radians(angle)
        location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
        return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))

def remove_numpy_arrays(obj):
    if isinstance(obj, np.ndarray):
        return remove_numpy_arrays(obj.tolist())
    elif isinstance(obj, list):
        return [remove_numpy_arrays(item) for item in obj]
    else:
        return obj

# Loads in the gains and other optimization values from a JSON file that saves the final values
# Param:
#   participant_id - the unique participant ID
# Return:
#   Returns the throttle PID, steering PID, safety buffer, and speed_adherance values
def load_gains(participant_id):
    filename = 'JSONBASE/'+participant_id + '_GAgains.json'
    try:
        # Load the gains from a JSON file
        with open(filename, 'r') as f:
            gains = json.load(f)
    except FileNotFoundError:
        print('No gains file found. Creating with default gains.')
        # Default gains
        gains = {
            'throttle_brake': {'kp': 0.5, 'ki': 0.1, 'kd': 0.1},
            'steering': {'kp': 0.5, 'ki': 0.1, 'kd': 0.1},
            'safety_buffer' : 0.8,
            'speed_adhere' :  0,
        }
        # Create the file with default gains
        with open(filename, 'w') as f:
            json.dump(gains, f)
    
    throttle_brake_pid = PIDController(kp=gains['throttle_brake']['kp'], ki=gains['throttle_brake']['ki'], kd=gains['throttle_brake']['kd'])
    steering_pid = PIDController(kp=gains['steering']['kp'], ki=gains['steering']['ki'], kd=gains['steering']['kd'])
    return throttle_brake_pid, steering_pid, gains['safety_buffer'], gains['speed_adhere']

# Saves the optimization values + PID for throttle and steering values to the JSON
# Params:
#   participant_id - the unique participant id
#   solution - the optimized solution
#   lastFit - the last associated with fitness values
#   lastSol - the best chosen solution
# Return:
#   No return
def update_pids_json(participant_id, solution, lastFit, lastSol,pareto_fronts):
    filename = 'JSONBASE/' + participant_id + '_GAgains.json'
    lastFit_list = lastFit.tolist() if isinstance(lastFit, np.ndarray) else lastFit
    lastSol_list = lastSol.tolist() if isinstance(lastSol, np.ndarray) else lastSol
    pareto_fronts_fix = remove_numpy_arrays(pareto_fronts)
    # print(pareto_fronts_fix)
    # New gains structure from the solution
    new_gains = {
        'throttle_brake': {'kp': solution[0], 'ki': solution[1], 'kd': solution[2]},
        'steering': {'kp': solution[3], 'ki': solution[4], 'kd': solution[5]},
        'safety_buffer' : solution[6],
        'speed_adhere' : solution[7],
        'lastFit': lastFit_list,
        'lastSol': lastSol_list,
        'pareto_front':pareto_fronts_fix,
    }
    # Update the JSON file with new gains
    with open(filename, 'w') as f:
        json.dump(new_gains, f)
    print("PID gains updated in JSON file.")

# For Sanity, this reward calculates less of a reward and more of an error. 
# Adjusted distance to be a positive value, since the error will be squared anyway, and distance is positive value
# The higher the positive value later, the worse the error calc will be when put in 1/error
# Params:
#   current_x - current x position of the car
#   current_y - current y position of the car
#   current_throttle - current throttle value of the car
#   current_brake - current brake value of the car
#   current_steering - current steering value of the car
#   current_velocity - current velocity of the car
#   track_data - the data structure that contains the participants track data including
#       the velocity, position, etc
# Return:
#   The calculated velocity and trajectory reward (error)
def calculate_reward(current_x, current_y, current_throttle, current_brake, current_steering, current_velocity, track_data):
        # Find the closest recorded data point
        closest_idx = np.argmin(np.sum((track_data[['x', 'y']].values - np.array([current_x, current_y]))**2, axis=1))
        closest_data = track_data.iloc[closest_idx]
        
        # Reward for matching velocity (assuming velocity is a function of throttle and brake)
        velocity_reward = abs(closest_data['vel_mps'] - current_velocity)

        # Reward for staying close to the trajectory
        distance = np.sqrt((closest_data['x'] - current_x)**2 + (closest_data['y'] - current_y)**2)
        trajectory_reward = distance
        return velocity_reward, trajectory_reward

# Get the target values for throttle and steering
# Params:
#   throttle_brake_pid - PID values for the throttle
#   steering_pid - PID values for the steering
#   current_velocity - car's current velocity
#   target_velocity - car's target velocity calculated elsewhere
#   current_heading - car's current heading
#   target_heading - car's target heading calculated elsewhere
#   delta_time - the change in time
#   waypoint - the selected waypoint
#   vehicle - the CARLA car
# Return:
#   target values for throttle and steering controls
def get_target_values(throttle_brake_pid, steering_pid, current_velocity, target_velocity, current_heading, target_heading, delta_time, waypoint, vehicle):
        # Calculate instantaneous velocity error
        velocity_error = target_velocity - current_velocity

        # Use PID controller for velocity
        # velocity_control = throttle_brake_pid.control(velocity_error, delta_time)
        velocity_control = throttle_brake_pid._pid_control(target_velocity, current_velocity)
        target_throttle = velocity_control

        # Use PID controller for steering
        # target_steering = steering_pid.control(heading_error, delta_time)
        vehicle_transform = vehicle.get_transform()
        target_steering = steering_pid._pid_control(waypoint, vehicle_transform)
        target_steering = max(min(target_steering, 1), -1)
        return target_throttle, target_steering

# Returns the track data of the participant
# Param:
#   participant_csv - a collected CSV file from a database
# Return:
#   df - the track data's data frame
def traj_loader(participant_csv):
    # Read the CSV file into a Pandas DataFrame, skipping the existing headers
    df = pd.read_csv(participant_csv, header=0)  # header=0 indicates that the first line contains headers, which will be replaced

    # Assign the correct column names
    df.columns = ['t', 'x', 'y', 'z', 'throttle', 'steering', 'brake', 'hand_brake','reverse','manual_gear_shift','gear']

    # Calculate velocity and add it to the DataFrame
    df['vel_mph'] = calculate_velocity(df)
    df['vel_mps'] = df['vel_mph'] / 2.23694
    return df

# Returns the track data of the baseline reference
# Param:
#   baseline_csv - a collected CSV file of the baseline
# Return:
#   base_traj - the baseline trajectory's data frame
def base_traj_loader(baseline_csv):
    # Read the CSV file into a Pandas DataFrame, skipping the existing headers
    base_traj = pd.read_csv(baseline_csv, header=0)  # header=0 indicates that the first line contains headers, which will be replaced

    # Assign the correct column names
    base_traj.columns = ['t', 'x', 'y', 'z','velocity','speed_limit']
    return base_traj

# Calculates the heading of the car
# Params:
#   closest_idx - the index value of the closest matched set of data in the track data
#   track_data - the passed in track data of either the baseline or participant
#   next_waypoint - the next waypoint beyond the current waypoint
# Return:
#   heading - the calculated heading value
def calculate_heading(closest_idx, track_data, next_waypoint):
    #calculate heading based on next waypoint
    heading = np.arctan2(next_waypoint.y - track_data['y'][closest_idx], next_waypoint.x - track_data['x'][closest_idx])#in radians
    #heading in degrees
    heading = heading*(-1)*180/np.pi# - 180
    return heading

# Calculates all velocity values in a dataframe using positional and time data existing in said dataframe
# Param:
#   df - a dataframe from pandas
# Return:
#   The velocity_mph column of a dataframe
def calculate_velocity(df):
    # Calculate differences in time, x, and y
    df['dt'] = df['t'].diff()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()

    # Calculate velocity in m/s and then convert to mph
    df['velocity_ms'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']
    df['velocity_mph'] = df['velocity_ms'] * 2.23694

    # Handle the first row (NaN due to diff) and any potential division by zero
    df['velocity_ms'].fillna(0, inplace=True)
    df['velocity_mph'].fillna(0, inplace=True)

    return df['velocity_mph']

# This is a PID controller class that holds the PID values of each controller component
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0

    def reset(self):
        self.integral = 0
        self.last_error = 0

    def control(self, error, delta_time):
        self.integral += error * delta_time
        derivative = (error - self.last_error) / delta_time
        # derivative = -(error - self.last_error) / delta_time if delta_time > 0 else 0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Made this data GLOBAL since it kind of needs to be accessible in a lot of places!
base_traj_path = 'baseline_trajectory.csv'

# Load the driver trajectories
base_traj_data = base_traj_loader(base_traj_path)

# This function is just debugging
def on_generation(ga_instance):
    print("GENERATION COMPLETED!")
    savefilename = "participantGA/"+participant_id
    
# The main function (technically) of the python program
# This is where all the genetic algorithm and PSO algorithm wrapping occurs
# The real gimmick is that it's set up to run with an accompanying bash script
# That automates the process and circumvents a memory leak in CARLA
if __name__ == "__main__":
    # Create an argparser and grab the arguments
    argparser = argparse.ArgumentParser(description='GA File Processor')
    argparser.add_argument(
            '-i', '--ID',
            metavar='I',
            type=ascii,
            default="AA1498",
            help='The participant ID from a file')
    argparser.add_argument(
            '-n', '--New',
            action='store_true',
            help='pass a true if new GA needs to be started')
    args = argparser.parse_args()
    global participant_path
    global participant_filename
    global participant_id
    global track_data
    args.ID = args.ID[1:-1]
    participant_path = 'participant_data/'+args.ID+'final.csv'
    participant_filename = os.path.basename(participant_path)
    participant_id = os.path.splitext(participant_filename)[0]
    track_data = traj_loader(participant_path)
    try:
        if args.New:
            # Generate reference embeddings if needed
            clean_id = args.ID.replace('final', '')
            reference_embeddings_path = f"{embedding_save_path}/reference_{clean_id}_blip_embeddings.pkl"
            if not os.path.exists(reference_embeddings_path):
                print("Generating reference embeddings first...")
                generate_reference_embeddings(args.ID)
            
            # Grab initial values for the participant
            throttle_brake_gains, steering_gains, safety_buffer, speed_adhere = load_gains(args.ID)
            numGener = 5
            numMat = 15
            initPop = np.random.rand(40,8)
            newC = initPop[:,-1] * 10
            rounded = np.round(newC)
            initPop[:,-1] = rounded 
            initPop[0] = np.array([throttle_brake_gains.kp, throttle_brake_gains.ki, throttle_brake_gains.kd,
                       steering_gains.kp, steering_gains.ki, steering_gains.kd, safety_buffer, speed_adhere])
            geneSpace = [{'low': 0, 'high': 5},{'low':0 , 'high': 5},{'low':0 , 'high': 5},{'low':0 , 'high': 5},{'low':0 , 'high': 5},{'low': 0, 'high': 5},{'low': 0, 'high': 5},{'low': -15, 'high': 15},]
            parenSel = "nsga2"
            ga_instance = pg.GA(num_generations = numGener,
                                initial_population=initPop,
                                num_parents_mating=numMat,
                                fitness_func=fitness_func,
                                on_generation=on_generation,
                                mutation_num_genes=2,
                                mutation_type="random",
                                mutation_by_replacement=True,
                                random_mutation_min_val=-1.0,
                                random_mutation_max_val=1.0,
                                parent_selection_type = parenSel,
                                gene_space = geneSpace,
                                )
            ga_instance.run()
            filename = "participantGA/"+args.ID
            ga_instance.save(filename)
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            lastFit = ga_instance.last_generation_fitness
            lastSol = ga_instance.last_generation_parents
            # print("PARETO FRONTIER?: ", ga_instance.pareto_fronts)
            # print(f"Parameters of the best solution : {solution}")
            # print(f"Fitness value of the best solution = {solution_fitness}")
            writepath = "participantPIDS/"+args.ID+"GAPIDS.txt"
            update_pids_json(args.ID, solution, lastFit, lastSol,ga_instance.pareto_fronts) # Update the JSON file with the new gains
            with open(writepath, "a") as theFile: # Writing to participantPIDS txt file
                for PID in solution:
                    print(type(PID))
                    theFile.write(str(PID) + " ")
                theFile.write("\n")
        else:
            # Rest of your existing code for loading GA instance
            loadname = "participantGA/"+args.ID
            load_instance = pg.load(loadname)
            load_instance.fitness_func = fitness_func
            load_instance.run()
            solution, solution_fitness, solution_idx = load_instance.best_solution(load_instance.last_generation_fitness)
            filename = "participantGA/"+args.ID
            load_instance.save(filename)
            lastFit = load_instance.last_generation_fitness
            lastSol = load_instance.last_generation_parents
            # print("PARETO FRONTIER?: ", load_instance.pareto_fronts)
            # print(f"Parameters of the best solution : {solution}")
            # print(f"Fitness value of the best solution = {solution_fitness}")
            writepath = "participantPIDS/"+args.ID+"GAPIDS.txt"
            update_pids_json(args.ID, solution, lastFit, lastSol,load_instance.pareto_fronts) # Update the JSON file with the new gains
            with open(writepath, "a") as theFile: # Writing to participantPIDS txt file
                for PID in solution:
                    theFile.write(str(PID) + " ")
                theFile.write("\n")
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        capture_frames = False
        for actor in actor_list:
            try:
                actor.destroy()
            except:
                pass
        actor_list.clear()
        print('\ndone.')
