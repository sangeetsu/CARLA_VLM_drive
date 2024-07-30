
import csv
import os

def log_to_csv(file_path, time_step, position, velocity):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['TimeStep', 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ'])
        writer.writerow([time_step, position.x, position.y, position.z, velocity.x, velocity.y, velocity.z])

import json
import os
import sys
import numpy as np
import carla
import argparse
import controller as myPID
import GA_PID
# import PSO_PID

sys.path.append('../')

# Loads the PID gains from the JSON + other critical values
# Param:
#   json_file_path - path to the JSON file for participant
# Return:
#   throttle_brake_pid - set of three PID values for throttle/brake
#   steering_pid - set of three PID values for steering
#   safety - safety buffer value
#   speed - speed adherance value
def load_pid_gains_from_json(json_file_path):
    """
    Loads PID gains from a specified JSON file.
    
    Parameters:
    - json_file_path: The path to the JSON file containing the PID gains.
    
    Returns:
    - A tuple containing two lists: throttle_brake_pid and steering_pid.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        throttle_brake_pid = [data['throttle_brake']['kp'], data['throttle_brake']['ki'], data['throttle_brake']['kd']]
        steering_pid = [data['steering']['kp'], data['steering']['ki'], data['steering']['kd']]
        safety = data['safety_buffer']
        speed = data['speed_adhere']
    return throttle_brake_pid, steering_pid, safety, speed

# A copy of the running instance from the optimizer
# Param:
#   PIDInput - the set of 6 PID values + safety buffer + speed adherance values
#   optimizer - the selected optimizer to know where to draw from
# Return:
#   No Return  
def run_carla_instance(PIDInput, optimizer, ID):
    """
    Runs a CARLA simulation instance with the given PID inputs.

    Parameters:
    - pid_input: A list containing PID gains [Kp, Ki, Kd] for throttle/brake and steering.

    Returns:
    - no return
    """
    global actor_list, vehicle, camera
    actor_list = [None, None]

    # Initialize CARLA client and world here
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    
    spectator = world.get_spectator()

    #Old timestep for 60 fps = 0.0167
    new_settings = world.get_settings()
    #new_settings.substepping = True
    #new_settings.max_substep_delta_time = 0.02
    #new_settings.max_substeps=16
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = .01
    world.apply_settings(new_settings) 

    if optimizer == "GA":
        vehicle, camera = GA_PID.spawn_actor(world)
    else:
        vehicle, camera = PSO_PID.spawn_actor(world)
    actor_list[0] = vehicle
    actor_list[1] = camera
    if optimizer == "GA":
        my_custom_waypoints = GA_PID.waypoint_gen(world, world.get_map())
    else:
        my_custom_waypoints = PSO_PID.waypoint_gen(world, world.get_map())

    physics_control = vehicle.get_physics_control()
    max_steer = physics_control.wheels[0].max_steer_angle
    rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
    offset = rear_axle_center - vehicle.get_location()
    wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
    vehicle.set_simulate_physics(True)
    throttle_brake_pid = myPID.PIDLongitudinalController(vehicle,PIDInput[0], PIDInput[1], PIDInput[2],world.get_settings().fixed_delta_seconds)
    steering_pid = myPID.PIDLateralController(vehicle,PIDInput[3], PIDInput[4], PIDInput[5],world.get_settings().fixed_delta_seconds)

    participant_path = 'participant_data/'+ ID + 'final.csv'
    participant_filename = os.path.basename(participant_path)
    participant_id = os.path.splitext(participant_filename)[0]
    if optimizer == "GA":
        track_data = GA_PID.traj_loader(participant_path)
    else:
        track_data = GA_PID.traj_loader(participant_path)
    base_traj_path = 'baseline_trajectory.csv'
    if optimizer == "GA":
        base_traj_data = GA_PID.base_traj_loader(base_traj_path)
    else:
        base_traj_data = PSO_PID.base_traj_loader(base_traj_path)
    
    #start recording
    client.start_recorder(participant_id + '.log', True)
    velAdh = PIDInput[7]
    # Simulation loop
    
    time_step = 0
    log_file_path = 'simulation_log.csv'

    while True:
        time_step += 1
        # Update camera view
        spectator.set_transform(camera.get_transform())
        if optimizer == "GA":
            waypoint = GA_PID.get_next_waypoint(world, vehicle, my_custom_waypoints, PIDInput[6])
        else:
            waypoint = PSO_PID.get_next_waypoint(world, vehicle, my_custom_waypoints, PIDInput[6])
        if waypoint is None:
            print("Faulty waypoint")
            break
        else:
            world.debug.draw_point(waypoint.transform.location, life_time=5)
            
            # Get the current transform of the ego vehicle
            transform = vehicle.get_transform()
            location = transform.location
            velocity = vehicle.get_velocity()
            
            # Log data to CSV
            log_to_csv(log_file_path, time_step, location, velocity)
            
    
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

            waypoint_location = waypoint.transform.location
            closest_idx = np.argmin(np.sum((base_traj_data[['x', 'y']].values - np.array([waypoint_location.x, waypoint_location.y]))**2, axis=1))
            closest_data = base_traj_data.iloc[closest_idx]
            target_velocity = closest_data['speed_limit']
            adhere = target_velocity + velAdh
            if target_velocity <= 0:
                adhere = 0
            elif target_velocity > 0 and adhere <= 0:
                adhere = target_velocity
            target_velocity = adhere
            if optimizer == "GA":
                target_heading = GA_PID.calculate_heading(closest_idx, track_data, waypoint.transform.location)
            else:
                target_heading = PSO_PID.calculate_heading(closest_idx, track_data, waypoint.transform.location)

            # Control vehicle's throttle and steering
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            if optimizer == "GA":
                target_throttle, pid_target_steer = GA_PID.get_target_values(throttle_brake_pid, steering_pid, current_velocity,target_velocity, current_heading, target_heading, world.get_settings().fixed_delta_seconds, waypoint, vehicle)
            else:
                target_throttle, pid_target_steer = PSO_PID.get_target_values(throttle_brake_pid, steering_pid, current_velocity,target_velocity, current_heading, target_heading, world.get_settings().fixed_delta_seconds, waypoint, vehicle)

            if target_throttle > 0:
                target_brake = 0
            elif target_throttle < 0:
                target_brake = -target_throttle
                target_throttle = 0
            else:
                target_brake = 0
            control = carla.VehicleControl(target_throttle, pid_target_steer, target_brake)
            vehicle.apply_control(control)

            # wait_for_tick is async. Tick is sync.
            world.tick()
    # Cleanup simulation and return errors
    print("stop recording")
    client.stop_recorder()   
    return 

def main(participant_path, optimizer, ID):
    # Load the PID gains
    json_file_path = participant_path  # Update with the correct path if necessary
    throttle_brake_pid, steering_pid, safety, speed = load_pid_gains_from_json(json_file_path)
    
    # Combine PID gains for simulation input
    PIDInput = throttle_brake_pid + steering_pid + [safety] + [speed]
    
    # Call the simulation with the loaded PID values
    # This part assumes you have a run_simulator function ready to use these values
    run_carla_instance(PIDInput, optimizer, ID)
    

if __name__ == "__main__":
    try:
        # Create an argparser and grab the arguments
        argparser = argparse.ArgumentParser(description='Evaluator Joy')
        argparser.add_argument(
                '-i', '--ID',
                metavar='I',
                type=ascii,
                default="DP8189",
                help='The participant ID from a file')
        argparser.add_argument(
                '-o', '--Optimizer',
                metavar='O',
                default="GA",
                help='GA for genetic algorithm or PSO for particle swarm optimization')
        args = argparser.parse_args()
        args.ID = args.ID[1:-1]
        if args.Optimizer == "GA":
            participant_path = 'JSONBASE/'+args.ID+'_GAgains.json'
        else:
            participant_path = 'Participant_Verify/'+args.ID+'_PSOgains.json'
        main(participant_path, args.Optimizer, args.ID)
    except KeyboardInterrupt:
        print("Exiting simulation")
        pass
    finally:
        print("Destroying actors")
        for actor in actor_list:
            actor.destroy()
        actor_list.clear()
        print("Simulation ended")

