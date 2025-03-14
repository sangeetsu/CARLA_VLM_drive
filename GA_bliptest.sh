#!/usr/bin/env bash
# Script to run multi-objective optimization with improved resource management

# Run first instance without an argument
file_path="participant_data/AR4924final.csv"
# Extract filename from the file path
filename=$(basename "$file_path")
# Retrieve ID
ID=${filename%final.csv}
echo "Opening CARLA Server"
/home/sangeetsu/sample33/LinuxNoEditor/CarlaUE4.sh -windowed > /dev/null 2>&1 &

# Store CARLA PIDs
SERVER_PID=$!
let "SERVER_PID2=$SERVER_PID+8"
sleep 6

# Create necessary directories if they don't exist
mkdir -p participantGA
mkdir -p participantPIDS
mkdir -p embeddings

echo "Initial Optimization Run with NSGA-II"
python3 blip_pid.py -i $ID --New &
OPT_PID=$!
wait $OPT_PID
echo "Initial run completed"

# Clean up processes
echo "Cleaning up CARLA processes"
kill -9 $SERVER_PID
kill -9 $SERVER_PID2
sleep 3

# Set the number of iterations
TOTAL_ITERS=30

# Run subsequent iterations
for n in $(seq 2 $TOTAL_ITERS); do
    echo "========================================="
    echo "Starting Iteration $n of $TOTAL_ITERS" 
    echo "========================================="
    
    # Launch CARLA server
    /home/sangeetsu/sample33/LinuxNoEditor/CarlaUE4.sh -windowed > /dev/null 2>&1 &
    SERVER_PID=$!
    let "SERVER_PID2=$SERVER_PID+8"
    sleep 3
    
    echo "Running Multi-Objective Optimizer (NSGA-II)"
    python3 blip_pid.py -i $ID &
    OPT_PID=$!
    wait $OPT_PID
    
    echo "Iteration $n completed"
    echo "Cleaning up CARLA processes"
    kill -9 $SERVER_PID
    kill -9 $SERVER_PID2
    sleep 3
    
    # Monitor disk space
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $DISK_SPACE -gt 90 ]; then
        echo "WARNING: Disk space usage is above 90%. Consider freeing up space."
    fi
    
    # Optional: Clean up temporary files to save space
    # find /tmp -name "carla*" -mtime +1 -delete 2>/dev/null
done

echo "========================================="
echo "Optimization completed with $TOTAL_ITERS iterations"
echo "========================================="
exit 0