#!/usr/bin/env bash
# Script exists to run the GA in an outer loop... 
# It uses a similar structure as my previous bash script as I know the previous script successfully restarts the
# server``

# Run first instance without an argument
file_path="participant_data/TT4525final.csv"
# Extract filename from the file path
filename=$(basename "$file_path")
# Retrieve ID
ID=${filename%final.csv}
echo "Opening Program"
/home/sangeetsu/sample33/LinuxNoEditor/CarlaUE4.sh -windowed > /dev/null 2>&1 &

PID=$!
let "PID2=$PID+8"
sleep 6
echo "Initial Optimization Run"
python3 blip_pid.py -i $ID --New &
PID3=$!
wait $PID3
kill -9 $PID
kill -9 $PID2
sleep 3
# for n in {2..50};
# do
# 	echo "Starting Iteration $n" 
# 	/home/sangeetsu/sample33/LinuxNoEditor/CarlaUE4.sh -windowed > /dev/null 2>&1 &
# 	PID=$!
# 	let "PID2=$PID+8"
# 	sleep 3
# 	echo "Running Optimizer"
# 	python3 blip_pid.py -i $ID &
# 	PID3=$!
# 	wait $PID3
# 	kill -9 $PID
# 	kill -9 $PID2
# 	sleep 3
# done
exit
