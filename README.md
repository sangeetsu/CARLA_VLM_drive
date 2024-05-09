# Virtuous PID Tuner
Howdy!
This is the PID tuner for a car built in CARLA. This particular endeavor has taken a long time, and it has many moving parts. Below, there will be a comprehensive explanation of installation, each file, each directory, operation, and line numbers for adjustments.

# Installation
1. Install [miniconda](https://docs.anaconda.com/free/miniconda/index.html). Follow the directions on their page up to setting up the first environment
2. Use this command to create the Virtuous Environment:
>conda env create -f Virtuous_Environment.yaml
3. There is a directory in the repository that contains files outside of a library that can be installed, but still vital. This is the directory called **agents**. Copy the folder to your environment's python3.8 directory. Refer below:
> Your directory should be located similarly: home/user/miniconda3/envs/Virtuous_Environment/lib/python3.8/site-packages
>cp -r agents/ /home/user/miniconda3/envs/Virtuous_Environment/lib/python3.8/site-packages/
4. Activate the environment
>conda activate Virtuous_Environment
5. Acquire the packaged map from [here](fixlater.later)
6. Extract the packaged map into the same repository as everything else
7. Enter **PSO_PID.py** and **GA_PID.py** and adjust lines 13 and 28 in both files respectively with the path to the package map *CarlaUE4.sh* executable
8. This should complete the installation process

# Files
|File Name|Purpose|
|----------------|-------------------------------|
|GA_Bash.sh|Automates the Genetic Algorithm over multiple sets of iterations|
|GA_PID.py|Runs Genetic Algorithm optimization over one set of iterations|
|PSO_Bash.sh|Automates the Particle Swarm Optimization over multiple sets of iterations|
|PSO_PID.py|Runs Particle Swarm Optimization over one set of iterations|
|Virtuous_Environment.yaml|Everything the conda environment needs to set itself up|
|baseline_trajectory.csv|A critical file that contains baseline data. Don't modify unless necessary.|
|controller.py|Contains functions necessary to run the PID controller|
|evaluator.py|Takes the optimal set of gains in either PSO or GA and runs an instance of carla|

# Directories
|Directory Name|Contents/Purpose|
|----------------|-------------------------------|
|JSONBASE|Stores the resulting JSON datafiles from both the PSO and GA|
|agents|Contains critical python files that need to be moved to the directory described above|
|participantGA|Stores pickle files containing pygad GA instances. Save data to allow additional runs on the same instance.|
|participantPIDS|History of PID values from GA|
|participant_data|The CSV files of the participants|
|participantPSO|Stores txt files that hold the current set of solutions for a PSO run. Acts as save data for PSO.|

# Operation
Operating this tool is quite simple. 
**For GA**:
The *GA_Bash.sh* file will run everything. Adjust line 7 to be the correct path, and it's as simple as running the following command:
>./GA_Bash.sh

To run just the optimizer once, just run the *GA_PID.py* file. This file takes two inputs:
1. -n will cause it to start a new generation from scratch
2. -i IDNumber will cause it to search for and use the data connected to the ID. For example:
> python3 GA_PID.py -i AB1234 -n

**For PSO** 
The *PSO_Bash.sh* file will run everything. Adjust line 7 to be the correct path, and it's as simple as running the following command:
>./PSO_Bash.sh

To run just the optimizer once, just run the *PSO_PID.py* file. This file takes two inputs:
1. -n will cause it to start a new generation from scratch
2. -i IDNumber will cause it to search for and use the data connected to the ID. For example:
> python3 PSO_PID.py -i AB1234 -n

**Evaluation**
Run the *evaluator.py* to evaluate the gains. This file takes in two inputs:
1. -i IDNumber will cause it to search for and use the data 
2. -o [GA|PSO] will choose which optimizer to run with
# Adjusting the Optimizers
Adjusting the optimizers requires entering the code at this time as arguments are not set up to be taken in. 
**FOR GA in GA_PID.py**
|Attribute|Purpose|Location|
|----------------|-------------------------------|----------------|
|numGener|Changes number of generations|Line 931|
|numMat|Number of parents mating|Line 932|
|initPop|Change number of population per set|Line 933, change first colum of numpy rand|
|geneSpace|constraints for the genes|Line 945|

**FOR GA in GA_Bash.sh**
|Attribute|Purpose|Location|
|----------------|-------------------------------|----------------|
|n|Multiply numGener by n|Line 26, {2..n} for a total of n times|

**FOR PSO**
|Attribute|Purpose|Location|
|----------------|-------------------------------|----------------|
|swarmSize|Track size of population|Line 940 and 965|
|options|Change c1, c2, and w for PSO|Line 939 and 964|
|initPop|Change number of population per set|Line 932, change first colum of numpy rand|
|iters|The number of iterations|Line 948 and 973|
|constraints|Change maximum and minimum boundaries for the population|Line 938 and 963|

**FOR PSO in PSO_Bash.sh**
|Attribute|Purpose|Location|
|----------------|-------------------------------|----------------|
|n|Multiply iters by n|Line 26, {2..n} for a total of n times|


# Future Work
1. Implement Arguments for easier optimizer adjustments
2. Add pareto frontier visualizer script
3. Add Evaluation Comparison Script
