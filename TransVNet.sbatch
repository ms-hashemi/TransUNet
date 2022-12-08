#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=32   # 32 processor core(s) per node 
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="TransVNet"
#SBATCH --mail-user=mhashemi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source activate mytorch
CUDA_VISIBLE_DEVICES=0,1,2,3
python train_deg.py --dataset Degradation