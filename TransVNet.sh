#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 32 processor core(s) per node 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=priority-a100    # gpu node(s)
#SBATCH --job-name="TransVNet"
#SBATCH --mail-user=mhashemi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
# export MASTER_PORT=12349
# export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source activate mytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3
# srun python train_deg.py --dataset Degradation
# python train_deg.py --dataset Degradation
# python test_deg.py --dataset Degradation
python train_deg.py --dataset Design --max_iterations 50 --vit_name Conv-ViT-Gen-B_16
# python test_deg.py --dataset Design
