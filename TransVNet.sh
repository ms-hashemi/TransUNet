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
# CUDA_VISIBLE_DEVICES=0,1,2,3 # Not needed given the current version of the script
# srun python train_deg.py --dataset Degradation # Not needed given the current version of the script
python train.py --dataset Design --img_size 64 --vit_patches_size 4 --vit_name Conv-ViT-Gen2-B_16 --gpu 4 --batch_size 64 # --pretrained_net_path '../model/TVG_Design[64, 64, 64]/TVG_encoderpretrained_Conv-ViT-Gen-B_16_vitpatch[4, 4, 4]_epo100_bs32_lr0.01_seed1234/epoch_99.pth'# Main training code
python test.py --dataset Design2 --img_size 64 --vit_patches_size 4 --vit_name Conv-ViT-Gen2-B_16 --gpu 4 --batch_size 64 --batch_size_test 64
# python train.py --dataset Design --vit_name Conv-ViT-Gen-B_16 --gpu 4 --batch_size 32 --pretrained_net_path '../model/TVG_Design[160, 160, 160]/TVG_encoderpretrained_Conv-ViT-Gen-B_16_vitpatch[8, 8, 8]_epo100_bs32_lr0.01_seed1234/epoch_99.pth'
# python test.py --dataset Design2 --vit_name Conv-ViT-Gen-B_16 --gpu 4 --batch_size 32 --batch_size_test 32
