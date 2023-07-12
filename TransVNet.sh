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
# python train.py --dataset Design --img_size 64 --vit_patches_size 1 --vit_name Conv-ViT-Gen2-B_16 --gpu 4 --batch_size 35 --base_lr 0.001 --max_epochs 200 --is_encoder_pretrained False # --pretrained_net_path '../model/TVG_Design[64, 64, 64]/TVG_encoderpretrained_Conv-ViT-Gen2-B_16_vitpatch[4, 4, 4]_epo100_bs64_lr0.01_seed1234/epoch_99.pth'
# python test.py --dataset Design2 --img_size 64 --vit_patches_size 1 --vit_name Conv-ViT-Gen2-B_16 --gpu 4 --batch_size 35 --base_lr 0.001 --max_epochs 200 --is_encoder_pretrained False --batch_size_test 35 # --net_path '../model/TVG_Design[64, 64, 64]/TVG_Conv-ViT-Gen2-B_16_vitpatch[1, 1, 1]_epo200_bs60_lr0.001_seed1234/epoch_39.pth'
# python train.py --dataset Degradation --img_size 64 --vit_patches_size 2 --vit_name Conv-ViT-B_16 --gpu 4 --batch_size 48 --base_lr 0.01 --max_epochs 4 --is_encoder_pretrained False # --pretrained_net_path '../model/TVG_Design[64, 64, 64]/TVG_encoderpretrained_Conv-ViT-Gen2-B_16_vitpatch[4, 4, 4]_epo100_bs64_lr0.01_seed1234/epoch_99.pth'
# python test.py --dataset Degradation --img_size 64 --vit_patches_size 2 --vit_name Conv-ViT-B_16 --gpu 4 --batch_size 48 --base_lr 0.01 --max_epochs 4 --is_encoder_pretrained False --batch_size_test 64 # --net_path '../model/TVG_Design[64, 64, 64]/TVG_Conv-ViT-Gen2-B_16_vitpatch[1, 1, 1]_epo200_bs60_lr0.001_seed1234/epoch_39.pth'
python train.py --dataset Degradation --img_size 160 --vit_patches_size 2 --vit_name Conv-ViT-B_16 --gpu 4 --batch_size 24 --base_lr 0.01 --max_epochs 4 --is_encoder_pretrained False --pretrained_net_path '../model/TVD_Degradation[64, 64, 64]/TVD_Conv-ViT-B_16_vitpatch[2, 2, 2]_epo4_bs48_lr0.01_seed1234/epoch_3.pth'
python test.py --dataset Degradation --img_size 160 --vit_patches_size 2 --vit_name Conv-ViT-B_16 --gpu 4 --batch_size 24 --base_lr 0.01 --max_epochs 4 --is_encoder_pretrained False --pretrained_net_path '../model/TVD_Degradation[64, 64, 64]/TVD_Conv-ViT-B_16_vitpatch[2, 2, 2]_epo4_bs48_lr0.01_seed1234/epoch_3.pth' --batch_size_test 24 # --net_path '../model/TVG_Design[64, 64, 64]/TVG_Conv-ViT-Gen2-B_16_vitpatch[1, 1, 1]_epo200_bs60_lr0.001_seed1234/epoch_39.pth'
