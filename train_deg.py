import argparse
import logging
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import builtins
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg, CONFIGS3D as CONFIGS_ViT_seg_3D
from trainer import trainer_synapse, trainer_deg, trainer_mat

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/deg/train_npz', help='root dir for data')
parser.add_argument('--root_path', type=str,
                    default='/work/sheidaei/mhashemi/data/mat', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Design', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Design', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=288, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--gpu', type=int, default=4, help='total gpu')
parser.add_argument('--world-size', default=-1, type=int, 
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, 
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, 
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, 
                    help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, 
                    help='local rank for distributed training')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=[160, 160, 160], help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=4, help='using number of skip-connect, default is 4')
parser.add_argument('--vit_name', type=str,
                    default='Conv-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=8, help='vit_patches_size, default is 8')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Degradation': {
            'root_path': '/work/sheidaei/mhashemi/data/deg',
            'list_dir': './lists/lists_Degradation',
            'num_classes': 2,
        },
        'Design': {
            'root_path': '/work/sheidaei/mhashemi/data/mat',
            'list_dir': './lists/lists_Design',
            'num_classes': 2,
        }
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TV_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TV')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg_3D[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # model    
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    model.load_from(weights=np.load(config_vit.pretrained_path))
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model)
        model.to(device)
        # raise NotImplementedError("Only DistributedDataParallel is supported.")

    trainer = {'Synapse': trainer_synapse, 'Degradation': trainer_deg, 'Design': trainer_mat}
    trainer[dataset_name](args, model, snapshot_path)
    # sys.exit(0)
