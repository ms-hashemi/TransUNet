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
# Following are our own codes/libraries
from networks.TransVNet_modeling import VisionTransformer as Net
from networks.TransVNet_modeling import CONFIGS, CONFIGS3D
from trainer import trainer_synapse, trainer_deg, trainer_mat

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Degradation', help='Experiment/dateset name')
parser.add_argument('--img_size', type=int, default=160, help='Input image size')
parser.add_argument('--vit_patches_size', type=int, default=8, help='The patch size which will be considered in the image sequentialization of the ViT input')

parser.add_argument('--vit_name', type=str, default='Conv-ViT-B_16', help='The name of the model/network architecture to be built/considered; detailed in "configs.py"')
parser.add_argument('--pretrained_net_path', type=str, default=False, help='If the training should start from a pretrained state/weights, the full path and name is given by this argument; otherwise (the default argument value of False), the training is started normally.')
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser.add_argument('--is_encoder_pretrained', type=str2bool, nargs='?', const=True, default=True, help='Whether the encoder or part(s) of it are pretrained; the default value is True')
parser.add_argument('--deterministic', type=int,  default=1, help='Whether to use deterministic inference')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Training batch size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='The initial learning rate of the optimizer (for SGD, not ADAM)')
parser.add_argument('--seed', type=int, default=1234, help='The random seed value')

parser.add_argument('--gpu', type=int, default=1, help='Total number of gpus')
parser.add_argument('--world-size', default=-1, type=int, help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='Node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='Url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='Distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for distributed training')

args = parser.parse_args()


if __name__ == "__main__":
    # Random or deterministic training
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

    # Preprocessing the input and the hyperparameters for training
    dataset_name = args.dataset
    train_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'dimension': 2,
            'prefix': 'TU', # TransUNet
        },
        'Degradation_local': {
            'root_path': '../data/deg/Selected', # On my local machine or CyBox,
            'list_dir': './lists/lists_Degradation',
            'num_classes': 2,
            'dimension': 3,
            'prefix': 'TVD', # TransVNetDegradation
        },
        'Degradation': {
            'root_path': '/work/sheidaei/mhashemi/data/deg',
            'list_dir': './lists/lists_Degradation',
            'num_classes': 2,
            'dimension': 3,
            'prefix': 'TVD', # TransVNetDegradation
        },
        'Design_local': {
            # 'root_path': '/work/sheidaei/mhashemi/data/mat',
            'root_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'num_classes': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
        },
        'Design': {
            'root_path': '/work/sheidaei/mhashemi/data/mat',
            # 'root_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'num_classes': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
        },
    }
    if isinstance(args.vit_patches_size, int): #len(args.vit_patches_size) == 1:
        args.vit_patches_size = [args.vit_patches_size] * train_config[dataset_name]['dimension']
    if isinstance(args.img_size, int): #len(args.img_size) == 1:
        args.img_size = [args.img_size] * train_config[dataset_name]['dimension']
    args.num_classes = train_config[dataset_name]['num_classes']
    args.root_path = train_config[dataset_name]['root_path']
    args.list_dir = train_config[dataset_name]['list_dir']
    args.exp = train_config[dataset_name]['prefix'] + '_' + dataset_name + str(args.img_size)

    # Set the training snapshot name
    snapshot_path = "../model/{}/{}".format(args.exp, train_config[dataset_name]['prefix'])
    if args.pretrained_net_path: # If the whole TransVNet has been trained, and the new training should start based on that trained model
        snapshot_path = snapshot_path + '_pretrained'
    else: # If only the encoder (or part of it) has been trained, and the new training should start based on that trained model
        snapshot_path = snapshot_path + '_encoderpretrained' if args.is_encoder_pretrained else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_seed' + str(args.seed) # if args.seed!=1234 else snapshot_path

    # Create a folder to save the training results and log if it does not exist
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # Get the config of the network to be built/considered in training
    if train_config[dataset_name]['dimension'] == 3:
        config = CONFIGS3D[args.vit_name]
    else:
        config = CONFIGS[args.vit_name]
    config.n_classes = args.num_classes
    if args.vit_name.upper().find('R50') != -1 or args.vit_name.upper().find('CONV') != -1: # If ResNet50/CNN is going to be used in a hybrid encoder
        grid = []
        for i in range(train_config[dataset_name]['dimension']):
            grid.append(int((args.img_size[i] // (2**(len(config.encoder_channels) - 1))) / args.vit_patches_size[i]))
        config.patches.grid = tuple(grid)
    else: # No hybrid encoder or only ViT as the encoder
        config.patches.size = args.vit_patches_size

    # Settings if distributed training is required (different GPUs on different Nodes)
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
    model = Net(config, img_size=args.img_size, num_classes=config.n_classes)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model)
        model.to(device)
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    if args.pretrained_net_path:
        pretrained_dict = torch.load(args.pretrained_net_path)
        model_dict = model.state_dict()
        # 1. filter mismatched keys (mismatch either in the key name or its value shape) 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(args.pretrained_net_path))
    if args.is_encoder_pretrained:
        if hasattr(model, "module"):
            model.module.load_from(weights=np.load(config.pretrained_path)) # No gradient calculation in (parts of the) encoder if there is a config.pretrained_path
        else:
            model.load_from(weights=np.load(config.pretrained_path)) # No gradient calculation in (parts of the) encoder if there is a config.pretrained_path

    trainer = {'Synapse': trainer_synapse, 'Degradation': trainer_deg, 'Degradation_local': trainer_deg, 'Design': trainer_mat, 'Design_local': trainer_mat}
    trainer[dataset_name](args, model, snapshot_path)
    # sys.exit(0)
