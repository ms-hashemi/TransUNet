import argparse
import logging
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import builtins
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume, test_multiple_volumes
from networks.TransVNet_modeling import VisionTransformer as Net
from networks.TransVNet_modeling import CONFIGS, CONFIGS3D
from torchvision import transforms
from datasets.dataset_3D import Degradation_dataset, Design_dataset, Resize

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Degradation', help='experiment_name')
parser.add_argument('--img_size', type=int, default=[160, 160, 160], help='input patch size of network input')

parser.add_argument('--vit_name', type=str, default='Conv-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=8, help='vit_patches_size, default is 8')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic inference')
parser.add_argument('--max_epochs', type=int, default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')

parser.add_argument('--gpu', type=int, default=4, help='total gpu')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

args = parser.parse_args()



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def inferrer_synapse(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"



def inferrer_deg(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, transform=transforms.Compose([Resize(output_size=args.img_size)]))
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = np.zeros(shape=(args.num_classes-1, 2))
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image_batch, label_batch, time_batch, case_name_batch = sampled_batch["image"], sampled_batch["label"], sampled_batch["time"], sampled_batch['case_name']
        image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
        metric_batch = test_multiple_volumes(image_batch, label_batch, time_batch, model, classes=args.num_classes, patch_size=args.img_size,
                                         test_save_path=test_save_path, case=case_name_batch, z_spacing=args.z_spacing)
        for i in range(1, args.num_classes):
            logging.info('i_batch %d mean_dice %f mean_hd95 %f' % (i_batch, metric_batch[i-1][0], metric_batch[i-1][1]))
        metric_list += np.array(metric_batch)
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"



def inferrer_mat(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, transform=transforms.Compose([Resize(output_size=args.img_size)]))
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = np.zeros(shape=(args.num_classes-1, 2))
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image_batch, label_batch, time_batch, case_name_batch = sampled_batch["image"], sampled_batch["label"], sampled_batch["time"], sampled_batch['case_name']
        image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
        metric_batch = test_multiple_volumes(image_batch, label_batch, time_batch, model, classes=args.num_classes, patch_size=args.img_size,
                                         test_save_path=test_save_path, case=case_name_batch, z_spacing=args.z_spacing)
        for i in range(1, args.num_classes):
            logging.info('i_batch %d mean_dice %f mean_hd95 %f' % (i_batch, metric_batch[i-1][0], metric_batch[i-1][1]))
        metric_list += np.array(metric_batch)
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"



if __name__ == "__main__":
    # Random or deterministic inference
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
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
            'dimension': 2,
            'prefix': 'TU', # TransUNet
        },
        'Degradation': {
            'Dataset': Degradation_dataset,
            'volume_path': '/work/sheidaei/mhashemi/data/deg',
            'list_dir': './lists/lists_Degradation',
            'num_classes': 2,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVD', # TransVNetDegradation
        },
        'Degradation': {
            'Dataset': Design_dataset,
            'volume_path': '/work/sheidaei/mhashemi/data/mat',
            'list_dir': './lists/lists_Design',
            'num_classes': 2,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
        }
    }
    if isinstance(args.vit_patches_size, int): #len(args.vit_patches_size) == 1:
        args.vit_patches_size = [args.vit_patches_size] * dataset_config[dataset_name]['dimension']
    if isinstance(args.img_size, int): #len(args.img_size) == 1:
        args.img_size = [args.img_size] * dataset_config[dataset_name]['dimension']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.exp = dataset_config[dataset_name]['prefix'] + dataset_name + str(args.img_size)

    # name the same snapshot defined in train script!
    snapshot_path = "../model/{}/{}".format(args.exp, dataset_config[dataset_name]['prefix'])
    if args.pretrained_net_path: # If the whole TransVNet has been trained, and the new training should start based on that trained model
        snapshot_path = snapshot_path + '_pretrained'
    else: # If only the encoder (or part of it) has been trained, and the new training should start based on that trained model
        snapshot_path = snapshot_path + '_encoderpretrained' if args.is_pretrain_encoder else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_seed' + str(args.seed) # if args.seed!=1234 else snapshot_path
    
    # Get the config of the network to be built/considered in training
    if dataset_config[dataset_name]['dimension'] == 3:
        config = CONFIGS3D[args.vit_name]
    else:
        config = CONFIGS[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1: # If ResNet50 is not used for the CNN feature extractor of the encoder
        config.patches.grid = []
        for i in range(dataset_config[dataset_name]['dimension']):
            config.patches.grid.append(int(args.img_size[i] / args.vit_patches_size[i]))
        config.patches.grid = tuple(config.patches.grid)

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
    net = Net(config, img_size=args.img_size, num_classes=config.n_classes)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
            net_without_ddp = net.module
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
            net_without_ddp = net.module
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = torch.nn.DataParallel(net)
        net.to(device)
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # Saving the test results
    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inferrer = {'Synapse': inferrer_synapse, 'Degradation': inferrer_deg, 'Design': inferrer_mat}
    inferrer[dataset_name](args, net, test_save_path)
    # sys.exit(0)
