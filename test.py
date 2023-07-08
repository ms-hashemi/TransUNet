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
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume, test_multiple_volumes_sequencing, test_multiple_volumes_generative, test_multiple_volumes_generative2
from networks.TransVNet_modeling import VisionTransformer as Net
from networks.TransVNet_modeling import CONFIGS, CONFIGS3D
from torchvision import transforms
from datasets.dataset_3D import Degradation_dataset, Design_dataset, Resize, Resize2


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Degradation', help='Experiment/dateset name')
parser.add_argument('--img_size', type=int, default=[160, 160, 160], help='Input image size')
parser.add_argument('--vit_patches_size', type=int, default=8, help='The patch size which will be considered in the image sequentialization of the ViT input')

parser.add_argument('--net_path', type=str, default=False, help='The path to the trained network file to be used for testing: the default value (False) means that it is not specified, so the path should be found by the following arguments. However, if the full path (including the file name) is specified by an input string, the program will find the network directly without using the following arguments.')
parser.add_argument('--vit_name', type=str, default='Conv-ViT-B_16', help='The name of the model/network architecture to be built/considered; detailed in "configs.py"')
parser.add_argument('--pretrained_net_path', type=str, default=False, help='If the training should start from a pretrained state/weights, the full path and name is given by this argument; otherwise (the default argument value of False), the training is started normally.') # '../model/TV_Design[160, 160, 160]/TV_pretrain_Conv-ViT-Gen-B_16_skip4_vitpatch[8, 8, 8]_epo100_bs24_lr0.01_seed1234/epoch_99.pth'
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
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='The initial learning rate of the optimizer (for SGD, not ADAM)')
parser.add_argument('--seed', type=int, default=1234, help='The random seed value')
parser.add_argument('--is_savenii', action="store_true", help='Whether to save the results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='Saving directory to be created for the inference results')

parser.add_argument('--gpu', type=int, default=1, help='Total number of gpus for testing')
parser.add_argument('--batch_size_test', type=int, default=2, help='Test batch size per gpu')
parser.add_argument('--world-size', default=-1, type=int, help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='Node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='Url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='Distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for distributed training')

args = parser.parse_args()


def seed_worker(worker_id):
    """Seeding the random sequences for different threads in dataloader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def inferrer_synapse(args, model, test_save_path=None):
    """Main inferrence function for the synapse dataset used in TransUNet paper"""
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
    """Main inferrence function for the degradation dataset used in our TransVNet for 3D-segmented sequence prediction"""
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, transform=transforms.Compose([Resize(output_size=args.img_size)]))
    testloader = DataLoader(db_test, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    # Matrix of the network performance: row: class; column: metric of performance (dice, hd95)
    metric_avg = np.zeros(shape=(args.num_classes-1, 2)) 
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image_batch, label_batch, time_batch, case_name_batch = sampled_batch["image"], sampled_batch["label"], sampled_batch["time"], sampled_batch['case_name']
        image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
        # Method in "utils.py" to run the model/network in the evaluation model on multiple inputs in parallel using GPU (at the end of it, the results are transferred to CPU for further calculations).
        name_batch, time_batch, metric_batch = test_multiple_volumes_sequencing(image_batch, label_batch, time_batch, model, classes=args.num_classes, patch_size=args.img_size,
                                         test_save_path=test_save_path, name_batch=case_name_batch, z_spacing=args.z_spacing)
        for i in range(len(name_batch)):
            formatting_tuple = tuple([name_batch[i]] + [time_batch[i]] + [metric_batch[i, j] for j in range(metric_avg.shape[0] * metric_avg.shape[1])])
            logging.info('name %7s time %8.6f dice_class1 %f hd95_class1 %f' % formatting_tuple)
        for i in range(1, args.num_classes):
            metric_avg[i - 1, :] += np.sum(metric_batch, axis=0)[2 * (i - 1):2 * (i - 1) + 2]
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_avg = metric_avg / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean metrics in class %d: dice: %f hd95: %f' % (i, metric_avg[i-1][0], metric_avg[i-1][1]))
    performance = np.mean(metric_avg, axis=0)[0]
    mean_hd95 = np.mean(metric_avg, axis=0)[1]
    logging.info('Testing performance in best val model: dice: %f hd95: %f' % (performance, mean_hd95))
    return "Testing Finished!"


def inferrer_mat(args, model, test_save_path=None):
    """Main inferrence function for the material design dataset used in our TransVNet as a 3D predictive and generative model"""
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, transform=transforms.Compose([Resize2(output_size=args.img_size)]))
    if args.index is None: # If the whole test cases in the testing list need to be tested.
        testloader = DataLoader(db_test, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)
    else: # If only one case or sample needs to be tested.
        testloader = db_test[args.index]
    logging.info("{} test iterations per epoch".format(len(testloader)))
    metric = torch.tensor([0.0, 0.0, 0.0]) # The metrics are (surrogate_model_error, generative_error, log_pxz).
    counter = 0 # Number of cases tested (for the average calculation)
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_batch, label_batch, time_batch, name_batch = sampled_batch["image"], sampled_batch["label"], sampled_batch["time"], sampled_batch['case_name']
        image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
        # Method in "utils.py" to run the model/network in the evaluation model on multiple inputs in parallel using GPU (at the end of it, the results are transferred to CPU for further calculations).
        name_batch, metric_batch = test_multiple_volumes_generative(image_batch, label_batch, time_batch, model, name_batch, test_save_path)
        for i in range(len(name_batch)):
            logging.info('name %7s surrogate_model_error %f generative_error %f reconstruction_loss %f' % (name_batch[i], metric_batch[i][0], metric_batch[i][1], metric_batch[i][2]))
            metric[0] = metric[0] + metric_batch[i][0]
            metric[1] = metric[1] + metric_batch[i][1]
            metric[2] = metric[2] + metric_batch[i][2]
            counter = counter + 1
    # Average metrics
    logging.info('name %s surrogate_model_error %f generative_error %f reconstruction_loss %f' % ('average', metric[0]/counter, metric[1]/counter, metric[2]/counter))
    return "Testing Finished!"


def inferrer_mat2(args, model, test_save_path=None):
    """The inferrence function for the material design dataset used in our TransVNet as a 3D predictive and generative model - case by case testing"""
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, transform=transforms.Compose([Resize2(output_size=args.img_size)]))
    if args.index is None: # If the whole test cases in the testing list need to be tested.
        testloader = DataLoader(db_test, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)
    else: # If only one case or sample needs to be tested.
        testloader = db_test[args.index]
    if args.number_of_samplings is not None: # If the whole test cases in the testing list need to be tested.
        number_of_samplings = args.number_of_samplings
    else:
        number_of_samplings = 6
    logging.info("{} test iterations per epoch".format(len(testloader)))
    # handler = logging.StreamHandler()
    # handler.terminator = ""
    metric_avg = torch.zeros(1, 3 + args.label_size*5).cuda() # The main metrics are (surrogate_model_error, generative_error, log_pxz). The rest are related to the labels.
    counter = 0 # Number of cases tested (for the average calculation)
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_batch, label_batch, time_batch, name_batch = sampled_batch["image"], sampled_batch["label"], sampled_batch["time"], sampled_batch['case_name']
        image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
        # Method in "utils.py" to run the model/network in the evaluation mode on multiple inputs in parallel using GPU (at the end of it, the results are transferred to CPU for further calculations).
        name_batch, metric_batch = test_multiple_volumes_generative2(image_batch, label_batch, time_batch, model, name_batch, test_save_path, number_of_samplings)
        for i in range(len(name_batch)):
            formatting_tuple = tuple([name_batch[i]] + [metric_batch[i, j] for j in range(metric_avg.shape[1])])
            logging.info('name %7s surrogate_model_error %13.6f generative_error %13.6f reconstruction_loss %13.6f C11 %13.6f C11_surrogate %13.6f C11_surrogate_error %13.6f C11_generative %13.6f C11_generative_error %13.6f C12 %13.6f C12_surrogate %13.6f C12_surrogate_error %13.6f C12_generative %13.6f C12_generative_error %13.6f C13 %13.6f C13_surrogate %13.6f C13_surrogate_error %13.6f C13_generative %13.6f C13_generative_error %13.6f C33 %13.6f C33_surrogate %13.6f C33_surrogate_error %13.6f C33_generative %13.6f C33_generative_error %13.6f C44 %13.6f C44_surrogate %13.6f C44_surrogate_error %13.6f C44_generative %13.6f C44_generative_error %13.6f C66 %13.6f C66_surrogate %13.6f C66_surrogate_error %13.6f C66_generative %13.6f C66_generative_error %13.6f e31 %13.6f e31_surrogate %13.6f e31_surrogate_error %13.6f e31_generative %13.6f e31_generative_error %13.6f e33 %13.6f e33_surrogate %13.6f e33_surrogate_error %13.6f e33_generative %13.6f e33_generative_error %13.6f e15 %13.6f e15_surrogate %13.6f e15_surrogate_error %13.6f e15_generative %13.6f e15_generative_error %13.6f gamma11 %13.6f gamma11_surrogate %13.6f gamma11_surrogate_error %13.6f gamma11_generative %13.6f gamma11_generative_error %13.6f gamma33 %13.6f gamma33_surrogate %13.6f gamma33_surrogate_error %13.6f gamma33_generative %13.6f gamma33_generative_error %13.6f' % formatting_tuple)
            # logging.info('name %7s surrogate_model_error %13.6f generative_error %13.6f reconstruction_loss %13.6f C33 %13.6f C33_predicted %13.6f C33_error %13.6f e33 %13.6f e33_predicted %13.6f e33_error %13.6f' % formatting_tuple)
            metric_avg = metric_avg + metric_batch[i, :]
            counter = counter + 1
    # Average metrics
    formatting_tuple = tuple(['average'] + [metric_avg[0, i]/counter for i in range(metric_avg.shape[1])])
    logging.info('name %7s surrogate_model_error %13.6f generative_error %13.6f reconstruction_loss %13.6f C11 %13.6f C11_surrogate %13.6f C11_surrogate_error %13.6f C11_generative %13.6f C11_generative_error %13.6f C12 %13.6f C12_surrogate %13.6f C12_surrogate_error %13.6f C12_generative %13.6f C12_generative_error %13.6f C13 %13.6f C13_surrogate %13.6f C13_surrogate_error %13.6f C13_generative %13.6f C13_generative_error %13.6f C33 %13.6f C33_surrogate %13.6f C33_surrogate_error %13.6f C33_generative %13.6f C33_generative_error %13.6f C44 %13.6f C44_surrogate %13.6f C44_surrogate_error %13.6f C44_generative %13.6f C44_generative_error %13.6f C66 %13.6f C66_surrogate %13.6f C66_surrogate_error %13.6f C66_generative %13.6f C66_generative_error %13.6f e31 %13.6f e31_surrogate %13.6f e31_surrogate_error %13.6f e31_generative %13.6f e31_generative_error %13.6f e33 %13.6f e33_surrogate %13.6f e33_surrogate_error %13.6f e33_generative %13.6f e33_generative_error %13.6f e51 %13.6f e51_surrogate %13.6f e51_surrogate_error %13.6f e51_generative %13.6f e15_generative_error %13.6f gamma11 %13.6f gamma11_surrogate %13.6f gamma11_surrogate_error %13.6f gamma11_generative %13.6f gamma11_generative_error %13.6f gamma33 %13.6f gamma33_surrogate %13.6f gamma33_surrogate_error %13.6f gamma33_generative %13.6f gamma33_generative_error %13.6f' % formatting_tuple)
    # logging.info('name %7s surrogate_model_error %13.6f generative_error %13.6f reconstruction_loss %13.6f C33 %13.6f C33_predicted %13.6f C33_error %13.6f e33 %13.6f e33_predicted %13.6f e33_error %13.6f' % formatting_tuple)
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
    test_config = {
        'Synapse': {
            'dataset_name': 'Synapse',
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
            'dimension': 2,
            'prefix': 'TU', # TransUNet
        },
        'Degradation_local': {
            'dataset_name': 'Degradation_local',
            'Dataset': Degradation_dataset,
            'volume_path': '../data/deg/Selected',
            'list_dir': './lists/lists_Degradation',
            'num_classes': 2,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVD', # TransVNetDegradation
        },
        'Degradation': {
            'dataset_name': 'Degradation',
            'Dataset': Degradation_dataset,
            'volume_path': '/work/sheidaei/mhashemi/data/deg',
            'list_dir': './lists/lists_Degradation',
            'num_classes': 2,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVD', # TransVNetDegradation
        },
        'Design_local': {
            'dataset_name': 'Design_local',
            'Dataset': Design_dataset,
            # 'volume_path': '/work/sheidaei/mhashemi/data/mat',
            'volume_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'num_classes': 1,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
        },
        'Design': {
            'dataset_name': 'Design',
            'Dataset': Design_dataset,
            'volume_path': '/work/sheidaei/mhashemi/data/mat',
            # 'volume_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'num_classes': 1,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
        },
        'Design2_local': {
            'dataset_name': 'Design_local',
            'Dataset': Design_dataset,
            # 'volume_path': '/work/sheidaei/mhashemi/data/mat',
            'volume_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'num_classes': 1,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
            'number_of_samplings': 6,
        },
        'Design2': {
            'dataset_name': 'Design',
            'Dataset': Design_dataset,
            'volume_path': '/work/sheidaei/mhashemi/data/mat',
            # 'volume_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'num_classes': 1,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
            'number_of_samplings': 6,
        },
        'Design_single': { # To test a single specific case in the test list
            'dataset_name': 'Design',
            'Dataset': Design_dataset,
            # 'volume_path': '/work/sheidaei/mhashemi/data/mat',
            'volume_path': '../data/mat/Results', # On my local machine or CyBox
            'list_dir': './lists/lists_Design',
            'case_index_in_list': 0,
            'num_classes': 1,
            'z_spacing': 1,
            'dimension': 3,
            'prefix': 'TVG', # TransVNetGenerative
        }
    }
    # Assigning index attribute to args if a single case is needed to be tested in the material design dataset
    if dataset_name == 'Design_single':
        args.index = test_config[dataset_name]['case_index_in_list']
    else:
        args.index = None
    if dataset_name == 'Design2' or dataset_name == 'Design2_local':
        args.number_of_samplings = test_config[dataset_name]['number_of_samplings']
    else:
        args.index = None
    # Repeating the arg to get the 2D/3D arg
    if isinstance(args.vit_patches_size, int): #len(args.vit_patches_size) == 1:
        args.vit_patches_size = [args.vit_patches_size] * test_config[dataset_name]['dimension']
    if isinstance(args.img_size, int): #len(args.img_size) == 1:
        args.img_size = [args.img_size] * test_config[dataset_name]['dimension']
    args.num_classes = test_config[dataset_name]['num_classes']
    args.volume_path = test_config[dataset_name]['volume_path']
    args.Dataset = test_config[dataset_name]['Dataset']
    args.list_dir = test_config[dataset_name]['list_dir']
    args.z_spacing = test_config[dataset_name]['z_spacing']
    args.exp = test_config[dataset_name]['prefix'] + '_' + test_config[dataset_name]['dataset_name'] + str(args.img_size)

    if args.net_path:
        snapshot = args.net_path
        snapshot_name = snapshot.split('/')[-2]
    else:
        # name the same snapshot defined in train script!
        snapshot_path = "../model/{}/{}".format(args.exp, test_config[dataset_name]['prefix'])
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
        # Determining the name of the last/best model trained previously
        snapshot = os.path.join(snapshot_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        snapshot_name = snapshot_path.split('/')[-1]
    
    # Get the config of the network to be built/considered in training
    if test_config[dataset_name]['dimension'] == 3:
        config = CONFIGS3D[args.vit_name]
    else:
        config = CONFIGS[args.vit_name]
    if config.classifier == 'gen':
        args.label_size = config.label_size
    config.n_classes = args.num_classes
    if args.vit_name.upper().find('R50') != -1 or args.vit_name.upper().find('CONV') != -1: # If ResNet50/CNN is going to be used in a hybrid encoder
        grid = []
        for i in range(test_config[dataset_name]['dimension']):
            grid.append(int((args.img_size[i] // (2**(len(config.encoder_channels) - 1))) / args.vit_patches_size[i]))
        config.patches.grid = tuple(grid)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = torch.nn.DataParallel(net)
        net.to(device)
        # raise NotImplementedError("Only DistributedDataParallel is supported.")

    net.load_state_dict(torch.load(snapshot)) # Loading the parameters from the training results

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

    inferrer = {'Synapse': inferrer_synapse, 'Degradation': inferrer_deg, 'Degradation_local': inferrer_deg, 'Design': inferrer_mat2, 'Design_local': inferrer_mat2, 'Design2': inferrer_mat2, 'Design2_local': inferrer_mat2}
    inferrer[dataset_name](args, net, test_save_path)
    # sys.exit(0)
