import logging
import os
import random
import sys
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms


def trainer_synapse(args, model, snapshot_path):
    """Trainer function for the synapse dataset, an example for the 2D-segmentation-tasked TransUNet"""
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 10 # 50 # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def seed_worker(worker_id):
    """A function to seed the workers/threads in the dataset loader methods"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def trainer_deg(args, model, snapshot_path):
    """Trainer function for the microstructure degradation dataset, an example for the 3D-segmented sequencing TransVNet"""
    from datasets.dataset_3D import Degradation_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.gpu
    # max_iterations = args.max_iterations
    db_train = Degradation_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size)]))
    print("The length of train set is: {}".format(len(db_train)))

    # g = torch.Generator()
    # g.manual_seed(args.seed)

    # train_sampler = distributed.DistributedSampler(db_train, shuffle=True)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=8, 
    #                         pin_memory=True, sampler=train_sampler, drop_last=True) #  worker_init_fn=seed_worker
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, 
                            pin_memory=True, worker_init_fn=seed_worker)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # Training epochs iterations
    for epoch_num in iterator:
        # np.random.seed(epoch_num)
        # random.seed(epoch_num)
        # # fix sampling seed such that each gpu gets different part of dataset
        # if args.distributed: 
        #     trainloader.sampler.set_epoch(epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, time_batch, label_batch = sampled_batch['image'], sampled_batch['time'], sampled_batch['label']
            image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch, time_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # Total loss value is the following composite function (each term is averaged among the input batch samples)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_ce, iter_num)

            logging.info('iteration %6d: loss: %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # Saving the intermediate training results
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                writer.add_images('train/Image', image, iter_num, None, 'CHWN')
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                writer.add_images('train/Prediction', outputs[1, ...], iter_num, None, 'CHWN')
                labs = label_batch[1, ...].unsqueeze(0)
                # writer.add_image('train/GroundTruth', labs, iter_num)
                writer.add_images('train/GroundTruth', labs, iter_num, None, 'CHWN')

        # Periodic saving of the trained model according to the current training epoch number
        save_interval = int(max_epoch / 5)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # Saving the final training results
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_mat(args, model, snapshot_path):
    """Trainer function for the material design dataset, an example for the generative TransVNet"""
    from datasets.dataset_3D import Design_dataset, RandomGenerator2
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.gpu
    db_train = Design_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                transform=transforms.Compose(
                                   [RandomGenerator2(output_size=args.img_size)]))
    print("The length of train set is: {}".format(len(db_train)))
    # g = torch.Generator()
    # g.manual_seed(args.seed)

    # train_sampler = distributed.DistributedSampler(db_train, shuffle=True)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=8, 
    #                         pin_memory=True, sampler=train_sampler, drop_last=True) #  worker_init_fn=seed_worker
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, 
                            pin_memory=True, worker_init_fn=seed_worker)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.train()
    loss_mse = torch.nn.MSELoss(reduction='none')
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    if hasattr(model, "module"):
        if len(model.module.config.patches.size) == 3:
            dim = (1, 2, 3)
        else:
            dim = (1, 2)
    else:
        if len(model.config.patches.size) == 3:
            dim = (1, 2, 3)
        else:
            dim = (1, 2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    if args.pretrained_net_path:
        max_epoch = args.max_epochs + 1 + int(os.path.basename(args.pretrained_net_path)[6:-4])
    else:
        max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(args.max_epochs), ncols=70)
    iterator2 = iterator

    # Annealing scheduler function for better training stability and final performance of the VAE
    # It oscillates between 0 and 1 and is a loss term multiplier; e.g., the KL contribution to the total network loss value changes accordingly. 
    def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5, suppress_from_cycle=4):
        if int(suppress_from_cycle) > n_cycle:
            suppress_from_cycle = n_cycle
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        
        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(int(suppress_from_cycle)):
            v , i = start , 0
            while v <= stop:
                L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
                v += step
                i += 1
        return L
    def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5, suppress_from_cycle=4):
        if int(suppress_from_cycle) > n_cycle:
            suppress_from_cycle = n_cycle
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        
        for c in range(int(suppress_from_cycle)):
            v , i = start , 0
            while v <= stop:
                if i < ((1 - ratio)/2) * period - 1:
                    L[int(i+c*period)] = start
                else:
                    L[int(i+c*period)] = v
                    v += step
                i += 1
        return L
    L = frange_cycle_linear(0.01, 1.0, max_epoch, 4, 0.5, 4)

    # Training epochs iterations
    if args.pretrained_net_path:
        iterator2 = range(int(os.path.basename(args.pretrained_net_path)[6:-4]) + 1, max_epoch)
    
    # # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
    # wd = 0.3
    
    for epoch_num in iterator2:
        # np.random.seed(epoch_num)
        # random.seed(epoch_num)
        # # fix sampling seed such that each gpu gets different part of dataset
        # if args.distributed: 
        #     trainloader.sampler.set_epoch(epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, time_batch, label_batch = sampled_batch['image'], sampled_batch['time'], sampled_batch['label']
            image_batch, time_batch, label_batch = image_batch.cuda(), time_batch.cuda(), label_batch.cuda()
            # logging.info('iteration %d: anomaly detection in image_batch: %f, time_batch: %f, label_batch: %f' % (iter_num, torch.isnan(image_batch).any() or torch.isinf(image_batch).any(), torch.isnan(time_batch).any() or torch.isinf(time_batch).any(), torch.isnan(label_batch).any() or torch.isinf(label_batch).any())) 
            predicted_labels, decoder_output, kl, log_pxz = model(image_batch, time_batch) # decoder_output is in fact the logits of the output image whose channels represent the categories/classes (each class = a material phase in this function)

            # Monte-Carlo estimation of the KL divergence loss (take average among the batch samples)
            kl = kl.mean()
            
            # Reconstruction loss in terms of log liklihood of seeing the output/decoder image given the input image (it is usually negative, so it will be negated in the total loss for minimization)
            log_pxz = log_pxz.mean()
            loss_reconstruction = -log_pxz
            # loss_ce = torch.sum(ce_loss(decoder_output, image_batch.squeeze(1).long()), dim=dim)
            # loss_reconstruction = loss_ce.mean()
            
            # MSE loss for label prediction in VAEs
            loss_pred = torch.sum(loss_mse(predicted_labels, label_batch), dim=1)
            loss_pred = loss_pred.mean()
            
            # Total loss value is the following composite function (each term is averaged among the input batch samples) (loss_reconstruction is also averaged among all voxels of the output image!)
            loss = 10*L[epoch_num]*kl + loss_reconstruction + 100*loss_pred
            # loss = 100*loss_reconstruction + loss_pred
            
            optimizer.zero_grad()

            loss.backward()
            # # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
            # for group in optimizer.param_groups:
            #     for param in group['params']:
            #         param.data = param.data.add(param.data, alpha=-wd * group['lr'])
            optimizer.step()
            # For SGD optimization
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)
            writer.add_scalar('info/annealing_multiplier', L[epoch_num], epoch_num)
            writer.add_scalar('info/loss_kl', kl, iter_num)
            writer.add_scalar('info/loss_recon', loss_reconstruction, iter_num)
            writer.add_scalar('info/loss_pred', loss_pred, iter_num)

            logging.info('iteration %6d: loss: %f, loss_kl: %f, loss_recon: %f, loss_pred: %f' % (iter_num, loss, kl, loss_reconstruction, loss_pred))
            # logging.info('iteration %d: loss: %f, loss_recon: %f, loss_pred: %f' % (iter_num, loss, loss_reconstruction, loss_pred))

            # Saving the intermediate training results
            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     # writer.add_image('train/Image', image, iter_num)
            #     writer.add_images('train/Image', image, iter_num, None, 'CHWN')
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     writer.add_images('train/Prediction', outputs[1, ...], iter_num, None, 'CHWN')
            #     labs = label_batch[1, ...].unsqueeze(0)
            #     # writer.add_image('train/GroundTruth', labs, iter_num)
            #     writer.add_images('train/GroundTruth', labs, iter_num, None, 'CHWN')

        # Periodic saving of the trained model according to the current training epoch number
        save_interval = int(max_epoch / 5)
        if (epoch_num + 1) % save_interval == 0: #epoch_num > int(max_epoch / 2) and 
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # Saving the final training results
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"