#########################
# SimCLR for building object representation learning
# Pretext
# Both pre and post-disaster bldg objects loaded on one side, used as baseline similar to SimCLR
# T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” in Proceedings of the 37th International Conference on Machine Learning, 2020, vol. 119, pp. 1597–1607.
#########################

from torchvision import transforms, utils
import torch
import time
import logging
import dataproc_double as dp
from utils import AverageMeter, show_tensor_img, set_logger, vis_ms, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import os
from simclr_net import SimCLR_Net as Model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='0301_xbd_bldg_simclr_prepost')
parser.add_argument('-t', '--train_batch_size', type=int, default=256)
parser.add_argument('-e', '--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-6, help="weight decay")
parser.add_argument('--pretrained_model', type=str, default=None, help="path to pre-trained model")
parser.add_argument('--backbone', type=str, default='resnet18', help="backbone of the image encoder")
parser.add_argument('--data_path', type=str, default= "/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors")
parser.add_argument('--csv_train', type=str, default='csvs_buffer/sub_train_wo_unclassified_prepost.csv', help='train csv sub-path within data path')
parser.add_argument('--print_freq', type=int, default=10, help="print evaluation for every n iterations")


def main(args):

    #######################################################
    # Create necessary folders for logging results
    #######################################################
    # create logs folder
    if not os.path.isdir('./logs_models'):
        os.mkdir('./logs_models')

    log_dir = "./logs_models/logs_{}".format(args.version)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = "./logs_models/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir)

    logging.info("-----------------------------------------------------------------------------")
    logging.info("---------------Spatial CRL of xBD Pre & Post Building Object Image Patches---------------")

    #######################################################
    # Log arguments
    #######################################################
    logging.info('version: {}'.format(args.version))
    logging.info('data_path: {}'.format(args.data_path))
    logging.info('csv_train: {}'.format(args.csv_train))
    logging.info('pretrained_model: {}'.format(args.pretrained_model))
    logging.info('backbone: {}'.format(args.backbone))
    logging.info('lr: {}'.format(args.lr))
    logging.info('wd: {}'.format(args.wd))
    logging.info('n_epochs: {}'.format(args.n_epochs))
    logging.info('train batch size: {}'.format(args.train_batch_size))
    logging.info('print_freq: {}'.format(args.print_freq))

    #######################################################
    # downstream model for image classification
    #######################################################
    # load computing device, cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model(h_dim = 512, backbone_net=args.backbone).to(device=device)
    logging.info(device)
    logging.info(net)

    #######################################################
    # Criterion, optimizer, learning rate scheduler
    # original SimCLR uses optimizer LARS with learning rate of 4.8
    # however, LARS is not implemented in PyTorch, so we use Adam
    # with learning rate of 1e-4 instead with weight decay 1e-6
    #######################################################
    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.BCELoss()
    #weight = torch.tensor([0.03087066637337608, 0.2944726031734042, 0.3531639351033918, 0.3214927953498281], device=device)
    #weight = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    #criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 45], gamma=0.1)
    #logging.info(criterion)
    logging.info(optimizer)
    logging.info(scheduler)

    #######################################################
    # train + valid dataset
    # Load both pre- & post- images to One Side, and copy to the other Side
    #######################################################
    # mean & std
    # same mean and std for pre and post images
    mean_pre = (0.39327543, 0.40631564, 0.32678495)
    std_pre = (0.16512179, 0.14379614, 0.15171282)
    mean_post = (0.39327543, 0.40631564, 0.32678495)
    std_post = (0.16512179, 0.14379614, 0.15171282)
    # train data
    transform_trn = transforms.Compose([
        dp.RandomResizedCrop(size=88, scale=(0.08, 1.0)),
        dp.RandomFlip(p=0.5),
        dp.ColorJitter_BCSH(0.8, 0.8, 0.8, 0.2, p=0.8),
        dp.RandomGrayScale(p=0.2),
        dp.GaussianBlur(kernel_size=9, sigma_range=(0.1, 2.0), p=0.5),
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
        ])
    trainset = dp.xBD_Building_Object_OneSide_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_train,
                                    transform=transform_trn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    logging.info(trainset)

    #######################################################
    # resume from pre-trained model
    #######################################################
    min_loss = float('inf') # initialize valid loss to a large number
    start_epoch = 0
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        net.load_state_dict(checkpoint['net_state_dict'])
        logging.info("resumed checkpoint at epoch {} with min loss {:.4f}".format(start_epoch, min_loss))

    #######################################################
    # start training + validation
    #######################################################
    t0 = time.time()
    for ep in range(args.n_epochs):
        logging.info('Epoch [{}/{}]'.format(start_epoch+ep + 1, start_epoch+args.n_epochs))

        # training
        t1 = time.time()
        loss_train = train(trainloader, net, optimizer, start_epoch+ep, writer, device, args.print_freq)
        t2 = time.time()
        logging.info('Train [Time: {:.2f} hours] [Loss: {:.4f}]'.format((t2 - t1) / 3600.0, loss_train))
        writer.add_scalars('training/Loss', {"train": loss_train}, start_epoch+ep + 1)

        logging.info('Time spent total at [{}/{}]: {:.2f}'.format(start_epoch + ep + 1, start_epoch + args.n_epochs, (t2 - t0) / 3600.0))

        # save the best model
        is_best = loss_train < min_loss
        min_loss = min(loss_train, min_loss)
        save_checkpoint({
            'epoch': start_epoch + ep + 1,
            'net_state_dict': net.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': min_loss,
        }, is_best, root_dir=model_dir, checkpoint_name='checkpoint_ep_{}'.format(start_epoch + ep + 1))

        # reschedule learning rate
        scheduler.step(loss_train) # if loss plateau
        #scheduler.step() # for multiple step LR change
        current_LR = optimizer.param_groups[0]['lr']
        logging.info('Current learning rate: {:.4e}'.format(current_LR))
        if current_LR < 1e-6:
            logging.info('**********Learning rate too small, training stopped at epoch {}**********'.format(start_epoch + ep + 1))
            break

    logging.info('Training Done....')


def train(dataloader, net, optimizer, epoch, writer, device='cpu', print_freq=5):

    logging.info('Training...')
    net.train()

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)
    #pdb.set_trace()

    for i, batch in enumerate(dataloader):

        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        batch_size = bldg_post.shape[0]

        #######################
        # visualize input
        show_batch = 1
        if show_batch:
            print('iter')
            grid = utils.make_grid(vis_ms(bldg_pre, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid)
            grid = utils.make_grid(vis_ms(bldg_post, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid)
        ########################

        # feed forward
        _, _, z_pre, z_post = net(bldg_pre, bldg_post)
        loss = net.NT_Xent_Loss_v3(z_pre, z_post)

        # log loss
        epoch_loss.update(loss.item(), batch_size)
        writer.add_scalar('training/train_loss', loss.item(), epoch*n_batches+i)

        grid_pre = utils.make_grid(vis_ms(bldg_pre[:16], 0, 1, 2), nrow=4, normalize=True)
        grid_post = utils.make_grid(vis_ms(bldg_post[:16], 0, 1, 2), nrow=4, normalize=True)
        writer.add_image('training/train_bldgs_pre', grid_pre, epoch*n_batches+i)
        writer.add_image('training/train_bldgs_post', grid_post, epoch*n_batches+i)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (n_batches // print_freq + 1) == 0:
            logging.info('[{}][{}/{}], loss={:.4f}'.format(epoch+1, i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)