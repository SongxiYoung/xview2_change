#########################
# Spatiotemporal Contrastive Learning for RSE 2021
# Pretext
# Both pre- and post-disaster bldg objects, two sides
#########################

from torchvision import transforms, utils
import torch
import torch.nn.functional as F
import time
import logging
import dataproc_double as dp
from utils import AverageMeter, show_tensor_img, set_logger, vis_ms, save_checkpoint, NT_Xent_Loss_v4, barlow_twins_loss
from torch.utils.tensorboard import SummaryWriter
import os
from stcrl_net import STCRL_Net as Model
import argparse
# Perform clustering on the embeddings
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='Tmp_tag')
parser.add_argument('--version', type=str, default='06162022_2212_tmp')
parser.add_argument('-t', '--train_batch_size', type=int, default=256)
parser.add_argument('-e', '--n_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-4, help="weight decay")
parser.add_argument('--pretrained_model', type=str, default=None, help="path to pre-trained model")
parser.add_argument('--backbone', type=str, default='resnet18', help="backbone of the image encoder")
parser.add_argument('--data_path', type=str, default= "/home/bpeng/data/xBD/xbd_disasters_building_polygons_neighbors")
parser.add_argument('--csv_train', type=str, default='csvs_buffer/train_tier3_test_hold_wo_unclassified.csv', help='train csv sub-path within data path')
parser.add_argument('--print_freq', type=int, default=10, help="print evaluation for every n iterations")
parser.add_argument('--gpu', action='store_true', default=False, help='use gpu for computing')


def main(args):

    #######################################################
    # Create necessary folders for logging results
    #######################################################
    # create logs folder
    experiments_dir = '../experiments'
    experiment_dir = os.path.join(experiments_dir, args.experiment)
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    version_dir = os.path.join(experiment_dir, args.version)
    if not os.path.isdir(version_dir):
        os.mkdir(version_dir)
    for folder in ['logs', 'outputs']:
        dir = os.path.join(version_dir, folder)
        if not os.path.isdir(dir):
            os.mkdir(dir)

    model_dir = os.path.join(version_dir, 'outputs')
    log_dir = os.path.join(version_dir, 'logs')

    set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir)

    logging.info("-----------------------------------------------------------------------------")
    logging.info("---------------Spatiotemporal CRL of xBD Building Object Image Patches---------------")

    #######################################################
    # Log arguments
    #######################################################
    logging.info('experiment: {}'.format(args.experiment))
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
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    net = Model(h_dim = 512, backbone_net=args.backbone).to(device=device)
    logging.info(device)
    logging.info(net)

    #######################################################
    # Criterion, optimizer, learning rate scheduler
    #######################################################
    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.BCELoss()
    #weight = torch.tensor([0.03087066637337608, 0.2944726031734042, 0.3531639351033918, 0.3214927953498281], device=device)
    #weight = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    #criterion = torch.nn.CrossEntropyLoss(weight=weight)
    criterion = barlow_twins_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    logging.info(criterion)
    logging.info(optimizer)
    logging.info(scheduler)

    #######################################################
    # train + valid dataset
    #######################################################
    # mean & std

    # different mean and std for pre and post images
    #mean_pre = (0.38768055, 0.40568469, 0.32679225)
    #std_pre = (0.16692868, 0.14560078, 0.15216715)
    #mean_post = (0.39971367, 0.40851469, 0.32853808)
    #std_post = (0.16498802, 0.14440385, 0.15470104)

    # same mean and std for pre and post images
    mean_pre = (0.39327543, 0.40631564, 0.32678495)
    std_pre = (0.16512179, 0.14379614, 0.15171282)
    mean_post = (0.39327543, 0.40631564, 0.32678495)
    std_post = (0.16512179, 0.14379614, 0.15171282)
    # train data
    transform_trn = transforms.Compose([
        dp.RandomFlip(p=0.5),
        dp.RandomRotate(angle=40),
        dp.RandomResizedCrop(size=88, scale=(0.8, 1.0)),
        #dp.ColorJitter_BCSH(0.2, 0.2, 0.2, 0.1, p=0.5),
        #dp.GaussianBlur(kernel_size=9, sigma_range=(0.1, 2.0), p=0.5),
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
        ])
    trainset = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_train,
                                    transform=transform_trn)
    
    # split to smaller datasets
    from torch.utils.data import random_split
    lengths = [len(trainset) // 10, len(trainset) - len(trainset) // 10]
    trainset, _ = random_split(trainset, lengths)

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
    kmeans = KMeans(n_clusters=4, random_state=0)  # set num_clusters as per your requirement
    for ep in range(args.n_epochs):
        logging.info('Epoch [{}/{}]'.format(start_epoch+ep + 1, start_epoch+args.n_epochs))

        # training
        t1 = time.time()
        loss_train = train(kmeans, trainloader, net, criterion, optimizer, start_epoch+ep, writer, device, args.print_freq)
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
        #scheduler.step(loss_train) # if valid loss plateau
        scheduler.step() # for multiple step LR change
        current_LR = optimizer.param_groups[0]['lr']
        logging.info('Current learning rate: {:.4e}'.format(current_LR))
        if current_LR < 1e-6:
            logging.info('**********Learning rate too small, training stopped at epoch {}**********'.format(start_epoch + ep + 1))
            break

    logging.info('Training Done....')


def train(clustering, dataloader, net, criterion, optimizer, epoch, writer, device='gpu', print_freq=5):

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
        show_batch = 0
        if show_batch:
            grid = utils.make_grid(vis_ms(bldg_pre, 0, 1, 2), nrow=8, normalize=True)
            show_tensor_img(grid)
            grid = utils.make_grid(vis_ms(bldg_post, 0, 1, 2), nrow=8, normalize=True)
            show_tensor_img(grid)
        ########################

        # feed forward
        _, _, z_pre, z_post = net(bldg_pre, bldg_post)
        loss = criterion(z_pre, z_post)

        # log loss
        epoch_loss.update(loss.item(), batch_size)
        writer.add_scalar('training/train_loss', loss.item(), epoch*n_batches+i)

        if i < 10:
            grid_pre = utils.make_grid(vis_ms(bldg_pre[:16], 0, 1, 2), nrow=4, normalize=True)
            grid_post = utils.make_grid(vis_ms(bldg_post[:16], 0, 1, 2), nrow=4, normalize=True)
            writer.add_image('training/train_bldgs_pre', grid_pre, epoch*n_batches+i)
            writer.add_image('training/train_bldgs_post', grid_post, epoch*n_batches+i)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize z_post and z_pre
        z_post = (z_post - z_post.mean()) / z_post.std()
        z_pre = (z_pre - z_pre.mean()) / z_pre.std()

        # After each batch, get embeddings for all data
        z_diff = z_post - z_pre
        embeddings = []
        with torch.no_grad():
            embeddings.append(z_diff) 
        embeddings = torch.cat(embeddings)
        # logging.info("-----------------embedding labels-----------------")
        # logging.info(embeddings)

        labels = clustering.fit_predict(embeddings.detach().numpy())
        labels = torch.from_numpy(labels)
        # logging.info("-----------------clustering labels-----------------")
        # logging.info(labels)

        # logging.info("-----------------batch labels-----------------")
        # Get bldg_damage_code for all data
        bldg_damage_codes = batch['label']
        # logging.info(bldg_damage_codes)

        # logging.info("-----------------comparison-----------------")
        # Compare bldg_damage_code with labels
        comparison = (bldg_damage_codes == labels).int()
        percentage = comparison.sum().item() / comparison.numel() * 100
        # logging.info(f"Match percentage: {percentage}%")

        if i % (n_batches // print_freq + 1) == 0:
            logging.info(f"Match percentage: {percentage}%")
            logging.info('[{}][{}/{}], loss={:.4f}'.format(epoch+1, i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)