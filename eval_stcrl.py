#########################
# Spatiotemporal Contrastive Learning for RSE 2021
# Evaluation
# Both pre- and post-disaster bldg objects, two sides
#########################

from torchvision import transforms, utils
import torch
import torch.nn.functional as F
import time
import logging
import dataproc_double as dp
import pandas as pd
from utils import AverageMeter, show_tensor_img, set_logger, vis_ms, save_checkpoint, NT_Xent_Loss_v4, barlow_twins_loss
from torch.utils.tensorboard import SummaryWriter
import os
from stcrl_net import STCRL_Net as Model
import argparse
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='Tmp_tag')
parser.add_argument('--version', type=str, default='06162022_2212_tmp')
parser.add_argument('-t', '--train_batch_size', type=int, default=256)
parser.add_argument('-e', '--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-4, help="weight decay")
parser.add_argument('--pretrained_model', type=str, default=None, help="path to pre-trained model")
parser.add_argument('--backbone', type=str, default='resnet18', help="backbone of the image encoder")
# parser.add_argument('--data_path', type=str, default= "/home/bpeng/data/xBD/xbd_disasters_building_polygons_neighbors")
parser.add_argument('--data_path', type=str, default= "/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors")
parser.add_argument('--csv_train', type=str, default='csvs_buffer/train_tier3_test_hold_wo_unclassified.csv', help='train csv sub-path within data path')
parser.add_argument('--csv_valid', type=str, default='csvs_buffer/test_hold_wo_unclassified.csv', help='valid csv sub-path within data path')
parser.add_argument('--print_freq', type=int, default=10, help="print evaluation for every n iterations")
parser.add_argument('--gpu', action='store_true', default=False, help='use gpu for computing')


def main(args):

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
    set_logger(os.path.join(log_dir, 'train.log'))
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
    logging.info(device)

    # Replace this with your model class
    model = Model()  
    model_path = model_dir + '/model_best.pth.tar' 
    net = load_model(model, model_path, device)
    logging.info(net)

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    logging.info(trainset)

    # Evaluation mode
    net.eval()

    features = []
    labels = []

    # 1. Extract features
    for i, batch in enumerate(trainloader):
        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        bldg_label = batch['label'].to(device=device, dtype=torch.int64)

        with torch.no_grad():
            # feed forward
            _, _, z_pre, z_post = net(bldg_pre, bldg_post)
            features.append(net.pairs_similarity(z_pre, z_post).cpu().numpy())
            labels.append(bldg_label.cpu().numpy())

    features = np.concatenate(features).reshape(-1, 1)
    labels = np.concatenate(labels)

    logging.info('Features and labels extracted....')

    # Create a DataFrame from features and labels
    df = pd.DataFrame(features, columns=['feature'])
    df['label'] = labels

    # Separate majority and minority classes
    df_majority = df[df.label==0]
    df_minority1 = df[df.label==1]
    df_minority2 = df[df.label==2]
    df_minority3 = df[df.label==3]

    # Get the number of samples in the largest class
    n_samples = df_majority.shape[0]

    # Upsample minority classes
    df_minority_upsampled1 = resample(df_minority1, replace=True, n_samples=n_samples, random_state=42)
    df_minority_upsampled2 = resample(df_minority2, replace=True, n_samples=n_samples, random_state=42)
    df_minority_upsampled3 = resample(df_minority3, replace=True, n_samples=n_samples, random_state=42)

    # Combine majority class with upsampled minority classes
    df_upsampled = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2, df_minority_upsampled3])

    # Now you can use df_upsampled to train your classifier
    features = df_upsampled['feature'].values.reshape(-1, 1)
    labels = df_upsampled['label'].values

    logging.info('Data resampled....')

    # 2. Supervised Classifier
    classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100),  # three layers of 100 nodes each
                           activation='tanh',  # tanh activation function
                           solver='sgd',  # stochastic gradient descent solver
                           alpha=0.0001,  # regularization term
                           learning_rate='adaptive',  # adaptive learning rate
                           max_iter=1000,  # maximum number of iterations
                           random_state=42)  # for reproducibility
    classifier.fit(features, labels)

    logging.info('Classifier trained....')

    # 3. Make predictions
    valset = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_valid,
                                    transform=transform_trn)
    # split to smaller datasets
    # lengths = [len(valset) // 30, len(valset) - len(valset) // 30]
    # valset, _ = random_split(valset, lengths)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    logging.info(valset)

    features = []
    labels = []

    for i, batch in enumerate(validloader):
        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        bldg_label = batch['label'].to(device=device, dtype=torch.int64)

        with torch.no_grad():
            # feed forward
            _, _, z_pre, z_post = net(bldg_pre, bldg_post)
            features.append(net.pairs_similarity(z_pre, z_post).cpu().numpy())
            labels.append(bldg_label.cpu().numpy())

    features = np.concatenate(features).reshape(-1, 1)
    labels = np.concatenate(labels)

    valid_preds = classifier.predict(features)
    # Compute metrics
    accuracy = accuracy_score(labels, valid_preds)
    precision = precision_score(labels, valid_preds, average='macro')
    recall = recall_score(labels, valid_preds, average='macro')
    f1 = f1_score(labels, valid_preds, average='macro')
    cm = confusion_matrix(labels, valid_preds)

    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F1 Score: {f1}')
    logging.info(f'Confusion Matrix: {cm}')

    logging.info('Validation Done....')

def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net_state_dict'])
    model = model.to(device)
    return model

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)