#########################
## Spatiotemporal Contrastive Learning for RSE 2021
#########################

import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from PIL import Image
import math
from sklearn.utils.class_weight import compute_class_weight


class xBD_Building_Polygon_TwoSides_PrePost(Dataset):

    def __init__(self, data_path, csv_file, transform=None):
        """
        Args:
            root_dir (string): directory with all images
            csv_file (string): csv file with image id's
            transform (callable, optional): optional transform to be applied to a sample image.
        """
        super(xBD_Building_Polygon_TwoSides_PrePost, self).__init__()
        self.data_path = data_path
        self.csv_file = csv_file
        self.df = pd.read_csv(os.path.join(data_path, csv_file))
        self.transform = transform

        # compute class weight
        class_labels = list(self.df['bldg_damage_code'])
        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        disaster_type = self.df.iloc[idx]['disaster_type']
        #image_id = self.df.iloc[idx]['image_id']
        bldg_uuid = self.df.iloc[idx]['bldg_uuid']
        bldg_damage = self.df.iloc[idx]['bldg_damage_code']

        # pre disaster
        bldg_name_pre = '{}_{}'.format(bldg_uuid, 'pre_disaster.png')
        bldg_path_pre = os.path.join(self.data_path, 'images_buffer', disaster_type, bldg_name_pre)
        bldg_pre = np.array(Image.open(bldg_path_pre), dtype=np.uint8)

        # post_disaster
        bldg_name_post = '{}_{}'.format(bldg_uuid, 'post_disaster.png')
        bldg_path_post = os.path.join(self.data_path, 'images_buffer', disaster_type, bldg_name_post)
        bldg_post = np.array(Image.open(bldg_path_post), dtype=np.uint8)

        sample = {'uuid': bldg_uuid,
                  'bldg_pre': bldg_pre,
                  'bldg_post': bldg_post,
                  'bldg_damage': bldg_damage}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return "Dataset, directory: {}\ncsv: {}\ntransform: {}\n class_weights: {}".format(self.data_path,
                                                                                           self.csv_file,
                                                                                           self.transform,
                                                                                           self.class_weights)


class xBD_Building_Object_OneSide_Post(Dataset):
    def __init__(self, data_path, csv_file, transform=None):
        """
        Dataset for simclr
        only post-disaster images on one side
        Args:
            root_dir (string): directory with all images
            csv_file (string): csv file with image id's
            transform (callable, optional): optional transform to be applied to a sample image.
        """
        super(xBD_Building_Object_OneSide_Post, self).__init__()
        self.data_path = data_path
        self.csv_file = csv_file
        self.df = pd.read_csv(os.path.join(data_path, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        disaster_type = self.df.iloc[idx]['disaster_type']
        #image_id = self.df.iloc[idx]['image_id']
        bldg_uuid = self.df.iloc[idx]['bldg_uuid']
        bldg_damage = self.df.iloc[idx]['bldg_damage_code']

        # pre disaster
        #bldg_name_pre = '{}_{}'.format(bldg_uuid, 'pre_disaster.png')
        #bldg_path_pre = os.path.join(self.data_path, 'images_buffer', disaster_type, bldg_name_pre)
        #bldg_pre = np.array(Image.open(bldg_path_pre), dtype=np.float32)

        # post_disaster
        bldg_name_post = '{}_{}'.format(bldg_uuid, 'post_disaster.png')
        bldg_path_post = os.path.join(self.data_path, 'images_buffer', disaster_type, bldg_name_post)
        bldg_post = np.array(Image.open(bldg_path_post), dtype=np.uint8)

        # pseudo pre_disaster for contrastive representation learning
        bldg_pre = bldg_post.copy()

        sample = {'uuid': bldg_uuid,
                  'bldg_pre': bldg_pre,
                  'bldg_post': bldg_post,
                  'bldg_damage': bldg_damage}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return "Dataset, directory: {}\ncsv: {}\ntransform: {}".format(self.data_path, self.csv_file, self.transform)


class xBD_Building_Object_OneSide_PrePost(Dataset):
    def __init__(self, data_path, csv_file, transform=None):
        """
        Dataset for simclr
        Pre- & Post- image are used together at One Side, instead of constructed as pairs
        Args:
            root_dir (string): directory with all images
            csv_file (string): csv file with image id's
            transform (callable, optional): optional transform to be applied to a sample image.
        """
        super(xBD_Building_Object_OneSide_PrePost, self).__init__()
        self.data_path = data_path
        self.csv_file = csv_file
        self.df = pd.read_csv(os.path.join(data_path, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        disaster_type = self.df.iloc[idx]['disaster_type']
        #image_id = self.df.iloc[idx]['image_id']
        bldg_uuid = self.df.iloc[idx]['bldg_uuid']
        bldg_uuid_patch = self.df.iloc[idx]['bldg_uuid_patch']
        bldg_damage = self.df.iloc[idx]['bldg_damage_code']

        bldg_path = os.path.join(self.data_path, 'images_buffer', disaster_type, bldg_uuid_patch)
        bldg = np.array(Image.open(bldg_path), dtype=np.uint8)

        sample = {'uuid': bldg_uuid,
                  'bldg_pre': bldg.copy(),
                  'bldg_post': bldg,
                  'bldg_damage': bldg_damage}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return "Dataset, directory: {}\ncsv: {}\ntransform: {}".format(self.data_path, self.csv_file, self.transform)


class RandomFlip(object):

    def __init__(self, p=0.5):
        self.p=p

    def __call__(self, sample):
        for s in ['bldg_pre', 'bldg_post']:
            if random.random() <= self.p:
                sample[s] = cv2.flip(sample[s], 1) # horizontal
            if random.random() <= self.p:
                sample[s] = cv2.flip(sample[s], 0) # vertical
        return sample

    def __repr__(self):
        return "Random flipping the image horizontally or vertically with probability [p={}].".format(self.p)


class Resize(object):

    def __init__(self, size=(64, 64)):
        assert isinstance(size, (int, tuple))
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, sample):
        for s in ['bldg_pre', 'bldg_post']:
            sample[s] = cv2.resize(sample[s], self.size, interpolation=cv2.INTER_LINEAR)
        return sample

    def __repr__(self):
        return "Resize image to {}.".format(self.size)


class RandomResizedCrop(object):

    def __init__(self, size=(64, 64), scale=(0.8, 1.0), ratio=(3.0/4, 4.0/3), num_trials=10):
        super(RandomResizedCrop, self).__init__()

        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio
        self.num_trials = num_trials
        self.resize_sample = Resize(self.size)

    def __call__(self, sample):

        for s in ['bldg_pre', 'bldg_post']:
            for attempt in range(self.num_trials):
                h, w =  sample[s].shape[:2] # width, height
                area = w * h
                target_area = random.uniform(self.scale[0], self.scale[1]) * area
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

                # aspect_ration = width / height, width * height = target_area
                h_crop = int(math.sqrt(target_area / aspect_ratio))
                w_crop = int(math.sqrt(target_area * aspect_ratio))

                if random.random() <= 0.5: # with probability of 0.5, aspect_ration = height / width
                    w_crop, h_crop = h_crop, w_crop

                # if the cropped image size not exceed the original image
                if h_crop <= h and w_crop <= w:
                    # sample the location of the left upper corner of the cropped image patch
                    top = random.randint(0, h - h_crop)  # row
                    left = random.randint(0, w - w_crop)  # column
                    sample[s] = sample[s][top:top+h_crop, left:left+w_crop]
                    break

            sample[s] = cv2.resize(sample[s], self.size, interpolation=cv2.INTER_LINEAR)

        return sample

    def __repr__(self):
        return "Random Resized Crop: [size={}] [scale={}] [ratio={}].".format(self.size, self.scale, self.ratio)


class RandomRotate(object):
    def __init__(self, angle = 10):
        """Rotate the given numpy array (around the image center) by a random degree.
        Args:
          degree_range (float): range of degree (-d ~ +d)
        """
        self.angle = angle

    def __call__(self, sample):

        for s in ['bldg_pre', 'bldg_post']:
            # sample rotation degree
            degree = np.random.uniform(-self.angle, self.angle)
            # ignore small rotations
            if np.abs(degree) <= 1.0:
                continue

            # get the max area rectangular within the rotated image
            # ref: stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
            h, w = sample[s].shape[:2]
            side_long = float(max([h, w]))
            side_short = float(min([h, w]))

            # since the solutions for angle, -angle and pi-angle are all the same,
            # it suffices to look at the first quadrant and the absolute values of sin,cos:
            sin_a = np.abs(np.sin(np.pi * degree / 180))
            cos_a = np.abs(np.cos(np.pi * degree / 180))

            if (side_short <= 2.0 * sin_a * cos_a * side_long):
                # half constrained case: two crop corners touch the longer side,
                # the other two corners are on the mid-line parallel to the longer line
                x = 0.5 * side_short
                if w >= h:
                    wr, hr = x / sin_a, x / cos_a
                else:
                    wr, hr = x / cos_a, x / sin_a
            else:
                # fully constrained case: crop touches all 4 sides
                cos_2a = cos_a * cos_a - sin_a * sin_a
                wr = (w * cos_a - h * sin_a) / cos_2a
                hr = (h * cos_a - w * sin_a) / cos_2a

            rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1.0)
            rot_mat[0,2] += (wr - w)/2.0
            rot_mat[1,2] += (hr - h)/2.0

            sample[s] = cv2.warpAffine(sample[s], rot_mat, (int(round(wr)), int(round(hr))), flags=cv2.INTER_LINEAR)

        return sample

    def __repr__(self):
        return "Image rotation with angle range of (-{}, +{}) degrees.".format(self.angle, self.angle)


class ColorJitter(object):
    """Perturb color channels of a given image
    Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
    The sampling is done independently for each channel.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """
    def __init__(self, color_range = 0.2):
        self.color_range = color_range

    def __call__(self, sample):
        for s in ['bldg_pre', 'bldg_post']:
            for i in range(sample[s].shape[2]):
                alpha = random.uniform(-self.color_range, self.color_range)
                sample[s][:, :, i] = sample[s][:, :, i] * (1 + alpha)
            sample[s] = sample[s].astype(np.uint8)
        return sample

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1-self.color_range, 1+self.color_range)


class ColorJitter_BCSH(object):
    """
    Perturb color brightness, contrast, saturation, hue of a given image
    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """
    def __init__(self, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8):
        self.brightness=brightness
        self.contrast= contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, sample):
        # convert numpy image to PIL
        for s in ['bldg_pre', 'bldg_post']:
            if random.random() <= self.p:
                sample[s] = Image.fromarray(sample[s]) # numpy array convert to pil image
                color_jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
                sample[s] = color_jitter(sample[s])
                sample[s] = np.array(sample[s]) # pil convert back to numpy array
        return sample

    def __repr__(self):
        return "Random Color Jitter based on PyTorch Implementation: [brightness={}] [contrast={}] [saturation={}] [hue={}] [p={}]".format(
            self.brightness, self.contrast, self.saturation, self.hue, self.p)


class RandomGrayScale(object):
    """
    Random color drop with probability p
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, sample):
        for s in ['bldg_pre', 'bldg_post']:
            sample[s] = Image.fromarray(sample[s])
            rnd_gray = transforms.RandomGrayscale(p=self.p)
            sample[s] = rnd_gray(sample[s])
            sample[s] = np.array(sample[s])
        return sample

    def __repr__(self):
        return "Random grayscale with probability p={}".format(self.p)


class GaussianBlur(object):
    """
    Random Gaussian Blur
    Args:
        kernel_size: 10% of the input image size (odd number)
        sigma_range: range of the sigmaX/Y to be uniformly sampled
    """
    def __init__(self, kernel_size=9, sigma_range = (0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.sigma_range = sigma_range
        self.p=p

    def __call__(self, sample):
        for s in ['bldg_pre', 'bldg_post']:
            if random.random() <= self.p:
                sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
                sample[s] = cv2.GaussianBlur(sample[s], self.kernel_size, sigma)
        return sample

    def __repr__(self):
        return "Random Gaussian Blur: [kernel={}] [sigma_range={}] [p={}]".format(
            self.kernel_size, self.sigma_range, self.p)


class ToTensor(object):
    """
    Convert numpy array to tensor
    """
    def __call__(self, sample):

        for s in ['bldg_pre', 'bldg_post']:
            sample[s] = sample[s]/255.0
            sample[s] = np.clip(sample[s], 0.0, 1.0) # clip out-range values
            sample[s] = torch.from_numpy(sample[s].transpose((2, 0, 1))) # transpose dimension

        sample['label'] = torch.tensor(sample['bldg_damage'].astype(int))

        return sample

    def __repr__(self):
        return "Numpy array image to tensor."


class Normalize_Std(object):
    """
    normalize tensor image to N(0,1)
    """
    def __init__(self, mean_pre, std_pre, mean_post, std_post):
        self.mean_pre = mean_pre
        self.std_pre = std_pre
        self.mean_post = mean_post
        self.std_post = std_post
        self.normalize_pre = transforms.Normalize(mean=mean_pre, std=std_pre)
        self.normalize_post = transforms.Normalize(mean=mean_post, std=std_post)

    def __call__(self, sample):
        sample['bldg_pre'] = self.normalize_pre(sample['bldg_pre'])
        sample['bldg_post'] = self.normalize_post(sample['bldg_post'])

        return sample

    def __repr__(self):
        return "Normalization to N(0,1),\n " \
               "mean-pre={}\n std-pre={}\n " \
               "mean-post={}\n std-post={}".format(self.mean_pre, self.std_pre, self.mean_post, self.std_post)