import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import zipfile
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    axes = {0:(0,1), 1:(1,2), 2:(0,2)}[np.random.randint(0, 3)]
    image = np.rot90(image, k, axes)
    label = np.rot90(label, k, axes)
    axis = np.random.randint(0, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    axes = {0:(0,1), 1:(1,2), 2:(0,2)}[np.random.randint(0, 3)]
    image = ndimage.rotate(image, angle, axes, order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes, order=0, reshape=False)
    return image, label


def random_rot_flip2(image, label):
    k = np.random.randint(0, 4)
    axes = (0,1)
    image = np.rot90(image, k, axes)
    axis = np.random.randint(0, 3)
    image = np.flip(image, axis=axis).copy()
    return image, label


def random_rotate2(image, label):
    angle = np.random.randint(-20, 20)
    axes = {0:(0,1), 1:(1,2), 2:(0,2)}[np.random.randint(0, 3)]
    image = ndimage.rotate(image, angle, axes, order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1] or z != self.output_size[2]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=3)  # why not 3?
            image[image>1] = 1
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=3)
            label[label>1] = 1
        image = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0) # check later if unsqueeze is needed for our 3D images
        label = torch.from_numpy(label.astype(np.uint8))
        sample['image'] = image.byte()
        sample['label'] = label.byte()
        return sample
    

# Almost the same as RandomGenerator, but with the physical limitations of the material design dataset
# I.e., there is only one image (label is actually numerical properties vector); rotation of the image/
# microstructure can only be around the axis which defines the orthotropic properties of the 
# microstructure
class RandomGenerator2(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image, label = random_rot_flip2(image, label)
        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1] or z != self.output_size[2]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=3)  # why not 3?
            image[image>1] = 1
        image = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0) # check later if unsqueeze is needed for our 3D images
        sample['image'] = image.byte()
        sample['label'] = label.to()
        return sample


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1] or z != self.output_size[2]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=3)  # why not 3?
            image[image>1] = 1
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=3)
            label[label>1] = 1
        image = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0) # check later if unsqueeze is needed for our 3D images
        label = torch.from_numpy(label.astype(np.uint8))
        sample['image'] = image.byte()
        sample['label'] = label.byte()
        return sample


class Degradation_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            volume_name, time = self.sample_list[idx].strip('\n').split()
            time = int(time)
            file_path = os.path.join(self.data_dir, volume_name)
            data = h5py.File(file_path, 'r').get('Adapted_Binary_Matrix_Degradation')
            image, label = data[:, :, :, time-1], data[:, :, :, time]
        else:
            volume_name, time = self.sample_list[idx].strip('\n').split()
            time = int(time)
            file_path = os.path.join(self.data_dir, volume_name)
            data = h5py.File(file_path, 'r').get('Adapted_Binary_Matrix_Degradation')
            image, label = data[:, :, :, time-1], data[:, :, :, time]

        sample = {'image': image, 'time': (float)(time-1)/(float)(36), 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')[8:-4]
        return sample


class Design_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        volume_name = self.sample_list[idx].strip('\n').split()
        property_path = os.path.join(self.data_dir, volume_name+'_64x64x64.mat')
        property_data = h5py.File(property_path, 'r').get('C_macro')
        C11 = property_data[0, 0]
        C12 = (property_data[0, 1] + property_data[1, 0])/2
        C13 = (property_data[0, 2] + property_data[2, 0])/2
        C33 = (property_data[1, 1] + property_data[2, 2])/2
        C44 = (property_data[3, 3] + property_data[4, 4])/2
        C66 = property_data[1, 1]
        gamma11 = (property_data[6, 6] + property_data[7, 7])/2
        gamma33 = property_data[8, 8]
        e31 = abs(property_data[0, 8]) + abs(property_data[8, 0])
        e33 = abs(property_data[0, 8]) + abs(property_data[8, 0])
        e15 = abs(property_data[0, 8]) + abs(property_data[8, 0])
        label = torch.FloatTensor([C11, C12, C13, C33, C44, C66, gamma11, gamma33, e31, e33, e15])
        image_path = os.path.join(self.data_dir, volume_name+'.zip')
        image_zip = zipfile.ZipFile(image_path)
        image = np.zeros((3, 150, 150, 150))
        index = 1
        for info in image_zip.infolist():
            if info.filename[-1:-4] == 'bmp':
                index = int(info.filename[-5:-8])
                assert index >= 1 and index <= 180, f"Index in the file name is out of bounds, got: {index}"
                if index <= 15 or index >= 166:
                    continue
            image_i = image_zip.open(info)
            image[:, index-16] = np.array(Image.open(image_i))[16:166, 16:166]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')[8:-4]
        return sample
