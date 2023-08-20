"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
import tqdm
import cv2
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from data.augment import Selection_crops
from torchvision import transforms
import random
from data.augment import select_crop


class COTTON(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, train_size, val_size, basefolder, root=MyPath.db_root_dir('cotton'), train=True, transform=None, 
                    download=False, im_size=None, preprocess = False , p = None):

        super(COTTON, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['Class 0', 'Class 1']
        self.preprocess = preprocess
        self.basefolder = basefolder
        
        self.data = []
        self.targets = []
        self.train_size = train_size
        self.val_size = val_size
        self.p = p
        if self.train:
            self.targets = np.loadtxt(os.path.join(self.root,self.basefolder, 'train_labels.txt'), delimiter=',', dtype=int)
            train_names = np.loadtxt(os.path.join(self.root,self.basefolder, 'train_names.txt'), delimiter=',', dtype=str)
            if self.preprocess:
                self.data = self.image_process(os.path.join(self.root, self.basefolder, 'Train/'), im_size, train_names)[:self.train_size]
                self.targets = self.targets[:self.train_size]
            else:
                self.data = np.load(os.path.join(self.root, self.basefolder,'train.npy'))[:self.train_size]
                self.targets = self.targets[:self.train_size]
                self.names = train_names[:self.train_size]
        else:
            self.targets = np.loadtxt(os.path.join(self.root, self.basefolder,'test_labels.txt'), delimiter=',', dtype=int)
            test_names = np.loadtxt(os.path.join(self.root, self.basefolder,'test_names.txt'), delimiter=',', dtype=str)
            if self.preprocess:
                self.data = self.image_process( os.path.join(self.root, self.basefolder, 'Test/'), im_size, test_names)[:self.val_size]
                self.targets = self.targets[:self.val_size]
            else:
                self.data = np.load(os.path.join(self.root, self.basefolder, 'test.npy'))[:self.val_size]        
                self.targets = self.targets[:self.val_size]
                self.names = test_names[:self.val_size]

        if self.p['augmentation_strategy'] == 'select_image' and self.train:
            unique_targets = np.unique(self.targets)
            self.indices_by_target = {target: np.where(self.targets == target)[0] for target in unique_targets}

        if self.p['augmentation_strategy'] in ['select_image_same_video', 'select_image_same_video_crop' ] and self.train:
            video_names = []
            for image_name in self.names:
                video_names.append(image_name[:image_name.rfind('-')])
            video_names = np.unique(video_names)
            # print(video_names, len(video_names))
            frame_count = {}
            for video in video_names:
                frame_list = []
                for frame in self.names:
                    # print(video, frame)
                    if frame.startswith(video + '-'):
                        frame_list.append(frame)
                if len(frame_list) > 0 :
                    frame_count[video] = frame_list 
                else:
                    print("found video name with no images. check the data")
                    print(video, frame, len(frame_count.keys()))
                    raise NotImplementedError
            self.vid_img_dict = frame_count

        if self.p['augmentation_strategy'] in ['select_crop', 'select_image_same_video_crop']:
            if self.train:
                imgnames = 'train_names_seg.txt'
                images = 'train_seg.npy'
            else:
                imgnames = 'test_names_seg.txt'
                images = 'test_seg.npy'
            self.seg_image_names = np.loadtxt(os.path.join(self.root, self.basefolder,imgnames), delimiter=',', dtype=str)
            self.seg_images = np.load(os.path.join(self.root, self.basefolder, images))
            image_names = []
            for name in self.seg_image_names:
                frame_index = name.find("frame")  # Find the index of "frame" in the input string
                if frame_index != -1:
                    substring = name[:frame_index-1]
                    image_names.append(substring)
            self.seg_image_names = image_names

            # print(self.segdata, self.segnames)
            # print(self.segdata.shape, len(self.segnames))
            # import matplotlib.pyplot as plt
            # plt.imshow(self.segdata[0])
            # plt.title(self.segnames[0])
            # plt.show()
            # dd

        # print(self.data[20].shape)
        # plt.imshow(self.data[10])
        # plt.title(self.targets[10])
        # plt.show()
        
        # print(self.classes)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        # print("inside __getitem__ cotton.py img_size : ",img_size, img.shape,img.dtype)
        img = Image.fromarray(img)
        class_name = self.classes[target]        

        if self.p['augmentation_strategy'] in ['select_crop', 'select_image_same_video_crop'] and not self.train:
            img = select_crop(index, self, self.p['augmentation_kwargs']['crop_size'], img)
        if self.transform is not None: # This will be None for training set always as we are wrapping augmented dataset onto this and are using transfomrs there.
            img = self.transform(img)
        # print(self.transform)
        # plt.imshow(img)
        # plt.show()
        # dd
        # # viewing the transformed image
        # img = np.array(img).astype(np.float)
        # img = Image.fromarray(img)
        # dd
        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def image_process(dataset_path, im_size, images_names):
        count = 0
        images = []
        for f in tqdm(images_names):    
            path =  Path(dataset_path + f )
            img = cv2.imread(dataset_path + f)
            img = cv2.resize(img, (im_size, im_size))
            x = np.expand_dims(img, axis=0)
            images.append(x)

        images = np.array(images).reshape(len(images), im_size, im_size,3)
        return images
