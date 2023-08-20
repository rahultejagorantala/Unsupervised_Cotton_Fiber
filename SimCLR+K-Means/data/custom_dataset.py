"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import cv2
import torchvision.transforms as transforms
from data.augment import select_crop
import copy
""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
def create_new_dataset(dataset, targets, class_indices):
    # Filter the dataset and targets based on the specified class indices
    filtered_dataset = dataset[np.isin(targets, class_indices)]
    filtered_targets = targets[np.isin(targets, class_indices)]
    
    return filtered_dataset, filtered_targets

def convert_labels(labels):
    class_mapping = {class_label: idx for idx, class_label in enumerate(set(labels))}
    binary_labels = [class_mapping[label] for label in labels]
    return binary_labels

class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        

        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']
            print(transform, "inside dict")
        else:
            self.image_transform = transform
            self.augmentation_transform = transform
            print(transform, "inside not dict")
        
        # dd

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if self.dataset.p['augmentation_strategy'] == 'selection_crops' and self.dataset.train:
            sample = self.dataset.__getitem__(index)
            image = sample['image']

            # import cProfile, pstats
            # profiler = cProfile.Profile()
            # profiler.enable()

            orig_img = image
            crop_name = self.dataset.names[index]
            frame_name = crop_name[:crop_name.rfind('_')]
            crops_with_same_frame = [crop for crop in self.dataset.names if crop[:crop_name.rfind('_')] == frame_name]
            random_crop = random.choice(crops_with_same_frame)
            index = np.where(self.dataset.names == random_crop)[0][0]
            # print(crop_name, '-' 'selected crop-', random_crop, "----", 'index is ',  index)
            img = self.dataset.data[index]
            img = Image.fromarray(img)

            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('cumtime').reverse_order()
            # stats.print_stats()
            # DD
            sample['image'] = self.image_transform(orig_img)
            sample['image_augmented'] = self.augmentation_transform(img)
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=1, ncols=2)
            # axes[0].imshow(orig_img)
            # axes[0].axis('off')
            # axes[0].set_title(crop_name, fontsize = 7)
            # axes[1].imshow(img)
            # axes[1].axis('off')
            # axes[1].set_title(random_crop, fontsize = 7)
            # plt.show()
            # dd
        elif self.dataset.p['augmentation_strategy'] == 'select_image' and self.dataset.train:
            sample = self.dataset.__getitem__(index)
            image = sample['image']

            # import cProfile, pstats
            # profiler = cProfile.Profile()
            # profiler.enable()

            random_select = random.choice(self.dataset.indices_by_target[sample['target']])
            # print(crop_name, '-' 'selected crop-', random_crop, "----", 'index is ',  index)
            img = self.dataset.data[random_select]
            img = Image.fromarray(img)

            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('cumtime').reverse_order()
            # stats.print_stats()
            sample['image'] = self.image_transform(image)
            sample['image_augmented'] = self.augmentation_transform(img)

        elif self.dataset.p['augmentation_strategy'] == 'select_crop' and self.dataset.train:
            sample = self.dataset.__getitem__(index)
            image = sample['image']
            

            size = self.dataset.p['augmentation_kwargs']['crop_size']
            no_of_crops = self.dataset.p['augmentation_kwargs']['num_of_crops']
            crop1 = select_crop(index, self.dataset, size, image, no_of_crops)
            crop2 = select_crop(index, self.dataset, size, image, no_of_crops)

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=1, ncols=3)
            # axes[0].imshow(sample['image'])
            # axes[0].set_title('original image')
            # axes[0].axis('off')
            # axes[1].imshow(crop1)
            # axes[1].set_title('crop image 1')
            # axes[1].axis('off')
            # axes[2].imshow(crop2)
            # axes[2].set_title('crop image 2')
            # axes[2].axis('off')
            # plt.show()
            # print(self.image_transform, self.augmentation_transform)
            sample['image'] = self.image_transform(crop1)
            sample['image_augmented'] = self.augmentation_transform(crop2)

            # import matplotlib.pyplot as plt
            # plt.imshow( np.transpose(sample['image'], (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(sample['image_augmented'], (1, 2, 0)))
            # plt.show()
            # dd
        elif self.dataset.p['augmentation_strategy'] in ['select_image_same_video' , "select_image_same_large_video"] and self.dataset.train:
            sample = self.dataset.__getitem__(index)
            old_ind = index
            image = sample['image']
            orig_img = image
            image_name = self.dataset.names[index]
            small_video_name = image_name[:image_name.rfind('-')]
            video_name = small_video_name.split('R')[0]
            images_with_same_video = copy.deepcopy(self.dataset.vid_img_dict[video_name])
            # if len(images_with_same_video) in [1] : # to check the impact of deepcopy
            #     print(len(self.dataset.data))
            #     print(images_with_same_video,image_name, video_name)
            images_with_same_video.remove(image_name)
            random_image = random.choice(images_with_same_video)
            index = np.where(self.dataset.names == random_image)[0][0]
            img = self.dataset.data[index]
            img = Image.fromarray(img)

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=1, ncols=2)
            # axes[0].imshow(orig_img)
            # axes[0].set_title('first image - ' + image_name + 'label - '+ str(self.dataset.targets[old_ind]), fontsize = 7)
            # axes[0].axis('off')
            # axes[1].imshow(img)
            # axes[1].set_title('second image -' + random_image +'label - '+ str(self.dataset.targets[index]), fontsize=7)
            # axes[1].axis('off')
            # plt.show()
            # dd
            sample['image'] = self.image_transform(orig_img)
            sample['image_augmented'] = self.augmentation_transform(img)
            
        elif self.dataset.p['augmentation_strategy'] == 'select_image_same_video_crop' and self.dataset.train:
            sample = self.dataset.__getitem__(index)
            image = sample['image']
            orig_img = image
            image_name = self.dataset.names[index]
            video_name = image_name[:image_name.rfind('-')]
            images_with_same_video = self.dataset.vid_img_dict[video_name]
            # images_with_same_video.remove(image_name)
            random_image = random.choice(images_with_same_video)
            index = np.where(self.dataset.names == random_image)[0][0]
            img = self.dataset.data[index]
            img = Image.fromarray(img)

            size = self.dataset.p['augmentation_kwargs']['crop_size']
            crop1 = select_crop(index, self.dataset, size, image)
            crop2 = select_crop(index, self.dataset, size, img)

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=1, ncols=4)
            # axes[0].imshow(orig_img)
            # axes[0].set_title('augmented image 1', fontsize = 5)
            # axes[0].axis('off')
            # axes[1].imshow(img)
            # axes[1].set_title('augmented image 2', fontsize = 5)
            # axes[1].axis('off')
            # axes[2].imshow(crop1)
            # axes[2].set_title('augmented cropped image 1', fontsize = 5)
            # axes[2].axis('off')
            # axes[3].imshow(crop2)
            # axes[3].set_title('augmented cropped image 2', fontsize = 5)
            # axes[3].axis('off')
            # plt.show()
            sample['image'] = self.image_transform(crop1)
            sample['image_augmented'] = self.augmentation_transform(crop2)
        else:
            sample = self.dataset.__getitem__(index)
            image = sample['image']

            sample['image'] = self.image_transform(image)
            sample['image_augmented'] = self.augmentation_transform(image)
        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        # print(self.indices.shape[0], len(self.dataset))
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output
