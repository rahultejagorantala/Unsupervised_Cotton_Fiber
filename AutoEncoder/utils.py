import os
import torch
import numpy as np
import errno
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, DataLoader
from DTDLoader import DTD
import matplotlib.pyplot as plt
from model import Encoder
from termcolor import colored
from cotton import COTTON



class CustomDTD(Dataset):
  def __init__(self, data, transforms=None):
    self.data = data
    self.transforms = transforms

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = self.data[idx]

    if self.transforms!=None:
      image = self.transforms(image)
    return image
  

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def extract_each_class(dataset, categories):
  """
  This function searches for and returns
  one image per class
  """
  images = []
  ITERATE = True
  classes = []
  while ITERATE:
    for image, label in dataset:
      # print(dataset.classes[label])
      if dataset.classes[label] in categories:

        img = cv2.resize(np.transpose(image.numpy(), (1,2,0)), (240, 240))
        images.append(img)
        classes.append(dataset.classes[label])
        categories.remove(dataset.classes[label])
        if categories == []:
          ITERATE = False
  print(f'\n classes {classes} found')
  return images

def get_dataset(p):
    training_images = []
    validation_images =[]
    training_labels = []
    validation_labels =[]
    if  p['train_db_name'] == 'DTD':
    #  loading training data
        training_set = DTD(root=p['dataset_dir'], download=True, split = 'train',
                                transform=transforms.ToTensor())
        for img, label in training_set:
            if training_set.classes[label] in p['categories']:
                    # Convert the NumPy array to a PIL Image
                img = cv2.resize(np.transpose(img.numpy(), (1,2,0)), (240, 240))
                training_images.append(img)
                training_labels.append(label)
        
        #  creating pytorch datasets
        training_data = CustomDTD(training_images, transforms=transforms.Compose([
                                                                              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                                        #  transforms.Resize((240, 240)),
                                                                              transforms.ToTensor(),]))
    elif p['train_db_name'] == 'cotton':
       dataset_path = os.path.join(p['root_dir'], 'Datasets')
       print(dataset_path)
       training_set = COTTON(p['train_length'], p['val_length'], dataset_path, train=True, transform=None, 
                    download=False, im_size=None, preprocess = False , p = None)

    if p['val_db_name'] == 'DTD':
    #  loading validation data
        validation_set = DTD(root=p['dataset_dir'], download=True,
                                    transform=transforms.ToTensor())
        #  extracting validation images
        for img, label in validation_set:
            if validation_set.classes[label] in p['categories']:
                img = cv2.resize(np.transpose(img.numpy(), (1,2,0)), (240, 240))
                validation_images.append(img)
                validation_labels.append(label)
        validation_data = CustomDTD(validation_images, transforms=transforms.Compose([
                                                                                  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                                            #  transforms.Resize((240, 240)),
                                                                                  transforms.ToTensor(),]))
        test_images = extract_each_class(validation_set, p['categories'])
        test_data = CustomDTD(test_images, transforms=transforms.Compose([
                                                                                  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                                #  transforms.Resize((240, 240)),
                                                                                  transforms.ToTensor(),]))
    return training_data, validation_data, test_data, training_labels, validation_labels

def save_plot(log_dict, p):
    plt.figure()
    loss_train = log_dict['training_loss_per_epoch']
    loss_val = log_dict['validation_loss_per_epoch']
    epochs = range(1, p['epochs']+1)
    plt.plot(epochs, loss_train[:p['epochs']], 'g', label='Training loss')
    plt.plot(epochs, loss_val[:p['epochs']], 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(p['progress_path'], 'loss vs epochs.png'))



