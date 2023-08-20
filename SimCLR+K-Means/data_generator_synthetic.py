import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os.path
from os import path
import cv2
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
random.seed(34)
import shutil
interval = 1
save = False
import argparse
# this code extracts frames from the videos and creates the dataset with images.

def check_forground(file_path, threshold):
    accepted_images = []
    with open(file_path, "r") as file:
        for line in file:
            image_name = line.strip().split(' - ')[0]
            foreground_ratio = float(line[line.index("-") + 21: line.index(",")])
            if foreground_ratio >= threshold :
                accepted_images.append(image_name + '.jpg')
    return accepted_images            

def gray_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create three copies of the grayscale image
    red = gray.copy()
    green = gray.copy()
    blue = gray.copy()

    # Create new numpy arrays for the channel images
    red_channel = np.zeros_like(gray)
    green_channel = np.zeros_like(gray)
    blue_channel = np.zeros_like(gray)

    # Set pixel values for each channel image
    red_channel[:, :] = red
    green_channel[:, :] = green
    blue_channel[:, :] = blue

    # Merge the three channel images into a single color image
    gray_img = cv2.merge((blue_channel, green_channel, red_channel))
    return gray_img


def preprocess(im_size, dataset_path):

    labels_0 = []
    labels_1 = []
    filenames = []
    names_0 = []
    names_1 = []
    threshold = 0.10
    images = []
    for folder in dataset_path:
        if str(folder) == r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_2_images_new':
            # file_path = r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_2_new.txt'
            # names_0 = check_forground(file_path, threshold)
            names_0 = names_0 + [i for i in os.listdir(folder)]
            labels_0 = [0]*len(names_0)
            images_names = names_0
            names_0 = ['texture_2_'+ i for i in names_0]
            
        else:
            # file_path = r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_4_new.txt'
            # names_1 = check_forground(file_path, threshold)
            # labels_1 = labels_1 + [1 for i in os.listdir(folder)]
            names_1 = names_1 + [i for i in os.listdir(folder)]
            labels_1 = [1]*len(names_1)
            images_names = names_1
            names_1 = ['texture_4_'+ i for i in names_1]

        for f in tqdm(images_names):    
            img = cv2.imread(os.path.join(folder, f))
            # img = gray_image(img)
            img = cv2.resize(img, (im_size, im_size)) 
            x = np.expand_dims(img, axis=0)
            images.append(x)

    images = np.array(images).reshape(len(images), im_size, im_size,3)

    names = names_0 + names_1
    labels = labels_0 + labels_1
    return labels, names, images

def test_train_val_split(labels, names, images):
    # Split input and output data into training and testing sets (70/30 split)
    output_train, output_test, filenames_train, filenames_test, images_train, images_test = train_test_split(labels, names, images, test_size=0.3, random_state=42, stratify=labels)

    print("Output shapes: train={}, test={}".format(len(output_train), len(output_test)))
    return output_train, output_test, filenames_train, filenames_test, images_train, images_test


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to the image")
    ap.add_argument("-s", "--size", required=True, help="Resize image size")
    ap.add_argument("-v", "--save", required=True, help="Indicator to save")
    args = vars(ap.parse_args())

    if os.path.exists(args['path']) == False:
        os.mkdir(args['path'])
        
    output_dir = args['path']
    save = args["save"]
    input_dir = dataset_path =[r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_2_images_new' , r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_4_images_new']

    labels, names, images = preprocess(int(args['size']), dataset_path)
    output_train, output_test, filenames_train, filenames_test, images_train, images_test = test_train_val_split(labels, names, images)
    if bool(save):  # set correct path before saving
        np.savetxt(os.path.join(output_dir, 'train_labels.txt'), output_train, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_dir,'test_labels.txt'), output_test, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_dir,'train_names.txt'), filenames_train, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_dir,'test_names.txt'), filenames_test, delimiter=',', fmt='%s')
        np.save(os.path.join(output_dir,'train'), images_train)
        np.save(os.path.join(output_dir,'test'), images_test)

# plt.imshow(images[120])
# plt.show()
# labels_train, images_names_train = extract_helper("Train/", filenames_train + filenames_val, input_dir, output_dir, output_train + output_val)
# labels_test, images_names_test = extract_helper("Test/", filenames_test, input_dir, output_dir, output_test)
# input_train = image_process( r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_2_images', im_size, filenames_train + filenames_val)
# input_test = image_process( r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_4_images', im_size, filenames_test)

# python data_generator_synthetic.py -p C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\Synthetic-500 -s 500 -v True
