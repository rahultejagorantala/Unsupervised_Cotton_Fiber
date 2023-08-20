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
save = True

dataset_path = [r'C:\Users\AVLguest\3089(DAIJA)\3089(DAIJA)' , r'C:\Users\AVLguest\3140(K)\3140(K)']

def preprocess(im_size, dataset_path):

    labels_0 = []
    labels_1 = []
    filenames = []
    names_0 = []
    names_1 = []

    count = 0
    for i in dataset_path:
        if str(i) == r'C:\Users\AVLguest\3089(DAIJA)\3089(DAIJA)':
            labels_0 = labels_0 + [0 for i in os.listdir(i)]
            names_0 = names_0 + [i for i in os.listdir(i)]
        else:
            labels_1 = labels_1 + [1 for i in os.listdir(i)]
            names_1 = names_1 + [i for i in os.listdir(i)]

    names_0_i = random.sample(names_0, 255)
    labels_0_i = random.sample(labels_0, 255)
    names = names_0_i + names_1
    labels = labels_0_i + labels_1
    return labels, names, im_size

def test_train_val_split(labels, names):
    # Split input and output data into training and testing sets (70/30 split)
    output_train, output_test, filenames_train, filenames_test = train_test_split(labels, names, test_size=0.3, random_state=42, stratify=labels)

    # Split training set into training and validation sets (80/20 split)
    output_train, output_val, filenames_train, filenames_val = train_test_split(output_train, filenames_train, test_size=0.2, random_state=42, stratify=output_train)

    # Print the shapes of each set
    print("Output shapes: train={}, val={}, test={}".format(len(output_train), len(output_val), len(output_test)))
    print("Filename shapes: train={}, val={}, test={}".format(len(filenames_train), len(filenames_val), len(filenames_test)))
    return output_train, output_test, output_val, filenames_train, filenames_test, filenames_val

def extract_frames(input_dir, filename, interval, output_dir):
        video_path = os.path.join(input_dir, filename)
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        counter = 0
        image_names = []
        while success:
            if count % interval == 0:
                file_path = output_dir + filename[:-4] + "-" + "frame" + str(count) + ".jpg"
                if path.exists(file_path) == False:
                    cv2.imwrite(file_path, image)
                counter+=1
                image_names.append(filename[:-4] + "-" + "frame" + str(count) + ".jpg")
            success, image = vidcap.read()
            count += 1
        return counter, image_names

def extract_helper(data_split, filenames, input_dir, output_dir, output):
        image_names = []
        output_path = os.path.join(output_dir, data_split)
        labels = []
        if os.path.exists(output_path) == True:
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        for i in tqdm(range(len(filenames))):
            file_path = os.path.join(input_dir[0], filenames[i])
            if os.path.isfile(file_path) and filenames[i].endswith("avi"):
                filepath = input_dir[0]
            else:
                filepath = input_dir[1]
            count, img_names = extract_frames(filepath, filenames[i], interval, output_path)
            labels = labels + count*[output[i]]
            image_names = image_names + img_names
        return labels, image_names

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

if os.path.exists(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/') == False:
    os.mkdir(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/')
    
output_dir = r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/"
input_dir = [ r"C:\Users\AVLguest\3140(K)\3140(K)/", r"C:\Users\AVLguest\3089(DAIJA)\3089(DAIJA)/" ]
    
labels, names, im_size = preprocess(240, dataset_path)
output_train, output_test, output_val, filenames_train, filenames_test, filenames_val = test_train_val_split(labels, names)
labels_train, images_names_train = extract_helper("Train/", filenames_train + filenames_val, input_dir, output_dir, output_train + output_val)
labels_test, images_names_test = extract_helper("Test/", filenames_test, input_dir, output_dir, output_test)
input_train = image_process( r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\Train/", im_size, images_names_train)
input_test = image_process( r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\Test/", im_size, images_names_test)

if save:  # set correct path before saving
    np.savetxt(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/train_labels.txt', labels_train, delimiter=',', fmt='%s')
    np.savetxt(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/test_labels.txt', labels_test, delimiter=',', fmt='%s')
    np.savetxt(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/train_names.txt', images_names_train, delimiter=',', fmt='%s')
    np.savetxt(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/test_names.txt', images_names_test, delimiter=',', fmt='%s')
    np.save(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/train', input_train)
    np.save(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton/test', input_test)