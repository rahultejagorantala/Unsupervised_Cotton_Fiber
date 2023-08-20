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

def preprocess(im_size, dataset_path, output_dir):

    
    for path in dataset_path: 
        images = []       
        images_names = [i for i in os.listdir(path)]
        for f in tqdm(images_names):    
                img = cv2.imread(os.path.join(path, f))
                img = cv2.resize(img, (im_size, im_size)) 
                x = np.expand_dims(img, axis=0)
                images.append(x)

        images = np.array(images).reshape(len(images), im_size, im_size,3)
        print(path)
        if 'Train_seg' in str(path):
            np.savetxt(os.path.join(output_dir,'train_names_seg.txt'), images_names, delimiter=',', fmt='%s')
            np.save(os.path.join(output_dir,'train_seg'), images)
        elif 'Test_seg' in str(path):
            np.savetxt(os.path.join(output_dir,'test_names_seg.txt'), images_names, delimiter=',', fmt='%s')
            np.save(os.path.join(output_dir,'test_seg'), images)
    # return images_names, images


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to the image")
    ap.add_argument("-s", "--size", required=True, help="Resize image size")
    ap.add_argument("-v", "--save", required=True, help="Indicator to save")
    args = vars(ap.parse_args())

    # if os.path.exists(args['path']) == False:
    #     os.mkdir(args['path'])
        
    output_dir = args['path']
    save = args["save"]
    input_dir = dataset_path =[r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\single_frame_1000\Train_seg' , r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\single_frame_1000\Test_seg']

    preprocess(int(args['size']), dataset_path, output_dir)

# python segment_data_generator.py -p C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\single_frame_1000 -s 1000 -v True
