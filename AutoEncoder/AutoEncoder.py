#  article dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchvision.utils import make_grid
from PIL import Image
import argparse
from config import create_config
from termcolor import colored
from utils import get_dataset, save_plot
from model import Autoencoder, Decoder , Encoder
from train import ConvolutionalAutoencoder
from k_means import loop_clustering

#  configuring device
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU', device)
else:
  device = torch.device('cpu')
  print('Running on the CPU')


parser = argparse.ArgumentParser(description='AutoEncoder')
parser.add_argument('--config_exp',
                    help='Config file for the environment')
args = parser.parse_args()

p = create_config(args.config_exp)
print(colored(p, 'red'))

training_data, validation_data, test_data, training_labels, validation_labels = get_dataset(p)

model = ConvolutionalAutoencoder(Autoencoder(Encoder(latent_dim=p['features_dim']), Decoder(latent_dim=p['features_dim']), device))

log_dict = model.train(nn.MSELoss(), epochs=p['epochs'], batch_size=p['batch_size'],
                       training_set=training_data, validation_set=validation_data,
                       test_set=test_data, device=device, p =p, training_labels = training_labels, validation_labels = validation_labels)

if log_dict['training_loss_per_epoch'] != []:
    save_plot(log_dict, p)
# save_embeddings(p,validation_data, validation_labels)
print(colored('Running K Means', 'blue'))
best_params = loop_clustering(p['embeddings_path'],p['features_dim'],p['num_classes'], 5, 100, faiss_ind = True)
print(best_params['Accuracy'], best_params['ari'], best_params['nmi'] )



