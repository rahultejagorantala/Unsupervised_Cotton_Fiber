import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from termcolor import colored
import numpy as np


class ConvolutionalAutoencoder():
  def __init__(self, autoencoder):
    self.network = autoencoder
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

  def train(self, loss_function, epochs, batch_size,
            training_set, validation_set, test_set, device, p,training_labels, validation_labels):

    #  creating log
    log_dict = {
        'training_loss_per_epoch': [],
        'validation_loss_per_epoch': [],
        'visualizations': []
    }

    #  defining weight initialization function
    def init_weights(module):
      if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
      elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

    #  initializing network weights
    self.network.apply(init_weights)

    #  creating dataloaders
    train_loader = DataLoader(training_set, batch_size)
    val_loader = DataLoader(validation_set, batch_size)
    test_loader = DataLoader(test_set, 10)

    # print(train_loader.dataset[0])
    # import pdb; pdb.set_trace()
    #  setting convnet to training mode
    self.network.train()
    self.network.to(device)

    if os.path.exists(p['checkpoint']):
            print(colored('Loading from checkpoint {}'.format(p['checkpoint']), 'blue'))
            checkpoint = torch.load(p['checkpoint'], map_location='cpu')
            # print(checkpoint.keys(), '/n')
            self.network.load_state_dict(checkpoint)
    else:
        for epoch in range(epochs + 1):
            train_losses = []
            self.network.train()
            #------------
            #  TRAINING
            #------------
            # print('training...')
            for images in train_loader:
                # print(images.shape)
                #  zeroing gradients
                self.optimizer.zero_grad()
                #  sending images to device
                images = images.to(device)
                #  reconstructing images
                output = self.network(images)
                #  computing loss
                loss = loss_function(output, images.view(-1, 3, 240, 240))
                #  calculating gradients
                loss.backward()
                #  optimizing weights
                self.optimizer.step()
            
            #--------------
            # LOGGING
            #--------------
            log_dict['training_loss_per_epoch'].append(loss.item())
            

            #--------------
            # VALIDATION
            #--------------
            # print('validating...')
            for val_images in val_loader:
                with torch.no_grad():
                    #  sending validation images to device
                    val_images = val_images.to(device)
                    #  reconstructing images
                    self.network.eval()
                    output = self.network(val_images)
                    #  computing validation loss
                    val_loss = loss_function(output, val_images.view(-1, 3, 240, 240))

            #--------------
            # LOGGING
            #--------------
            log_dict['validation_loss_per_epoch'].append(val_loss.item())


            #--------------
            # VISUALISATION
            #--------------
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}')
                print(f'training_loss: {round(loss.item(), 4)} validation_loss: {round(val_loss.item(), 4)}')
                # Save the model's state dictionary to the specified file path
                torch.save(self.network.state_dict(), p['checkpoint'])

                for test_images in test_loader:
                    #  sending test images to device
                    test_images = test_images.to(device)
                    with torch.no_grad():
                        #  reconstructing test images
                        self.network.eval()
                        reconstructed_imgs = self.network(test_images)
                        #  sending reconstructed and images to cpu to allow for visualization
                        reconstructed_imgs = reconstructed_imgs.cpu()
                        test_images = test_images.cpu()

                #  visualisation
                imgs = torch.stack([test_images.view(-1, 3, 240, 240), reconstructed_imgs],
                                    dim=1).flatten(0,1)
                grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
                grid = grid.permute(1, 2, 0)
                plt.figure(dpi=170)
                plt.title('Original/Reconstructed')
                plt.imshow(grid)
                #   log_dict['visualizations'].append(grid)
                plt.axis('off')
                plt.savefig(os.path.join(p['progress_path'], "epoch - " + str(epoch) ))

    self.save_embeddings(p, val_loader, validation_labels, device)
    print(colored("Training Completed", 'blue'))
    return log_dict

  def autoencode(self, x):
    return self.network(x)

  def encode(self, x):
    encoder = self.network.encoder
    return encoder(x)

  def decode(self, x):
    decoder = self.network.decoder
    return decoder(x)
  
  def save_embeddings(self, p, val_loader, validation_labels, device):
    embeddings = []
    for images in val_loader:
        self.network.eval()
        images = images.to(device)
        x = self.encode(images)
        embeddings.append(x.cpu().detach().numpy())

    np.save(os.path.join(p['embeddings_path'], 'features.npy'),np.vstack(embeddings))
    np.save(os.path.join(p['embeddings_path'], 'targets.npy'),validation_labels)
    print(colored('Saved Embeddings', 'blue'))