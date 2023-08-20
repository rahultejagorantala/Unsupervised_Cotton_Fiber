"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import matplotlib.pyplot as plt
# import cProfile
import sys

def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """

    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    # print("inside simclr")
    model.train()
    # print("inside simclr2")
    # for i, batch in enumerate(train_loader):
    #     print("batch :", type(batch))
    #     dd
    
    for i, batch in enumerate(train_loader):
        # print("train_loader keys : ", batch.keys())
        # print("train_loader length : ", len(batch))
        # print("target sample : ", batch["target"][0:10])
        # print(batch.keys(), batch['meta']['class_name'], batch['meta']['index'], batch['target'])
        
        images = batch['image']
        images_augmented = batch['image_augmented']
        # print("images.shape, images_augmented.shape",images.shape, images_augmented.shape)
        
        # print(images_augmented)
        # image_np = images_augmented[0].detach().numpy()
        # print(image_np.shape)
        # print(np.max(image_np), np.min(image_np))
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # image_np = np.transpose(image_np, (1, 2, 0))
        # images_np1 = images[0].detach().numpy()
        # images_np1 = np.transpose(images_np1, (1, 2, 0))

        # axes[0].imshow(image_np)
        # axes[0].axis('off')  # Remove axis ticks and labels
        # axes[0].set_title('augmented image')
        # axes[1].imshow(images_np1)
        # axes[1].set_title('original image')
        # axes[1].axis('off')  # Remove axis ticks and labels
        # plt.show()
        # if (images_np1 == image_np).all():
        #     print("both are equal")
        # dd

        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        # print("input shape", input_.shape)   # adding two image arrays and making them (1024, 3, 32, 32) shape to feed into the network(if batch size is 512)

        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(input_).view(b, 2, -1)

        # print("output shape", output.shape) #model predicting and returning the features array in the shape of (512, 2, 128).
        #128  is the length of features and 2 is for images and aurmented and 512 is the nmber of images
        loss = criterion(output)  #based on the features outputed by the model, the loss is computed.
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    return loss


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    This method takes the data, extracts the anchors and its neighbors and passes them to the 
    clustering model. this results in a logits matrix of size(batch size, no. of clusters)
    based on these, it calculates the losses, backpropagates and updates the model. 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN
    # print("train_loader :", type(train_loader))
    for i, batch in enumerate(train_loader):
        # Forward pass
        # print("batch.keys()", batch.keys())
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        # print("anchors, neighbors :", anchors.shape, neighbors.shape) # why is there only 1 neighbor for each anchor i.e., both sizes are (128,3,32,32)
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)  # ouput of size (128(batch size), 10(num of classes))
            neighbors_output = model(neighbors)     
        # print("anchors_output :", len(anchors_output[0]))
        # print("neighbors_output :", len(neighbors_output[0]))
        # print("anchors_output :", (anchors_output))
        # print("\n anchors_output :", len(anchors_output[0][0]))
        # print("neighbors_output :", (neighbors_output))
        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            # print("anchors_output_subhead :",anchors_output_subhead.shape)
            # print("neighbors_output_subhead :", neighbors_output_subhead.shape) # using zip to just to remove the list outside bounder 
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        # here in the loss function the model parameters are tuned only on the augmented data
        # here when torch.no_grad() is used on model(images), then when loss.backward() is called then gradients are not calculated
        # it is done to reduce overfitting. for augmented data we find the gradients. and update them using 
        # cross entropy loss
        # when optmizer.step() is called then the model parameters are updated based on the gradients, input images and images_augmented
        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)
    return loss