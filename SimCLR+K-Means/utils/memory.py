"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        
        # perform weighted knn
        # print("Inside weighted Knn")
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device) 
        batchSize = predictions.shape[0]
        #self.features has features of all the images where as predictions has the features of only the batch
        correlation = torch.matmul(predictions, self.features.t()) # multiplying the batch features and total features
        # print("predictions :", predictions.shape, "  self.features :", self.features.shape)
        # correlation matrix of shape (512 x 50000 , 512 x 128 for predictions and 50000 x 128)
        # finding the max 100 values in the the correlation array in dimension 1. i.e., you get 100 valued array and its indices array for all the 512 images in the batch. 
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True) 
        candidates = self.targets.view(1,-1).expand(batchSize, -1) #  expand the size of a tensor by replicating its elements along specified dimensions
        # print("correlation indices :", yi.shape, "     candidates :", candidates.shape)
        # print("correlation indices :", yi, "     candidates :", candidates)
        retrieval = torch.gather(candidates, 1, yi)  # place the candidates array values i.e., augmented targets in the same places ars the correlation indices refer to.
        # print("retrieval shape: ", retrieval.shape, retrieval, np.unique(retrieval.cpu().numpy()) )
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_() # size (512 * 100, 20(in case of cipher20))
        # print("retrieval_one_hot: ", retrieval_one_hot.shape)
        # print(retrieval_one_hot.shape, retrieval.shape)
        # print("before", retrieval_one_hot.shape, retrieval.view(-1, 1).shape , retrieval.shape, np.unique(retrieval.cpu().numpy()), np.unique(retrieval_one_hot.cpu().numpy()))
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1) #assigning value 1 to the retreval indices location into retrieval_hot array.
        # print("retrieval_one_hot : ", retrieval_one_hot.shape)
        # print("after", retrieval_one_hot)
        yd_transform = yd.clone().div_(self.temperature).exp_() # taking the top 100 values from the feature matrix of images and dividing it with temperature and then calculating the log values.
        # print("retrieval_one_hot.view(batchSize, -1 , self.C) :", retrieval_one_hot.view(batchSize, -1 , self.C).shape)
        # multiplying the reshaped retrieval_one_hot (512, 100, 20) and reshaped yd (512,100,1) and resulted in (512,100,20) then summed along dimension 1 to give (512,20)
        #i.e.,for a image, for a class, summing all the 100 values to get a single value. so each image has 20 values corresponding to 20 classes
        # print("after after", yd_transform)
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1) 
        # print("after after after", probs)
        _, class_preds = probs.sort(1, True) # sorting along dimension 1, the 20 class values of a image
        class_pred = class_preds[:, 0] # getting the first value, which is the prediction for that image
        # print("inside", class_pred)
        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def save_features(self, path):
        import os
        features = self.features.cpu().numpy()
        targets = self.targets.cpu().numpy()
        np.save(os.path.join(path, 'features.npy'), features)
        np.save(os.path.join(path, 'targets.npy'), targets)
        return

    def return_features(self):
        import os
        features = self.features.cpu().numpy()
        targets = self.targets.cpu().numpy()
        return features,targets     


    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        # print("inside mine_nearest_neighbors")
        import faiss
        features = self.features.cpu().numpy() # self.features is 50000 x 128 for ciphar20
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            # print("neighbor_targets :", neighbor_targets.shape)
            # print("anchor_targets :", anchor_targets.shape)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            # print("accuracy : ", accuracy)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)  # picking the size of the features array i.e., batch size
        
        assert(b + self.ptr <= self.n) #Checking that the pointer is lower than the total images count
        #copying the features of the batch to a bigger self.features array for future use
        self.features[self.ptr:self.ptr+b].copy_(features.detach()) 
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach()) #same with this array
        self.ptr += b #changing the pointer for next call.

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
