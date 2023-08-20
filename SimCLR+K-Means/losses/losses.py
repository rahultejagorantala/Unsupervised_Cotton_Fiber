"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS) #EPS is a very small value. it is set as a min value in the x array to avoid log(0) scenarios
        #multiplying the modified anchor probabilities  with its corresponding log values.
        b =  x_ * torch.log(x_)
    else:
        # if only anchor output logits are given then converting to probabilities and then use the same formula as above.
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)
    # then summing the values 
    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        this method takes the anchor and neighbor logits and calculates the probabilities of each class.
        by multiplying these anchor and neighbor probabilities we get similarty matrix.
        with BCE Loss calculation of similarity and a ones matrix we get consistency loss.
        By calculating the sum of products of anchor logits and its log values we get entropy loss.
        returning them at the end.
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes] 
            The logit can be thought of as a measure of the relative evidence or 
            strength for a particular class in a binary classification problem.
            these are got by predicting on the images using the clustering model.
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        # print("anchors sample :", anchors[0])
        # calculating the softmax probabilities for the anchors and neighbors
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
        # print("anchors_prob :", anchors_prob.shape)
        # print("positives_prob :", positives_prob.shape)
        # print("anchors prob sample :", anchors_prob[0])
        # Similarity in output space
        # making the anchors shape (128,1, 10) and positvies shape (128, 10,1). bmm is batch matrix multiplication
        # multiplying the anchors and its neighbors probabilites one by one and results in a similarity array of length 128(batch size)
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity) # this is the desired values for all same images.
        # print("similarity :", similarity)
        # caculating the binary cross entropy loss on the actual simlarity and the target similarity .
        # BCE loss measures the dissimilarity or discrepancy between the predicted probabilities p and the true labels y
        # we aim to reduce this loss
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        # print("Inside SimCLRLoss: ", type(nn.Module))
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """
        # caller = inspect.currentframe().f_back.f_code
        # caller_name = caller.co_name
        # print("The calling function is:", caller_name)
        # print("Inside SimCLRLoss Forward:", features.shape)
        
        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda() #creating the identity matrix of size b x b (512 x 512)
        # print("features : ", features.shape)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0) # appending the original features 
                                                                            # and augmented features back to back. 
        anchor = features[:, 0] # original features extracted from the model
        # print("contrast_features :", contrast_features.shape)  # (batch size *2, 128)
        # print("anchor :", anchor.shape)  #(batch size, 128)
        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature #(batch size, 2*batch size)
        # print("dot_product.shape :",dot_product.shape)
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        # print("logits_max :", logits_max.shape)  # calculating the max from each feature(corresponding to each photo) and 
        # print("logits :", logits.shape)          # finding the maximum and then subtracting in the original array for stability
        
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        # logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1), 0)
        # print("mask :", mask.shape, "   logits_mask :", logits_mask.shape)
        mask = mask * logits_mask
        # print(mask)
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean() #not clear with all the above code.

        return loss
