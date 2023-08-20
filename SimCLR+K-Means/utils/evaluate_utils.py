"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter, confusion_matrix
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy
import matplotlib.pyplot as plt


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = model(images) # getting the features from the model prediction
        # print(output.shape, target.shape, output, target)
        output = memory_bank.weighted_knn(output) 
        # print("after", output.shape, target.shape, output, target)
        acc1 = 100*torch.mean(torch.eq(output, target).float()) # calculating accuracy of each batch 
        top1.update(acc1.item(), images.size(0)) #updating the accuracy and then using for calculations

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()
    
    if isinstance(dataloader.dataset, NeighborsDataset): # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        # print(res['features'].shape)
        # dd
        output = res['output']
        # print('res[output]',res.keys())
        # print('res[features] :', len(res['features']), len(res['features'][0]))
        if return_features:
            features[ptr: ptr+bs] = res['features']
            ptr += bs
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
            # print(predictions[i], probs[i], output_i)

        targets.append(batch['target'])
        if include_neighbors:
            neighbors.append(batch['possible_neighbors'])

    predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def scan_evaluate(predictions):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()
        
        # Consistency loss       
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()
        
        # Total loss
        total_loss = - entropy_loss + consistency_loss
        
        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None, 
                        compute_purity=True, compute_confusion_matrix=True,
                        confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    # print("type(head) :", head.keys())
    # print("subhead index :",subhead_index ,subhead_index)
    targets = head['targets'].cuda()
    # print("targets :", targets.shape ,targets[:10])
    predictions = head['predictions'].cuda()
    # print("predictions :", predictions.shape,predictions[:10])
    probs = head['probabilities'].cuda()
    # print("probs :", probs.shape,probs[:10])
    num_classes = torch.unique(targets).numel()
    # print("num_classes :", num_classes)
    num_elems = targets.size(0)
    # print("num_elems :", num_elems)
    # print("neighbors :", head["neighbors"].shape, head["neighbors"][:10])
    # dd
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    # probs has the shape(batch size, num of classes). .topk() picks the top two probabilities and assigns its classes to preds_top5 for each image.
    # for example like preds_top5 = [[2,0],[1,2]] in case of preds = [[0.3, 0.2, 0.5], [0.1, 0.5, 0.4]]
    if num_classes == 2:
        _, preds_top5 = probs.topk(1, 1, largest=True) #2 for our dataset doesnt make sense because you get 100% accuracy irrepective of the output 
    else:
        _, preds_top5 = probs.topk(5, 1, largest=True) #5 for other datasets

    # print("preds_top5.shape :",preds_top5.shape)

    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)

    # after reordering the cluster labels of the predictions, they are compared with the actual target clusters to calculate accuracy
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    # print("correct_top5_binary.shape :", correct_top5_binary.shape, correct_top5_binary[:10])
    # print("targets : ", targets.shape, targets[:10])
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), 
                            class_names, confusion_matrix_file)

    if num_classes == 2:
        return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-1': top5, 'hungarian_match': match}
    else :
        return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]
    # print(flat_preds.shape, flat_preds[:10])
    # print(flat_targets.shape, flat_targets[:10])
    # dd
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
        
    # print("num_correct :", num_correct)

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))
    # print("match : ", match)

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    # print("res :", res)
    return res


def plot_training_curves(simclr_stats, scan_stats, hungarian_stats, path):
    if simclr_stats is not None:
        plt.figure()
        loss_train = simclr_stats['loss']
        epochs = simclr_stats['epochs']
        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.title('Training loss vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path + '/Loss vs epochs.png')

        plt.figure()
        acc_val = simclr_stats['accuracy_train']
        epochs = simclr_stats['epochs']
        plt.plot(epochs, acc_val, 'g', label='Training Accuracy')
        plt.title('Training Accuracy vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path + '/Training Accuracy vs epochs.png')
        plt.figure()

    if hungarian_stats is not None:
        plt.figure()
        loss_train = hungarian_stats['loss']
        epochs = hungarian_stats['epochs']

        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.title('Training loss vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path + '/Loss vs epochs.png')

        plt.figure()
        # acc_val = hungarian_stats['accuracy_val']
        acc_train = hungarian_stats['accuracy_train']

        # plt.plot(epochs, acc_val, 'g', label='Validation Accuracy')
        plt.plot(epochs, acc_train, 'b', label='Training Accuracy')
        plt.title('Accuracy vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path + '/Accuracy vs epochs.png')

        plt.figure()
        plt.plot(epochs, hungarian_stats["ari"], 'g', label='ARI')
        plt.plot(epochs, hungarian_stats["nmi"], 'b', label='NMI')
        plt.title('ARI, NMI vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('ARI, NMI')
        plt.legend()
        plt.savefig(path + '/ARI, NMI vs epochs.png')
        
        plt.figure()
    
    if scan_stats is not None:
        plt.figure()
        loss_train = scan_stats['total_loss']
        consistancy_loss = scan_stats["consistancy_loss"]
        entropy_loss = scan_stats["entropy_loss"]
        epochs = scan_stats['epochs']

        plt.plot(epochs, loss_train, 'g', label='Training Loss')
        plt.plot(epochs, consistancy_loss, 'b', label='Consistency Loss')
        plt.plot(epochs, entropy_loss, 'r', label='Entropy Loss')

        plt.title('Training loss vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path + '/Loss vs epochs.png')

        plt.figure()
        # acc_val = scan_stats['accuracy_val']
        acc_train = scan_stats['accuracy_train']

        # plt.plot(epochs, acc_val, 'g', label='Validation Accuracy')
        plt.plot(epochs, acc_train, 'b', label='Training Accuracy')
        plt.title('Accuracy vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path + '/Accuracy vs epochs.png')

        plt.figure()
        plt.plot(epochs, scan_stats["ari"], 'g', label='ARI')
        plt.plot(epochs, scan_stats["nmi"], 'b', label='NMI')
        plt.title('ARI, NMI vs epochs')
        plt.xlabel('Epochs')
        plt.ylabel('ARI, NMI')
        plt.legend()
        plt.savefig(path + '/ARI, NMI vs epochs.png')
        
        plt.figure()


