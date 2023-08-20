"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate, plot_training_curves
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
# from utils.k_means import plot_visualize_tsne

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    import torch.nn as nn
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    if p['get_embeddings']:
        dummy_model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()
    # dummy_model = dummy_model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                        split='train+unlabeled') # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)

    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    if p['get_embeddings']:
        memory_bank_val = MemoryBank(len(val_dataset),
                                p['embedding_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    else:
        memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {} ,'.format(criterion.__class__.__name__), 'TYPE IS', type(criterion))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)
 
    # Checkpoint
    p['pretext_checkpoint'] =r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\results\resnet50-1x.pth"
    if os.path.exists(p['pretext_checkpoint']):
        
            print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
            checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
            # print(checkpoint['model'].keys(), '/n')
            # for i in model.parameters():
            #     print(i)
            # print(checkpoint['optimizer']["param_groups"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # print(optimizer)
            model.load_state_dict(checkpoint['model'])
            if p['get_embeddings']:
                dummy_model.load_state_dict(checkpoint['model'])
                dummy_model.contrastive_head = nn.Identity()
                print(dummy_model)
                dummy_model.cuda()
            model.cuda()
            start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()
    dd
    # Training
    train_loss = []
    epochs = []
    accuracy_train = []
    accuracy_val = []
    print(colored('Starting main loop', 'blue'))
    for epoch in tqdm(range(start_epoch, p['epochs'])):
        print(colored('Epoch %d/%d' %(epoch + 1 , p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        loss = simclr_train(train_dataloader, model, criterion, optimizer, epoch)
        train_loss.append(loss.item())
        epochs.append(epoch)

        fill_memory_bank(base_dataloader, model, memory_bank_base, None, p) 
        indices, acc = memory_bank_base.mine_nearest_neighbors(5)
        accuracy_train.append(acc)
        print('Top-5 Accuracy of kNN evaluation on train set is %.2f' %(acc))

        # fill_memory_bank(val_dataloader, model, memory_bank_val)
        # indices, acc = memory_bank_val.mine_nearest_neighbors(5)
        # accuracy_val.append(acc)

        # Fill memory bank
        # print('Fill memory bank for kNN...')
        # fill_memory_bank(base_dataloader, model, memory_bank_base)

        # # Evaluate (To monitor progress - Not for validation)
        # print('Evaluate ...')
        # top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        # accuracy.append(top1)
        # print('Result of kNN evaluation is %.2f' %(top1))
         
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint'])
        

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])


    # printing stats.
    if len(epochs) > 0:
        plot_training_curves({"epochs" : epochs , "loss" : train_loss, "accuracy_train" : accuracy_train, "accuracy_val" : accuracy_val}, None, None, os.path.dirname(p['pretext_checkpoint']))

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    if not p['get_embeddings']:
        print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
        fill_memory_bank(base_dataloader, model, memory_bank_base, None, p)
        topk = 20
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
        np.save(p['topk_neighbors_train_path'], indices)  
        
        # added to see top-5 accuracy in train set.
        print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
        fill_memory_bank(base_dataloader, model, memory_bank_base, None, p)
        topk = 5
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc)) 

   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    if not p['get_embeddings']:
        dummy_model = None
    fill_memory_bank(val_dataloader, model, memory_bank_val, dummy_model, p)
    if p['get_embeddings']:
        folder = 'embeddings'
    else:
        folder = 'features'

    save_path = os.path.join(r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\results", p["train_db_name"], "pretext", folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    memory_bank_val.save_features(save_path)
    # plot_visualize_tsne(save_path)

    if not p['get_embeddings']:
        topk = 5
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
        np.save(p['topk_neighbors_val_path'], indices)   

 
if __name__ == '__main__':
    main()
