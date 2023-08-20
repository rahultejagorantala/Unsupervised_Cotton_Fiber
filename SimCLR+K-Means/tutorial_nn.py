"""
Authors: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import numpy as np
import torch

from utils.config import create_config
from utils.common_config import get_model, get_train_dataset, \
                                get_val_dataset, \
                                get_val_dataloader, \
                                get_val_transformations \
                                
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Eval_nn')
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
    print(dummy_model)
    model = model.cuda()
    dummy_model = dummy_model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} val samples'.format(len(val_dataset)))
    
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

    # Checkpoint
    assert os.path.exists(p['pretext_checkpoint'])
    print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint)
    if p['get_embeddings']:
        dummy_model.load_state_dict(checkpoint)
        dummy_model.contrastive_head = nn.Identity()
        print(dummy_model)
        dummy_model.cuda()
    model.cuda()
    
    # Save model
    torch.save(model.state_dict(), p['pretext_model'])

    if not p['get_embeddings']:
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
        print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
        fill_memory_bank(base_dataloader, model, memory_bank_base)
        topk = 20
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
        np.save(p['topk_neighbors_train_path'], indices)   

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val, dummy_model, p)

    if p['get_embeddings']:
        folder = 'embeddings'
    else:
        folder = 'features'
        
    save_path = os.path.join(r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\results", p["train_db_name"], "pretext", folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path)

    memory_bank_val.save_features(save_path)

    if not p['get_embeddings']:
        topk = 5
        print('Mine the nearest neighbors (Top-%d)' %(topk)) 
        indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
        np.save(p['topk_neighbors_val_path'], indices)   

 
if __name__ == '__main__':
    main()
