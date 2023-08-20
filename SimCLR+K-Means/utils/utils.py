"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
import errno

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # print("inside", val, n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, dummy_model, p):
    model.eval()
    memory_bank.reset()
    num_of_crops = 1
    if p['predict_kwargs']['multi_crop_predict'] : 
        num_of_crops = p['predict_kwargs']['num_of_crops']
    for j in range(0, num_of_crops):
        for i, batch in enumerate(loader):
            images = batch['image'].cuda(non_blocking=True)
            targets = batch['target'].cuda(non_blocking=True)
        
            if p is not None:
                if p["get_embeddings"] :
                    output_embedd = dummy_model(images)
                else:
                    output = model(images)
            if p is not None:
                if p["get_embeddings"]:
                    memory_bank.update(output_embedd, targets)
                else:
                    memory_bank.update(output, targets)


            if i % 100 == 0 and num_of_crops == 1:
                print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
        if num_of_crops > 1:
            print('Fill Memory Bank for multiple crops [%d/%d]' %(j+1,num_of_crops))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    # confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    class_accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    total_accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
    
    # fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.xticks([i for i in range(len(class_names))], class_names)
    plt.yticks([i for i in range(len(class_names))], class_names)
    # plt.x   (class_names, ha='right', fontsize=8, rotation=40)
    # plt.yticklabels(class_names, ha='right', fontsize=8)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix - Test Accuracy ' + str(round(total_accuracy*100, 2)) + "%")
    plt.colorbar()
    thresh = confusion_matrix.max() / 2
    for (i, j), z in np.ndenumerate(confusion_matrix):
        # print(i,j,z)

        # axes.text(j, i, '%d' %(z), ha='center', va='center', color='black', fontsize=6)
        if i == j:
            plt.text(j, i, f'{confusion_matrix[i, j]}\n({class_accuracies[i] * 100:.1f}%)', ha='center', va='center',
                color='red' if confusion_matrix[i, j] > thresh else 'black')
        else:
            plt.text(j, i, f'{confusion_matrix[i, j]}', ha='center', va='center',
                color='red' if confusion_matrix[i, j] > thresh else 'black')
            

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def save_features(features, targets, path):
        import os
        np.save(os.path.join(path, 'features.npy'), features)
        np.save(os.path.join(path, 'targets.npy'), targets)
        return 
