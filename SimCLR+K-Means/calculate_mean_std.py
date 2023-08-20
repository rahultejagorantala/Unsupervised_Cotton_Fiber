import torch
from torchvision import datasets, transforms as T

transform = None

# T.Compose([
#     T.CenterCrop(240),
#     T.ToTensor()
# ])
# dataset = datasets.ImageFolder("path/to/your/dataset", transform=transform)
from data.cotton import COTTON
# from data.cifar import CIFAR10
base_folder = r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\Synthetic'
dataset = COTTON(100000, 30000, base_folder,train=False, transform=transform, download=True, preprocess = False)
# dataset = CIFAR10(train=False, transform=transform, download=True, class_indices = 'All')

means = []
stds = []
means_list = []
std_list = []
for i in range(3):
    for img in dataset.data:
        # import pdb;
        # pdb.set_trace()
        means.append(torch.mean(torch.from_numpy(img[:,:,i]).to(torch.float32))) 
        stds.append(torch.std(torch.from_numpy(img[:,:,i]).to(torch.float32)))

    means_list.append(torch.mean(torch.tensor(means))/255)
    std_list.append(torch.mean(torch.tensor(stds))/255)


print(means_list, std_list)