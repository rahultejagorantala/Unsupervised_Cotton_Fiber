import numpy as np
from sklearn.metrics import accuracy_score
knn_indices = np.load(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\results\cotton\pretext\topk-val-neighbors.npy')

# print(knn_indices[:20])

targets = np.loadtxt(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\test_labels.txt' , delimiter=',', dtype=int)
image_names = np.loadtxt(r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\test_names.txt' , delimiter=',', dtype=str)

print(len(targets))
# print(image_names[:20])
# data = np.load(os.path.join(self.root, self.basefolder, 'test.npy'))[:self.val_size]        
# targets = self.targets[:self.val_size]
# print(type(targets))

def accuracy():
    inferred_targets = []
    for i, index in enumerate(knn_indices):
        # print(targets[index[0]])
        neighbor_count = 0
        neighbors = index[1:6]
        for neighbor in neighbors:
            neighbor_target = targets[neighbor]
            neighbor_count += neighbor_target
        if neighbor_count >=3:
            inferred_target = 1
        else:
            inferred_target = 0
        inferred_targets.append(inferred_target)
    accuracy = accuracy_score(targets, inferred_targets) * 100
    print(f"Accuracy: {accuracy}%")


def top_5_accuracy():
    neighbor_targets = np.take(targets, knn_indices[:,1:], axis=0) # Exclude sample itself for eval
    anchor_targets = np.repeat(targets.reshape(-1,1), 5, axis=1)
    # print("neighbor_targets :", neighbor_targets[:10])
    # print("anchor_targets :", anchor_targets[:10])
    print(neighbor_targets == anchor_targets)
    accuracy = np.mean(neighbor_targets[:2] == anchor_targets[:2])
    # print("accuracy : ", accuracy)
    print(f"Top-5 Accuracy: {accuracy}%")


# accuracy()
top_5_accuracy()
