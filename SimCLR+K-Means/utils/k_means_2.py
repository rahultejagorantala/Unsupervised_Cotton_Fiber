# highly sensitive to initialization.

from sklearn.cluster import KMeans
import numpy as np
import os
# from utils.evaluate_utils import _hungarian_match
import torch
from scipy.optimize import linear_sum_assignment
import faiss
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import centroid_initialization as cent_init
from sklearn import metrics

def smooth_scatter(x, y):
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)
    plt.show()

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


def plot_visualize_tsne_no_labels(input_dir, perplexity = 30):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    features = np.load(os.path.join(input_dir, 'features.npy')) # loading cropped data
    targets = np.load(os.path.join(input_dir, 'targets.npy')) # loading cropped data
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter = 5000)
    reduced_data = tsne.fit_transform(features)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = 'black')
    plt.title("t-SNE Visualization of Clusters - Ground Truth" +  "-no labels - " + str(perplexity))
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    plt.savefig(os.path.join(input_dir, "t-SNE-Ground Truth" + "-no labels - " + str(perplexity) +".png"))


def plot_visualize_tsne(input_dir, k_means_labels = None, display = False, name = None, marked_indices = None, perplexity = 30):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    features = np.load(os.path.join(input_dir, 'features.npy')) # loading cropped data
    targets = np.load(os.path.join(input_dir, 'targets.npy')) # loading cropped data
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter = 5000)
    reduced_data = tsne.fit_transform(features)

   # Define a custom colormap with distinct colors
    custom_cmap = plt.cm.get_cmap('tab10')  # Choose a colormap, such as 'tab10'

    # Plot the scatter plot with the custom colormap

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=targets, cmap=custom_cmap)
    # Mark the specified points with a different marker style or color
    plt.title("t-SNE Visualization of Clusters - Ground Truth" +  "-perplexity - " + str(perplexity))
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    if display == True:
        plt.show()
    else:
        plt.savefig(os.path.join(input_dir, "t-SNE-Ground Truth" + "-perplexity - " + str(perplexity) +".png"))


    if  k_means_labels is not None:
        print(marked_indices.shape)
        # dd
        reduced_data = np.array(reduced_data)
        # print(reduced_data.shape)
        # print(reduced_data.shape)
        # print(reduced_data[:10])
        # print(marked_indices[0].flatten(), type(marked_indices[0].flatten()))
        # print(np.array(reduced_data[marked_indices[0].flatten()])[:4])
        # marked_points = np.array(reduced_data[marked_indices.flatten()])
        # print(marked_points)
        # Combine the coordinates
        # smooth_scatter(marked_points[:, 0], marked_points[:, 1])
        # dd
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=k_means_labels, cmap=custom_cmap, )  # 'labels' represent the actual cluster assignments
        # plt.scatter(marked_points[:, 0], marked_points[:, 1], color='white', marker = 'x', label='Marked Points', s=100)  # Set s parameter to increase size

        plt.title("t-SNE Visualization of Clusters  " + name + "-perplexity - " + str(perplexity))
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        if display == True:
            plt.show()
        else:
            plt.savefig(os.path.join(input_dir, "t-SNE " + name + "-perplexity - " + str(perplexity) + ".png"))

def kmeans(input_dir, dim, k, display_plot, perplexity,create_tsne = False):
    # Create KMeans index
    import centroid_initialization as cent_init

    features = np.load(os.path.join(input_dir, 'features.npy')) # loading cropped data
    targets = np.load(os.path.join(input_dir, 'targets.npy')) # loading cropped data

    # plus_centroids = cent_init.plus_plus(features, k )
    # print(plus_centroids)

    kmeans = KMeans(n_clusters=k,init = 'random', max_iter=5000, n_init=10) 
    kmeans.fit(features)
    # print(kmeans.cluster_centers_)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    index = faiss.IndexFlatL2 (dim)
    index.add (features)
    _, indices = index.search(centroids, 1)

    print("normal k-means accuracy:")
    accuracy = hungarian(targets,labels, k )
    if create_tsne:
        plot_visualize_tsne(input_dir,labels,display_plot, "Normal K-means,  Acc - " + str(round(accuracy, 2)), indices, perplexity)


def faiss_k_means(input_dir,dim, k,display_plot, perplexity, create_tsne = False):
    import faiss
    import numpy as np
    import centroid_initialization as cent_init
    features = np.load(os.path.join(input_dir, 'features.npy')) # loading cropped data
    targets = np.load(os.path.join(input_dir, 'targets.npy')) # loading cropped data

    # Create an index and add data
    
    index = faiss.IndexFlatL2(dim)  # 128 is the dimension of your data points
    index.add(features)
    # Perform clustering
    plus_centroids, plus_plus_seed = cent_init.plus_plus(features, k , 392)
    kmeans = faiss.Kmeans(dim, k,seed = 91, nredo = 5)  # 128 is the dimension of your data points
    kmeans.train(features, init_centroids=plus_centroids)

    # Get cluster assignments
    _, clusters = kmeans.index.search(features, 1)

    index = faiss.IndexFlatL2 (dim)
    index.add (features)
    _, indices = index.search(np.array(kmeans.centroids), 1)

    accuracy, nmi, ari = hungarian(targets,clusters.flatten(), k)
    print(accuracy, nmi, ari)
    if create_tsne:
        plot_visualize_tsne(input_dir, clusters.flatten(), display_plot, "faiss K-means, Acc - " + str(round(accuracy, 2)), indices, perplexity)

def hungarian(predictions, targets, num_classes):
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = np.zeros(len(targets), dtype=predictions.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(len(targets))
    nmi = metrics.normalized_mutual_info_score(targets, predictions)
    ari = metrics.adjusted_rand_score(targets, predictions)
    # print(acc, match)
    return acc, nmi, ari

def hungarian_match(target_labels, predicted_labels):
    import numpy as np

    # Assuming you have target labels as `target_labels` and predicted labels as `predicted_labels`
    # Make sure `target_labels` and `predicted_labels` have the same length and represent the same data points.

    # Create a confusion matrix
    num_classes = len(np.unique(target_labels))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(target_labels, predicted_labels):
        confusion_matrix[t, p] += 1

    # Apply the Hungarian algorithm to find the best matching between target and predicted labels
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    # Create a dictionary to map the matched labels
    label_mapping = {t: p for t, p in zip(row_ind, col_ind)}

    # Map the predicted labels using the matched labels
    matched_predicted_labels = [label_mapping[p] for p in predicted_labels]

    # Calculate the accuracy using the matched labels
    accuracy = np.sum(target_labels == matched_predicted_labels) / len(target_labels)
    # print(matched_predicted_labels)
    # print("Hungarian Match Accuracy:", accuracy)
    return accuracy

def loop_clustering(input_dir,dim,k, redo, seed_count, faiss_ind=True):
    features = np.load(os.path.join(input_dir, 'features.npy')) # loading cropped data
    targets = np.load(os.path.join(input_dir, 'targets.npy')) # loading cropped data
    print(features.shape)
    best_accuracy = 0
    plus_plus_seed = 0
    avg_accuracy = 0
    avg_nmi = avg_ari = 0
    for i in tqdm(range(seed_count)):
        for re in range(redo):
            for j in ['random', 'k-means++']:
                if not faiss_ind:
                    if j == 'random' and re == 0:
                        kmeans = KMeans(n_clusters=k, random_state = i, max_iter=1000, init = j)
                    else:
                        plus_centroids, plus_plus_seed  = cent_init.plus_plus(features, k)
                        kmeans = KMeans(n_clusters=k, random_state = i, max_iter=1000, init = plus_centroids )
                    kmeans.fit(features)
                    labels = kmeans.labels_
                else:
                    # Create an index and add data
                    index = faiss.IndexFlatL2(dim)  # 128 is the dimension of your data points
                    index.add(features)
                    if j == 'random' and re == 0:
                        kmeans = faiss.Kmeans(dim, k,seed = i, niter=1000)  # 128 is the dimension of your data points
                        kmeans.train(features)
                    else:
                        plus_centroids, plus_plus_seed = cent_init.plus_plus(features, k )
                        kmeans = faiss.Kmeans(dim, k,seed = i, niter=1000)
                        kmeans.train(features, init_centroids = plus_centroids)
                    _, labels = kmeans.index.search(features, 1)

                accuracy, nmi, ari = hungarian(targets,labels.flatten(), k )
                avg_accuracy+=accuracy
                avg_nmi+=nmi
                avg_ari+=ari
                # hung_accuracy = hungarian_match(targets, labels.flatten())  # Compare with ground truth labels
                print(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_labels = labels
                    best_params = {'random_seed' : i,'plus_plus_seed' : plus_plus_seed, 'init' : j, 'Accuracy' :  best_accuracy, 'best_labels' : best_labels, 'nmi': nmi, 'ari': ari}
        print('\n best accuracy so far after seed ',i ," : ", round(best_accuracy,2))
    avg_accuracy = avg_accuracy /(seed_count*redo*2)
    avg_nmi = avg_nmi /(seed_count*redo*2)
    avg_ari = avg_ari /(seed_count*redo*2)
    print(best_params, "avg_accuracy :", avg_accuracy, "avg_ari :", ari, "avg_nmi :", nmi)
    # # Create an index and add data
    # index = faiss.IndexFlatL2(dim)  # 128 is the dimension of your data points
    # index.add(features)
    # # Perform clustering
    # kmeans = faiss.Kmeans(dim, k,seed = 42, nredo = 1)  # 128 is the dimension of your data points
    # kmeans.train(features)
root = r"C:\Users\AVLguest\Desktop"
basefolder = r"cotton\K-Means\pretext - no aug, 240x240, resnet50, epochs 10\embeddings"

# root = r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master"
# basefolder = r"results\stl-10\pretext\embeddings"

input_dir = os.path.join(root, basefolder)
if 'cotton' not in basefolder and 'embeddings' in basefolder:
    dim = 512
elif 'cotton' in basefolder and 'embeddings' in basefolder:
    dim = 2048
else:
    dim = 128
if 'cotton' in basefolder:
    k = 2
elif 'cifar-20' in basefolder:
    k = 20
else:
    k=10

redo = 5
seed_count = 100
perplexity = 500
# loop_clustering(input_dir,dim,k, redo, seed_count, faiss_ind = True)
faiss_k_means(input_dir, dim, k, False, perplexity, create_tsne = True)
# kmeans(input_dir, 128,2, False, 50, create_tsne = False)
# plot_visualize_tsne_no_labels(input_dir, perplexity = 50)


# print(len(targets), len(features))
# kmeans = KMeans(n_clusters=10) 

# kmeans.fit(features)

# # print(kmeans.cluster_centers_)

# labels = kmeans.labels_
# print(labels)

# #check how many of the samples were correctly labeled

# correct_labels = sum(targets == labels)

# print("Result: %d out of %d samples were correctly labeled." % (correct_labels, targets.size))

# print('Accuracy score: {0:0.2f}'. format(correct_labels/float(targets.size)))

# # plot_visualize_tsne(input_dir, k_means_labels = None, display = True)
