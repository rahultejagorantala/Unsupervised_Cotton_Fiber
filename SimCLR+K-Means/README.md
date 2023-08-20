# SimCLR_K-Means Implementation for Cotton Fiber Dataset.


This repo contains the Pytorch implementation of _[SimCLR](https://arxiv.org/pdf/2002.05709.pdf)_ paper applying some additional modifications to fit out dataset :
> _and the main source code is available [here](https://github.com/wvangansbeke/Unsupervised-Classification/tree/master)_.


## Installation requirements
The code runs with recent Pytorch versions, e.g. 1.4. 
Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
```shell
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install matplotlib scipy scikit-learn   # For evaluation and confusion matrix visualization
conda install faiss-gpu                       # For efficient nearest neighbors search 
conda install pyyaml easydict                 # For using config files
conda install termcolor                       # For colored print statements
```
We refer to the `requirements.txt` file for an overview of the packages in the environment we used to produce our results.



run the following command to perform this method on CIFAR10:
```shell
python simclr.py --config_env configs/your_env.yml --config_exp configs/pretext/simclr_cifar10.yml
```
run the following command to perform this method on Cotton Dataset:
```shell
python simclr-cotton.py --config_env configs/your_env.yml --config_exp configs/pretext/simclr_cotton.yml
```

After Running the SimCLR technique, the embeddings from the trained model are saved in the folder 'results'. They are clustered using
```shell
python utils/k_means.py
```

The above command saves the T-SNE plot of the embeddings formed. Below is one example.
