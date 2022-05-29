# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                  TODO: Implement the functions in this file                 #
#                                                                             #
###############################################################################


from numpy.core.fromnumeric import partition
from sklearn import utils
import util
import matplotlib.pyplot as plt
import numpy as np
from time import time

def numpy_single_nn(xtrain, xstest, k):
    distances = xtrain - xstest
    distances = np.power(distances, 2)
    distances = np.sum(distances, axis=1)
    distances = np.sqrt(distances)

    queue =  dict()
    max_distance = 0
    for i, distance in enumerate(distances):
        if len(queue) < k:
            queue[distance] = i
            max_distance = max(distance, max_distance)
            continue
        if distance < max_distance:
            queue.pop(max_distance)
            queue[distance] = i
            max_distance = max(queue.keys())

    distance_list, index_list = zip(*sorted(zip(queue.keys(), queue.values())))
    return np.array(index_list), np.array(distance_list)



def numpy_nn_get_neighbors(xtrain, xtest, k):
    """
    Compute the `k` nearest neighbors in `xtrain` of each instance in `xtest`

    This method should return a pair `(indices, distances)` of (N x k)
    matrices, with `N` the number of rows in `xtest`. The `j`th column (j=0..k)
    should contain the indices of and the distances to the `j`th nearest
    neighbor for each row in `xtest` respectively.
    """
    indices = np.zeros((xtest.shape[0], k), dtype=int)
    distances = np.zeros((xtest.shape[0], k), dtype=float)
    for i, x in enumerate(xtest):
        indices[i], distances[i] = numpy_single_nn(xtrain, x, k)
    return indices, distances


def compute_accuracy(ytrue, ypredicted):
    """
    Return the fraction of correct predictions.
    """
    correct_samples = np.sum(ytrue == ypredicted)
    return correct_samples / len(ytrue)

def time_and_accuracy_task(dataset, k, n, seed):
    """
    Measure the time and accuracy of ProdQuanNN, NumpyNN, and SklearnNN on `n`
    randomly selected examples

    Make sure to keep the output format (a tuple of dicts with keys 'pqnn',
            'npnn', and 'sknn') unchanged!
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)
    pqnn, npnn, sknn = util.get_nn_instances(dataset, xtrain, ytrain)

    accuracies = { "pqnn": 0.0, "npnn": 0.0, "sknn": 0.0 }
    times = { "pqnn": 0.0, "npnn": 0.0, "sknn": 0.0 }

    # TODO use the methods in the base class `BaseNN` to classify the instances
    # in `xsample`. Then compute the accuracy with your implementation of
    # `compute_accuracy` above using the true labels `ysample` and your
    # predicted values.
    
    classifiers = {"npnn":npnn, "sknn":sknn, "pqnn":pqnn}
    for clf_name in classifiers:
        predictions, times[clf_name] = classifiers[clf_name].timed_classify(xsample, k)
        print(predictions)
        accuracies[clf_name] = compute_accuracy(ysample, predictions)
    

    return accuracies, times

def distance_absolute_error_task(dataset, k, n, seed):
    """
    Compute the mean absolute error between the distances computed by product
    quantization and the distances computed by scikit-learn.

    Return a single real value.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
            cache_partitions=True)

    _, sknn_prediction = sknn.get_neighbors(xsample, k)
    _, pqnn_prediction = pqnn.get_neighbors(xsample, k)
    mean_abs_dist = np.mean(np.abs(sknn_prediction - pqnn_prediction))

    return mean_abs_dist

def retrieval_task(dataset, k, n, seed):
    """
    How often is scikit-learn's nearest neighbor in the top `k` neighbors of
    ProdQuanNN?

    Important note: neighbors with the exact same distance to the test instance
    are considered the same!

    Return a single real value between 0 and 1.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    retrieval_rate = 1.0 # all present in top-k of ProdQuanNN
    retrieval_rate = 0.0 # none present in top-k of ProdQuanNN


    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain, cache_partitions=True)
    pqnn_idx, _ = pqnn.get_neighbors(xsample, k)
    sknn_idx, _ = sknn.get_neighbors(xsample, 1)
    
    matches = (pqnn_idx == sknn_idx)
    print(pqnn_idx, sknn_idx)
    retrieval_rate = np.sum(matches) / len(matches)
    return retrieval_rate

def fix_partition_count(n_partitions, n_features):
    step_size = int(np.ceil(n_features / n_partitions))
    if step_size * (n_partitions-1) <= n_features - 2:
        return n_partitions
    return fix_partition_count(n_partitions - 1, n_features)

def hyperparam_task(dataset, k, n, seed):
    """
    Optimize the hyper-parameters `npartitions` and  `nclusters` of ProdQuanNN.
    Produce a plot that shows how each parameter setting affects the NN
    classifier's accuracy.

    What is the effect on the training time?

    Make sure `n` is large enough. Keep `k` fixed.

    You do not have to return anything in this function. Use this place to do
    the hyper-parameter optimization and produce the plots. Feel free to add
    additional helper functions if you want, but keep them all in this file.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    # TODO optimize the hyper parameters of ProdQuanNN and produce plot
    partition_range = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    cluster_range = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Partition of 1 fix:
    n_train_samples, n_features = xsample.shape
    partition_range = list(filter(lambda x: x < n_features, partition_range))
    cluster_range = list(filter(lambda x: x < n_train_samples, cluster_range))
    partition_range = [fix_partition_count(x, n_features) for x in partition_range]

    accuracy_data = np.full((len(partition_range), len(cluster_range)), -1.0)
    time_data = np.full((len(partition_range), len(cluster_range)), -1.0)
    for j, c in enumerate(cluster_range):
        for i, p in enumerate(partition_range):
            print("Clusters:", c, "|", "Partitions:", p)
            pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=p, nclusters=c, cache_partitions=False)
            predictions, exec_time = pqnn.timed_classify(xsample, k)
            
            accuracy_data[i][j] = compute_accuracy(ysample, predictions) * 100
            time_data[i][j] = exec_time

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    cmap_acc = plt.cm.viridis.with_extremes(under="w")
    im_acc = ax1.imshow(accuracy_data, cmap=cmap_acc, vmin=0, vmax=100, origin="lower")
    cbar = fig.colorbar(im_acc, ax=ax1, fraction=0.047)
    cbar.set_label('Accuracy (%)')
    ax1.set_xlabel("number of clusters")
    ax1.set_ylabel("number of partititions")
    ax1.set_xticks(range(0, len(cluster_range)))
    ax1.set_yticks(range(0, len(partition_range)))
    ax1.set_xticklabels(cluster_range)
    ax1.set_yticklabels(partition_range)
    for (j,i),value in np.ndenumerate(accuracy_data):
        rounded = round(value * 100) / 100
        ax1.text(i,j,rounded,ha='center',va='center')
    ax1.set_title("Impact of hyperparameters on accuracy")

    cmap_time = plt.cm.viridis.with_extremes(under="w")
    im_time = ax2.imshow(time_data, cmap=cmap_time, vmin=0, origin="lower")
    cbar_time = fig.colorbar(im_time, ax=ax2, fraction=0.047)
    cbar_time.set_label('time (s)')
    ax2.set_xlabel("number of clusters")
    ax2.set_ylabel("number of partititions")
    ax2.set_xticks(range(0, len(cluster_range)))
    ax2.set_yticks(range(0, len(partition_range)))
    ax2.set_xticklabels(cluster_range)
    ax2.set_yticklabels(partition_range)
    ax2.set_title("Impact of hyperparameters on inference time")
    
    fig.tight_layout()
    plt.show()


def plot_task(dataset, k, n, seed):
    """
    This is a fun function for you to play with and visualize the resutls of
    your implementations (emnist and emnist_orig only).
    """
    if dataset != "emnist" and dataset != "emnist_orig":
        raise ValueError("Can only plot emnist and emnist_orig")

    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)

    if n > 10:
        n = 10
        print(f"too many samples to plot, showing only first {n}")

    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
            cache_partitions=True)
    pqnn_index, _ = pqnn.get_neighbors(xsample, k)
    sknn_index, _ = sknn.get_neighbors(xsample, k)

    # `emnist` is a transformed dataset, load the original `emnist_orig` to
    # plot the result (the instances are in the same order)
    if dataset == "emnist":
        xtrain, xtest, ytrain, ytest = util.load_dataset("emnist_orig")
        xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    for index, title in zip([pqnn_index, sknn_index], ["pqnn", "sknn"]):
        fig, axs = plt.subplots(xsample.shape[0], 1+k)
        fig.suptitle(title)
        for i in range(xsample.shape[0]):
            lab = util.decode_emnist_label(ysample[i])
            axs[i, 0].imshow(xsample[i].reshape((28, 28)).T, cmap="binary")
            axs[i, 0].set_xlabel(f"label {lab}")
            for kk in range(k):
                idx = index[i, kk]
                lab = util.decode_emnist_label(ytrain[idx])
                axs[i, kk+1].imshow(xtrain[idx].reshape((28, 28)).T, cmap="binary")
                axs[i, kk+1].set_xlabel(f"label {lab} ({idx})")
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0, 0].set_title("Query")
        for kk, ax in enumerate(axs[0, 1:]):
            ax.set_title(f"Neighbor {kk}")
    plt.show()
