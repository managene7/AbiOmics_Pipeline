import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_heatmap(matrix, title):
    fig = plt.figure(figsize=(15, 3))
    ax = plt.gca()
    im = ax.imshow(matrix.T, aspect='auto')
    plt.xlabel('Gene')
    plt.ylabel('Sample')
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    # cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.close()


def plot_kscores(wcss, scores, k_range, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(wcss.keys(), wcss.values(), 'go-')
    ax[0].grid(linestyle='--')
    ax[0].set_xticks(k_range)
    ax[0].set_xlabel('K')
    ax[0].set_title('WCSS')

    ax[1].plot(scores.keys(), scores.values(), 'ro-')
    ax[1].grid(linestyle='--')
    ax[1].set_xticks(k_range)
    ax[1].set_xlabel('K')
    ax[1].set_title('Silhouette Score (higher better)')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_cluster_size(mapped_cluster_name, n_clusters, title):
    cluster_size = [len(mapped_cluster_name[i]) for i in range(n_clusters)]

    plt.figure()
    plt.bar(range(n_clusters), cluster_size)
    plt.xticks(range(n_clusters))
    plt.title('Number of Samples in Clusters: ' + title)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.show()
    plt.close()


def plot_confusion(cm, dict_label):
    n = len(dict_label)
    figsize = (max(5, n), max(5, n))  # scale with number of classes
    plt.figure(figsize=figsize)
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(ticks=range(len(dict_label)), labels=dict_label.values(), rotation=90)
    plt.yticks(ticks=range(len(dict_label)), labels=dict_label.values())

    n = cm.shape[0]  # infer class count from confusion matrix shape
    for k in range(n * n):
        i, j = k // n, k % n
        plt.text(i-0.1, j+0.05, float(cm[j, i]))  # text displays as (x, y), so y is row

    plt.xlabel('Inference')
    plt.ylabel('Ground Truth')
    plt.show()
    plt.close()


def plot_loss_graph(train_loss_list, val_loss_list, test_loss_list, early_stop_epoch, cv_fold_number):
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.axvline(early_stop_epoch, color="grey", linestyle="--", label="Early Stop Epoch")
    
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"./Loss_graph/Loss_graph_CV-{cv_fold_number}")
    plt.show()  
    plt.close()