import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import random
import csv
import sys
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel
from transformers import  BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def debug(data, manager_p, manager, args):

    print('-----------------Data--------------------')
    data_attrs = ["data_dir","n_known_cls","num_labels","all_label_list","known_label_list"]

    for attr in data_attrs:
        attr_name = attr
        attr_value = data.__getattribute__(attr)
        print(attr_name,':',attr_value)

    print('-----------------Args--------------------')
    for k in list(vars(args).keys()):
        print(k,':',vars(args)[k])

    print('-----------------Manager_pretrain--------------------')
    manager_p_attrs = ["device","num_train_optimization_steps","best_eval_score"]

    for attr in manager_p_attrs:
        attr_name = attr
        attr_value = manager_p.__getattribute__(attr)
        print(attr_name,':',attr_value)

    print('-----------------Manager--------------------')
    manager_attrs = ["device","best_eval_score","test_results"]

    for attr in manager_attrs:
        attr_name = attr
        attr_value = manager.__getattribute__(attr)
        print(attr_name,':',attr_value)
    
    if manager.predictions is not None:
        print('-----------------Predictions--------------------')
        show_num = 100
        for i,example in enumerate(data.test_examples):
            if i >= show_num:
                break
            sentence = example.text_a
            true_label = manager.true_labels[i]
            predict_label = manager.predictions[i]
            print(i,':',sentence)
            print('Pred:{}; True:{}'.format(predict_label,true_label))


def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
          
    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['Known'] = f_seen
    result['Open'] = f_unseen
    result['F1-score'] = f
        
    return result

def plot_confusion_matrix(cm, classes, save_name, normalize=False, title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues, save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.switch_backend('agg')
    
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        plt.savefig(save_name)
    
def draw(x, y):
    from matplotlib.colors import ListedColormap
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    print("TSNE: fitting start...")
    tsne = TSNE(2, n_jobs=4, perplexity=30)
    Y = tsne.fit_transform(x)

    vis_x = Y[:, 0]
    vis_y = Y[:, 1]

    vis_x1 , vis_y1 = [],[]
    vis_x2 , vis_y2 = [],[]
    y_1,y_2 = [],[]
    mask = [i==58 for i in y]
    
    for i,j in enumerate(mask):
        if j== False:
            vis_x1.append(vis_x[i])
            vis_y1.append(vis_y[i])
            y_1.append(y[i])
        else:
            vis_x2.append(vis_x[i])
            vis_y2.append(vis_y[i])
            y_2.append(y[i])
            
    #plt.scatter(vis_x1, vis_y1, c=y_1,cmap=plt.cm.get_cmap("jet", 4), marker='.')
    #plt.scatter(vis_x2, vis_y2, c=y_2,cmap=plt.cm.get_cmap("jet", 5), marker='x')

    plt.scatter(vis_x1, vis_y1, c=y_1,cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.scatter(vis_x2, vis_y2, c=y_2, cmap=plt.cm.get_cmap("Set3"),marker='x',alpha=0.3)


    plt.colorbar(ticks=range(5))
    plt.clim(0, 5)
    plt.legend(('known classes', 'open class'), loc='upper right')
    #plt.show()
    plt.savefig('t-sne.pdf')


def plot_curve(points):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    centers = [[] for x in range(len(points[0]))]
    print('centers',centers)
    for clusters in points:
        clusters = clusters.cpu().detach().numpy()
        for i,c in enumerate(clusters):
            centers[i].append(c)
    print('centers',centers)
    plt.figure()
    plt.grid(alpha=0.4)
    markers = ['o', '*', 's', '^', 'x', 'd', 'D', 'H', 'v', '>', 'h', 'H', 'v', '>', 'v', '<', '>', '1', '2', '3', '4', 'p']
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
    
    x = np.arange(-0.02, len(centers[0]) + 0.01).astype(dtype=np.str)
    for i,y in enumerate(centers):
        plt.plot(x,y,label=labels[i], marker=markers[i])
    
    plt.xlim(0, 20, 1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Decision Boundary $\Delta$', fontsize=12)
    plt.legend()
    plt.title('50% Known Classes on StackOverflow')
    plt.show()
    plt.savefig('curve.pdf')
    

