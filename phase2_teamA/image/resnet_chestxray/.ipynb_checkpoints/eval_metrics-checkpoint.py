'''
Author: Ruizhi Liao

Script for evaluation metric methods
'''

from scipy.stats import logistic
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score
import numpy as np
import sklearn
from scipy.special import softmax
import matplotlib.pyplot as plt

def compute_pairwise_auc(labels, preds):
    ''' Compute pariwise AUCs given
        labels (a batch of 4-class one-hot labels) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    pairwise_aucs = {}

    def _pairwise_auc(y_all, pred_all, channel0, channel1):
        num_datapoints = np.shape(pred_all)[0]

        y = []
        pred = []
        for j in range(num_datapoints):
            if y_all[j][channel0] == 1 or y_all[j][channel1] == 1: # Only includer "relavent" predictions/labels
                y.append(y_all[j][channel1])
                pred.append(pred_all[j][channel1]/(pred_all[j][channel0]+pred_all[j][channel1]))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        return sklearn.metrics.auc(fpr, tpr)

    pairwise_aucs['0v1'] = _pairwise_auc(labels, preds, 0, 1)
    pairwise_aucs['0v2'] = _pairwise_auc(labels, preds, 0, 2)
    pairwise_aucs['0v3'] = _pairwise_auc(labels, preds, 0, 3)
    pairwise_aucs['1v2'] = _pairwise_auc(labels, preds, 1, 2)
    pairwise_aucs['1v3'] = _pairwise_auc(labels, preds, 1, 3)
    pairwise_aucs['2v3'] = _pairwise_auc(labels, preds, 2, 3)

    return pairwise_aucs

def compute_ordinal_auc(labels, preds):
    ''' Compute ordinal AUCs given
        labels (a batch of 4-class one-hot labels) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    #assert np.shape(labels) == np.shape(preds) # size(labels)=(N,C);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]
    cutoff_channels = num_channels-1

    ordinal_aucs = [] # 0v123, 01v23, 012v3
    for i in range(cutoff_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(sum(labels[j][i+1:])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
            pred.append(sum(preds[j][i+1:])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        ordinal_aucs.append(sklearn.metrics.auc(fpr, tpr))
    
    return ordinal_aucs

def compute_multiclass_auc(labels, preds):
    ''' Compute multiclass AUCs given
        labels (a batch of C-class one-hot labels) and
        preds (a batch of predictions as C-class probabilities)
    '''

    #assert np.shape(labels) == np.shape(preds) # size(labels)=(N,C);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]

    labels = np.array(labels)
    preds = np.array(preds)
    aucs = []
    for i in range(num_channels):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels[:,i], preds[:,i], pos_label=1)
        aucs.append(sklearn.metrics.auc(fpr, tpr))

    return aucs

def compute_ordinal_acc_f1_metrics(labels, preds):
    ''' Compute ordinal AUCs given
        labels (a batch of 4-class one-hot labels) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    #assert np.shape(labels) == np.shape(preds) # size(labels)=(N,C);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]
    cutoff_channels = num_channels-1

    ordinal_precision = [] # 0v123, 01v23, 012v3
    ordinal_recall = [] # 0v123, 01v23, 012v3
    ordinal_accuracy = [] # 0v123, 01v23, 012v3
    for i in range(cutoff_channels):
        dichotomized_labels = []
        pred_classes = []
        for j in range(num_datapoints):
            y = sum(labels[j][i+1:]) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
            dichotomized_labels.append(y)
            pred_prob = sum(preds[j][i+1:]) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
            pred_classes.append(np.argmax([1-pred_prob, pred_prob]))
        precision, recall, f1, _ = precision_recall_fscore_support(dichotomized_labels, 
                                                                   pred_classes)
        accuracy = accuracy_score(dichotomized_labels, pred_classes)
        ordinal_precision.append(precision[1])
        ordinal_recall.append(recall[1])
        ordinal_accuracy.append(accuracy)
    
    return {
        "ordinal_precision": ordinal_precision,
        "ordinal_recall": ordinal_recall,
        "ordinal_accuracy": ordinal_accuracy}

def compute_acc_f1_metrics(labels, preds):
    ''' Compute accuracy, F1, and other metrics given
        labels (a batch of integers between 0 and 3) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    #assert len(labels) == np.shape(preds)[0] # size(labels)=(N,1);size(preds)=(N,C)

    pred_classes = np.argmax(preds, axis=1)
   # print(pred_classes)
    #pred_classes = preds
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_classes)
    accuracy = accuracy_score(labels, pred_classes)
    macro_f1 = np.mean(f1)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        'macro_f1': macro_f1 
    }, labels, pred_classes 

def compute_mse(labels, preds):
    ''' Compute MSE given
        labels (a batch of integers between 0 and 3) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    #assert len(labels) == np.shape(preds)[0] # size(labels)=(N,1);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]

    expect_preds = np.zeros(num_datapoints)
    for i in range(num_datapoints):
        for j in range(num_channels):
            expect_preds[i] += j * preds[i][j]

    return mse(labels, expect_preds)

def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

def compute_binary_auc(labels, preds):
    ''' Compute multiclass AUCs given
        labels (a batch of C-class one-hot labels) and
        preds (a batch of predictions as C-class probabilities)
    '''

    #assert np.shape(labels) == np.shape(preds) # size(labels)=(N,C);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]

    labels = np.array(labels)
    preds = np.array(preds)
    aucs = []

#     for i in range(num_channels):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, preds[:,0], pos_label=1)
    print(fpr,tpr,thresholds)
    
    # plot the roc curve for the model
#     plt.plot(fpr, tpr, marker='.', label='Test')
#     # axis labels
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     # show the legend
#     plt.legend()
#     # show the plot
#     plt.show()

    #scores = [f1_score(labels, to_labels(preds, thresh)) for thresh in thresholds]
    # get best threshold
    #ind = np.argmax(scores)
    gmeans = np.sqrt(tpr * (1-fpr))
    gmeans[thresholds < 5e-2] = 0
    ind = np.argmax(gmeans)
    print(gmeans)
    print(ind)
    print('Threshold=%.3f' % thresholds[ind])
    print('Accuracy=%.3f' %accuracy_score(labels, to_labels(preds, thresholds[ind])))
    aucs.append(sklearn.metrics.auc(fpr, tpr))
    #results_acc_f1, _, _ = compute_acc_f1_metrics(labels, to_labels(preds, thresholds[ind]))
    #results_acc_f1, _, _ = compute_acc_f1_metrics(labels, to_labels(preds, thresholds[ind]))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, to_labels(preds, thresholds[ind]))
    #print(results_acc_f1, _, _)
    print('precision' % precision)
    print('recall' % recall)
    print('f1' % f1)
    print('support' %_)
    print(precision, recall, f1, _)
    return aucs


