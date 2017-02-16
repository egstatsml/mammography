#!/usr/bin/python

"""
metrics.py

Description
Will contain some functions for determining performance of the system

Eg. Accuracy, Sensitivity, Specificity, AUROC, etc.

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc


"""
metric_accuracy()

Description:
Output the accuracy of the classifier

Accuracy = (TP + FP)/( TP + FP + FP + FN)
 
@param true = true labels in validation set
@param predicted = predicted labels in validation set


"""

def metric_accuracy(true, predicted):
    
    return accuracy_score(true, predicted)



"""
metric_sensitivity()

Description:
output overall sensitivity of the system

Sensitivity = TP/(TP + FN)

@param true = true labels in validation set
@param predicted = predicted labels in validation set

"""

def metric_sensitivity(true, predicted):
    
    TP = np.sum(np.logical_and(true, predicted))
    FN = np.sum(np.logical_and(true, np.logical_not(predicted)))
    
    return np.divide(TP, TP + FN)
    



"""
metric_specificity()

Description:
output overall specificity of the system

specificity = TN/(TN + FP)

@param true = true labels in validation set
@param predicted = predicted labels in validation set

"""

def metric_specificity(true, predicted):
    
    TN = np.sum(np.logical_and(np.logical_not(true), np.logical_not(predicted)))
    FP = np.sum(np.logical_and(np.logical_not(true), predicted))
    
    return np.divide(TN, TN + FP)



"""
metric_auroc()

Description:
Is a measure of the Area Under the Receiver Operating Curve (AUROC)

Basically a measure of how well the system is performing
More information at:

https://en.wikipedia.org/wiki/Receiver_operating_characteristic


@param true = true labels in validation set
@param predicted_score = predicted_score that an image is positive or negative

@retval AUROC score
"""


def metric_auroc(true, predicted_score):
    
    return roc_auc_score(true, predicted_score)



"""
metric_plot_roc()

Description:
Will just plot the ROC of the classifier and save it to the current directory

Code mostly taken from Sklearn example
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


@param true = true labels in validation set
@param predicted_score = predicted_score that an image is positive or negative

"""
def metric_plot_roc(true, predicted_score):
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, fnr = roc_curve(true, predicted_score)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('./roc.png')
    
    
    
    
    
    
