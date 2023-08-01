# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:01:29 2021

@author: lauracue
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch

from typing import Union

def evaluate_metrics(pred:Union[np.ndarray, torch.Tensor], gt:Union[np.ndarray, torch.Tensor], val = 0) -> dict:
    """Calculte the metrics:
    Accuracy, F1 Score, Precision, and Recall.
    Calculate based on the pred with highest probability and the gt - ground truth segmentation.

    Parameters
    ----------
    pred : Union[np.ndarray, torch.Tensor]
        Tensor with shape [row, cols, num_class]
        This tensor has one matrix of confidence for each label class.
        The function uses the class with highest probability/confidence to evaluate metric
    gt : Union[np.ndarray, torch.Tensor]
        Tensor with shape [row, cols]
        This tensor has the ground truth segmentation matrix
    val : int, optional
        , by default 0

    Returns
    -------
    dict
        Return the metrics:
        - Accuracy, as Accuracy
        - F1 Score, as avgF1
        - Precision, as avgPre
        - Recall, as avgRec
    """

    accu_criteria = dict()

    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()

    c = pred.shape[1]

    # Get the class with highest probability
    pred = np.argmax(pred,axis=1)

    # Create to just the place where the ground_truth_segmentation is non zero
    mask = np.where(gt>0)

    # Apply Mask
    gt = gt[mask][:]

    pred = pred[mask][:]+1

    #### CALCULATE METRICS WITH SKLEARN ####
    accuracy = accuracy_score(gt, pred)*100
    accu_criteria["Accuracy"] = np.round(accuracy,2)
    
    f1 = f1_score(gt, pred, average=None, zero_division=True)

    pre = precision_score(gt, pred, average=None, zero_division=True)

    rec = recall_score(gt, pred, average=None, zero_division=True)


    accu_criteria["avgF1"] = np.round(np.sum(f1)*100/c,2)
    accu_criteria["avgPre"] = np.round(np.sum(pre)*100/c,2)
    accu_criteria["avgRec"] = np.round(np.sum(rec)*100/c,2)
    # accu_criteria["F1"] = np.round(np.array(f1)*100,2)
    # accu_criteria["Pre"] = np.round(np.array(pre)*100,2)
    # accu_criteria["Rec"] = np.round(np.array(rec)*100,2)

    return accu_criteria


