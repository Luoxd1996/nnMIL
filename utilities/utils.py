import os
import random
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             log_loss, roc_auc_score)
from tqdm import tqdm
from scipy.special import softmax

# zeroshot prompt templates
TEMPLATES = [
    "CLASSNAME.",
    "an image of CLASSNAME.",
    "the image shows CLASSNAME.",
    "the image displays CLASSNAME.",
    "the image exhibits CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "I observe CLASSNAME.",
    "the pathology image shows CLASSNAME.",
    "a pathology image shows CLASSNAME.",
    "the pathology slide shows CLASSNAME.",
    "shows CLASSNAME.",
    "contains CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "CLASSNAME is observed.",
    "the pathology image reveals CLASSNAME.",
    "a microscopic image of showing CLASSNAME.",
    "histology shows CLASSNAME.",
    "CLASSNAME can be seen.",
    "the tissue shows CLASSNAME.",
    "CLASSNAME is identified.",
]

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    unique_classes: Optional[List[int]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.

    Args:
        targets_all (array-like): True target values.
        preds_all (array-like): Predicted target values.
        probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
        get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
        prefix (str, optional): Prefix to add to the result keys. Defaults to "".
        roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.

    Returns:
        dict: Dictionary containing the evaluation metrics.

    """
    unique_classes = unique_classes if unique_classes is not None else np.unique(targets_all)
    bacc = balanced_accuracy_score(targets_all, preds_all) if len(targets_all) > 1 else 0
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    nw_kappa = cohen_kappa_score(targets_all, preds_all, weights="linear")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0, labels=unique_classes)

    eval_metrics = {
        f"{prefix}/acc": acc,
        f"{prefix}/bacc": bacc,
        f"{prefix}/kappa": kappa,
        f"{prefix}/nw_kappa": nw_kappa,
        f"{prefix}/weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if probs_all is not None:
        if len(np.unique(targets_all)) > 1:
            # Convert to numpy array if needed
            probs_all = np.array(probs_all)
            targets_all = np.array(targets_all)
            
            # Ensure probs_all is 2D: (n_samples, n_classes)
            if probs_all.ndim == 1:
                # Binary classification: expand to (n_samples, 2)
                probs_all = np.column_stack([1 - probs_all, probs_all])
            
            # Check if probs_all has correct number of classes
            n_classes_probs = probs_all.shape[1]
            n_classes_labels = len(unique_classes)
            
            if n_classes_probs != n_classes_labels:
                # If dimensions don't match, use the actual number of classes in probs
                # This handles cases where model output doesn't match label classes
                if n_classes_probs >= n_classes_labels:
                    # Use probs as-is, but ensure labels match
                    labels_for_loss = list(range(n_classes_probs))
                else:
                    # Pad probs if needed (shouldn't happen, but handle gracefully)
                    raise ValueError(
                        f"Probability shape mismatch: probs has {n_classes_probs} classes, "
                        f"but labels have {n_classes_labels} classes. "
                        f"Probs shape: {probs_all.shape}, Unique labels: {unique_classes}"
                    )
            else:
                labels_for_loss = unique_classes
            
            try:
                loss = log_loss(targets_all, probs_all, labels=labels_for_loss)
                eval_metrics[f"{prefix}/loss"] = loss
                
                # Calculate ROC AUC - handle binary and multi-class
                # Binary: probs_all has 2 columns AND unique_classes has exactly 2 classes
                is_binary = (probs_all.shape[1] == 2) and (len(unique_classes) == 2)
                
                if is_binary:
                    # Binary classification: use probabilities of positive class (class 1)
                    # Need at least 2 classes present in the data to calculate AUC
                    unique_in_data = np.unique(targets_all)
                    if len(unique_in_data) == 2:
                        roc_auc = roc_auc_score(targets_all, probs_all[:, 1])
                    else:
                        # Only one class present in data, cannot calculate AUC
                        roc_auc = np.nan
                else:
                    # Multi-class: use one-vs-rest (ovr) with macro average
                    # Need at least 2 classes present in the data
                    unique_in_data = np.unique(targets_all)
                    if len(unique_in_data) > 1:
                        roc_auc = roc_auc_score(
                            targets_all, 
                            probs_all, 
                            labels=unique_classes,
                            multi_class='ovr',
                            average='macro',
                            **roc_kwargs
                        )
                    else:
                        # Only one class present in data, cannot calculate AUC
                        roc_auc = np.nan
                eval_metrics[f"{prefix}/auroc"] = roc_auc
            except Exception as e:
                # If calculation fails, skip these metrics but continue
                import warnings
                warnings.warn(f"Could not calculate loss/roc_auc: {e}. Skipping these metrics.")
                pass

    return eval_metrics


def seed_torch(device, seed=0):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def merge_dict(dict1, dict2):
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k].append(dict2[k])
        else:
            dict1[k] = [dict2[k]]
    return dict1

    
def bootstrap(results_dict=None, preds_all=None, targets_all=None, probs_all=None, n=1000, alpha=0.95, format_as_str=False):
    if results_dict is not None:
        targets_all = results_dict['targets']
        probs_key = 'logits' if 'logits' in results_dict.keys() else 'probs'
        if probs_key == 'logits':
            results_dict[probs_key] = softmax(results_dict[probs_key], axis=1)
        probs_all = results_dict[probs_key] if probs_key in results_dict.keys() else None
        preds_all = results_dict['preds'] if 'preds' in results_dict.keys() else None
        if probs_all is None:
            assert 'preds' in results_dict.keys()
            preds_all = results_dict['preds']
        if preds_all is None:
            preds_all = np.argmax(probs_all, axis=1)
    
    num_classes = len(np.unique(targets_all))
    if probs_all is not None and len(probs_all.shape) == 2:
        probs_all = probs_all[:, 1] if num_classes == 2 else probs_all
    roc_kwargs = {'average': 'macro', 'multi_class': 'ovo'} if num_classes > 2 else {}
    overall_scores = get_eval_metrics(probs_all=probs_all, preds_all=preds_all, targets_all=targets_all, roc_kwargs=roc_kwargs)

    all_scores = {}
    for seed in tqdm(range(n)):
        np.random.seed(seed)
        bootstrap_ind = list(pd.Series(targets_all).sample(n=len(targets_all), replace=True, random_state=seed).index)
        collision = 0
        while len(np.unique(targets_all[bootstrap_ind])) != num_classes:
            bootstrap_ind = list(pd.Series(targets_all).sample(n=len(targets_all), replace=True, random_state=seed+collision+n).index)
            collision += 1
            if collision % 100 == 0:
                print(collision)
        sample_targets_all = targets_all[bootstrap_ind]
        sample_preds_all = preds_all[bootstrap_ind] if preds_all is not None else None
        sample_probs_all = probs_all[bootstrap_ind] if probs_all is not None else None
        results = get_eval_metrics(probs_all=sample_probs_all, preds_all=sample_preds_all, targets_all=sample_targets_all, roc_kwargs=roc_kwargs)
        merge_dict(all_scores, results)
    
    ci_dict = {}
    ci_as_str_dict = {}
    mean_dict = {}
    std_dict = {}
    for k in all_scores.keys():
        scores = np.array(all_scores[k])
        mean_dict[k] = scores.mean()
        std_dict[k] = scores.std()
        
    return mean_dict, std_dict


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """Copied from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py
    """
    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length
    
    def _assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = new_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = new_lr
    
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster