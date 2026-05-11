
'''
splitting.py — Stratified 5-fold CV
'''

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(y, df=None, test_size=0.15, val_size=0.15, random_state=42):
    idx = np.arange(len(y))
    idx_train_val, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []
    y_train_val = y[idx_train_val]
    for train_idx, val_idx in skf.split(idx_train_val, y_train_val):
        orig_train = idx_train_val[train_idx]
        orig_val = idx_train_val[val_idx]
        splits.append((orig_train, orig_val, idx_test))
    return splits
