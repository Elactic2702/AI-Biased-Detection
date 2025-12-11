# src/evaluate.py (snippet)
import pandas as pd
from sklearn.metrics import confusion_matrix

def group_tpr(df, group_col='gender'):
    res = {}
    for g, sub in df.groupby(group_col):
        tn, fp, fn, tp = confusion_matrix(sub['label'], sub['pred'], labels=[0,1]).ravel()
        res[g] = tp / (tp+fn) if (tp+fn)>0 else None
    return res
