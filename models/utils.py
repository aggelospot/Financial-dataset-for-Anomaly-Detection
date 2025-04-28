from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, average_precision_score, \
    roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import numpy as np


def group_train_test_split(
    df, group_col='cik', label_col='label',
    test_size=0.2, random_state=42
):
    """
    Splits df so that entire groups (ciks) end up in train or test.
    Ensures at least one positive in each split if possible.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col]
    y = df[label_col]

    for train_idx, test_idx in gss.split(df, y, groups):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]
        # Check presence of positives in each split
        if train_df[label_col].sum() > 0 and test_df[label_col].sum() > 0:
            return train_df, test_df

    # Fallback if the above condition isn't satisfied
    return train_df, test_df


def evaluate_bankruptcy_model(y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_proba)
    top_indices = np.argsort(y_proba)[::-1][:100]
    recall_at_100 = (
        y_true.iloc[top_indices].sum() / y_true.sum()
        if y_true.sum() > 0 else 0.0
    )
    roc_auc = (
        roc_auc_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1 else 0.0
    )
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': cr,
        'average_precision': avg_precision,
        'recall_at_100': recall_at_100,
        'roc_auc': roc_auc
    }