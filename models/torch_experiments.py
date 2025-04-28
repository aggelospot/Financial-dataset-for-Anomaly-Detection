import os

import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from tools import config

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score
)
import numpy as np

def evaluate_bankruptcy_model(y_true, y_pred, y_proba):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, zero_division=0)  # suppress warning
    avg_precision = average_precision_score(y_true, y_proba)

    # Top 100 predicted probabilities
    top_indices = np.argsort(y_proba)[::-1][:100]
    recall_at_100 = (
        y_true[top_indices].sum() / y_true.sum()
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



class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze(1)



def run_bankruptcy_experiments_torch(df, numeric_cols, model_class=MLPClassifier,
                                      n_runs=5, test_size=0.2, drop_na=True,
                                      smote=True, epochs=20, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_accuracies = []
    all_avgprec = []
    all_recall100 = []
    all_rocauc = []
    all_accuracies_train = []
    all_avgprec_train = []
    all_recall100_train = []
    all_rocauc_train = []

    for seed in range(n_runs):
        print(f"\n===== RANDOM SEED: {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_df = df[df['bankruptcy_prediction_split'] == 'train']
        test_df = df[df['bankruptcy_prediction_split'] == 'test']

        if drop_na:
            train_df = train_df.dropna(subset=numeric_cols)
            test_df = test_df.dropna(subset=numeric_cols)

        X_train = train_df[numeric_cols]
        y_train = train_df['label']
        X_test = test_df[numeric_cols]
        y_test = test_df['label']

        if smote:
            rus = RandomUnderSampler(sampling_strategy=0.01, random_state=seed)
            X_rus, y_rus = rus.fit_resample(X_train, y_train)

            sm = SMOTE(sampling_strategy='minority', random_state=seed)
            X_train_res, y_train_res = sm.fit_resample(X_rus, y_rus)
        else:
            X_train_res, y_train_res = X_train, y_train

        y_train_res = y_train_res.astype(int)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Scale
        scaler = StandardScaler()
        X_train_res = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = scaler.transform(X_train)

        # Prepare tensors
        X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Model init
        model = model_class(input_dim=X_train_tensor.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Training loop
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        def get_preds(X):
            with torch.no_grad():
                return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

        # TEST SET
        y_proba_test = get_preds(X_test_scaled)
        y_pred_test = (y_proba_test >= 0.5).astype(int)

        metrics_dict = evaluate_bankruptcy_model(
            y_true=y_test.values,
            y_pred=y_pred_test,
            y_proba=y_proba_test
        )
        all_accuracies.append(metrics_dict['accuracy'])
        all_avgprec.append(metrics_dict['average_precision'])
        all_recall100.append(metrics_dict['recall_at_100'])
        all_rocauc.append(metrics_dict['roc_auc'])

        # TRAIN SET
        y_proba_train = get_preds(X_train_scaled)
        y_pred_train = (y_proba_train >= 0.5).astype(int)

        metrics_dict_train = evaluate_bankruptcy_model(
            y_true=y_train.values,
            y_pred=y_pred_train,
            y_proba=y_proba_train
        )
        all_accuracies_train.append(metrics_dict_train['accuracy'])
        all_avgprec_train.append(metrics_dict_train['average_precision'])
        all_recall100_train.append(metrics_dict_train['recall_at_100'])
        all_rocauc_train.append(metrics_dict_train['roc_auc'])

    # Final results
    print(f"\n===== MODEL: {model_class.__name__} (PyTorch) =====")
    print(f"\n===== TEST DATA =====")
    print("Mean Accuracy:           {:.2f}%, std: {:.4f}%".format(np.mean(all_accuracies) * 100, np.std(all_accuracies) * 100))
    print("Mean Average Precision:  {:.2f}%, std: {:.4f}%".format(np.mean(all_avgprec) * 100, np.std(all_avgprec) * 100))
    print("Mean Recall@100:         {:.2f}%, std: {:.4f}%".format(np.mean(all_recall100) * 100, np.std(all_recall100) * 100))
    print("Mean ROC AUC:            {:.2f}%, std: {:.4f}%".format(np.mean(all_rocauc) * 100, np.std(all_rocauc) * 100))

    print(f"\n===== TRAIN DATA =====")
    print("Mean Accuracy:           {:.2f}%, std: {:.4f}%".format(np.mean(all_accuracies_train) * 100, np.std(all_accuracies_train) * 100))
    print("Mean Average Precision:  {:.2f}%, std: {:.4f}%".format(np.mean(all_avgprec_train) * 100, np.std(all_avgprec_train) * 100))
    print("Mean Recall@100:         {:.2f}%, std: {:.4f}%".format(np.mean(all_recall100_train) * 100, np.std(all_recall100_train) * 100))
    print("Mean ROC AUC:            {:.2f}%, std: {:.4f}%".format(np.mean(all_rocauc_train) * 100, np.std(all_rocauc_train) * 100))


df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags.csv'))

# Drop irrelevant columns
columns_to_drop = [
    'company','period_of_report','gvkey','filing_date','year','accessionNumber','reportDateIndex',
    'datadate', 'filename', 'can_label', 'qualified', 'cik_year', 'gc_list',
    'bankruptcy_date_1','bankruptcy_date_2','bankruptcy_date_3','form','primaryDocument',
    'isXBRL','net_increase_decrease_in_cash'
]
df.drop(columns=columns_to_drop, axis=1, inplace=True)
print("cols remaining: ", df.columns)
numeric_cols = [
   'revenues','operating_expenses', 'operating_income', 'net_income',
   'earnings_per_share_basic', 'earnings_per_share_diluted',
   'total_current_assets', 'total_noncurrent_assets', 'total_assets',
   'total_current_liabilities', 'total_noncurrent_liabilities',
   'total_liabilities', 'stockholders_equity', 'total_liabilities_equity',
   'net_cash_from_operating_activities','net_cash_from_investing_activities',
   'net_cash_from_financing_activities', 'cash','other_comprehensive_income'
]
run_bankruptcy_experiments_torch(df, numeric_cols, model_class=MLPClassifier, n_runs=5, smote=False)
