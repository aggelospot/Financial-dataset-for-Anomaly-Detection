import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import ModelWrapper
from models.utils import evaluate_bankruptcy_model

import os
import sys
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from tools import config

def run_bankruptcy_experiments(model, df, numeric_cols, n_runs=5, test_size=0.2, drop_na=True, smote=True):
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

        # model = model_factory(seed)

        train_df = df[df['bankruptcy_prediction_split'] == 'train']
        test_df = df[df['bankruptcy_prediction_split'] == 'test']

        if drop_na:
            train_df = train_df.dropna(subset=numeric_cols)
            test_df = test_df.dropna(subset=numeric_cols)

        X_train = train_df.drop(columns=['cik','label','bankruptcy_prediction_split'], errors='ignore')
        y_train = train_df['label']
        X_test = test_df.drop(columns=['cik', 'label','bankruptcy_prediction_split'], errors='ignore')
        y_test = test_df['label']

        # print("TRAIN Label Distribution:\n", y_train.value_counts())
        # print("TEST Label Distribution:\n", y_test.value_counts())

        # =========== Under-sampling + SMOTE (for imbalance) ===========
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

        model.fit(X_train_res.values, y_train_res.values)

        # =========== Evaluate on test ===========
        y_proba_test = model.predict_proba(X_test.values)[:, 1]

        y_pred_test = model.predict(X_test.values)
        metrics_dict = evaluate_bankruptcy_model(
            y_true=y_test,
            y_pred=y_pred_test,
            y_proba=y_proba_test
        )
        all_accuracies.append(metrics_dict['accuracy'])
        all_avgprec.append(metrics_dict['average_precision'])
        all_recall100.append(metrics_dict['recall_at_100'])
        all_rocauc.append(metrics_dict['roc_auc'])

        # =========== Evaluate on train ===========
        y_proba_train = model.predict_proba(X_train.values)[:, 1]

        y_pred_train = model.predict(X_train.values)
        metrics_dict_train = evaluate_bankruptcy_model(
            y_true=y_train,
            y_pred=y_pred_train,
            y_proba=y_proba_train
        )
        all_accuracies_train.append(metrics_dict_train['accuracy'])
        all_avgprec_train.append(metrics_dict_train['average_precision'])
        all_recall100_train.append(metrics_dict_train['recall_at_100'])
        all_rocauc_train.append(metrics_dict_train['roc_auc'])

    print(f"\n===== AVERAGE METRICS ACROSS SEEDS =====")
    model_name = model.__class__.__name__
    print(f"\n===== MODEL: {model_name} =====")
    print(f"\n===== TEST DATA =====")
    print("Accuracy ", all_accuracies)
    print("Mean Accuracy:           {:.2f}%, std: {:.2f}%".format(np.mean(all_accuracies) * 100, np.std(all_accuracies) * 100))
    print("Mean Average Precision:  {:.2f}%, std: {:.2f}%".format(np.mean(all_avgprec) * 100, np.std(all_avgprec) * 100))
    print("Mean Recall@100:         {:.2f}%, std: {:.2f}%".format(np.mean(all_recall100) * 100, np.std(all_recall100) * 100))
    print("Mean ROC AUC:            {:.2f}%, std: {:.2f}%".format(np.mean(all_rocauc) * 100, np.std(all_rocauc) * 100))

    print(f"\n===== TRAIN DATA =====")
    print("Mean Accuracy:           {:.2f}%, std: {:.2f}%".format(np.mean(all_accuracies_train) * 100, np.std(all_accuracies_train) * 100))
    print("Mean Average Precision:  {:.2f}%, std: {:.2f}%".format(np.mean(all_avgprec_train) * 100, np.std(all_avgprec_train) * 100))
    print("Mean Recall@100:         {:.2f}%, std: {:.2f}%".format(np.mean(all_recall100_train) * 100, np.std(all_recall100_train) * 100))
    print("Mean ROC AUC:            {:.2f}%, std: {:.2f}%".format(np.mean(all_rocauc_train) * 100, np.std(all_rocauc_train) * 100))


df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags.csv'))

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


# model=LogisticRegression(max_iter=10000, solver='liblinear', random_state=42)
# run_bankruptcy_experiments(model, df, numeric_cols, n_runs=5, smote=True)

# model=XGBClassifier(eval_metric='logloss', random_state=42)
# run_bankruptcy_experiments(model, df, numeric_cols, n_runs=5, drop_na=False, smote=False)

# model=AdaBoostClassifier(n_estimators=100, random_state=42)
# run_bankruptcy_experiments(model, df, numeric_cols, n_runs=5, smote=True)



# ------------------------------------------------------
# CatBoost
# catboost_model = CatBoostClassifier(
#     iterations=200,
#     learning_rate=0.05,
#     depth=6,
#     verbose=0,
#     random_state=42
# )
# run_bankruptcy_experiments(catboost_model, df, numeric_cols, n_runs=5)

# ------------------------------------------------------
# Stacking: Decision Tree + Random Forest → RidgeClassifier
stacking_model = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    passthrough=True,
    cv=5
)
run_bankruptcy_experiments(stacking_model, df, numeric_cols, n_runs=5)



# ------------------------------------------------------
# 1) Logistic Regression–style PyTorch model
# print("===Logistic regression===")
# torch_lr_model = ModelWrapper.GeneralizedTorchModel(
#     input_dim=len(numeric_cols),
#     architecture='logistic',
#     hidden_dim=0,      # not used in logistic
#     epochs=8000,
#     lr=1e-3,
#     # random_state=1
# )
# run_bankruptcy_experiments(torch_lr_model, df, numeric_cols, n_runs=5)

# 2) MLP–style PyTorch model
# print("===MLP===")
# torch_mlp_model = GeneralizedTorchModel(
#     input_dim=len(numeric_cols),
#     architecture='mlp',
#     hidden_dim=32,
#     epochs=20,
#     lr=1e-3,
#     random_state=42
# )
# run_bankruptcy_experiments(torch_mlp_model, df, numeric_cols, n_runs=5)

# You can easily add more architectures by expanding GeneralizedTorchModel.
