import pandas as pd
import os

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import RILE, GALTAN, CLF, compute_spearman_rho
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    task = CLF
    target = GALTAN  # RILE / GALTAN
    print('Target:', target)

    # data_path = os.path.join(mount, 'marpor_data', 'df_with_rl_gt_categories.csv')
    data_path = os.path.relpath(r'C:\Uni Stuttgart\thesis\joint_rile_galtan\marpor_data\df_with_rl_gt_categories.csv')
    # print(data_path)
    df = pd.read_csv(data_path)

    train_df = df[df['year'] < 2019]
    test_df = df[df['year'] >= 2019].reset_index(drop=True)

    maj_label = train_df[f'{target}_label'].mode()[0]
    print(f'Train set majority label: {maj_label}')

    test_df[f'{target}_pred'] = maj_label

    acc = accuracy_score(test_df[f'{target}_label'], test_df[f'{target}_pred'])
    macro_f1 = f1_score(
        test_df[f'{target}_label'], test_df[f'{target}_pred'],
        average='macro')
    weighted_f1 = f1_score(
        test_df[f'{target}_label'], test_df[f'{target}_pred'],
        average='weighted')
    rho = compute_spearman_rho(test_df, task, target)

    print(f"Accuracy: {acc:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")
    print(f"Spearman's rho: {rho:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
