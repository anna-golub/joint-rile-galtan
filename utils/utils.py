import pandas as pd
import numpy as np
import random
import os
import torch

from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torchmetrics.wrappers import MultioutputWrapper

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.macleginn_utils import compute_rile_simple, compute_gal_tan_simple

# TODO: make parameter
# mount_w_backup = r'C:\Uni Stuttgart\thesis\joint_rile_galtan'
# mount_no_backup = r'C:\Uni Stuttgart\thesis\joint_rile_galtan'

mount_w_backup = r'/projekte/tcl/tclext/golubaa'
mount_no_backup = r'/mount/arbeitsdaten/tcl/tclext/golubaa/thesis'

# task
CLF = 'clf'  # classification
CONTR = 'contr'  # contrastive pre-training
RGR = 'rgr'  # regression

# target
RILE = 'rile'
GALTAN = 'gal_tan'
JOINT = 'joint'

label_str2num = {
    'right': 0,
    'left': 1,
    'authoritarian': 0,
    'libertarian': 1,
    'neutral': 2
}


def label_num2str(label, target):
    if target == RILE:
        mapping = {
            0: 'right',
            1: 'left',
            2: 'neutral'
        }
    elif target == GALTAN:
        mapping = {
            0: 'authoritarian',
            1: 'libertarian',
            2: 'neutral'
        }
    return mapping[label]


def unpack_tokenize(batch, tokenizer, device, max_length=None):
    # if static_emb:
    #     inputs = {'idx': torch.tensor(batch['idx']).to(device),
    #               'train_val_test': train_val_test}
    #     print(f'{inputs=}')
    # else:
    # input_ids = batch['input_ids'].to(device)
    # attention_mask = batch['attention_mask'].to(device)

    if max_length:  # pad to max_length
        tokenized = tokenizer(
            batch['texts'],
            max_length=max_length, padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    else:  # pad to match longest sequence in batch
        tokenized = tokenizer(
            batch['texts'],
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

    # inputs = {'input_ids': tokenized['input_ids'].to(device),
    #               'attention_mask': tokenized['attention_mask'].to(device)}

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    del tokenized
    return input_ids, attention_mask


def get_preds(logits, target):  # for label aggr.
    if target in (RILE, GALTAN):
        dim = 1
    elif target == JOINT:
        dim = 2
    pred_probs = torch.softmax(logits, dim=dim)
    preds = torch.argmax(pred_probs, dim=dim)
    return preds


def get_metric_trackers(target, device):  # for label aggr.
    if target in (RILE, GALTAN):
        train_acc = MulticlassAccuracy(num_classes=3).to(device)
        train_weighted_f1 = MulticlassF1Score(num_classes=3, average='weighted').to(device)
        train_macro_f1 = MulticlassF1Score(num_classes=3, average='macro').to(device)
        val_acc = MulticlassAccuracy(num_classes=3).to(device)
        val_weighted_f1 = MulticlassF1Score(num_classes=3, average='weighted').to(device)
        val_macro_f1 = MulticlassF1Score(num_classes=3, average='macro').to(device)
    elif target == JOINT:
        train_acc = MultioutputWrapper(
            MulticlassAccuracy(num_classes=3), 2).to(device)
        train_weighted_f1 = MultioutputWrapper(
            MulticlassF1Score(num_classes=3, average='weighted'), 2).to(device)
        train_macro_f1 = MultioutputWrapper(
            MulticlassF1Score(num_classes=3, average='macro'), 2).to(device)
        val_acc = MultioutputWrapper(
            MulticlassAccuracy(num_classes=3), 2).to(device)
        val_weighted_f1 = MultioutputWrapper(
            MulticlassF1Score(num_classes=3, average='weighted'), 2).to(device)
        val_macro_f1 = MultioutputWrapper(
            MulticlassF1Score(num_classes=3, average='macro'), 2).to(device)
    return train_acc, train_weighted_f1, train_macro_f1, \
        val_acc, val_weighted_f1, val_macro_f1


def compute_spearman_rho(test_df, task, target):
    if task == CLF:
        if target == RILE:
            compute_simple = compute_rile_simple
        elif target == GALTAN:
            compute_simple = compute_gal_tan_simple

        df_scores_true = test_df.groupby('manifesto_id')[f'{target}_label'].apply(
            compute_simple).reset_index(name='scores_true')
        df_scores_pred = test_df.groupby('manifesto_id')[f'{target}_pred'].apply(
            compute_simple).reset_index(name='scores_pred')

    elif task == RGR:
        # print(f'{test_df.columns=}')
        df_scores_true = test_df.groupby('manifesto_id')[
            f'chunk_{target}_score'].mean().reset_index(name='scores_true')
        df_scores_pred = test_df.groupby('manifesto_id')[
            f'chunk_{target}_pred'].mean().reset_index(name='scores_pred')

    df_scores = pd.merge(
        df_scores_true,
        df_scores_pred,
        on='manifesto_id',
        how='outer')

    rho = df_scores[f'scores_true'].corr(df_scores[f'scores_pred'], method='spearman')
    return rho


def set_random_seed(random_seed=7):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


def get_model_name_short(emb_model_name):
    mapping = {
        'answerdotai/ModernBERT-base': 'ModernBERT',
        'sentence-transformers/all-mpnet-base-v2': 'all-mpnet-base-v2',
        'all-mpnet-base-v2': 'all-mpnet-base-v2',
        'google/bigbird-roberta-base': 'BigBird',
        'allenai/Olmo-3-7B-Think': 'Olmo-3-7B'
    }
    return mapping[emb_model_name]


def get_model_name_full(emb_model_name):
    mapping = {
        'ModernBERT': 'answerdotai/ModernBERT-base',
        'sbert': 'sentence-transformers/all-mpnet-base-v2',
        'BigBird': 'google/bigbird-roberta-base',
        'Olmo-3-7B': 'allenai/Olmo-3-7B-Think',
    }
    return mapping[emb_model_name]


def get_model_path(task, target, random_seed,
                   emb_model_name, freeze_enc_weights,
                   lr, train_batch_size, with_lr_schedule=None,
                   contr_pretr_id=None, whiten=None, dropout_p=None,
                   max_sent=None
                   ):
    model_name_short = get_model_name_short(emb_model_name)
    models_dir = os.path.join(mount_no_backup, 'models',
                              task, target, model_name_short, f'seed_{random_seed}')
    os.makedirs(models_dir, exist_ok=True)

    lr_str = f'lr{lr}'
    if with_lr_schedule:
        lr_str += '_sched'

    freeze_str = '_freeze' if freeze_enc_weights else ''
    cpid_str = f'_cpid{contr_pretr_id}' if contr_pretr_id else ''
    whiten_str = '_whiten' if whiten else ''
    dropout_str = f'_dropout{dropout_p}' if dropout_p else ''
    ms_str = f'_ms{max_sent}' if max_sent else ''

    model_filename = f'state_dict_{model_name_short}{freeze_str}{cpid_str}{whiten_str}{ms_str}_{lr_str}_bs{train_batch_size}{dropout_str}_' + 'ep{}.pth'
    model_path = os.path.join(models_dir, model_filename)

    return model_path


def get_contr_pretr_path(emb_model_name, contr_pretr_id, random_seed):
    model_dir = f'{emb_model_name}_cpid{contr_pretr_id}_seed{random_seed}'
    model_path = os.path.join(mount_no_backup, 'models', 'st', model_dir)
    return model_path


def get_train_stats_path(task, target, random_seed,
                         emb_model_name, freeze_enc_weights,
                         lr, train_batch_size, n_epochs, with_lr_schedule=None,
                         contr_pretr_id=None, whiten=None, dropout_p=None,
                         max_sent=None
                         ):
    model_name_short = get_model_name_short(emb_model_name)
    stats_dir = os.path.join(mount_w_backup, 'train_stats',
                             task, target, model_name_short, f'seed_{random_seed}')
    os.makedirs(stats_dir, exist_ok=True)

    lr_str = f'lr{lr}'
    if with_lr_schedule:
        lr_str += '_sched'

    freeze_str = '_freeze' if freeze_enc_weights else ''
    cpid_str = f'_cpid{contr_pretr_id}' if contr_pretr_id else ''
    whiten_str = '_whiten' if whiten else ''
    dropout_str = f'_dropout{dropout_p}' if dropout_p else ''
    ms_str = f'_ms{max_sent}' if max_sent else ''

    train_stats_path = os.path.join(
        stats_dir,
        f'{model_name_short}{freeze_str}{cpid_str}{whiten_str}{ms_str}_{lr_str}_bs{train_batch_size}{dropout_str}_ep{n_epochs}.csv')
    return train_stats_path


def get_preds_path(task, target, random_seed,
                   emb_model_name, freeze_enc_weights,
                   lr, train_batch_size, ep, with_lr_schedule=None,
                   contr_pretr_id=None, whiten=None, dropout_p=None,
                   max_sent=None
                   ):
    model_name_short = get_model_name_short(emb_model_name)
    preds_dir = os.path.join(mount_w_backup, 'preds', task, target, f'seed_{random_seed}')
    os.makedirs(preds_dir, exist_ok=True)

    lr_str = f'lr{lr}'
    if with_lr_schedule:
        lr_str += '_sched'

    freeze_str = '_freeze' if freeze_enc_weights else ''
    cpid_str = f'_cpid{contr_pretr_id}' if contr_pretr_id else ''
    whiten_str = '_whiten' if whiten else ''
    dropout_str = f'_dropout{dropout_p}' if dropout_p else ''
    ms_str = f'_ms{max_sent}' if max_sent else ''

    filename = f'pred_test_{model_name_short}{freeze_str}{cpid_str}{whiten_str}{ms_str}_{lr_str}_bs{train_batch_size}{dropout_str}_ep{ep}.csv'
    preds_path = os.path.join(preds_dir, filename)
    return preds_path


def print_ep_stats(train_stats: dict, ep, target):
    if target == JOINT:
        # one value for both targets
        print(f"Train loss = {train_stats[ep]['train_loss']:.3f}")
        print(f"Val loss = {train_stats[ep]['val_loss']:.3f}")
        print()

        # target-specific values
        for i, t in enumerate([RILE, GALTAN]):
            print(t.upper())
            # print(f"Val acc = {train_stats[ep]['val_acc'][i]:.3f}")
            # print(f"Val weighted f1 = {train_stats[ep]['val_weighted_f1'][i]:.3f}")
            # print(f"Val macro f1 = {train_stats[ep]['val_macro_f1'][i]:.3f}")
            for metric in train_stats[ep]:
                if 'loss' in metric:
                    continue
                metric_str = metric.capitalize().replace('_', ' ')
                print(f"{metric_str} = {train_stats[ep][metric][i]:.3f}")
            print()
    else:
        # print(f"Val acc = {train_stats[ep]['val_acc']:.3f}")
        # print(f"Val weighted f1 = {train_stats[ep]['val_weighted_f1']:.3f}")
        # print(f"Val macro f1 = {train_stats[ep]['val_macro_f1']:.3f}")
        for metric in train_stats[ep]:
            metric_str = metric.capitalize().replace('_', ' ')
            print(f"{metric_str} = {train_stats[ep][metric]:.3f}")
        print()


def postprocess_joint_train_stats(train_stats: pd.DataFrame):
    orig_cols = [col for col in list(train_stats.columns) if 'loss' not in col]
    for i, t in enumerate([RILE, GALTAN]):
        # train_stats[f'train_acc_{t}'] = train_stats['train_acc'].apply(lambda x: x[i])
        # train_stats[f'train_weighted_f1_{t}'] = train_stats['train_weighted_f1'].apply(lambda x: x[i])
        # train_stats[f'train_macro_f1_{t}'] = train_stats['train_macro_f1'].apply(lambda x: x[i])
        # train_stats[f'val_acc_{t}'] = train_stats['val_acc'].apply(lambda x: x[i])
        # train_stats[f'val_weighted_f1_{t}'] = train_stats['val_weighted_f1'].apply(lambda x: x[i])
        # train_stats[f'val_macro_f1_{t}'] = train_stats['val_macro_f1'].apply(lambda x: x[i])

        for col in orig_cols:
            train_stats[f'{col}_{t}'] = train_stats[col].apply(lambda x: x[i])

    # train_stats = train_stats.drop([
    #     'train_acc', 'train_weighted_f1', 'train_macro_f1',
    #     'val_acc', 'val_weighted_f1', 'val_macro_f1'
    # ], axis=1)
    train_stats = train_stats.drop(orig_cols, axis=1)

    # train_stats = train_stats[[
    #     'train_loss',
    #     'train_acc_rile', 'train_weighted_f1_rile', 'train_macro_f1_rile',
    #     'train_acc_gal_tan', 'train_weighted_f1_gal_tan', 'train_macro_f1_gal_tan',
    #     'val_loss',
    #     'val_acc_rile', 'val_weighted_f1_rile', 'val_macro_f1_rile',
    #     'val_acc_gal_tan', 'val_weighted_f1_gal_tan', 'val_macro_f1_gal_tan',
    # ]]

    train_stats = train_stats[
        ['train_loss'] + \
        [f'{col}_rile' for col in orig_cols if 'train' in col] + \
        [f'{col}_gal_tan' for col in orig_cols if 'train' in col] + \
        ['val_loss'] + \
        [f'{col}_rile' for col in orig_cols if 'val' in col] + \
        [f'{col}_gal_tan' for col in orig_cols if 'val' in col]
        ]

    return train_stats


# https://forum.pyro.ai/t/a-trick-to-debug-tensor-memory/556
def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape),)
        for o in gc.get_objects()
        if torch.is_tensor(o)
        # and str(o.device) == 'cuda:0'
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

# import gc
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
# import time
#
#
# def clear_gpu_memory():
#     torch.cuda.empty_cache()
#     gc.collect()
#     # del variables
#
#
# def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
#
#     for _ in range(max_retries):
#         info = nvmlDeviceGetMemoryInfo(handle)
#         if info.free >= min_memory_available:
#             print(f'{info.free=}')
#             break
#         print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
#         time.sleep(sleep_time)
#     else:
#         raise RuntimeError(
#             f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")
