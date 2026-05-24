from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from datasets import Dataset

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.cmp_dataset import CMPDataset


def get_data_filename(task, emb_model_name, max_tokens, max_sent):
    if task == CLF:
        return 'df_with_rl_gt_categories.csv'
    elif task == RGR:
        model_name_short = get_model_name_short(emb_model_name)
        mst = f'_ms{max_sent}' if max_sent else f'_mt{max_tokens}'
        return f'{model_name_short}_chunks_df{mst}.csv'


def get_col_names(task):
    if task == CLF:
        return 'text_translated', 'rile_label_num', 'gal_tan_label_num'
    elif task == RGR:
        return 'chunk_text_translated', 'chunk_rile_score', 'chunk_gal_tan_score'


def get_train_val_set(df, task, target, train_batch_size, random_seed):
    # X-TIME train / val set
    # TODO: X-COUNTRY

    train_df = df[df['year'] < 2019]
    text_col, rile_col, gal_tan_col = get_col_names(task)
    X_train = train_df[text_col].to_list()
    y_train = train_df[[rile_col, gal_tan_col]]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        shuffle=True, random_state=random_seed)

    print(f'Train: {len(X_train)}')
    train_dataset = CMPDataset(X_train, y_train, task=task, target=target)

    print(f'Val: {len(X_val)}')
    val_dataset = CMPDataset(X_val, y_val, task=task, target=target)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=train_batch_size,
                                shuffle=False)

    return train_dataloader, val_dataloader


def get_test_set(df, task, target, test_batch_size):
    # X-TIME test set
    # TODO: X-COUNTRY

    test_df = df[df['year'] >= 2019].reset_index(drop=True)
    text_col, rile_col, gal_tan_col = get_col_names(task)
    X_test = test_df[text_col].to_list()
    y_test = test_df[[rile_col, gal_tan_col]]

    print(f'Test: {len(X_test)}')
    test_dataset = CMPDataset(X_test, y_test, task=task, target=target)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False)
    return test_df, test_dataloader


def load_data(
        task,
        target, train_or_test, mount, test_batch_size,
        train_batch_size=None, random_seed=None,
        emb_model_name=None,
        max_tokens=None, max_sent=None
):
    # read data from disk
    print('Reading data...')
    data_filename = get_data_filename(task, emb_model_name, max_tokens, max_sent)
    data_path = os.path.join(mount, 'marpor_data', data_filename)
    # data_path = os.path.relpath(r'C:\Uni Stuttgart\thesis\joint_rile_galtan\marpor_data\df_with_rl_gt_categories.csv')
    df = pd.read_csv(data_path)

    # sanity check
    # df = df.sample(100 if task == CLF else 50, random_state=random_seed).reset_index(drop=True)

    # for clf, map string labels to numbers
    if task == CLF:
        df[f'rile_label_num'] = df[f'rile_label'].map(label_str2num)
        df[f'gal_tan_label_num'] = df[f'gal_tan_label'].map(label_str2num)

    # get datasets & dataloaders
    if train_or_test == 'train':
        return get_train_val_set(df, task, target, train_batch_size, random_seed)
    elif train_or_test == 'test':
        return get_test_set(df, task, target, test_batch_size)


def get_datasets_st(label_basis, random_seed):
    # read data from disk
    print('Reading data...')
    data_path = os.path.join(mount_no_backup, 'marpor_data', 'df_with_rl_gt_categories.csv')
    df = pd.read_csv(data_path)

    # sanity check
    # df = df.sample(100, random_state=random_seed).reset_index(drop=True)

    # generate contrastive labels
    if label_basis == 'all':
        df['st_label'] = df.apply(
            lambda row: f"{row['party']}-{row['rile_label']}-{row['gal_tan_label']}",
            axis=1
        )
    elif label_basis == 'rile_gal-tan':
        df['st_label'] = df.apply(
            lambda row: f"{row['rile_label']}-{row['gal_tan_label']}",
            axis=1
        )
    elif label_basis == 'party':
        df['st_label'] = df['party']
    print(f"Total classes: {df['st_label'].nunique()}")

    # map string labels to numbers
    label2num = {lb: i for i, lb in enumerate(list(df['st_label'].unique()))}
    df['st_label_num'] = df['st_label'].map(label2num)

    # X-TIME train / val / test split
    train_df = df[df['year'] < 2019][['text_translated', 'st_label_num']].rename(
        columns={'text_translated': 'sentence', 'st_label_num': 'label'}
    )
    test_df = df[df['year'] >= 2019][['text_translated', 'st_label_num']].rename(
        columns={'text_translated': 'sentence', 'st_label_num': 'label'}
    )

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    ds_dict = train_dataset.train_test_split(test_size=0.1, shuffle=True, seed=random_seed)
    train_dataset, val_dataset = ds_dict['train'], ds_dict['test']
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    print('Train:', train_dataset)
    print('Val:', val_dataset)
    print('Val:', test_dataset)

    return train_dataset, val_dataset, test_dataset
