# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from tqdm import tqdm
import time

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.prep_dataset import get_datasets_st


def compute_kernel_bias(vecs, k=None):
    """kernel bias
    y = (x + bias).dot(kernel)
    Code taken from: https://github.com/bojone/BERT-whitening
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))

    if k:
        return W[:, :k], -mu
    else:
        return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """
    Code taken from: https://github.com/bojone/BERT-whitening
    """

    if not (kernel is None or bias is None):
        # vecs = (vecs + bias).dot(kernel)          # numpy
        vecs = torch.matmul(vecs + bias, kernel)    # torch

    # output = vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5     # numpy
    output = vecs / (vecs ** 2).sum(dim=1, keepdim=True) ** 0.5         # torch

    return output


def encode_sentences(dataloader):
    emb = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            batch_emb = model.encode(batch['sentence'])
        emb.append(batch_emb)
    emb = np.concatenate(emb, axis=0)
    return emb


def get_average_emb_dist(emb_matrix):
    sample_idx = np.random.choice(emb_matrix.shape[0], int(0.05 * emb_matrix.shape[0]))
    sample_emb = emb_matrix[sample_idx]

    euclidean_dists = euclidean_distances(sample_emb, emb_matrix)
    cosine_dists = cosine_similarity(sample_emb, emb_matrix)

    del sample_idx, sample_emb

    return np.mean(euclidean_dists), np.mean(cosine_dists)


if __name__ == "__main__":
    emb_model_name = 'all-mpnet-base-v2'
    contr_pretr_id = 5
    label_basis = 'party'
    random_seed = 7
    train_batch_size = 16

    print(f'contrastive pre-training ID = {contr_pretr_id}')
    print(f'label basis = {label_basis}')
    print(f'random seed = {random_seed}')
    print(f'train batch size = {train_batch_size}')
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')
    set_random_seed(random_seed=random_seed)
    model_name_short = get_model_name_short(emb_model_name)

    # load model
    print('Loading model...')
    checkpoint = get_contr_pretr_path(emb_model_name, contr_pretr_id)
    model = SentenceTransformer(checkpoint, device=device)

    # load data
    train_dataset, val_dataset, test_dataset = get_datasets_st(
        label_basis=label_basis,
        random_seed=random_seed
    )
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

    # save datasets to csv
    # ds_dir = f'{mount_no_backup}/hf_ds/seed{random_seed}'
    # train_dataset.to_csv(f'{ds_dir}/train.csv', index=False)
    # val_dataset.to_csv(f'{ds_dir}/val.csv', index=False)
    # test_dataset.to_csv(f'{ds_dir}/test.csv', index=False)

    # embed all sentences
    print('Encode sentences...')
    print('Train set')
    train_emb = encode_sentences(train_dataloader)
    print('\nVal set')
    val_emb = encode_sentences(val_dataloader)
    print('\nTest set')
    test_emb = encode_sentences(test_dataloader)
    print()

    # calculate average emb dist
    print(f'Before whitening')
    aver_euclid, aver_cos_dist = get_average_emb_dist(train_emb)
    print(f'Train set average euclidean dist = {aver_euclid:.3f}')
    print(f'Train set average cosine dist = {aver_cos_dist:.3f}')
    print()

    # apply whitening transformation -- adapted from Ceron et al. (2022)
    print('Whitening...')
    # k=None - no dimensionality reduction
    train_kernel, train_bias = compute_kernel_bias(train_emb, None)
    train_emb = transform_and_normalize(train_emb, train_kernel, train_bias)

    # use train kernel and bias for val and test emb
    val_emb = transform_and_normalize(val_emb, train_kernel, train_bias)
    test_emb = transform_and_normalize(test_emb, train_kernel, train_bias)
    print()

    # save embeddings
    print('Save embeddings to disk...')
    emb_dir = f'{mount_no_backup}/emb/whiten/{model_name_short}_cpid{contr_pretr_id}'
    os.makedirs(emb_dir, exist_ok=True)

    train_emb = pd.DataFrame(train_emb)
    train_emb.to_csv(f'{emb_dir}/train_emb.csv', index=False)

    val_emb = pd.DataFrame(val_emb)
    val_emb.to_csv(f'{emb_dir}/val_emb.csv', index=False)

    test_emb = pd.DataFrame(test_emb)
    test_emb.to_csv(f'{emb_dir}/test_emb.csv', index=False)

    # read emb
    emb_dir = f'{mount_no_backup}/emb/whiten/{model_name_short}_cpid{contr_pretr_id}'
    train_emb = pd.read_csv(f'{emb_dir}/train_emb.csv').to_numpy()
    print(train_emb.shape)

    # calculate average emb dist
    print(f'After whitening')
    aver_euclid, aver_cos_dist = get_average_emb_dist(train_emb)
    print(f'Train set average euclidean dist = {aver_euclid:.3f}')
    print(f'Train set average cosine dist = {aver_cos_dist:.3f}')
    print()
