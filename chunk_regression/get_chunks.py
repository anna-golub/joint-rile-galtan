from transformers import AutoTokenizer
from tqdm import tqdm

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *


def get_chunks(mid_df):
    if max_sent:
        return get_chunks_max_sent(mid_df)
    return get_chunks_max_tokens(mid_df)


def get_chunks_max_sent(mid_df):
    # current chunk data
    n_sent, n_tokens = 0, 0
    chunk_sent, chunk_rile_labels, chunk_gal_tan_labels = [], [], []

    # all chunks
    chunks, rile_scores, gal_tan_scores, \
        chunk_n_sent, chunk_n_tokens = [], [], [], [], []

    for ind, row in mid_df.iterrows():
        sent = row['text_translated']
        tokens = tokenizer(sent)

        # reached max length => record chunk, clear buffer
        if n_sent + 1 > max_sent or n_tokens + len(tokens['input_ids']) > max_tokens:
            chunks.append(' '.join(chunk_sent))
            chunk_rile = compute_rile_simple(chunk_rile_labels)
            rile_scores.append(chunk_rile)
            chunk_gal_tan = compute_gal_tan_simple(chunk_gal_tan_labels)
            gal_tan_scores.append(chunk_gal_tan)
            chunk_n_sent.append(n_sent)
            chunk_n_tokens.append(n_tokens)

            n_sent, n_tokens = 0, 0
            chunk_sent, chunk_rile_labels, chunk_gal_tan_labels = [], [], []

        # add current chunk to buffer
        n_sent += 1
        chunk_sent.append(sent)
        chunk_rile_labels.append(row['rile_label'])
        chunk_gal_tan_labels.append(row['gal_tan_label'])
        n_tokens += len(tokens['input_ids'])

    # record incomplete last chunk if test set or chunk long enough
    year = mid_df['year'].to_list()[0]
    # print(f'{year=}')
    if n_sent >= max_sent // 4:  # or year >= 2019:
        chunks.append(' '.join(chunk_sent))
        chunk_rile = compute_rile_simple(chunk_rile_labels)
        rile_scores.append(chunk_rile)
        chunk_gal_tan = compute_gal_tan_simple(chunk_gal_tan_labels)
        gal_tan_scores.append(chunk_gal_tan)
        chunk_n_sent.append(n_sent)
        chunk_n_tokens.append(n_tokens)

    # create dataframe
    mid_chunks_df = pd.DataFrame()
    mid_chunks_df['chunk_text_translated'] = chunks
    mid_chunks_df['chunk_rile_score'] = rile_scores
    mid_chunks_df['chunk_gal_tan_score'] = gal_tan_scores
    mid_chunks_df['chunk_n_sent'] = chunk_n_sent
    mid_chunks_df['chunk_n_tokens'] = chunk_n_tokens

    return mid_chunks_df


def get_chunks_max_tokens(mid_df):
    # current chunk data
    n_tokens = 0
    chunk_sent, chunk_rile_labels, chunk_gal_tan_labels = [], [], []

    # all chunks
    chunks, rile_scores, gal_tan_scores, \
        chunk_n_sent, chunk_n_tokens = [], [], [], [], []

    for ind, row in mid_df.iterrows():
        sent = row['text_translated']
        tokens = tokenizer(sent)

        # reached max length => record chunk, clear buffer
        if n_tokens + len(tokens['input_ids']) > max_tokens:
            # print(f'{n_tokens=} {len(chunk_sent)=}')
            chunks.append(' '.join(chunk_sent))
            chunk_rile = compute_rile_simple(chunk_rile_labels)
            rile_scores.append(chunk_rile)
            chunk_gal_tan = compute_gal_tan_simple(chunk_gal_tan_labels)
            gal_tan_scores.append(chunk_gal_tan)
            chunk_n_sent.append(len(chunk_sent))
            chunk_n_tokens.append(n_tokens)

            n_tokens = 0
            chunk_sent, chunk_rile_labels, chunk_gal_tan_labels = [], [], []

        # add current chunk to buffer
        n_tokens += len(tokens['input_ids'])
        chunk_sent.append(sent)
        chunk_rile_labels.append(row['rile_label'])
        chunk_gal_tan_labels.append(row['gal_tan_label'])

    # record incomplete last chunk if length >= 1000 (logic from Nikolaev et al.) or tset set
    year = mid_df['year'].to_list()[0]
    # print(f'{year=}')
    if n_tokens >= 1000:  # or year >= 2019:
        chunks.append(' '.join(chunk_sent))
        chunk_rile = compute_rile_simple(chunk_rile_labels)
        rile_scores.append(chunk_rile)
        chunk_gal_tan = compute_gal_tan_simple(chunk_gal_tan_labels)
        gal_tan_scores.append(chunk_gal_tan)
        chunk_n_sent.append(len(chunk_sent))
        chunk_n_tokens.append(n_tokens)

    # create dataframe
    mid_chunks_df = pd.DataFrame()
    mid_chunks_df['chunk_text_translated'] = chunks
    mid_chunks_df['chunk_rile_score'] = rile_scores
    mid_chunks_df['chunk_gal_tan_score'] = gal_tan_scores
    mid_chunks_df['chunk_n_sent'] = chunk_n_sent
    mid_chunks_df['chunk_n_tokens'] = chunk_n_tokens

    return mid_chunks_df


if __name__ == "__main__":
    emb_model_name = 'answerdotai/ModernBERT-base'
    # emb_model_name = 'google/bigbird-roberta-base'

    model_name_short = get_model_name_short(emb_model_name)

    print(f'{emb_model_name=}')

    # context window: BigBird - 4096, ModernBERT - 8192
    max_tokens = 4096 if 'bigbird' in emb_model_name else 8192
    print(f'{max_tokens=}')

    # maximum number of sentences
    # set to None to generate chunks with the max number tokens each
    max_sent = None
    print(f'{max_sent=}')
    print()

    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)

    print('Reading data...')
    data_path = os.path.join(mount_no_backup, 'marpor_data', 'df_with_rl_gt_categories.csv')
    df = pd.read_csv(data_path)

    print('Split each manifesto into chunks and compute chunk ground truth scores...')
    chunks_df = []
    for i, mid in tqdm(enumerate(df['manifesto_id'].unique()), total=df['manifesto_id'].nunique()):
        mid_chunks_df = get_chunks(mid_df=df[df['manifesto_id'] == mid])
        mid_chunks_df['manifesto_id'] = mid
        # print(f'{mid_chunks_df.shape=}')
        chunks_df.append(mid_chunks_df)

        # if i == 1:  # sanity check
        #     break
    print()

    chunks_df = pd.concat(chunks_df)
    print(f'Total chunks: {chunks_df.shape[0]}')
    print()

    # reorder columns
    chunks_df = chunks_df[[
        'manifesto_id',
        'chunk_text_translated',
        'chunk_rile_score', 'chunk_gal_tan_score',
        'chunk_n_sent', 'chunk_n_tokens'
    ]]

    # add back manifesto metadata
    metadata_df = df[[
        'manifesto_id', 'year', 'month', 'party', 'country', 'source_language'
    ]].groupby(['manifesto_id', 'year', 'month', 'party', 'country', 'source_language']
               ).count().reset_index()
    chunks_df = pd.merge(
        chunks_df,
        metadata_df,
        on='manifesto_id',
        how='inner'
    )

    print('test set manifestos:', chunks_df[chunks_df['year'] >= 2019]['manifesto_id'].nunique())

    # save to disk
    print('Saving to file...')
    mst = f'_ms{max_sent}' if max_sent else f'_mt{max_tokens}'
    filename = f'{model_name_short}_chunks_df{mst}.csv'
    chunks_path = os.path.join(mount_no_backup, 'marpor_data', filename)
    chunks_df.to_csv(chunks_path, index=False)
    print('Saved!')
