import re
from sklearn.metrics import accuracy_score, f1_score
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.prep_dataset import load_data
from lookup import str2label


def post_process_joint(s):
    s = s.lstrip().rstrip()
    if len(s) <= 1:
        return None, None

    label_rile, label_galtan = None, None

    for v1 in ('right', 'right-wing', 'left', 'left-wing', 'neutral', 'A', 'B', 'C'):
        for v2 in ('traditional-authoritarian-nationalist', 'traditional', \
                   'authoritarian', 'nationalist',
                   'green-alternative-liberal', 'green', 'alternative', 'liberal',
                   'neutral', 'D', 'E', 'F'):
            pattern = f'({v1}|{v1.capitalize()})' + r'(.|\n)*' + f'({v2}|{v2.capitalize()})'
            if re.search(pattern, s):
                label_rile = v1
                label_galtan = v2

    return label_rile, label_galtan


if __name__ == "__main__":
    task = CLF
    target = GALTAN
    model_name = 'allenai/Olmo-3-7B-Instruct'
    random_seed = 7
    temperature = 0.85  # default 0.6
    top_p = 0.85  # default 0.95
    max_new_tokens = 30
    num_return_sequences = 5

    # read data
    test_df, _ = load_data(
        task=task,
        target=target,
        train_or_test='test',
        mount=mount_no_backup,
        test_batch_size=16,
        random_seed=random_seed)

    # read predictions
    preds_path = get_preds_path_llm(task, target, model_name,
                                    temperature, top_p, max_new_tokens, num_return_sequences)
    preds = pd.read_csv(preds_path, index_col=0).rename(
        columns={'0': 'sent', '1': f'{target}_llm_output'})
    preds['sent_id'] = np.array([[i] * num_return_sequences
                                 for i in range(test_df.shape[0])]).flatten()

    # keep only new tokens
    split_str = 'Correct options:\nassistant\n' if target == JOINT \
        else 'Correct option:\nassistant\n'
    preds[f'{target}_llm_output'] = preds[f'{target}_llm_output'].apply(
        lambda x: x.split(split_str)[1].lstrip().rstrip()
    )

    # extract RILE and GAL-TAN categories
    if target == JOINT:
        preds['joint_pred'] = preds[f'{target}_llm_output'].apply(post_process_joint)
        preds['rile_pred'] = preds['joint_pred'].apply(lambda x: x[0]).map(str2label[target])
        preds['gal_tan_pred'] = preds['joint_pred'].apply(lambda x: x[1]).map(str2label[target])
        target_arr = [RILE, GALTAN]
    else:
        preds[f'{target}_pred'] = preds[f'{target}_llm_output'].map(str2label[target])
        target_arr = [target]

    for t in target_arr:
        print(t)
        print(preds[f'{t}_pred'].value_counts(dropna=False))
        print()

    # process multiple responses per sentence
    if target == JOINT:
        preds = preds.groupby(['sent_id', 'sent'])[
            ['rile_pred', 'gal_tan_pred']].aggregate(lambda x: x).reset_index()
        preds[[f'rile_pred{i}' for i in range(num_return_sequences)]] = pd.DataFrame(
            preds['rile_pred'].tolist())
        preds[[f'gal_tan_pred{i}' for i in range(num_return_sequences)]] = pd.DataFrame(
            preds['gal_tan_pred'].tolist())
        preds = preds.rename(columns={'rile_pred': 'rile_pred_arr',
                                      'gal_tan_pred': 'gal_tan_pred_arr'})
    else:
        preds = preds.groupby(['sent_id', 'sent'])[
            f'{target}_pred'].aggregate(lambda x: x).reset_index()
        preds[[f'{target}_pred{i}' for i in range(num_return_sequences)]] = pd.DataFrame(
            preds[f'{target}_pred'].tolist())
        preds = preds.rename(columns={f'{target}_pred': f'{target}_pred_arr'})

    # calculate variance of responses
    for t in target_arr:
        preds[f'{t}_pred_variance'] = preds[f'{t}_pred_arr'].apply(lambda x: len(set(x)))
        print(t)
        print(preds[f'{t}_pred_variance'].value_counts(normalize=True))
        print()

    # compute metrics
    metrics_df = {i: {} for i in range(num_return_sequences)}
    for t in target_arr:
        for i in range(num_return_sequences):
            test_df[f'{t}_pred'] = preds[f'{t}_pred{i}'].to_list()
            test_df_i = test_df[~test_df[f'{t}_pred'].isna()]
            metrics_df[i].update({
                f'{t}_test_examples': test_df_i.shape[0] / test_df.shape[0],
                f'{t}_acc': accuracy_score(test_df_i[f'{t}_label'], test_df_i[f'{t}_pred']),
                f'{t}_weighted_f1': f1_score(test_df_i[f'{t}_label'], test_df_i[f'{t}_pred'],
                                             average='weighted'),
                f'{t}_macro_f1': f1_score(test_df_i[f'{t}_label'], test_df_i[f'{t}_pred'],
                                          average='macro'),
                f'{t}_rho': compute_spearman_rho(test_df_i, task=task, target=t)
            })

    metrics_df = pd.DataFrame.from_dict(metrics_df, orient='index')
    metrics_df.loc['mean'] = metrics_df.mean()
    print(metrics_df)

    mean_values = metrics_df.loc['mean'].to_dict()
    print('Mean:')
    for k in mean_values:
        print(f'{k}: {mean_values[k]:.2f}')
