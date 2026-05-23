from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.prep_dataset import load_data
from utils.hyperparameters import ModelHyperparams

# RANDOM_SEEDS = (7, 42, 158, 1293, 8888)
# RANDOM_SEEDS = (7, 42, 158, 1293)
RANDOM_SEEDS = [7]

def get_preds_ground_truth(m):
    # read ground truth
    df, _ = load_data(
        task=m.task,
        target=m.train_target,
        train_or_test='test',
        mount=mount_no_backup,
        test_batch_size=4,
        emb_model_name=m.emb_model_name,
        max_tokens=m.max_tokens,
        max_sent=m.max_sent
    )

    if m.task == CLF:
        df = df[['manifesto_id', f'{m.eval_target}_label']].rename(
            columns={f'{m.eval_target}_label': f'{m.eval_target}_true{m.model_i}'})
    else:
        df = df[['manifesto_id', f'chunk_{m.eval_target}_score']].rename(
            columns={f'chunk_{m.eval_target}_score': f'{m.eval_target}_true{m.model_i}'})

    for seed in RANDOM_SEEDS:
        # read preds
        preds_path = get_preds_path(m.task, m.train_target, seed, m.emb_model_name,
                                    m.freeze_enc_weights, m.lr, m.train_batch_size,
                                    m.eval_ep,
                                    contr_pretr_id=m.contr_pretr_id,
                                    whiten=m.whiten, max_sent=m.max_sent)
        preds = pd.read_csv(preds_path, index_col=0)

        if m.task == CLF:
            preds[f'{m.eval_target}_pred'] = preds[f'{m.eval_target}_pred_num'].apply(
                lambda x: label_num2str(x, m.eval_target))
        else:
            preds[f'{m.eval_target}_pred'] = preds[f'chunk_{m.eval_target}_pred']

        df[f'{m.eval_target}_pred{m.model_i}_{seed}'] = preds[f'{m.eval_target}_pred'].to_list()

    # compute manifesto-level scores
    pred_cols = list(df.columns)
    pred_cols.remove('manifesto_id')
    if m.task == CLF:
        compute_simple = compute_rile_simple if m.eval_target == RILE \
            else compute_gal_tan_simple
        scores = df.groupby('manifesto_id')[pred_cols].agg(compute_simple).reset_index()
        return df, scores

    scores = df.groupby('manifesto_id')[pred_cols].mean().reset_index()
    return df, scores


def compute_metrics_avg_seed(m, df, scores):
    # compute metrics (normally) per random seed
    metrics_df = {f'seed{seed}': {} for seed in RANDOM_SEEDS}

    for seed in RANDOM_SEEDS:
        if m.task == CLF:
            acc = accuracy_score(df[f'{m.eval_target}_true{m.model_i}'],
                                 df[f'{m.eval_target}_pred{m.model_i}_{seed}'])

            weighted_f1 = f1_score(
                df[f'{m.eval_target}_true{m.model_i}'],
                df[f'{m.eval_target}_pred{m.model_i}_{seed}'],
                average='weighted')

            macro_f1 = f1_score(
                df[f'{m.eval_target}_true{m.model_i}'],
                df[f'{m.eval_target}_pred{m.model_i}_{seed}'],
                average='macro')

            rho = scores[f'{m.eval_target}_true{m.model_i}'].corr(
                scores[f'{m.eval_target}_pred{m.model_i}_{seed}'],
                method='spearman')

            metrics_df[f'seed{seed}'].update({
                'acc': acc,
                'weighted_f1': weighted_f1,
                'macro_f1': macro_f1,
                'rho': rho
            })

        elif m.task == RGR:
            mse_chunk = mean_squared_error(df[f'{m.eval_target}_true{m.model_i}'],
                                           df[f'{m.eval_target}_pred{m.model_i}_{seed}'])

            mse_manifesto = mean_squared_error(scores[f'{m.eval_target}_true{m.model_i}'],
                                     scores[f'{m.eval_target}_pred{m.model_i}_{seed}'])

            rho = scores[f'{m.eval_target}_true{m.model_i}'].corr(
                scores[f'{m.eval_target}_pred{m.model_i}_{seed}'],
                method='spearman')

            metrics_df[f'seed{seed}'].update({
                'mse_chunk': mse_chunk,
                'mse_manifesto': mse_manifesto,
                'rho': rho
            })

    metrics_df = pd.DataFrame.from_dict(metrics_df, orient='index')
    metrics_df.loc['mean'] = metrics_df.mean()

    return metrics_df


def get_metrics_avg_path(task, train_target, eval_target,
                         emb_model_name, freeze_enc_weights, ep,
                         contr_pretr_id=None, whiten=None, max_sent=None):
    model_name_short = get_model_name_short(emb_model_name)
    metrics_dir = os.path.join(mount_w_backup, 'metrics', eval_target)
    os.makedirs(metrics_dir, exist_ok=True)

    train_target_str = f'{train_target}' if train_target == JOINT \
        else f'{train_target}_ind'

    freeze_str = '_freeze' if freeze_enc_weights else ''
    cpid_str = f'_cpid{contr_pretr_id}' if contr_pretr_id else ''
    whiten_str = '_whiten' if whiten else ''
    ms_str = f'_ms{max_sent}' if max_sent else ''

    filename = f'metrics_{task}_{train_target_str}_{model_name_short}{freeze_str}{cpid_str}{whiten_str}{ms_str}_ep{ep}.csv'
    metrics_path = os.path.join(metrics_dir, filename)
    return metrics_path


if __name__ == "__main__":
    # choose model
    model = ModelHyperparams(
        model_i='',
        task=RGR,
        train_target=JOINT,
        eval_target=GALTAN,
        emb_model_name='ModernBERT',
        n_epochs=5,
        lr=1e-5,
        train_batch_size=4,
        eval_ep=4,
        freeze_enc_weights=False,
        contr_pretr_id=None,
        whiten=None,
        max_sent=100,
        max_tokens=None
    )
    print(model)
    print()

    df, scores = get_preds_ground_truth(model)

    metrics_df = compute_metrics_avg_seed(model, df, scores)
    print(metrics_df)
    print()

    mean_values = metrics_df.loc['mean'].to_dict()
    print('Mean:')
    for k in mean_values:
        if 'mse' in k:
            print(f'{k}: {mean_values[k]:.3f}')
        else:
            print(f'{k}: {mean_values[k]:.2f}')
    print()

    # save to file
    metrics_path = get_metrics_avg_path(
        model.task, model.train_target, model.eval_target,
        model.emb_model_name, model.freeze_enc_weights, model.eval_ep,
        contr_pretr_id=model.contr_pretr_id, whiten=model.whiten, max_sent=model.max_sent)
    metrics_df.to_csv(metrics_path)
