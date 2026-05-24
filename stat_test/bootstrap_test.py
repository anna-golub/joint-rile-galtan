from scipy.stats import bootstrap
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from stat_test.metrics_avg_seed import *


# bootstrap statistic function
def rho_diff(sample_index):
    # example from scipy docs
    # mean1 = np.mean(sample1, axis=axis)
    # mean2 = np.mean(sample2, axis=axis)
    # return mean1 - mean2

    sample_scores = scores.set_index('manifesto_id').loc[list(sample_index)]

    # calculate rho for each random seed value in sample 1 and 2
    rho1_arr, rho2_arr = [], []
    for seed in RANDOM_SEEDS:
        rho1 = sample_scores[f'{m1.eval_target}_true1'].corr(
            sample_scores[f'{m1.eval_target}_pred1_{seed}'],
            method='spearman')
        rho1_arr.append(rho1)

        rho2 = sample_scores[f'{m2.eval_target}_true2'].corr(
            sample_scores[f'{m2.eval_target}_pred2_{seed}'],
            method='spearman')
        rho2_arr.append(rho2)

    # average over all random seeds
    rho1 = np.mean(rho1_arr)
    rho2 = np.mean(rho2_arr)

    # return diff. btw. rho 1 and 2
    return rho1 - rho2


if __name__ == "__main__":
    # model 1
    m1 = ModelHyperparams(
        model_i=1,
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

    # model 2
    m2 = ModelHyperparams(
        model_i=2,
        task=RGR,
        train_target=JOINT,
        eval_target=GALTAN,
        emb_model_name='ModernBERT',
        n_epochs=5,
        lr=1e-5,
        train_batch_size=2,
        eval_ep=3,
        freeze_enc_weights=False,
        contr_pretr_id=None,
        whiten=None,
        max_sent=200,
        max_tokens=None
    )

    for m in (m1, m2):
        print(f'Model {m.model_i}: {m}')

        # read and preprocess predictions and ground truth
        m.df, m.scores = get_preds_ground_truth(m)

        # compute metrics (normally) and average over random seed
        metrics_df = compute_metrics_avg_seed(m, m.df, m.scores)
        mean_values = metrics_df.loc['mean'].to_dict()
        print(f'Model {m.model_i} mean:')
        for k in mean_values:
            if 'mse' in k:
                print(f'{k}: {mean_values[k]:.3f}')
            else:
                print(f'{k}: {mean_values[k]:.2f}')
        print()

        # save metrics to file
        metrics_path = get_metrics_avg_path(
            m.task, m.train_target, m.eval_target,
            m.emb_model_name, m.freeze_enc_weights, m.eval_ep,
            contr_pretr_id=m.contr_pretr_id, whiten=m.whiten, max_sent=m.max_sent)
        metrics_df.to_csv(metrics_path)

    # combine 2 model-specific tables into one
    print(f'Model 1 test set: {m1.scores["manifesto_id"].nunique()} manifestos')
    print(f'Model 2 test set: {m2.scores["manifesto_id"].nunique()} manifestos')
    manifestos_intersect = list(
        set(list(m1.scores['manifesto_id'].unique())) & \
        set(list(m2.scores['manifesto_id'].unique()))
    )
    print(f'Intersection: {len(manifestos_intersect)}')
    m1.scores = m1.scores[m1.scores['manifesto_id'].isin(manifestos_intersect)]
    m2.scores = m2.scores[m2.scores['manifesto_id'].isin(manifestos_intersect)]
    scores = pd.merge(m1.scores, m2.scores, on='manifesto_id', how='outer')
    print(f'{scores.shape=} {scores.isna().sum().sum()=}')

    # run bootstrap test
    res = bootstrap(
        data=(scores['manifesto_id'],),
        statistic=rho_diff,
        method='percentile',
        n_resamples=10000,
        vectorized=False,
        confidence_level=0.95,
        alternative='two-sided',
        rng=7  # random state
    )

    # print results
    print(f'Confidence interval: [{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]')

    if res.confidence_interval.low <= 0 <= res.confidence_interval.high:
        print('models ON PAR')
    elif 0 < res.confidence_interval.low:
        print('model 1 BETTER than model 2')
    elif 0 > res.confidence_interval.high:
        print('model 1 WORSE than model 2')
