from torch.nn import DataParallel
from transformers import AutoTokenizer

from tqdm import tqdm
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.hyperparameters import *
from utils.prep_dataset import load_data
from regression_model import RegressionModel


def batch_test(batch):
    input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                max_length=max_tokens)

    with torch.no_grad():
        pred_scores = model(input_ids, attention_mask)

    # remove extra nesting e.g. for target=JOINT: (batch size, 2, 1) => (batch size, 2)
    pred_scores = torch.squeeze(pred_scores, dim=-1)

    del input_ids, attention_mask

    return pred_scores.cpu().numpy()


def do_test():
    print('Test...')
    y_pred = []
    model.eval()

    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_pred = batch_test(batch)
        y_pred.append(batch_pred)
        del batch

    y_pred = np.concatenate(y_pred, axis=0)
    y_pred = pd.DataFrame(y_pred)
    if target == JOINT:
        y_pred = y_pred.rename(columns={
            0: f'chunk_{RILE}_pred',
            1: f'chunk_{GALTAN}_pred'})
    else:
        y_pred = y_pred.rename(columns={0: f'chunk_{target}_pred'})

    # save predictions
    preds_path = get_preds_path(task, target, random_seed,
                                emb_model_name, freeze_enc_weights,
                                lr, train_batch_size, ep, max_sent=max_sent)
    y_pred.to_csv(preds_path)

    return y_pred


def eval_preds():
    target_arr = [RILE, GALTAN] if target == JOINT else [target]
    results = dict()

    print('Test set:')

    for t in target_arr:
        # compute rho
        rho = compute_spearman_rho(test_df, task, t)
        results[t] = {'rho': rho}

        print(t.upper())
        print(f"Spearman's rho (manifesto level): {rho:.3f}")
        print()

    return results


if __name__ == "__main__":
    print('Chunk-level regression: eval')
    # context window: BigBird - 4096, ModernBERT - 8192

    task = RGR
    target, emb_model_name, random_seed, n_epochs, lr, \
        train_batch_size, test_batch_size, freeze_enc_weights, \
        max_tokens, max_sent, ep = read_command_line(task=RGR, mode='eval')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')
    set_random_seed(random_seed=random_seed)

    # load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)

    # load data
    test_df, test_dataloader = load_data(
        task=task,
        target=target,
        train_or_test='test',
        mount=mount_no_backup,
        test_batch_size=test_batch_size,
        emb_model_name=emb_model_name,
        max_tokens=max_tokens,
        max_sent=max_sent,
    )

    # load model
    model_path = get_model_path(task, target, random_seed,
                                emb_model_name, freeze_enc_weights,
                                lr, train_batch_size, max_sent=max_sent)
    checkpoint = model_path.format(ep)
    print(f'{checkpoint=}')

    print('Loading model...')
    model = RegressionModel(
        target=target,
        emb_model_name=emb_model_name,
        freeze_enc_weights=freeze_enc_weights,
        # dropout_p=dropout_p,
        emb_dim=768,
    )
    model.to(device)

    # if model was trained with DP
    # model = DataParallel(model, device_ids=[0, 1, 2, 3])

    checkpoint = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint)

    # if model was trained w/o DP, but now it's needed
    model = DataParallel(model, device_ids=[0, 1, 2, 3])

    # get test set predictions
    y_pred = do_test()

    # or load predictions from disk
    # preds_path = get_preds_path(task, target, random_seed,
    #                emb_model_name, freeze_enc_weights,
    #                lr, train_batch_size, ep, max_sent=max_sent)
    # y_pred = pd.read_csv(preds_path, index_col=0)

    # get test set metrics
    test_df = pd.merge(test_df, y_pred, left_index=True, right_index=True)
    test_results = eval_preds()
