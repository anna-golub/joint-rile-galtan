from torch.nn import DataParallel
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *

from utils.prep_dataset import load_data
from label_aggregation_model import LabelAggregationModel
from utils.hyperparameters import *


def batch_test(batch):
    input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                max_length=max_length)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = get_preds(logits, target)

    del input_ids, attention_mask, logits

    return preds.cpu().numpy()


def do_test():
    print('Test...')
    y_pred = []
    model.eval()

    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_pred = batch_test(batch)
        y_pred.append(batch_pred)
        del batch
        # if i == 1:  # sanity check
        #     break

    y_pred = np.concatenate(y_pred, axis=0)
    y_pred = pd.DataFrame(y_pred)
    if target == JOINT:
        y_pred = y_pred.rename(columns={
            0: f'{RILE}_pred_num',
            1: f'{GALTAN}_pred_num'})
    else:
        y_pred = y_pred.rename(columns={0: f'{target}_pred_num'})

    # save predictions
    preds_path = get_preds_path(task, target, random_seed,
                                emb_model_name, freeze_enc_weights,
                                lr, train_batch_size, ep,
                                contr_pretr_id=contr_pretr_id, whiten=whiten)
    y_pred.to_csv(preds_path)

    return y_pred


def eval_preds():
    target_arr = [RILE, GALTAN] if target == JOINT else [target]
    results = dict()

    print('Test set:')

    for t in target_arr:
        # map label num to str
        test_df[f'{t}_pred'] = test_df[f'{t}_pred_num'].apply(
            lambda x: label_num2str(x, t))

        # compute metrics
        acc = accuracy_score(test_df[f'{t}_label_num'], test_df[f'{t}_pred_num'])
        macro_f1 = f1_score(
            test_df[f'{t}_label_num'], test_df[f'{t}_pred_num'],
            average='macro')
        weighted_f1 = f1_score(
            test_df[f'{t}_label_num'], test_df[f'{t}_pred_num'],
            average='weighted')
        rho = compute_spearman_rho(test_df, task, t)
        results[t] = {'acc': acc,
                      'weighted_f1': weighted_f1,
                      'rho': rho,
                      'macro_f1': macro_f1}

        print(t.upper())
        print(f"Accuracy: {acc:.3f}")
        print(f"Weighted F1: {weighted_f1:.3f}")
        print(f"Spearman's rho: {rho:.3f}")
        print(f"Macro F1: {macro_f1:.3f}")
        print()

    if target == JOINT:
        acc_joint = (results[RILE]['acc'] + results[GALTAN]['acc']) / 2
        f1_joint = (results[RILE]['weighted_f1'] + results[GALTAN]['weighted_f1']) / 2
        results[JOINT] = {'acc': acc_joint, 'f1': f1_joint}
        print(f"Joint accuracy: {acc_joint:.3f}")
        print(f"Joint weighted F1: {f1_joint:.3f}")

    return results


if __name__ == "__main__":
    print('Label Aggregation: eval')

    task = CLF
    target, emb_model_name, random_seed, n_epochs, lr, \
        train_batch_size, test_batch_size, freeze_enc_weights, \
        contr_pretr_id, whiten, ep = read_command_line(task=CLF, mode='eval')

    # target, emb_model_name, contr_pretr_id, whiten, \
    #     random_seed, n_epochs, lr, with_lr_schedule, \
    #     train_batch_size, test_batch_size, freeze_enc_weights, \
    #     ep = get_hyperparameters(task=CLF, mode='eval')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')
    set_random_seed(random_seed=random_seed)

    # load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    max_length = 100

    # load data
    test_df, test_dataloader = load_data(
        task=task,
        target=target,
        train_or_test='test',
        mount=mount_no_backup,
        test_batch_size=test_batch_size,
        # tokenizer=tokenizer
    )

    # load model
    model_path = get_model_path(task, target, random_seed,
                                emb_model_name, freeze_enc_weights,
                                lr, train_batch_size,
                                contr_pretr_id=contr_pretr_id, whiten=whiten)
    checkpoint = model_path.format(ep)
    print(f'{checkpoint=}')

    print('Loading model...')
    model = LabelAggregationModel(
        target=target,
        emb_model_name=emb_model_name,
        emb_dim=768,
        n_classes=3,
        freeze_enc_weights=freeze_enc_weights,
        contr_pretr_id=contr_pretr_id,
        random_seed=random_seed
    )
    model.to(device)

    if whiten:
        train_dataloader, _ = load_data(
            task=task,
            target=target,
            train_or_test='train',
            mount=mount_no_backup,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )

        model.emb_model.get_whiten_kernel_bias(
            dataloader=train_dataloader,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device
        )

    model = DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5])
    checkpoint = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint)

    # get test set predictions
    y_pred = do_test()

    # or load predictions from disk
    # preds_path = get_preds_path(task, target,
    #                             emb_model_name, freeze_enc_weights,
    #                             lr, train_batch_size, ep,
    #                             with_lr_schedule=with_lr_schedule,
    #                             contr_pretr_id=contr_pretr_id, whiten=whiten)
    # y_pred = pd.read_csv(preds_path, index_col=0)
    # if '0' in y_pred.columns:
    #     y_pred = y_pred.rename(columns={'0': f'{target}_pred_num'})

    # get test set metrics
    test_df = pd.merge(test_df, y_pred, left_index=True, right_index=True)
    test_results = eval_preds()
