from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn import CrossEntropyLoss, DataParallel
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from tqdm import tqdm

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *

from utils.prep_dataset import load_data
from label_aggregation_model import LabelAggregationModel
from utils.hyperparameters import read_command_line


def get_cross_entropy_loss(logits, labels):
    if target in (RILE, GALTAN):
        return cross_entropy(logits, labels)
    elif target == JOINT:
        # CE_rile + CE_galtan
        return cross_entropy(logits[:, 0, :], labels[:, 0]) + \
            cross_entropy(logits[:, 1, :], labels[:, 1])


def batch_train(batch):
    input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                max_length=max_length)

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids, attention_mask)

    labels = batch['ground_truth'].to(device)
    loss = get_cross_entropy_loss(logits, labels)
    # TODO: gradient clipping ?
    loss.backward()
    loss_item = loss.item()
    del loss, input_ids, attention_mask

    optimizer.step()

    # if with_lr_schedule:
    #     lr_scheduler.step()

    preds = get_preds(logits, target)

    # update metric trackers
    batch_acc = train_acc(preds=preds, target=labels)
    batch_weighted_f1 = train_weighted_f1(preds=preds, target=labels)
    batch_macro_f1 = train_macro_f1(preds=preds, target=labels)

    del labels, logits, preds

    return loss_item


def batch_val(batch):
    input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                max_length=max_length)
    labels = batch['ground_truth'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = get_preds(logits, target)

    loss = get_cross_entropy_loss(logits, labels)
    loss_item = loss.item()
    del loss, input_ids, attention_mask

    batch_acc = val_acc(preds=preds, target=labels)
    batch_weighted_f1 = val_weighted_f1(preds=preds, target=labels)
    batch_macro_f1 = val_macro_f1(preds=preds, target=labels)
    del logits, labels, preds

    return loss_item


def do_train():
    model_path = get_model_path(task, target, random_seed,
                                emb_model_name, freeze_enc_weights,
                                lr, train_batch_size,
                                contr_pretr_id=contr_pretr_id, whiten=whiten)
    train_stats = {i: dict() for i in range(n_epochs)}

    # training loop

    for ep in range(n_epochs):
        print(f'Epoch {ep}:')

        # train
        print('Train...')
        train_acc.reset()
        train_weighted_f1.reset()
        train_macro_f1.reset()
        train_loss = 0
        model.train()

        for i, batch in tqdm(enumerate(train_dataloader),
                             total=len(train_dataloader)):
            batch_loss = batch_train(batch)
            train_loss += batch_loss
            del batch

        train_loss /= len(train_dataloader)

        # if with_lr_schedule:
        #     lr_scheduler.step()
        # print('lr =', optimizer.param_groups[0]["lr"])

        # val
        print('Val...')
        val_acc.reset()
        val_weighted_f1.reset()
        val_macro_f1.reset()
        val_loss = 0
        model.eval()

        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            batch_loss = batch_val(batch)
            val_loss += batch_loss
            del batch

        val_loss /= len(val_dataloader)

        # record metrics
        train_stats[ep] = {
            'train_loss': train_loss,
            'train_acc': train_acc.compute().cpu().numpy(),
            'train_weighted_f1': train_weighted_f1.compute().cpu().numpy(),
            'train_macro_f1': train_macro_f1.compute().cpu().numpy(),

            'val_loss': val_loss,
            'val_acc': val_acc.compute().cpu().numpy(),
            'val_weighted_f1': val_weighted_f1.compute().cpu().numpy(),
            'val_macro_f1': val_macro_f1.compute().cpu().numpy()
        }

        # save checkpoint
        print('Saving checkpoint:', model_path.format(ep))
        torch.save(model.state_dict(), model_path.format(ep))

        # print epoch stats
        print_ep_stats(train_stats, ep, target)

    # save train stats
    train_stats = pd.DataFrame.from_dict(train_stats, orient='index')
    train_stats = train_stats.reset_index(names='epoch').set_index('epoch')
    if target == JOINT:
        train_stats = postprocess_joint_train_stats(train_stats)
    train_stats_path = get_train_stats_path(
        task, target, random_seed, emb_model_name, freeze_enc_weights,
        lr, train_batch_size, n_epochs,
        contr_pretr_id=contr_pretr_id, whiten=whiten)
    train_stats.to_csv(train_stats_path)
    return train_stats


if __name__ == "__main__":
    print('Label Aggregation: train')

    task = CLF
    target, emb_model_name, random_seed, n_epochs, lr, \
        train_batch_size, test_batch_size, freeze_enc_weights, \
        contr_pretr_id, whiten = read_command_line(task=CLF, mode='train')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')
    set_random_seed(random_seed=random_seed)

    # load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    max_length = 100

    # load train & val data
    train_dataloader, val_dataloader = load_data(
        task=task,
        target=target,
        train_or_test='train',
        mount=mount_no_backup,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        random_seed=random_seed,
        # tokenizer=tokenizer
    )

    # load model
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
        model.emb_model.get_whiten_kernel_bias(
            dataloader=train_dataloader,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device
        )

    model = DataParallel(
        model,
        device_ids=[1, 2, 3, 4, 5, 6]
    )
    model.to(device)
    print()

    # set up optimization
    cross_entropy = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    # if with_lr_schedule:
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=len(train_dataloader) * n_epochs,
    # )
    # lr_scheduler = LinearLR(
    #     optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs)

    # init metric trackers
    train_acc, train_weighted_f1, train_macro_f1, \
        val_acc, val_weighted_f1, val_macro_f1 = get_metric_trackers(target, device)

    print()

    # train model
    train_stats = do_train()
