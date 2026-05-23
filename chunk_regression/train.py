import torch.autograd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn import MSELoss, DataParallel
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torchmetrics.regression import SpearmanCorrCoef

import gc
from tqdm import tqdm
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.hyperparameters import *
from utils.prep_dataset import load_data
from regression_model import RegressionModel


def get_mse_loss(pred_scores, ground_truth):
    if target in (RILE, GALTAN):
        return mse(pred_scores, ground_truth)
    elif target == JOINT:
        # MSE_rile + MSE_galtan
        return mse(pred_scores[:, 0], ground_truth[:, 0]) + \
            mse(pred_scores[:, 1], ground_truth[:, 1])


def batch_train(batch):
    # print('START of batch_train')
    # debug_memory()

    input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                max_length=max_tokens)
    ground_truth = batch['ground_truth'].to(device)

    optimizer.zero_grad(set_to_none=True)

    pred_scores = model(input_ids, attention_mask)
    # remove extra nesting e.g. for target=JOINT: (batch size, 2, 1) => (batch size, 2)
    # print(f'{pred_scores.shape=}')
    pred_scores = torch.squeeze(pred_scores, dim=-1)

    loss = get_mse_loss(pred_scores, ground_truth)

    # TODO: gradient clipping ?
    loss.backward()
    loss_item = loss.item()
    del loss, input_ids, attention_mask

    optimizer.step()

    # if with_lr_schedule:
    #     lr_scheduler.step()

    # update metric trackers
    batch_rho = train_rho(preds=pred_scores, target=ground_truth)

    del pred_scores, ground_truth
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()

    # print('END of batch_train')
    # debug_memory()

    return loss_item


def batch_val(batch):
    input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                max_length=max_tokens)
    ground_truth = batch['ground_truth'].to(device)

    with torch.no_grad():
        pred_scores = model(input_ids, attention_mask)

    # remove extra nesting e.g. for target=JOINT: (batch size, 2, 1) => (batch size, 2)
    pred_scores = torch.squeeze(pred_scores, dim=-1)

    loss = get_mse_loss(pred_scores, ground_truth)
    loss_item = loss.item()
    del loss, input_ids, attention_mask

    # update metric trackers
    batch_rho = val_rho(preds=pred_scores, target=ground_truth)

    del pred_scores, ground_truth
    torch.cuda.empty_cache()
    gc.collect()

    return loss_item


def do_train():
    model_path = get_model_path(task, target, random_seed,
                                emb_model_name, freeze_enc_weights,
                                lr, train_batch_size, max_sent=max_sent)
    train_stats = {i: dict() for i in range(n_epochs)}

    # training loop

    # for ep in range(restart_ep + 1, restart_ep + 1 + n_epochs):
    for ep in range(n_epochs):
        print(f'Epoch {ep}:')

        # train
        print('Train...')
        train_rho.reset()
        train_loss = 0
        model.train()

        for i, batch in tqdm(enumerate(train_dataloader),
                             total=len(train_dataloader)):
            batch_loss = batch_train(batch)
            train_loss += batch_loss
            del batch
            # if i == 1:    # sanity check
            #     break
            # print(f'{i=}: done')

            # min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB
            # clear_gpu_memory()
            # wait_until_enough_gpu_memory(min_memory_available)

        train_loss /= len(train_dataloader)

        # if with_lr_schedule:
        #     lr_scheduler.step()
        # print('lr =', optimizer.param_groups[0]["lr"])

        # val
        print('Val...')
        val_rho.reset()
        val_loss = 0
        model.eval()

        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            batch_loss = batch_val(batch)
            val_loss += batch_loss
            del batch
            # if i == 1:    # sanity check
            #     break

        val_loss /= len(val_dataloader)

        # record metrics
        train_stats[ep] = {
            'train_loss': train_loss,
            'train_rho_chunk-lvl': train_rho.compute().cpu().numpy(),
            'val_loss': val_loss,
            'val_rho_chunk-lvl': val_rho.compute().cpu().numpy(),
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
    train_stats_path = get_train_stats_path(task, target, random_seed,
                                            emb_model_name, freeze_enc_weights,
                                            lr, train_batch_size, n_epochs,
                                            max_sent=max_sent)
    train_stats.to_csv(train_stats_path)
    return train_stats


if __name__ == "__main__":
    print('Chunk-level regression: train')

    task = RGR
    target, emb_model_name, random_seed, n_epochs, lr, \
        train_batch_size, test_batch_size, freeze_enc_weights, \
        max_tokens, max_sent = read_command_line(task=RGR, mode='train')

    # target, emb_model_name, max_tokens, max_sent, \
    #     random_seed, n_epochs, lr, with_lr_schedule, \
    #     train_batch_size, test_batch_size, freeze_enc_weights, \
    #     dropout_p, _ = get_hyperparameters(task=task, mode='train')

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')
    set_random_seed(random_seed=random_seed)

    # load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)

    # context window: BigBird - 4096, ModernBERT - 8192
    # max_length = 4096 if 'bigbird' in emb_model_name else 8192

    # load train & val data
    train_dataloader, val_dataloader = load_data(
        task=task,
        target=target,
        train_or_test='train',
        mount=mount_no_backup,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        random_seed=random_seed,
        emb_model_name=emb_model_name,
        max_tokens=max_tokens,
        max_sent=max_sent,
        # tokenizer=tokenizer,
    )

    # load model
    print('Loading model...')
    model = RegressionModel(
        target=target,
        emb_model_name=emb_model_name,
        freeze_enc_weights=freeze_enc_weights,
        # dropout_p=dropout_p,
        emb_dim=768,
    )
    # print(model)
    model.to(device)

    # ModernBERT is trained on nandu w/o DP
    # if 'bigbird' in emb_model_name:
    #     model = DataParallel(model, device_ids=[0, 1, 2, 3])
    #     model.to(device)
    print()

    # continue training
    # model_path = get_model_path(task, target, emb_model_name, freeze_enc_weights,
    #                             lr, train_batch_size, with_lr_schedule,
    #                             dropout_p=dropout_p)
    # restart_ep = 4  # restart from epoch 4
    # checkpoint = model_path.format(restart_ep)
    # print(f'{checkpoint=}')
    #
    # checkpoint = torch.load(checkpoint, weights_only=True, map_location=device)
    # model.load_state_dict(checkpoint)

    # set up optimization
    mse = MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    # if with_lr_schedule:
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=len(train_dataloader) * n_epochs,
    # )
    # lr_scheduler = LinearLR(
    #     optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs)

    # get metric trackers
    train_rho = SpearmanCorrCoef(num_outputs=2 if target == JOINT else 1)
    val_rho = SpearmanCorrCoef(num_outputs=2 if target == JOINT else 1)

    print()

    # train model
    # train_stats = do_train()

    try:
        train_stats = do_train()
    except torch.OutOfMemoryError:
        print('OutOfMemoryError')
        debug_memory()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
