import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *

from argparse import ArgumentParser, BooleanOptionalAction


def read_command_line(task, mode='train'):
    parser = ArgumentParser()

    # general hyperparameters
    if task in (CLF, RGR):
        parser.add_argument('--target', type=str,
                            choices=[RILE, GALTAN, JOINT])
    parser.add_argument('--emb_model', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)

    # task-specific hyperparameters
    if task == CLF:
        parser.add_argument('--freeze_encoder', action=BooleanOptionalAction)
        parser.add_argument('--cpid', type=int)
        parser.add_argument('--whiten',
                            action=BooleanOptionalAction, default=False)

    elif task == CONTR:
        parser.add_argument('--cpid', type=int)
        parser.add_argument('--label_basis', type=str)
        parser.add_argument('--margin', type=float)
        parser.add_argument('--lr_scheduler', type=str)
        parser.add_argument('--warmup_steps', type=int)

    elif task == RGR:
        parser.add_argument('--freeze_encoder', action=BooleanOptionalAction)
        parser.add_argument('--max_tokens', type=int)
        parser.add_argument('--max_sent', type=int)

    # evaluation
    if mode == 'eval':
        parser.add_argument('--eval_ep', type=int)

    args = parser.parse_args()
    args.emb_model = get_model_name_full(args.emb_model)
    args = vars(args)  # turn into dict

    for k, v in args.items():
        print(f'{k}: {v}')
    print(args.values())

    return args.values()


class ModelHyperparams:
    def __init__(self, model_i, task, train_target, eval_target, emb_model_name,
                 n_epochs, lr, train_batch_size, eval_ep, freeze_enc_weights,
                 contr_pretr_id=None, whiten=None,
                 max_sent=None, max_tokens=None):
        self.model_i = model_i  # id for bootstrap test
        self.task = task
        self.train_target = train_target
        self.eval_target = eval_target
        self.emb_model_name = get_model_name_full(emb_model_name)
        self.n_epochs = n_epochs
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.eval_ep = eval_ep
        self.freeze_enc_weights = freeze_enc_weights
        self.contr_pretr_id = contr_pretr_id
        self.whiten = whiten
        self.max_sent = max_sent
        self.max_tokens = max_tokens

    def __repr__(self):
        return f'task={self.task} train_target={self.train_target} ' + \
            f'eval_target={self.eval_target} enc={self.emb_model_name} ' + \
            f'eval_ep={self.eval_ep} freeze={self.freeze_enc_weights} ' + \
            f'cpid={self.contr_pretr_id} whiten={self.whiten} ' + \
            f'max_sent={self.max_sent} max_tokens={self.max_tokens}'


def get_hyperparameters(task, mode=None):
    if task == CLF:
        return get_hyperparameters_clf(mode=mode)
    if task == CONTR:
        return get_hyperparameters_contr()
    if task == RGR:
        return get_hyperparameters_rgr(mode=mode)


def get_hyperparameters_rgr(mode='train'):
    # TODO: read command line arguments

    target = JOINT

    emb_model_name = 'answerdotai/ModernBERT-base'
    # emb_model_name = 'google/bigbird-roberta-base'

    max_tokens = None
    max_sent = 1

    freeze_enc_weights = False  # set to True for baseline

    random_seed = 7  # 42 in Nikolaev et al. 2023

    # n_epochs = 1  # sanity check
    n_epochs = 5  # 5 in Nikolaev et al. 2023

    ep = None
    if mode == 'eval':
        ep = 1  # epoch to evaluate

    lr = 1e-5  # 1e-5 in Nikolaev et al. 2023
    with_lr_schedule = False

    # Nikolaev et al. 2023: batch size = 4 (for BigBird, Longformer)
    train_batch_size = 32
    test_batch_size = 128

    dropout_p = None

    print(f'target = {target}')
    print(f'emb model = {emb_model_name}')
    print(f'max tokens = {max_tokens}')
    print(f'max sentences = {max_sent}')
    print(f'freeze encoder weights = {freeze_enc_weights}')
    print(f'random seed = {random_seed}')
    print(f'n_epochs = {n_epochs}')
    print(f'lr = {lr}')
    print(f'lr scheduler = {with_lr_schedule}')
    print(f'train batch size = {train_batch_size}')
    print(f'dropout prob = {dropout_p}')

    if mode == 'eval':
        print(f'ep = {ep}')

    print()

    return target, emb_model_name, max_tokens, max_sent, \
        random_seed, n_epochs, lr, with_lr_schedule, \
        train_batch_size, test_batch_size, freeze_enc_weights, dropout_p, ep


def get_hyperparameters_contr():
    # TODO: read command line arguments

    # ! ! !    C H A N G E      T H I S    ! ! !
    contr_pretr_id = 11

    # emb_model_name = 'all-mpnet-base-v2'
    emb_model_name = "answerdotai/ModernBERT-base"

    # all / rile_gal-tan / party
    label_basis = 'party'

    margin = 1

    random_seed = 7

    # n_epochs = 1  # sanity check
    n_epochs = 5  # 5 in Ceron et al. 2022

    lr = 1e-5  # 5e-5 in Ceron et al. 2022 (default in hf trainer)
    lr_scheduler_type = 'linear'  # linear (default) in Ceron et al. (2022)
    warmup_steps = 100  # 100 in Ceron et al. 2022

    # train_batch_size=8 in Ceron et al. 2022 (default in hf trainer)
    # w/o parallelization: max 8
    train_batch_size = 16
    test_batch_size = 16

    print(f'contrastive pre-training ID = {contr_pretr_id}')
    print(f'emb model = {emb_model_name}')
    print(f'label basis = {label_basis}')
    print(f'margin = {margin}')
    print(f'random seed = {random_seed}')
    print(f'n_epochs = {n_epochs}')
    print(f'lr = {lr}')
    print(f'lr scheduler type = {lr_scheduler_type}')
    print(f'warmup steps = {warmup_steps}')
    print(f'train batch size = {train_batch_size}')
    print()

    return contr_pretr_id, emb_model_name, label_basis, margin, random_seed, \
        n_epochs, lr, lr_scheduler_type, warmup_steps, train_batch_size, test_batch_size


def get_hyperparameters_clf(mode='train'):
    # TODO: read command line arguments

    target = RILE

    # emb_model_name = "answerdotai/ModernBERT-base"
    emb_model_name = 'sentence-transformers/all-mpnet-base-v2'

    contr_pretr_id = None
    freeze_enc_weights = False
    whiten = False

    random_seed = 7

    # n_epochs = 1  # sanity check
    n_epochs = 5  # 5 in Nikolaev et al. 2023

    ep = None
    if mode == 'eval':
        ep = 2  # epoch to evaluate

    lr = 1e-5  # 1e-5 in Nikolaev et al. 2023
    with_lr_schedule = False

    # Nikolaev et al. 2023: batch_size = 32 * 8 = 256 if sbert else 16 * 8 = 128
    # train_batch_size = 64  # w/o DataParallel
    train_batch_size = 64
    test_batch_size = 256

    print(f'target = {target}')
    print(f'emb model = {emb_model_name}')
    print(f'contrastive pre-training ID = {contr_pretr_id}')
    print(f'whiten embeddings = {whiten}')
    print(f'freeze encoder weights = {freeze_enc_weights}')
    print(f'random seed = {random_seed}')
    print(f'n_epochs = {n_epochs}')
    print(f'lr = {lr}')
    print(f'lr scheduler = {with_lr_schedule}')
    print(f'train batch size = {train_batch_size}')

    if mode == 'eval':
        print(f'ep = {ep}')

    print()

    return target, emb_model_name, contr_pretr_id, whiten, \
        random_seed, n_epochs, lr, with_lr_schedule, \
        train_batch_size, test_batch_size, freeze_enc_weights, ep
