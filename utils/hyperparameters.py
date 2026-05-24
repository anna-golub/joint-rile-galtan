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
