import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, \
    SentenceTransformerTrainingArguments
from sentence_transformers.losses import BatchAllTripletLoss

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.prep_dataset import get_datasets_st
from utils.hyperparameters import *

if __name__ == "__main__":
    print('Starting...')

    emb_model_name, random_seed, n_epochs, lr, train_batch_size, test_batch_size, \
        contr_pretr_id, label_basis, margin, \
        lr_scheduler, warmup_steps = read_command_line(task=CONTR, mode='train')

    # contr_pretr_id, emb_model_name, label_basis, margin, random_seed, \
    #     n_epochs, lr, lr_scheduler_type, warmup_steps, \
    #     train_batch_size, test_batch_size = get_hyperparameters(task=CONTR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')
    set_random_seed(random_seed=random_seed)

    # load data
    train_dataset, val_dataset, test_dataset = get_datasets_st(
        label_basis=label_basis,
        random_seed=random_seed
    )

    # model
    model = SentenceTransformer(emb_model_name, device=device)
    model.max_seq_length = 100

    # loss
    loss = BatchAllTripletLoss(model, margin=margin)  # default: Euclid. dist., margin=5

    # output dir
    model_name_short = get_model_name_short(emb_model_name)
    model_dir = f'{model_name_short}_cpid{contr_pretr_id}'
    output_dir = os.path.join(mount_no_backup, 'st_output', model_dir)
    os.makedirs(output_dir, exist_ok=True)

    # training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,

        num_train_epochs=n_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,  # linear (default) in Ceron et al. (2022)
        warmup_steps=warmup_steps,

        fp16=True,

        eval_strategy="epoch",
        # eval_steps=100,
        save_strategy="epoch",
        # save_steps=100,
        # save_total_limit=2,
        logging_strategy='epoch',
        # logging_steps=100,
    )

    # train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )
    trainer_output = trainer.train()
    # print(f'{trainer_output=}')

    # save model
    print('Saving model...')
    model_path = get_contr_pretr_path(model_name_short, contr_pretr_id, random_seed)
    print(f'{model_path=}')
    model.save_pretrained(model_path)
