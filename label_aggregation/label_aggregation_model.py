import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from whiten import compute_kernel_bias, transform_and_normalize


class StaticEmbeddingsModel(nn.Module):
    def __init__(self,
                 model_name,  # HF model name
                 freeze_enc_weights,  # set to True to only train clf head(s)
                 contr_pretr_id,  # contrastive pretraining setup ID
                 ):
        super().__init__()
        model_name_short = get_model_name_short(model_name)

        # load emb from disk
        emb_dir = f'{mount_no_backup}/emb/whiten/{model_name_short}_cpid{contr_pretr_id}'
        self.emb = dict()
        self.emb['train'] = pd.read_csv(f'{emb_dir}/train_emb.csv').to_numpy()
        self.emb['val'] = pd.read_csv(f'{emb_dir}/val_emb.csv').to_numpy()
        self.emb['test'] = pd.read_csv(f'{emb_dir}/test_emb.csv').to_numpy()
        print(f'{self.train_emb.shape=}, {self.val_emb.shape=}, {self.test_emb.shape=}')

        # freeze weights
        if freeze_enc_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        train_val_test = inputs['train_val_test']
        idx = inputs['idx']
        output = self.emb[train_val_test][idx]
        print(f'{output.shape=}')
        return output


class EmbeddingModel(nn.Module):
    def __init__(self,
                 model_name,  # HF model name
                 n_classes,  # 3 for 3 RILE/GALTAN categories
                 freeze_enc_weights,  # set to True to only train clf head(s)
                 contr_pretr_id,  # contrastive pretraining setup ID
                 random_seed
                 ):
        super().__init__()

        self.model_name = model_name
        self.model_name_short = get_model_name_short(model_name)
        self.n_classes = n_classes

        self.whiten = False
        self.register_buffer('whiten_kernel', None)
        self.register_buffer('whiten_bias', None)

        if contr_pretr_id:  # load pre-trained ST from checkpoint
            self._load_st(contr_pretr_id, random_seed)
        else:
            # load model from hf
            self._load_hf()

        # freeze weights
        if freeze_enc_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_st(self, contr_pretr_id, random_seed):
        # load backbone transformer of the contrast. pretrained ST
        checkpoint = get_contr_pretr_path(self.model_name_short, contr_pretr_id,
                                          random_seed)
        print(f'contrastive pre-training checkpoint = {checkpoint}')
        self.model = SentenceTransformer(
            checkpoint,
            device='cpu',
            # compatibility w/ torch data parallel
            config_kwargs={'reference_compile': self.model_name_short != 'ModernBERT'}
        )[0].auto_model

    def _load_hf(self):
        # config = AutoConfig.from_pretrained(self.model_name)

        if self.model_name_short == 'ModernBERT':
            # config.attention_dropout = 0.2
            # config.classifier_dropout = 0.2
            # config.embedding_dropout = 0.2
            # config.mlp_dropout = 0.2
            self.model = AutoModel.from_pretrained(
                self.model_name,
                # num_labels=self.n_classes,
                # config=config,
                reference_compile=False  # compatibility w/ torch data parallel
            )
        else:  # sbert MPNet
            # config.attention_probs_dropout_prob = 0.4
            # config.hidden_dropout_prob = 0.4
            self.model = AutoModel.from_pretrained(
                self.model_name,
                # num_labels=self.n_classes,
                # config=config
            )

    def forward(self, input_ids, attention_mask):
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if 'sentence-transformers' in self.model_name:  # ST
            emb = mean_pooling(model_output, attention_mask)
            # advised by hf but not in Nikolaev et al. code:
            # emb = F.normalize(emb, p=2, dim=1)
        else:  # transformer
            # sent emb is last hidden state repr of CLS token
            emb = model_output.last_hidden_state[:, 0, :]

        # whitening transformation
        if self.whiten:
            emb = transform_and_normalize(emb, self.whiten_kernel, self.whiten_bias)

        return emb

    def get_whiten_kernel_bias(self, dataloader, tokenizer, max_length, device):
        print('Computing whitening kernel and bias...')
        emb = []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids, attention_mask = unpack_tokenize(batch, tokenizer, device,
                                                        max_length=max_length)
            with torch.no_grad():
                batch_emb = self.forward(input_ids, attention_mask).cpu().numpy()
            emb.append(batch_emb)
            # if i == 1:  # sanity check
            #     break

        # > shape (80, 768)
        emb = np.concatenate(emb, axis=0)

        # > shape (80, 768)
        # emb = torch.cat(emb, dim=0)

        kernel, bias = compute_kernel_bias(emb, k=None)
        self.whiten_kernel = torch.tensor(kernel, dtype=torch.float32).to(device)
        self.whiten_bias = torch.tensor(bias, dtype=torch.float32).to(device)
        self.whiten = True


class ClassificationHead(nn.Module):
    def __init__(self, n_classes, emb_dim, inner_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, n_classes)
        # self.dropout = nn.Dropout(0.2)

        # self.linear1 = nn.Linear(emb_dim, 256)
        # self.linear2 = nn.Linear(256, 32)
        # self.linear3 = nn.Linear(32, n_classes)

    def forward(self, emb):
        output = self.linear1(emb)

        output = torch.tanh(output)
        # output = F.leaky_relu(output, negative_slope=0.1)

        # output = self.dropout(output)

        output = self.linear2(output)

        # output = self.linear1(emb)
        # output = torch.tanh(output)
        # output = self.linear2(output)
        # output = torch.tanh(output)
        # output = self.linear3(output)

        return output


class LabelAggregationModel(nn.Module):
    def __init__(self, target,
                 emb_model_name, freeze_enc_weights=False,
                 contr_pretr_id=None, random_seed=None,
                 n_classes=3, emb_dim=768, inner_dim=1024):
        super().__init__()
        self.target = target
        self.emb_model = EmbeddingModel(
            emb_model_name, n_classes, freeze_enc_weights, contr_pretr_id, random_seed)

        if self.target in (RILE, GALTAN):
            self.clf = ClassificationHead(n_classes, emb_dim, inner_dim)
        elif self.target == JOINT:
            self.clf_rile = ClassificationHead(n_classes, emb_dim, inner_dim)
            self.clf_galtan = ClassificationHead(n_classes, emb_dim, inner_dim)

    def forward(self, input_ids, attention_mask):
        emb_model_output = self.emb_model(input_ids, attention_mask)
        if self.target in (RILE, GALTAN):
            clf_output = self.clf(emb_model_output)
        elif self.target == JOINT:
            # same shape as labels: (batch size, 2, 3)
            clf_output = torch.stack((self.clf_rile(emb_model_output),
                                      self.clf_galtan(emb_model_output)), dim=1)
        return clf_output


# code from https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# mean pooling - take attention mask into account for correct averaging
def mean_pooling(emb_model_output, attention_mask):
    token_embeddings = emb_model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
