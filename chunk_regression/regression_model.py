import torch.nn as nn
from transformers import AutoModel, AutoConfig

from utils.utils import *


class EmbeddingModel(nn.Module):
    def __init__(self,
                 model_name,
                 freeze_enc_weights,  # set to True to only train rgr head(s)
                 dropout_p,
                 ):
        super().__init__()
        self.model_name = model_name
        self.model_name_short = get_model_name_short(model_name)

        if self.model_name_short == 'ModernBERT':
            config = AutoConfig.from_pretrained(self.model_name)
            config.reference_compile = False  # compatibility with torch.compile

            if dropout_p is not None:
                config.attention_dropout = dropout_p
                config.classifier_dropout = dropout_p
                config.embedding_dropout = dropout_p
                config.mlp_dropout = dropout_p

            self.model = AutoModel.from_pretrained(
                self.model_name,
                config=config,
                # reference_compile=False  # compatibility with torch.compile
            )
        else:  # BigBird
            self.model = AutoModel.from_pretrained(self.model_name)

        # freeze weights
        if freeze_enc_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # chunk emb is last hidden state repr of CLS token
        output = output.last_hidden_state[:, 0, :]
        return output


class RegressionHead(nn.Module):
    def __init__(self, emb_dim, inner_dim):
        super().__init__()

        self.linear1 = nn.Linear(emb_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)

        # map output to [-1, 1]
        x = torch.tanh(x)

        return x


class RegressionModel(nn.Module):
    def __init__(self, target,
                 emb_model_name, freeze_enc_weights=False,
                 emb_dim=768, inner_dim=1024, dropout_p=0.0
                 ):
        super().__init__()

        self.target = target
        self.emb_model = EmbeddingModel(emb_model_name, freeze_enc_weights, dropout_p)

        if self.target in (RILE, GALTAN):
            self.rgr = RegressionHead(emb_dim, inner_dim)
        elif self.target == JOINT:
            self.rgr_rile = RegressionHead(emb_dim, inner_dim)
            self.rgr_galtan = RegressionHead(emb_dim, inner_dim)

    def forward(self, input_ids, attention_mask):
        emb_model_output = self.emb_model(input_ids, attention_mask)

        if self.target in (RILE, GALTAN):
            rgr_output = self.rgr(emb_model_output)
        elif self.target == JOINT:
            rgr_output = torch.stack((self.rgr_rile(emb_model_output),
                                      self.rgr_galtan(emb_model_output)), dim=1)
        del emb_model_output
        return rgr_output
