# -*- coding: utf-8 -*-

from easydict import EasyDict as edict

config = edict()
config.max_seq_len = 32
config.num_hidden_layers = 4
config.label_num = 4
config.batch_size = 18
config.model_type = 'bert_roberta'