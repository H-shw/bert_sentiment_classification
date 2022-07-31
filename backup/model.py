# -- coding:UTF-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


# bert encoder + linear classifier
class bert_classficator(nn.Module):
    def __init__(self, config):
        super(bert_classficator, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, text, token, mask):
        # batch = text.size()[0]
        y = self.bert.forward(text, token, mask)
        res = self.fc(y[1])
        return res, y


class student_bert_classfier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conf = BertConfig(vocab_size=config.stu_vocab_size, hidden_size=config.stu_hidden_size,
                               num_hidden_layers=config.stu_hidden_layers)
        self.bert = BertModel(self.conf)

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.stu_hidden_size, config.class_num)

    def forward(self, text, token, mask):
        # batch = text.size()[0]
        y = self.bert.forward(text, token, mask)

        res = self.fc(y[1])
        return res, y


class dim_converter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.stu_hidden_size, config.tea_hidden_size)

    def forward(self, x):
        return self.fc(x)
