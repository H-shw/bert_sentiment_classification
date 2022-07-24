# -- coding:UTF-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# bert encoder + linear classifier
class bert_classficator(nn.Module):
    def __init__(self, config):
        super(bert_classficator, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, text, token, mask):
        # batch = text.size()[0]
        y = self.bert.forward(text, token, mask)
        y = self.fc(y[1])
        return y
