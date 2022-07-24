# -- coding:UTF-8 --
import csv
import os
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch

# bert Tokenizer get input_ids,token_type_ids,attention_mask
def encoder(max_len, vocab_path, text_list):
    # 将 text_list embedding 成 bert 模型可用的输入形式
    # 加载分词模型
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
        # 返回的类型为pytorch tensor
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask


def load_data(path, config, mode):
    # csv -> dataset
    # [] -> tenser
    csvFileObj = open(path)
    readerObj = csv.reader(csvFileObj)
    text_list = []
    labels = []
    ids = []
    # train/valid/test with labels
    if mode != 'predict':
        for row in readerObj:
            # 跳过表头
            if readerObj.line_num == 1:
                continue
            label = int(row[2])+1
            text = row[1].replace(' ', '')
            id = int(row[0])
            text_list.append(text)
            labels.append(label)
            ids.append(id)
        # 调用encoder函数，获得预训练模型的三种输入形式
        input_ids, token_type_ids, attention_mask = encoder(max_len=150, vocab_path=config.vocab_path,
                                                            text_list=text_list)
        labels = torch.tensor(labels)
        ids = torch.tensor(ids)

        # 将encoder的返回值以及label封装为Tensor的形式
        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, ids)
    else:
        # predict without labels
        for row in readerObj:
            # 跳过表头
            if readerObj.line_num == 1:
                continue
            text = row[1].replace(' ', '')
            id = int(row[0])
            text_list.append(text)
            ids.append(id)
        # 调用encoder函数，获得预训练模型的三种输入形式
        input_ids, token_type_ids, attention_mask = encoder(max_len=150, vocab_path=config.vocab_path,
                                                            text_list=text_list)
        ids = torch.tensor(ids)
        # 将encoder的返回值以及label封装为Tensor的形式
        data = TensorDataset(input_ids, token_type_ids, attention_mask, ids)
    return data


def init_train_dataset(config):
    # return train/dev/test dataloder

    batch_size = config.batch_size

    # 调用load_data函数，将数据加载为Tensor形式
    train_data = load_data(config.train_path, mode='train', config=config)
    dev_data = load_data(config.valid_path, mode='train', config=config)
    test_data = load_data(config.test_path, mode='train', config=config)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=config.shuffle,num_workers=config.reader_num)
    valid_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=config.shuffle,num_workers=config.reader_num)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=config.shuffle,num_workers=config.reader_num)

    return train_loader, valid_loader, test_loader


def init_test_dataset(config):
    # return predict dataloder
    batch_size = config.batch_size
    predict_data = load_data(config.test, mode='predict', config=config)
    predict_loader = DataLoader(dataset=predict_data, batch_size=batch_size, shuffle=False)
    return predict_loader
