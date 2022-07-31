# -- coding:UTF-8 --

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW
from utils import dev_eval
from model import bert_classficator
from data import init_train_dataset


def train_start(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    # set up the model
    model = bert_classficator(config)
    train_loader, dev_loader, test_loader = init_train_dataset(config)
    train(model, train_loader, dev_loader, device, config)


def train(model, train_loader, dev_loader, device, config):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_params = {'lr': config.lr, 'eps': config.eps, 'correct_bias': config.correct_bias}
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
                                  threshold=0.0001, eps=1e-08)
    # t_total = len(train_loader)

    total_epochs = config.epoch
    bestAcc = 0
    correct = 0
    total = 0

    print('Training & verification :')

    for epoch in range(total_epochs):
        for step, (input_ids, token_type_ids, attention_mask, labels, ids) in enumerate(train_loader):
            # dataloder
            input_ids, token_type_ids, attention_mask, labels, ids = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device), ids.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # model
            out_put = model(input_ids, token_type_ids, attention_mask)
            # loss
            loss = criterion(out_put, labels)
            # return max value and index
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            # print information
            if (step + 1) % 2 == 0:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs,
                                                                                          step + 1, len(train_loader),
                                                                                          train_acc * 100, loss.item()))
            # eval with dev dataset : 200
            if (step + 1) % 200 == 0:
                train_acc = correct / total
                # evaluate and save
                acc,f1 = dev_eval(model, dev_loader, device)
                if bestAcc < acc:
                    bestAcc = acc
                    # save the model
                    path = config.save_path
                    torch.save(model, path)
                print(
                    "DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,f1{:3f} ,dev_acc{:.6f} %,loss:{:.6f}".format(
                        epoch + 1, total_epochs, step + 1, len(train_loader), train_acc * 100, bestAcc * 100, f1,
                        acc * 100,
                        loss.item()))
        scheduler.step(bestAcc)
