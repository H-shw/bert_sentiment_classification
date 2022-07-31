# -- coding:UTF-8 --

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW
from utils import dev_eval, soft_cross_entropy
from model import bert_classficator, student_bert_classfier
from data import init_train_dataset
from sklearn.metrics import f1_score


def stu_train_start(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    # set up the model
    teacher_model = bert_classficator(config)
    student_model = student_bert_classfier(config)
    teacher_model.bert.config.output_hidden_states = True
    teacher_model.bert.config.output_attentions = True
    student_model.bert.config.output_attentions = True
    student_model.bert.config.output_hidden_states = True
    train_loader, dev_loader, test_loader = init_train_dataset(config)
    train(teacher_model, student_model, train_loader, dev_loader, device, config)


def train(teacher_model, student_model, train_loader, dev_loader, device, config):
    teacher_model.to(device)
    student_model.to(device)

    student_model.train()
    hard_criterion = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    param_optimizer = list(student_model.named_parameters())
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
            tea_res, tea_hidden_state, tea_attention = teacher_model(input_ids, token_type_ids, attention_mask)
            stu_res, stu_hidden_state, stu_attention = student_model(input_ids, token_type_ids, attention_mask)

            #           print(tea_hidden_state[2])
            #           print(tea_hidden_state[2].size())
            #           print(len(tea_hidden_state))
            #           print(tea_attention[2])
            #           print(tea_attention[2].size())
            #           print(len(tea_attention))
            #           print(len(stu_attention))

            att_loss = 0.
            rep_loss = 0.

            tea_attention = [x.detach() for x in tea_attention]
            tea_hidden_state = [x.detach() for x in tea_hidden_state]

            tea_layer_num = len(tea_attention)
            stu_layer_num = len(stu_attention)

            assert tea_layer_num % stu_layer_num == 0

            layers_per_block = int(tea_layer_num / stu_layer_num)
            new_teacher_atts = [tea_attention[i * layers_per_block + layers_per_block - 1]
                                for i in range(stu_layer_num)]

            for student_att, teacher_att in zip(stu_attention, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)
                att_loss += loss_mse(student_att, teacher_att)

            new_teacher_reps = [tea_hidden_state[i * layers_per_block] for i in range(stu_layer_num + 1)]
            new_student_reps = stu_hidden_state

            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                rep_loss += loss_mse(student_rep, teacher_rep)

            # loss
            soft_loss = att_loss + rep_loss
            hard_loss = hard_criterion(stu_res, labels)
            loss = config.softloss_factor * soft_loss + (1 - config.softloss_factor) * hard_loss

            cls_loss = soft_cross_entropy(stu_res / config.temperature, tea_res / config.temperature)

            # loss = criterion(out_put, labels)
            # return max value and index
            _, predict = torch.max(stu_res.data, 1)
            correct += (predict == labels).sum().item()
            f1 = f1_score(labels.cpu().numpy(), predict.cpu().numpy(), average='micro')
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            # print information
            if (step + 1) % 20 == 0:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,train_f1{:.6f} ,loss:{:.6f}".format(epoch + 1,
                                                                                                          total_epochs,
                                                                                                          step + 1,
                                                                                                          len(train_loader),
                                                                                                          train_acc * 100,
                                                                                                          f1,
                                                                                                          loss.item()))
            # eval with dev dataset : 200
            if (step + 1) % 200 == 0:
                train_acc = correct / total
                # evaluate and save
                acc, f1 = dev_eval(student_model, dev_loader, device)
                if bestAcc < acc:
                    bestAcc = acc
                    # save the model
                    path = config.save_path
                    torch.save(student_model, path)
                print(
                    "DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,f1{:3f} ,dev_acc{:.6f} %,loss:{:.6f}".format(
                        epoch + 1, total_epochs, step + 1, len(train_loader), train_acc * 100, bestAcc * 100, f1,
                        acc * 100,
                        loss.item()))
        scheduler.step(bestAcc)
