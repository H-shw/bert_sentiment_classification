# -- coding:UTF-8 --

import torch.nn.functional as F
import torch
from data import init_test_dataset, init_train_dataset


# test dataset evaluate
def test_start(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = torch.load(config.save_path)
    train_loader, dev_loader, test_loader = init_train_dataset(config)
    test_eval(model, test_loader, device)


# predict labels
def predict_start(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = torch.load(config.save_path)
    predict_loader = init_test_dataset(config)
    predict(model, predict_loader, device, config)


# predict labels
def predict(model, test_loader, device, config):
    model.to(device)
    model.eval()

    res = {}
    with torch.no_grad():

        for step, (input_ids, token_type_ids, attention_mask, ids) in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, ids = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), ids.to(device)
            out_put = model(input_ids, token_type_ids, attention_mask)

            _, predict = torch.max(out_put.data, 1)

            for x1, x2 in zip(ids, predict):
                res[x1] = x2 - 1
    # 写入预测结果
    f = open(config.res_path, 'w', encoding='utf8')
    f.write('id,y \n')
    for (key, value) in res.items():
        f.write(f'{key},{value}\n')
    f.close()


# test dataset evaluate
def test_eval(model, test_loader, device):
    model.to(device)
    model.eval()
    predicts = []
    predict_probs = []
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels, ids) in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, labels, ids = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device), ids.to(device)
            out_put = model(input_ids, token_type_ids, attention_mask)

            _, predict = torch.max(out_put.data, 1)

            # result (list) 和标签比较
            pre_numpy = predict.cpu().numpy().tolist()
            predicts.extend(pre_numpy)
            probs = F.softmax(out_put).detach().cpu().numpy().tolist()
            predict_probs.extend(probs)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        print('predict_Accuracy : {} %'.format(100 * res))
        # 返回预测结果和预测的概率
        return predicts, predict_probs
