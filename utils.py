# -- coding:UTF-8 --

import torch
from tqdm import tqdm

# calculate the accruacy on valid dataset
def dev_eval(model, dev_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels, ids) in tqdm(enumerate(dev_loader),
                                                                                   desc='Dev Itreation:'):
            input_ids, token_type_ids, attention_mask, labels, ids = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device), ids.to(device)
            out_put = model(input_ids, token_type_ids, attention_mask)
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        return res
