# -- coding:UTF-8 --

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


# calculate the accruacy on valid dataset
def dev_eval(model, dev_loader, device):
    model.to(device)
    model.eval()
    f1 = 0.
    count = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels, ids) in enumerate(dev_loader):
            input_ids, token_type_ids, attention_mask, labels, ids = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device), ids.to(device)
            out_put,_ = model(input_ids, token_type_ids, attention_mask)
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            f1 += f1_score(labels.cpu().numpy(), predict.cpu().numpy(),average='micro')
            count += 1
            total += labels.size(0)
        res = correct / total
        f1 = f1 / count
        return res, f1


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()