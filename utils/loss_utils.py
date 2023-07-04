# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import one_hot
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()

        logpt = F.log_softmax(input,dim = -1)
      
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class auxiliary_loss(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.deta_const:
            self.deta = 0.2
        else:
            self.deta = nn.Parameter(torch.Tensor([0.2]).float())
    def forward(self,y_pred,labels):
        y_pred = y_pred.reshape(1,-1)
        labels = labels.reshape(1,-1)
        pos_num = torch.sum(labels)
        neg_num = len(labels)-pos_num
        if len(labels) > neg_num > 0:
            pos_pred = y_pred[labels.bool()]
            neg_pred = y_pred[(1-labels).bool()]
            loss = self.deta*torch.sum(neg_pred - pos_pred.reshape(-1,1))/(pos_num*neg_num)
        else: 
            loss = 0
        return loss 
class PolyLoss_FL(torch.nn.Module):
    """
    Implementation of poly loss FOR FL.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    """

    def __init__(self, num_classes=2, epsilon=1.0,gamma = 2.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = FocalLoss(gamma = 2.0,size_average = False)
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, output, target):
        fl = self.criterion(output, target)
        pt = one_hot(target.long(), num_classes=self.num_classes) * self.softmax(output)
        return (fl + self.epsilon * torch.pow(1.0 - pt.sum(dim=-1),self.gamma + 1)).mean()


class PolyLoss_CE(torch.nn.Module):
    """
    Implementation of poly loss for CE.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    """

    def __init__(self, num_classes=2, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes

    def forward(self, output, target):
        ce = self.criterion(output, target.long())
        pt = one_hot(target.long(), num_classes=self.num_classes) * self.softmax(output)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()
if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
    