import numpy as np
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss
from itertools import product


class ReweightedUnivariateLoss(nn.Module):
    def __init__(self,mode="u"):
        super().__init__()
        self.mode = mode
        self.bce = BCEWithLogitsLoss()

    def forward(self, pred_y, true_y):
        # pred_y: tensor, shape: (N,C) N is the batch size and C is the class numbers
        # true_y: tensor, shape: (N,C) N is the batch size and C is the class numbers
        device = true_y.device
        losses = []
        K = true_y.shape[-1]
        n = true_y.shape[0]
        S_pos = torch.where(true_y == 1)
        S_neg = torch.where(true_y == 0)

        # univariate loss
        if self.mode=="u":
            l = self.bce(pred_y,true_y)
            
        # reweighted univariate loss
        elif self.mode=="ru":
            for i in range(K):
                S_i_pos = torch.where(true_y[:,i] == 1)[0]
                S_i_neg = torch.where(true_y[:,i] == 0)[0]
                e1 = self.loss(pred_y[S_i_pos,i]) / len(S_pos[0])
                e2 = self.loss(-pred_y[S_i_neg, i]) / len(S_neg[0])
                re = torch.add(torch.sum(e2),torch.sum(e1))
                losses.append(re)
            l = torch.sum(torch.stack(losses))

        # pair wise loss
        elif self.mode=="pairwise":
            for i in range(K):
                for j in range(K):
                    S_i_pos = torch.stack((S_pos[0][torch.where(S_pos[1] == i)[0]],S_pos[1][torch.where(S_pos[1] == i)[0]]),dim=1).tolist()
                    S_j_neg = torch.stack((S_neg[0][torch.where(S_neg[1] == j)[0]],S_neg[1][torch.where(S_neg[1] == j)[0]]),dim=1).tolist()
                    elems = product(*[S_i_pos,S_j_neg])
                    for elem in elems:
                        losses.append(self.loss(pred_y[elem[0][0],elem[0][1]] - pred_y[elem[1][0],elem[1][1]]))
            l = torch.sum(torch.stack(losses)) / torch.tensor(len(S_pos[0])*len(S_neg[0])).to(device)

        # the faster implementation of pairwise loss
        elif self.mode=="pairwise2":
            for i in range(K):
                for j in range(K):
                    mask_i_pos = true_y[:,i]==1
                    mask_j_neg = true_y[:,j]==0
                    p = torch.masked_select(pred_y[:,i],mask=mask_i_pos)
                    q = torch.masked_select(pred_y[:, j], mask=mask_j_neg)
                    prod = torch.cartesian_prod(p,q)
                    losses.append(torch.sum(self.loss(prod[:,0]-prod[:,1])))
            l = torch.sum(torch.stack(losses)) / torch.tensor(len(S_pos[0])*len(S_neg[0])).to(device)
        else:
            raise NotImplementedError

        return l

    def loss(self, ele):
        return torch.log(torch.add(1, torch.exp(-ele)))
