# import sys
# sys.path.append('../')

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from gnn_layer import Learner
from sklearn import metrics
from chemical import Chemical
from scaling_sgc import Scaling
from translation_sgc import Translation


class Meta(nn.Module):
    def __init__(self, config, config_chemi, config_scal, config_trans, args, num_attri, label_dim):
        super(Meta, self).__init__()
        self.lamda = args.lamda
        self.chemical_lr = args.chemical_lr
        self.scaling_lr = args.scaling_lr
        self.translation_lr = args.translation_lr
        self.meta_lr = args.meta_lr

        self.net = Learner(config)
        self.chemical = Chemical(config_chemi)
        self.scaling = Scaling(config_scal, args, num_attri, label_dim)
        self.translation = Translation(config_trans, args, num_attri, label_dim)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.chemi_optim = optim.Adam(self.chemical.parameters(), lr=self.chemical_lr)
        self.scaling_optim = optim.Adam(self.scaling.parameters(), lr=self.scaling_lr)
        self.translation_optim = optim.Adam(self.translation.parameters(), lr=self.translation_lr)
        self.dataset = args.dataset

    def forward(self, x_spt, y_spt, x_qry, y_qry, chemical_bond_vectors, l2_coef,
                update_step, len_graphs_index, batch_size,
                dataset, train, training, epoch):
        training = training
        batch_size = batch_size
        task_num = len(x_spt)
        update_step = update_step
        Losses_q = [0 for _ in range(update_step + 1)]
        accs = 0
        update_lr = self.lamda
        for j in range(int(task_num / batch_size) if task_num % batch_size == 0 else int(task_num / batch_size) + 1):
            start_idx = j * batch_size
            end_idx = min(start_idx + batch_size, task_num)
            losses_q = [0 for _ in range(update_step + 1)]
            for i in range(start_idx, end_idx):
                chemical = self.chemical(chemical_bond_vectors[i])
                scaling = self.scaling(chemical)
                translation = self.translation(chemical)
                adapted_prior = []
                for s in range(len(scaling)):
                    adapted_prior.append(torch.mul(self.net.parameters()[s], (scaling[s] + 1)) + translation[s])
                logits = self.net(x_spt[i], adapted_prior)
                if self.dataset == 'Cuneiform':
                    loss = torch.nn.BCEWithLogitsLoss()
                    loss = loss(logits, y_spt[i])
                else:
                    loss = torch.nn.functional.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, adapted_prior)
                fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, adapted_prior)))

                if update_step == 1:
                    logits_q = self.net(x_qry[i], fast_weights)
                    if self.dataset == 'Cuneiform':
                        loss_q = torch.nn.BCEWithLogitsLoss()
                        loss_q = loss_q(logits_q, y_qry[i])
                    else:
                        loss_q = F.cross_entropy(logits_q, y_qry[i])
                    l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                    l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                    l2_loss = l2_loss * l2_coef
                    losses_q[1] += (loss_q + l2_loss)
                    Losses_q[1] += (loss_q + l2_loss)
                else:
                    for k in range(1, update_step):
                        logits = self.net(x_spt[i], fast_weights)
                        if self.dataset == 'Cuneiform':
                            loss = torch.nn.BCEWithLogitsLoss()
                            loss = loss(logits, y_spt[i])
                        else:
                            loss = F.cross_entropy(logits, y_spt[i])
                        grad = torch.autograd.grad(loss, fast_weights)
                        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, fast_weights)))
                        if k == update_step - 1:
                            logits_q = self.net(x_qry[i], fast_weights)
                            if self.dataset == 'Cuneiform':
                                loss_q = torch.nn.BCEWithLogitsLoss()
                                loss_q = loss_q(logits_q, y_qry[i])
                            else:
                                loss_q = F.cross_entropy(logits_q, y_qry[i])
                            l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                            l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                            l2_loss = l2_loss * l2_coef
                            losses_q[k + 1] += (loss_q + l2_loss)
                            Losses_q[k + 1] += (loss_q + l2_loss)
                with torch.no_grad():
                    if self.dataset == 'Cuneiform':
                        pred_q = torch.sigmoid(logits_q)
                        pred_q = torch.round(pred_q)
                        y_true = []
                        y_pred = []
                        for m in range(len(y_qry[i])):
                            for n in range(7):
                                y_true.append((y_qry[i])[m, n].cpu())
                        for m in range(len(pred_q)):
                            for n in range(7):
                                y_pred.append(pred_q[m, n].cpu())
                        accs += metrics.accuracy_score(y_true, y_pred, normalize=True)
                    else:
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        accs += metrics.accuracy_score(y_qry[i].cpu(), pred_q.cpu(), normalize=True)
            loss_q = losses_q[-1] / batch_size
            if training == True:
                self.meta_optim.zero_grad()
                self.chemi_optim.zero_grad()
                self.scaling_optim.zero_grad()
                self.translation_optim.zero_grad()
                loss_q.backward()
                self.meta_optim.step()
                self.chemi_optim.step()
                self.scaling_optim.step()
                self.translation_optim.step()
        acc = accs / task_num
        Loss_q = Losses_q[-1] / task_num
        return acc, Loss_q

