import torch
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def weight_init(self):
        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        for name, param in self.config:
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'relu':
                x = torch.relu(x)
            elif name is 'tanh':
                x = torch.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars