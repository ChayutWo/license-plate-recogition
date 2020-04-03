# Import module and packages
from torch import optim
from utils.variables import *

# This code is obtained from Pytorch tutorial of ECE deep learning class
def make_optimizer(optimizer_name, model, **kwargs):
    """
    optimizer_name = Adam/SGD
    lr = learning rate
    momentum, weight_decay: for SGD only
    """
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], momentum=kwargs['momentum'],
                              weight_decay=kwargs['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(scheduler_name, optimizer, **kwargs):
    """
    milestones (list) – List of epoch indices. Must be increasing.
    factor (python:float) – Multiplicative factor of learning rate decay. Default: 0.1
    """
    if scheduler_name == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs['milestones'], gamma=kwargs['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler