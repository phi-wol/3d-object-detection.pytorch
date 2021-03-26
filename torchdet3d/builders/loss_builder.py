import torch
from torchdet3d.losses import DiagLoss, ADD_loss

__AVAILABLE_LOSS = ['smoothl1', 'cross_entropy', 'diag_loss', 'mse', 'add_loss']


def build_loss(cfg):
    "build losses in right order"
    regress_criterions = []
    class_criterions = []
    for loss_name in cfg.loss.names:
        assert loss_name in __AVAILABLE_LOSS
        if loss_name == 'cross_entropy':
            class_criterions.append(torch.nn.CrossEntropyLoss())
        elif loss_name == 'smoothl1':
            regress_criterions.append(torch.nn.SmoothL1Loss(reduction='mean', beta=cfg.loss.smoothl1_beta))
        elif loss_name == 'mse':
            regress_criterions.append(torch.nn.MSELoss())
        elif loss_name == 'add_loss':
            regress_criterions.append(ADD_loss())
        elif loss_name == 'diag_loss':
            regress_criterions.append(DiagLoss())

    return regress_criterions, class_criterions
