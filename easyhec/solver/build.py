import numpy as np
import torch
from dl_ext.pytorch_ext import OneCycleScheduler
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from .lr_scheduler import WarmupMultiStepLR, WarmupCosineLR, ExponentialScheduler, WarmupExponentialScheduler


# from ..modeling.models.barf.barf import BaRF

def make_optimizer(cfg, model):
    params = []
    lr = cfg.solver.max_lr
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = cfg.solver.weight_decay
        if "bias" in key:
            lr = cfg.solver.max_lr * cfg.solver.bias_lr_factor
            weight_decay = cfg.solver.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.solver.optimizer == 'SGD':
        optimizer = SGD(params, lr, momentum=cfg.solver.momentum)
    elif cfg.solver.optimizer == 'Adam':
        optimizer = Adam(params, lr)
    else:
        raise NotImplementedError()
    return optimizer


def make_lr_scheduler(cfg, optimizer, max_iter):
    if cfg.solver.scheduler == 'WarmupMultiStepLR':
        return WarmupMultiStepLR(
            optimizer,
            cfg.solver.steps,
            cfg.solver.gamma,
            warmup_factor=cfg.solver.warmup_factor,
            warmup_iters=cfg.solver.warmup_iters,
            warmup_method=cfg.solver.warmup_method,
        )
    elif cfg.solver.scheduler == 'OneCycleScheduler':
        return OneCycleScheduler(
            optimizer,
            cfg.solver.max_lr,
            max_iter
        )
    elif cfg.solver.scheduler == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif cfg.solver.scheduler == 'ConstantScheduler':
        from dl_ext.pytorch_ext.optim import ConstantScheduler
        return ConstantScheduler(optimizer)
    elif cfg.solver.scheduler == 'ExponentialScheduler':
        return ExponentialScheduler(optimizer, cfg.solver.gamma, cfg.solver.lrate_decay, cfg=cfg)
    elif cfg.solver.scheduler == 'WarmupExponentialScheduler':
        return WarmupExponentialScheduler(optimizer,
                                          cfg.solver.warmup_iters,
                                          cfg.solver.gamma,
                                          cfg.solver.lrate_decay)
    elif cfg.solver.scheduler == 'ExponentialStep':
        scheduler = LambdaLR(
            optimizer,
            ExponentialSchedulerLambda(
                total_steps=cfg.solver.num_iters,
                min_factor=cfg.solver.min_factor
            )
        )
        return scheduler
    elif cfg.solver.scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, cfg.solver.num_epochs, 1e-8)
    else:
        raise NotImplementedError()


def ExponentialSchedulerLambda(total_steps, min_factor=0.1):
    assert 0 <= min_factor < 1

    def lambda_fn(epoch):
        t = np.clip(epoch / total_steps, 0, 1)
        learning_factor = np.exp(t * np.log(min_factor))
        return learning_factor

    return lambda_fn
