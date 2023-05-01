from bisect import bisect_right
from typing import List

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class ExponentialScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, lrate_decay, last_epoch=-1, cfg=None):
        self.gamma = gamma
        self.lrate_decay = lrate_decay
        super().__init__(optimizer, last_epoch)
        self.cfg = cfg
        self.do_decay = [lr > cfg.solver.scheduler_decay_thresh for lr in self.base_lrs]  # todo:put in cfg

    def get_lr(self):
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            if hasattr(self, 'do_decay'):
                do_decay = self.do_decay[i]
            else:
                do_decay = True
            if do_decay:
                lrs.append(base_lr * self.gamma ** (self.last_epoch / self.lrate_decay))
            else:
                lrs.append(base_lr)
        # print(lrs[0])
        return lrs


class WarmupExponentialScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, gamma, lrate_decay, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        self.lrate_decay = lrate_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            lrs = [
                base_lr * self.last_epoch / self.warmup_iters
                * (self.gamma ** (self.last_epoch / self.lrate_decay))
                for base_lr in self.base_lrs
            ]
        else:
            lrs = [
                base_lr * (self.gamma ** (self.last_epoch / self.lrate_decay))
                for base_lr in self.base_lrs
            ]
        return lrs
