import numpy as np
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1, verbose=False, summary=None):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.summary = summary

        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr
        )

        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / float(self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return self.cosine_scheduler.get_last_lr()

    def step(self, epoch=None):
        super(WarmupCosineScheduler, self).step(epoch)

        if self.last_epoch >= self.warmup_steps:
            self.cosine_scheduler.step()

        if self.summary is not None:
            lr = self.get_last_lr()
            if isinstance(lr, list):
                lr = lr[0]  # Extract scalar if list

            self.summary.add_scalar("learning_rate", np.array(lr), self.last_epoch + 1)
