import numpy as np

from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR


class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1, verbose=False, summary=None):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cosine_scheduler = None
        self.summary = summary
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            return [
                base_lr * (self.last_epoch + 1) / float(self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay phase
            if self.cosine_scheduler is None:
                self.cosine_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.total_steps - self.warmup_steps,
                    eta_min=self.min_lr
                )
                self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_steps

            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        # Advance the scheduler's internal epoch counter
        super(WarmupCosineScheduler, self).step(epoch)
        # Also step the cosine scheduler after the warmup phase
        if self.last_epoch >= self.warmup_steps:
            self.cosine_scheduler.step(epoch)

        if self.summary is not None:
            #lr =  self.get_lr()
            #if type(lr) is list:
            #    lr = lr[0]

            self.summary.add_scalar("learning_rate", np.array(self.get_lr()), self.last_epoch + 1)