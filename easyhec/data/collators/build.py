from easyhec.registry import BATCH_COLLATORS
from .default_batch_collator import DefaultBatchCollator
from .extended_batch_collator import ExtendedBatchCollator


@BATCH_COLLATORS.register('DefaultBatchCollator')
def build_default_batch_colloator(cfg):
    return DefaultBatchCollator()


@BATCH_COLLATORS.register('ExtendedBatchCollator')
def build(cfg):
    return ExtendedBatchCollator()


def make_batch_collator(cfg):
    return BATCH_COLLATORS[cfg.dataloader.collator](cfg)
