from torch.utils.data._utils.collate import default_collate


class DefaultBatchCollator(object):
    def __call__(self, batch):
        return default_collate(batch)
