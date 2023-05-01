from enum import IntEnum

import torch


def to_cuda(x):
    if hasattr(x, 'cuda'):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    elif isinstance(x, (str, int, float)):
        return x
    elif hasattr(x, 'to'):
        return x.to('cuda')
    else:
        raise NotImplementedError()


def to_cpu(x):
    if x is None:
        return None
    elif hasattr(x, 'cpu'):
        return x.cpu()
    elif hasattr(x, 'to'):
        return x.to(device='cpu')
    elif isinstance(x, (list, tuple)):
        return [to_cpu(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    elif isinstance(x, (str, int, float)):
        return x
    elif hasattr(x, 'to'):
        return x.to('cpu')
    else:
        raise RuntimeError(f'x:{type(x)} is not supported for to_cpu().')


def batch_gpu(batch):
    x, y = batch
    # print_with_rank('before to cuda x', x)
    x_cuda = to_cuda(x)
    # print_with_rank('after to cuda x')
    y_cuda = to_cuda(y)
    # print_with_rank('after to cuda y')
    return x_cuda, y_cuda


def format_time(t_):
    t = int(t_)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    if h != 0:
        return f'{h}:{m:02d}:{s:02d}'
    else:
        if m != 0:
            return f'{m:02d}:{s:02d}'
        else:
            return f'{t_:.2f}s'


class TrainerState(IntEnum):
    BASE = 1
    PARALLEL = 2
    DISTRIBUTEDPARALLEL = 3


def split_list(vals, skip_start: int, skip_end: int):
    return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]


def cat_outputs(outputs):
    # pdb.set_trace()
    if isinstance(outputs[0], torch.Tensor):
        # if outputs[0].ndim == 0:
        return torch.stack(outputs)
        # else:
        #     return torch.cat(outputs)
    # elif isinstance(outputs[0], ts.SparseTensor):
    #     ts.cat(outputs)
    elif isinstance(outputs[0], dict):
        outs = {}
        for k in outputs[0].keys():
            outs[k] = cat_outputs([o[k] for o in outputs])
        return outs
    else:
        return outputs
