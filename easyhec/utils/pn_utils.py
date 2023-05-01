import math

import numpy as np
import torch
import trimesh
from trimesh.caching import TrackedArray

from easyhec.utils.os_utils import NotCheckedError


def norm(x, dim=None, keepdims=False):
    """
    compute L2 norm of a tensor or a numpy array
    :param x:
    :return:
    """
    # x, ord=None, axis=None, keepdims=False
    # input, p="fro", dim=None, keepdim=False, out=None, dtype=None
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x, axis=dim, keepdims=keepdims)
    elif isinstance(x, torch.Tensor):
        return torch.norm(x.float(), dim=dim, keepdim=keepdims)
    else:
        raise TypeError()


def to_array(x, dtype=float):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, list):
        return [to_array(a) for a in x]
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}
    elif isinstance(x, TrackedArray):
        return np.array(x)
    else:
        return x


def exp(x):
    if isinstance(x, np.ndarray):
        return np.exp(x)
    elif isinstance(x, torch.Tensor):
        return torch.exp(x)
    else:
        return math.exp(x)


def to_same_type(x, y, dtype=float):
    """
    return x with the same type of y
    :param x:
    :param y:
    :param dtype:
    :return:
    """
    if isinstance(y, np.ndarray):
        return to_array(x, dtype)
    elif isinstance(y, torch.Tensor):
        if isinstance(x, torch.Tensor):  # suppress warnings
            return x.to(device=y.device, dtype=dtype)
        else:
            return torch.tensor(x, device=y.device, dtype=dtype)
    else:
        raise TypeError()


def to_same_type_if_present(x, y, dtype=float):
    if x is None:
        return None
    else:
        return to_same_type(x, y, dtype=y.dtype)


def from_numpy_if_present(x, dtype=torch.float):
    if x is None:
        return None
    return torch.from_numpy(x).to(dtype=dtype)


def min_max(x, dim=None):
    if isinstance(x, np.ndarray):
        return x.min(axis=dim), x.max(axis=dim)
    elif isinstance(x, torch.Tensor):
        return x.min(dim=dim).values, x.max(dim=dim).values
    else:
        raise TypeError()


def ptp(x, dim=None):
    if isinstance(x, np.ndarray):
        return x.ptp(axis=dim)
    elif isinstance(x, torch.Tensor):
        return x.max(dim=dim).values - x.min(dim=dim).values
    else:
        raise TypeError()


def random_choice(x, size, dim=None, replace=True):
    if dim is None:
        assert len(x.shape) == 1
        n = x.shape[0]
        idxs = np.random.choice(n, size, replace)
        return x[idxs], idxs
    else:
        n = x.shape[dim]
        idxs = np.random.choice(n, size, replace)
        if isinstance(x, np.ndarray):
            swap_function = np.swapaxes
        elif isinstance(x, torch.Tensor):
            swap_function = torch.transpose
        else:
            raise TypeError()
        x_ = swap_function(x, 0, dim)
        x_ = x_[idxs]
        x_ = swap_function(x_, 0, dim)
        return x_, idxs


def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    # x_masked[mask == 0] = -30

    return torch.softmax(x_masked, **kwargs)


def as_tensor_if_present(data, dtype=None, device=None):
    if data is None:
        return None
    else:
        return torch.as_tensor(data, dtype=dtype, device=device)


def to_tensor_if_present(data, dtype=None, device=None):
    if data is None:
        return None
    else:
        if isinstance(data, torch.Tensor):
            return data.clone().to(dtype=dtype, device=device)
        else:
            return torch.tensor(data, dtype=dtype, device=device)


def to_tensor(data, dtype=None, device=None):
    if isinstance(data, torch.Tensor):
        return data.clone().to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


def to_tensor_if_present_and_not_ndarray(data, dtype=None, device=None):
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data
    else:
        if isinstance(data, torch.Tensor):
            return data.clone().to(dtype=dtype, device=device)
        else:
            return torch.tensor(data, dtype=dtype, device=device)


def numpy_if_present(data):
    if data is None:
        return None
    else:
        return to_array(data)


def to_device_if_present(data, device):
    if data is None:
        return None
    else:
        return data.to(device=device)


def clone_if_present(data):
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data.copy()
    elif isinstance(data, torch.Tensor):
        return torch.clone(data)
    elif isinstance(data, (int, float, str)):
        return data
    elif isinstance(data, dict):
        return {k: clone_if_present(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clone_if_present(i) for i in data]
    elif isinstance(data, trimesh.Trimesh):
        return data.copy()
    else:
        raise NotImplementedError()


def padded_stack(tensors, dim=0, *, out=None):
    # change to point tensors.
    # assert dim == 0
    max_dim = max(t.shape[dim] for t in tensors)
    padded_tensors = []
    for t in tensors:
        shape_to_pad = list(t.shape[0:dim]) + [max_dim - t.shape[dim]] + list(t.shape[dim + 1:])
        padded_tensors.append(
            torch.cat((t, torch.full(shape_to_pad, -1000, dtype=t.dtype, device=t.device)), dim=dim)
        )
    return torch.stack(padded_tensors, dim=dim, out=out)


def setdiff1d_pytorch(ar1, ar2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the unique values in `ar1` that are not in `ar2`.

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        Input comparison array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    setdiff1d : ndarray
        1D array of values in `ar1` that are not in `ar2`. The result
        is sorted when `assume_unique=False`, but otherwise only sorted
        if the input is sorted.

    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 2, 4, 1])
    >>> b = np.array([3, 4, 5, 6])
    >>> np.setdiff1d(a, b)
    array([1, 2])

    """
    if assume_unique:
        ar1 = ar1.reshape(-1)
    else:
        ar1 = torch.unique(ar1)
        ar2 = torch.unique(ar2)
    return ar1[in1d_pytorch(ar1, ar2, assume_unique=True, invert=True)]


def in1d_pytorch(ar1, ar2, assume_unique=False, invert=False):
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    We recommend using :func:`isin` instead of `in1d` for new code.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.

        .. versionadded:: 1.8.0

    Returns
    -------
    in1d : (M,) ndarray, bool
        The values `ar1[in1d]` are in `ar2`.

    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``np.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to an array, in those cases
    ``asarray(ar2)`` is an object array rather than the expected array of
    contained values.

    .. versionadded:: 1.4.0

    Examples
    --------
    >>> test = np.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = np.in1d(test, states)
    >>> mask
    array([ True, False,  True, False,  True])
    >>> test[mask]
    array([0, 2, 0])
    >>> mask = np.in1d(test, states, invert=True)
    >>> mask
    array([False,  True, False,  True, False])
    >>> test[mask]
    array([1, 5])
    """
    # Ravel both arrays, behavior for the first array could be different
    ar1 = ar1.reshape(-1)
    ar2 = ar2.reshape(-1)

    # Check if one of the arrays may contain arbitrary objects
    # contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145:

        if invert:
            mask = torch.ones(len(ar1), dtype=torch.bool, device=ar1.device)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = torch.zeros(len(ar1), dtype=torch.bool, device=ar1.device)
            for a in ar2:
                mask |= (ar1 == a)
        return mask
    print('large condition')
    raise NotCheckedError()
    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = torch.unique(ar1, return_inverse=True)
        ar2 = torch.unique(ar2)

    ar = torch.cat((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = torch.argsort(ar, )
    # order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = torch.cat((bool_ar, [invert]))
    ret = torch.empty(ar.shape, dtype=torch.bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return [clone(el) for el in x]
    elif isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    elif isinstance(x, (int, float, str)):
        return x
    else:
        raise TypeError()


def stack(l, dim=0):
    if len(l) == 0:
        return l
    if isinstance(l[0], np.ndarray):
        return np.stack(l, dim)
    else:
        return torch.stack(l, dim)


def floor(x):
    if isinstance(x, np.ndarray):
        return np.floor(x)
    elif isinstance(x, torch.Tensor):
        return torch.floor(x)
    else:
        return math.floor(x)


def ceil(x):
    if isinstance(x, np.ndarray):
        return np.ceil(x)
    elif isinstance(x, torch.Tensor):
        return torch.ceil(x)
    else:
        return math.ceil(x)


def intersect1d_pytorch(ar1, ar2, assume_unique=False, return_indices=False):
    if not assume_unique:
        if return_indices:
            ar1, inv_ind1 = torch.unique(ar1, return_inverse=True)
            perm = torch.arange(inv_ind1.size(0), dtype=inv_ind1.dtype, device=inv_ind1.device)
            inverse, perm = inv_ind1.flip([0]), perm.flip([0])
            ind1 = inverse.new_empty(ar1.size(0)).scatter_(0, inverse, perm)
            ar2, inv_ind2 = torch.unique(ar2, return_inverse=True)
            perm = torch.arange(inv_ind2.size(0), dtype=inv_ind2.dtype, device=inv_ind2.device)
            inverse, perm = inv_ind2.flip([0]), perm.flip([0])
            ind2 = inverse.new_empty(ar2.size(0)).scatter_(0, inverse, perm)
        else:
            ar1 = torch.unique(ar1)
            ar2 = torch.unique(ar2)
    else:
        ar1 = ar1.reshape(-1)
        ar2 = ar2.reshape(-1)

    aux = torch.cat((ar1, ar2))
    if return_indices:
        # aux_sort_indices = torch.argsort(aux)
        aux_sort_indices = torch.from_numpy(np.argsort(to_array(aux), kind='mergesort'))
        aux = aux[aux_sort_indices]
    else:
        aux, _ = aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.numel()
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d


#
# def torch_all_close(x, y, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
#     if isinstance(x, torch.Tensor):
#         return torch.allclose(x.detach().cpu(), y.detach().cpu(), rtol, atol, equal_nan)
#     elif isinstance(x,)

def numel(x):
    if isinstance(x, torch.Tensor):
        return x.numel()
    elif isinstance(x, np.ndarray):
        return x.size
    else:
        raise TypeError()
