import re
import random
from functools import wraps

import numpy as np
import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def dict_from_var_names(*args):
    return dict(((k, eval(k)) for k in args))


def find_all_str_matches(s, subs):
    """
    Finds "whole words only" matches of expressions in a given string
    :param s: The string in which to search for the given expressions in subs.
    :param subs: A list of expressions to search for in s.
    :return: All matches as spans (from, to) in s.
    """
    spans = []
    for p in subs:
        matches = re.finditer(p, s, re.IGNORECASE | re.MULTILINE)
        for m in matches:
            spans.append(m.span())
    return spans


def is_substring_in_group(c, group):
    for substring in group:
        if substring in c:
            return True
    return False


def match_sorted_array_to_another_sorted_array(I, J, matching_function=None):
    if matching_function is None:
        def match(I, i, J, j):
            return I[i] >= J[j]

        matching_function = match
    result = np.empty(len(I))
    i = 0
    j = 0
    N_I = len(I)
    N_J = len(J)
    while True:
        if j >= N_J:
            break
        if i >= N_I:
            break
        match = matching_function(I, i, J, j)
        if match > 0:
            j += 1
        elif match < 0:
            i += 1
        else:
            result[i] = j - 1

    return result


def try_except(success, failure, *exceptions):
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure


def robust_div(numerator, denominator, default_val=np.nan):
    return numerator / denominator if denominator is not None and denominator != 0 else default_val


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
        except:
            pass


def dict_to_device(batch, device, filter_props=None):
    if filter_props is None:
        return {k: v.to(device) for k, v in batch.items()}
    else:
        return {k: v.to(device) for k, v in batch.items() if k in filter_props}


def with_device(device="cpu"):
    def decorator(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            cur_devices = {
                o: v.device
                for o, v in kwargs.items()
                if hasattr(v, "device")
            }
            result = None
            try:
                for o, v in kwargs.items():
                    if not hasattr(v, "device"): continue
                    v = v.to(device)
                result = fn(*args, **kwargs)
            finally:
                for o, v in kwargs.items():
                    if not hasattr(v, "device"): continue
                    v = v.to(cur_devices[o])
            return result
        return wrapped_fn
    return decorator

if __name__ == '__main__':
    I = [1, 1, 1, 1, 2, 2.1, 2.2, 2.7, 3, 3.6, 7, 7.9, 12]
    J = [0, 1, 2, 3, 4, 6, 7, 8, 9]

    result = match_sorted_array_to_another_sorted_array(I, J)
    print(result)
