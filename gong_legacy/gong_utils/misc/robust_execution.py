import numpy as np

def try_except(success, failure, *exceptions):
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure

def robust_div(numerator, denominator, default_val=np.nan):
    return numerator / denominator if denominator is not None and denominator != 0 else default_val
