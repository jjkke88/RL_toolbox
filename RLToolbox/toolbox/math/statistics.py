import numpy as np

def min_max_norm(array_target):
    min_val = np.min(array_target)
    max_val = np.max(array_target)
    if min_val != max_val:
        result = []
        for val in array_target:
            val = (val - min_val) / (max_val - min_val)
            result.append(val)
        return np.array(result)
    else:
        return array_target / array_target.shape[0]