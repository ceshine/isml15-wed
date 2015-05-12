"""The metrics module implements functions assessing prediction error for specific purposes."""

import numpy as np


def trapz(x, y):
    """Trapezoidal rule for integrating
    the curve defined by x-y pairs.
    Assume x and y are in the range [0,1]
    """
    assert len(x) == len(y), 'x and y need to be of same length'
    x = np.concatenate([x, np.array([0.0, 1.0])])
    y = np.concatenate([y, np.array([0.0, 1.0])])
    sort_idx = np.argsort(x)
    sx = x[sort_idx]
    sy = y[sort_idx]
    area = 0.0
    for ix in range(len(x) - 1):
        area += 0.5 * (sx[ix + 1] - sx[ix]) * (sy[ix + 1] + sy[ix])
    return area


def auc(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    if not np.all(np.in1d(y_true, [0, 1])):
        raise ValueError('y_true should only contain 0 and 1!')
    tps, fps = np.array([]), np.array([])
    n = len(y_true)
    for threshold in np.sort(y_score)[:-1]:
        tp = np.sum(np.absolute((y_score <= threshold) - y_score))
        np.append(tps, tp / n)
        np.append(fps, (n - tp) / n)
    return trapz(fps, tps)
