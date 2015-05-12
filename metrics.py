"""The metrics module implements functions assessing prediction error for specific purposes."""

import numpy as np
import pandas as pd


def trapz(x, y):
    """Trapezoidal rule for integrating
    the curve defined by x-y pairs.
    Assume x and y are in the range [0,1]
    """
    assert len(x) == len(y), 'x and y need to be of same length'
    df = pd.DataFrame({'x': np.concatenate([x, np.array([0.0, 1.0])]),
                       'y': np.concatenate([y, np.array([0.0, 1.0])])})
    df = df.sort(['x', 'y'])
    area = 0.0
    for ix in range(len(df) - 1):
        area += (0.5 *
                 (df['x'].iloc[ix + 1] - df['x'].iloc[ix]) *
                 (df['y'].iloc[ix + 1] + df['y'].iloc[ix]))
    return area


def auc(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    if not np.all(np.in1d(y_true, [0, 1])):
        raise ValueError('y_true can only contain 0 and 1!')
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    fps, tps = np.array([]), np.array([])
    for threshold in np.sort(y_score):
        pred = (y_score > threshold)
        fp = np.sum(np.logical_and(pred == 1, y_true == 0))
        tp = np.sum(pred) - fp
        fps = np.append(fps, fp / n_neg)
        tps = np.append(tps, tp / n_pos)
    return trapz(fps, tps)
