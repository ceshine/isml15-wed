"""Test examples for AUC"""

from metrics import auc

def test_u1234567():
    y_true = [0., 0., 1., 1.]
    y_score = [-20., 0.2, 0.1, 0.9]
    print('Expected AUC   = 0.75')
    print('Calculated AUC = %f' % auc(y_true, y_score))

if __name__ == '__main__':
    test_u1234567()
