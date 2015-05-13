import math

import numpy as np
import scipy.optimize as opt


class LogisticRegression():

    def _sigmoid(self, z):
        s = 1.0 / (1.0 + np.exp(-1.0 * z))
        return s

    def _cost(self, w, X, y):
        """ Return the cost give the parameter w and data"""
        t = y
        y = self._sigmoid(np.dot(w, X.transpose()))
        return np.sum(np.multiply(t, np.log(y)) + np.multiply(1 - t, np.log(1 - y))) * -1.0

    def _gradient(self, w, X, y):
        t = y
        y = self._sigmoid(np.dot(w, X.transpose()))
        diff = y - t
        return np.apply_along_axis(sum, 0,
                                   np.apply_along_axis(
                                       lambda a: np.multiply(diff, a), 0, X))

    def train(self, X, y):
        w0 = np.random.rand(X.shape[1])
        xopt = opt.fmin_bfgs(self._cost, w0,
                             self._gradient, args=(X, y), maxiter=100)
        self.w = xopt

    def predict(self, X, prob=False):
        if getattr(self, 'w', None) is None:
            raise RuntimeError("Train before predicting.")
        threshold = 0.5
        y = self._sigmoid(np.dot(self.w, X.transpose()))
        if prob:
            return y
        return (y > 0.5) * 1


def test_logistic_regression_train():
    import pandas as pd
    names = ["Y", "PREG", "GLUCOSE", "PRESS",
             "SKIN", "INSULIN", "BMI", "PEDIGREE", "AGE"]
    data = pd.read_csv('tests/data/diabetes_scale.csv', names=names)
    data['Y'] = (data['Y'] + 1) / 2
    data['ones'] = np.ones((data.shape[0], 1))  # add a column of ones
    model = LogisticRegression()
    model.train(data.drop("Y", 1).values, data["Y"])
    assert model._cost(
        model.w, data.drop('Y', 1), data["Y"]) - 361.722692813 < 1e-5

    def cost(y, pred):
        return np.sum(np.multiply(y, np.log(pred)) + np.multiply(1 - y, np.log(1 - pred))) * -1.0
    assert cost(data["Y"], model.predict(
        data.drop("Y", 1).values, prob=True)) - 361.722692813 < 1.e-5

if __name__ == '__main__':
    test_logistic_regression_train()
