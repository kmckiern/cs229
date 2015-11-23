#!/usr/bin/env python

import argparse
from sklearn import linear_model, svm


def square_resid(y_predict, y_true):
    return (y_predict - y_true)**2


def linear_regression(X_train, Y_train, X_test, Y_test):
    opt = linear_model.LinearRegression()
    opt.fit(X_train,Y_train)
    return opt, square_resid(opt.predict(X_test), Y_test)


def SVR(X_train, Y_train, X_test, Y_test, kernel='linear'):
    opt = svm.SVR(kernel=kernel)
    opt.fit(X,Y)
    return opt, square_resid(opt.predict(X_test), Y_test)

