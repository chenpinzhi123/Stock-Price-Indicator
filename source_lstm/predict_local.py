from predict import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':

    model = model_fn("../model")

    input_data = np.array(pd.read_csv("../data/test.csv"))

    X_test = input_data[:, 1:]
    y_test = input_data[:, 0]

    y_pred = predict_fn(X_test, model)

    y_pred = y_pred.reshape(y_pred.shape[0])
    y_test = y_test.reshape(y_pred.shape[0])

    accuracy = ( np.logical_or( np.logical_and(y_pred >= 0, y_test >= 0),
                                np.logical_and(y_pred < 0, y_test < 0) ) ).sum() / len(y_test)
    print("Accuracy =", accuracy)

    plt.figure()
    plt.plot(y_test, y_pred, ls=' ', marker='.')
    plt.show()


