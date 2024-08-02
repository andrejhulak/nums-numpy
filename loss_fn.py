import numpy as np

def cross_entropy_loss(y_pred, y_true):

    assert len(y_true) == len(y_pred)

    ret = 0

    for i in range(len(y_true)):

        ret -= y_true[i] * np.log(y_pred[i])

    return ret