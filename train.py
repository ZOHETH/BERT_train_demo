import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import pickle
from sklearn.externals import joblib

from evaluate import Evaluate

TRAIN_DATA_PATH='dataset/data_t.npy'


def train(use_train=False):
    model = None
    data = np.load(TRAIN_DATA_PATH)
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if use_train:
        model = LogisticRegression(C=5, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                                   multi_class='ovr', penalty='l2',
                                   random_state=None, solver='lbfgs', tol=0.0001, verbose=3,
                                   warm_start=False)
        model.fit(X_train, y_train)
        joblib.dump(model, 'model.pkl')
    else:
        model = joblib.load('model.pkl')
    y_pred = model.predict(X_test)
    print("acc\t:{}\nrecall\t:{}\nf1\t:{}".format(accuracy_score(y_test, y_pred),
                                                  recall_score(y_test, y_pred, average='weighted'),
                                                  f1_score(y_test, y_pred, average='weighted')))
    e=Evaluate()
    e.show_need_scores(y_test,y_pred)


if __name__ == '__main__':
    train(True)
