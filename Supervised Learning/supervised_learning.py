# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


class SupervisedLearning(object):

    def __init__(self) -> None:
        self.epsilon = [0.01, 0.05, 0.1]
        self.delta = self.epsilon  # [0.05, 0.1, 0.15]

    def vc_dimension(self, x, alg, degree=1):
        [n, f] = x.shape
        if alg == 'linear':
            vc_dim = f+1
        elif alg == 'svm_linear':
            vc_dim = f+1
        elif alg == 'svm_poli':
            vc_dim = degree+f-1
        return vc_dim

    def optimal_training_set(self, epsilon, delta, vc_dim):
        n = (1/epsilon)*(np.log(vc_dim)+np.log(1/delta))
        return n

    def optimal_training_set_tree(self, epsilon, delta, depth, m):
        n = (np.log(2)/(2*epsilon**2)) * \
            ((2**depth-1)*(1+np.log2(m))+1+np.log(1/delta))
        return n

    def linear_regression(self, X_train, X_test, y_train, y_test):
        vc_dim = self.vc_dimension(X_train, 'linear')
        N = []
        Y = []
        SCORE = []
        for i in range(len(self.epsilon)):
            epsilon = self.epsilon[i]
            delta = self.delta[i]
            n = self.optimal_training_set(epsilon, delta, vc_dim)
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = LinearRegression().fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            score = model.score(X_test, y_test)
            SCORE.append(score)
        return N, Y, SCORE

    def decision_tree(self, X_train, X_test, y_train, y_test):
        N = []
        Y = []
        SCORE = []
        m = 1
        for i in range(len(self.epsilon)):
            param_grid = {'max_depth': [8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                          'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
            grid_search = GridSearchCV(
                DecisionTreeClassifier(), param_grid,  cv=10, scoring='accuracy')
            # Ajustar el modelo
            grid_search.fit(X_train, y_train)
            # Obtener los hiperparámetros óptimos
            best_params = grid_search.best_params_
            print(best_params)
            n = self.optimal_training_set_tree(
                self.epsilon[i], self.delta[i], best_params['max_depth'], m)
            N.append(n)
            if -float('inf') < n < float('inf'):
                new_x = X_train[:int(n)]
                new_y = y_train[:int(n)]
            else:
                new_x = X_train
                new_y = y_train
            # Use best model
            clf = grid_search.best_estimator_
            clf.fit(new_x, new_y)
            y_pred = clf.predict(X_test)
            Y.append(y_pred)
            score = clf.score(X_test, y_test)
            SCORE.append(score)
        return N, Y, SCORE

    def svm_linear_kernel(self, X_train, X_test, y_train, y_test):
        vc_dim = self.vc_dimension(X_train, 'svm_linear')
        N = []
        Y = []
        SCORE = []
        for i in range(len(self.epsilon)):
            epsilon = self.epsilon[i]
            delta = self.delta[i]
            n = self.optimal_training_set(epsilon, delta, vc_dim)
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='linear').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            score = model.score(X_test, y_test)
            SCORE.append(score)
        return N, Y, SCORE

    def svm_polinomical_kernel(self, X_train, X_test, y_train, y_test):
        vc_dim = self.vc_dimension(X_train, 'svm_poli', degree=3)
        N = []
        Y = []
        SCORE = []
        for i in range(len(self.epsilon)):
            epsilon = self.epsilon[i]
            delta = self.delta[i]
            n = self.optimal_training_set(epsilon, delta, vc_dim)
            N.append(n)
            new_x = X_train[:int(n)]
            new_y = y_train[:int(n)]
            model = SVC(kernel='poly').fit(new_x, new_y)
            y_pred = model.predict(X_test)
            Y.append(y_pred)
            score = model.score(X_test, y_test)
            SCORE.append(score)
        return N, Y, SCORE

    def svm_radial_base_kernel(self, X_train, X_test, y_train, y_test):
        model = SVC(kernel='linear').fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        return y_pred, score

    def main(self, X_train, X_test, y_train, y_test):
        n_linear, y_linear, score_linear = self.linear_regression(
            X_train, X_test, y_train, y_test)
        n_svm_linear, y_svm_linear, score_svm_linear = self.svm_linear_kernel(
            X_train, X_test, y_train, y_test)
        n_svm_poli, y_svm_poli, score_svm_poli = self.svm_polinomical_kernel(
            X_train, X_test, y_train, y_test)
        y_svm_radial, score_svm_radial = self.svm_radial_base_kernel(
            X_train, X_test, y_train, y_test)
        n_svm_tree, y_svm_tree, score_svm_tree = self.decision_tree(
            X_train, X_test, y_train, y_test)
        return n_linear, y_linear, score_linear, n_svm_linear, y_svm_linear, score_svm_linear, n_svm_poli, y_svm_poli, score_svm_poli, y_svm_radial, score_svm_radial, n_svm_tree, y_svm_tree, score_svm_tree

# %%

def read_iris_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def read_original_dataset():
    df = pd.read_excel(r'.\Rice_Cammeo_Osmancik.xlsx', header=0,  names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "y1", "y2", "y3"])
    df.pop('y3')
    df_normalized = (df-df.min())/(df.max()-df.min())
    x1 = df_normalized["x1"]
    x2 = df_normalized["x2"]
    x3 = df_normalized["x3"]
    x4 = df_normalized["x4"]
    x5 = df_normalized["x5"]
    x6 = df_normalized["x6"]
    x7 = df_normalized["x7"]
    x = np.transpose(np.array([x1, x2, x3, x4, x5, x6, x7]))
    y1 = df_normalized["y1"]
    y2 = df_normalized["y2"]
    y = np.transpose(np.array(y1))
    return x, y

def read_hd_dataset():
    df = pd.read_excel(r'.\hd.xlsx', header=0,  names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "y1", "y2"])
    df_normalized = (df-df.min())/(df.max()-df.min())
    x1 = df_normalized["x1"]
    x2 = df_normalized["x2"]
    x3 = df_normalized["x3"]
    x4 = df_normalized["x4"]
    x5 = df_normalized["x5"]
    x6 = df_normalized["x6"]
    x7 = df_normalized["x7"]
    x8 = df_normalized["x8"]
    x9 = df_normalized["x9"]
    x10 = df_normalized["x10"]
    x11 = df_normalized["x11"]
    x12 = df_normalized["x12"]
    x = np.transpose(np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]))
    y1 = df_normalized["y1"]
    y2 = df_normalized["y2"]
    y = np.transpose(np.array(y1))
    return x, y

def read_ld_dataset():
    df = pd.read_excel(r'.\ld.xlsx', header=0,  names=["x1", "x2", "x3", "y1", "y2"])
    df_normalized = (df-df.min())/(df.max()-df.min())
    x1 = df_normalized["x1"]
    x2 = df_normalized["x2"]
    x3 = df_normalized["x3"]
    x = np.transpose(np.array([x1, x2, x3]))
    y1 = df_normalized["y1"]
    y2 = df_normalized["y2"]
    y = np.transpose(np.array(y1))
    return x, y

def read_and_split_data():
    # x, y = read_original_dataset()
    # x, y = read_hd_dataset()
    x, y = read_ld_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = read_and_split_data()
    sl = SupervisedLearning()
    n_linear, y_linear, score_linear, n_svm_linear, y_svm_linear, score_svm_linear, n_svm_poli, y_svm_poli, score_svm_poli, y_svm_radial, score_svm_radial, n_svm_tree, y_svm_tree, score_svm_tree = sl.main(
        X_train, X_test, y_train, y_test)
# %%
