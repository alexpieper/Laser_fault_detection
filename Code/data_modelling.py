import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

'''
Each of these Classes represent one Model Type
They all have the methods train_model() and predict(), which train the model, given X_train instances and predict Outcomes, given X_test instances 
'''

class NaiveModel():
    def __init__(self, X_train, y_train, forward_diff, name):
        self.X_train = X_train
        self.y_train = y_train
        self.forward_diff = forward_diff
        self.name = name

        self.fig_loc = os.path.join('figures')
        if not os.path.exists(self.fig_loc):
            os.makedirs(self.fig_loc)


    def train_model(self):
        max_diffs_correct = []
        max_diffs_faulty = []
        i = 0
        for y_value in self.y_train:
            if y_value == 1:
                max_diffs_correct += [np.max(self.X_train[i][0:len(self.X_train[i]) - self.forward_diff] - self.X_train[i][self.forward_diff:])]
            else:
                max_diffs_faulty += [np.max(self.X_train[i][0:len(self.X_train[i]) - self.forward_diff] - self.X_train[i][self.forward_diff:])]
            i += 1


        fig, ax = plt.subplots(1,1, figsize = (8,5))
        ax.hist(max_diffs_faulty, color = 'red', alpha = 0.4, bins = np.linspace(5,35,13))
        ax.hist(max_diffs_correct, color = 'green', alpha = 0.4, bins = np.linspace(5,35,13))
        fig.tight_layout()
        plt.savefig(os.path.join(self.fig_loc, f'hist_naive_rule_{self.forward_diff}.png'))
        plt.close()

        # from this histogram, a great discriminant can be chosen
        self.cutoff = (np.mean([np.median(max_diffs_correct), np.median(max_diffs_faulty)]))
        self.cutoff = 15


    def predict(self, X_test):
        predictions = np.empty(shape = (X_test.shape[0],1))
        i = 0
        for X_obs in X_test:
            if np.max(X_obs[0:len(X_obs) - self.forward_diff] - X_obs[self.forward_diff:]) > self.cutoff:
                predictions[i] = [-1]
            else:
                predictions[i] = [1]
            i += 1
        self.predictions = predictions.ravel()
        return self.predictions



class LinearModel():
    def __init__(self, X_train, y_train, name):
        self.X_train = X_train
        self.y_train = y_train
        self.name = name
        self.fig_loc = os.path.join('figures', 'logistic_regression')
        if not os.path.exists(self.fig_loc):
            os.makedirs(self.fig_loc)

    def train_model(self):
        self.model = LogisticRegression(random_state=0, max_iter = 500, verbose = 0)
        self.model.fit(self.X_train, self.y_train.ravel())


    def predict(self, X_test):
        self.predictions = self.model.predict(X_test)
        return self.predictions


    def explain_model(self):
        fig, ax = plt.subplots(1,1,figsize = (12,6))
        ax.plot(range(len(self.model.coef_[0])), self.model.coef_[0])
        ax.set_xlabel('t [in seconds]')
        ax.set_ylabel('Theta(t)')
        fig.tight_layout()
        plt.savefig(os.path.join(self.fig_loc, f'theta_values.png'))
        plt.close()





class NeuralNetSklearn():
    def __init__(self, X_train, y_train, name):
        self.X_train = X_train
        self.y_train = y_train
        self.name = name

    def train_model(self):
        self.model = MLPClassifier(hidden_layer_sizes=(24, 12, 6), activation='logistic', solver='adam', max_iter=2500, random_state = 42)
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        self.predictions = self.model.predict(X_test)
        return self.predictions




class ExplainableClassifier():
    def __init__(self, X_train, y_train, name):
        self.X_train = X_train
        self.y_train = y_train
        self.name = name
        self.fig_loc = os.path.join('figures', 'interpret_ml')
        if not os.path.exists(self.fig_loc):
            os.makedirs(self.fig_loc)

    def train_model(self):
        self.model = ExplainableBoostingClassifier()
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        self.predictions = self.model.predict(X_test)
        return self.predictions

    def explain_model(self):
        ebm_global = self.model.explain_global()
        for index, value in enumerate(self.model.feature_groups_):
            plotly_fig = ebm_global.visualize(index)
            plotly_fig.write_image(os.path.join(self.fig_loc, f'explainability_{index}.png'))





class XGBoost():
    def __init__(self, X_train, y_train, name):
        self.X_train = X_train
        self.y_train = y_train
        self.name = name

    def train_model(self):
        self.model = XGBClassifier(n_estimators=100, max_depth = 3, eval_metric = 'error', random_state = 42)
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        self.predictions = self.model.predict(X_test)
        return self.predictions



class DecisionTree():
    def __init__(self, X_train, y_train, name):
        self.X_train = X_train
        self.y_train = y_train
        self.name = name

    def train_model(self):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        self.predictions = self.model.predict(X_test)
        return self.predictions





class RandomForest():
    def __init__(self, X_train, y_train, name):
        self.X_train = X_train
        self.y_train = y_train
        self.name = name

    def train_model(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        self.predictions = self.model.predict(X_test)
        return self.predictions



class Perceptron():
    def __init__(self, X_train, y_train, learn_rate, epochs, name):
        self.X_train = X_train
        self.y_train = y_train
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.name = name


    def train_model(self):
        self.theta, n_miss_list = self.perceptron(self.X_train, self.y_train, self.learn_rate, self.epochs)

    def predict(self, X_test):
        # apply sigmoid, so we get the percentages out
        self.predictions = np.array([1 if i > 0.5 else -1 for i in self.sigmoid(np.dot(X_test, self.theta))])
        return self.predictions

    def sigmoid(self,X):
        # sometimes, this throws the warning: RuntimeWarning: overflow encountered in exp
        # the round sometimes avoids this
        return np.round(1 / (1 + np.exp(-X)), 5)

    def step_func(self, X):
        return 1.0 if (X > 0) else 0.0

    def perceptron(self, X, y, lr, epochs):
        '''
        This is a modified version of this code: https://towardsdatascience.com/perceptron-algorithm-in-python-f3ac89d2e537
        :param X:
        :param y:
        :param lr:
        :param epochs:
        :return:
        '''

        # X --> Inputs.
        # y --> labels/target.
        # lr --> learning rate.
        # epochs --> Number of iterations.

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Initializing parapeters(theta) to zeros.
        # +1 in n+1 for the bias term.
        theta = np.zeros((n, 1)).ravel()

        # Empty list to store how many examples were
        # misclassified at every iteration.
        n_miss_list = []

        # Training.
        for epoch in range(epochs):
            # variable to store #misclassified.
            n_miss = 0

            # looping for every example.
            for idx, x_i in enumerate(X):

                # Insering 1 for bias, X0 = 1.
                # x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

                # Calculating prediction/hypothesis.
                # y_hat = (self.sigmoid(np.dot(x_i.reshape(-1, 1).T, theta)) * 2) - 1
                y_hat = (self.step_func(np.dot(x_i.reshape(-1, 1).T, theta)) * 2) - 1

                # Updating if the example is misclassified.
                if (np.squeeze(y_hat) >= 0) & (y[idx] == [-1]) | (np.squeeze(y_hat) < 0) & (y[idx] == [1]):
                    # print((np.squeeze(y_hat) - y[idx]))
                    theta += lr * ((y[idx] - y_hat) * x_i)

                    # Incrementing by 1.
                    n_miss += 1

            # Appending number of misclassified examples
            n_miss_list.append(n_miss)

        # the n_miss list, is something like the loss functions of the epochs.
        return theta, n_miss_list



