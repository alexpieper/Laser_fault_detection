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
        ax.set_xlabel(f'Maximum {self.forward_diff} second difference')
        ax.set_ylabel('Number of Occurences')
        fig.tight_layout()
        plt.savefig(os.path.join(self.fig_loc, f'hist_naive_rule_{self.forward_diff}.png'))
        plt.close()

        # from this histogram, a great discriminant can be chosen
        # could also be calculated like this, but i set it to 15 here
        # self.cutoff = (np.mean([np.median(max_diffs_correct), np.median(max_diffs_faulty)]))
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
        # If the max_iter parameter = 100 (default), then the normalization has huge impact on the performance
        # also other parameter have huge impact (like fit_intercept)
        self.model = LogisticRegression(random_state=0, max_iter = 500)
        self.model.fit(self.X_train, self.y_train.ravel())


    def predict(self, X_test):
        # this return (dotproduct of X and \theta^t + intercept) > 0 as integer
        self.predictions = self.model.predict(X_test)
        return self.predictions


    def explain_model(self):
        # the self.model.coef_ is \theta
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
        # that is the Neural network funcitonfrom sklearn.
        # parameters are quite selvexplaining in my opinion.
        self.model = MLPClassifier(hidden_layer_sizes=(24, 12, 6), activation='logistic', solver='adam', max_iter=2500, random_state = 42)
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        # this is basically one forward propagation through the fully trained neural network.
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
        # This model was developed by Microsoft and is supposed to tackle the blackbox nature of most algorithms.
        # It is a GAM and uses techniques like bagging, gradient boosting, and automatic interaction detection.
        self.model = ExplainableBoostingClassifier()
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self, X_test):
        # this funciton calculates the score of the determined features, and basically adds them (generalized ADDITIVE model)
        # then it return the class with the highes score (here, binary classification: 1 if score>0)
        self.predictions = self.model.predict(X_test)
        return self.predictions

    def explain_model(self):
        ebm_global = self.model.explain_global()
        for index, value in enumerate(self.model.feature_groups_):
            plotly_fig = ebm_global.visualize(index)
            plotly_fig.write_image(os.path.join(self.fig_loc, f'explainability_{index}.png'))



