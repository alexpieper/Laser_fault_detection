import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random


class DataPreprocessing():
    def __init__(self):
        self.data_loc = os.path.join('data', 'laser.mat')
        self.data = scipy.io.loadmat(self.data_loc)
        self.X = self.data['X']
        self.y = self.data['Y']
        self.N_obs = self.X.shape[0]
        self.N_feat = self.X.shape[1]

        self.fig_loc = os.path.join('figures')
        if not os.path.exists(self.fig_loc):
            os.makedirs(self.fig_loc)

    def get_stats_of_data(self):
        '''
        This function is designed to get a first grasp of the data and create some plots for the analysis
        :return:
        '''
        # look at the first three observations, to get a first understanding
        print(pd.DataFrame(self.X).head(3))
        # here we validate, that there are no missing values:
        print(f'We have {pd.DataFrame(self.X).isna().sum().sum()} missing values in our dataset.')
        print(f'we have {self.N_obs} observations with {self.N_feat} features for each entry.')
        df = pd.DataFrame(self.X)
        df['std'] = df.std(axis = 1)
        df['y'] = self.y.ravel()
        positives = df[df['y'] == 1]
        negatives = df[df['y'] == -1]
        # Here we can look at the summary statistics of the positive and negative-labelled data each
        print(positives.describe())
        print(negatives.describe())
        pos = positives.describe().drop(columns = ['std', 'y']).transpose()
        neg = negatives.describe().drop(columns = ['std', 'y']).transpose()
        fig, ax = plt.subplots(2,1, sharex=True,figsize = (12,7))
        ax[0].plot(pos['25%'], color = 'green', alpha = 0.4, label = '25% quantile')
        ax[0].plot(pos['mean'], color = 'green', alpha = 1, label = 'mean')
        ax[0].plot(pos['75%'], color = 'green', alpha = 0.4, label = '75% quantile')
        ax[1].plot(neg['25%'], color = 'red', alpha = 0.4, label = '25% quantile')
        ax[1].plot(neg['mean'], color = 'red', alpha = 1, label = 'mean')
        ax[1].plot(neg['75%'], color = 'red', alpha = 0.4, label = '75% quantile')
        ax[0].legend()
        ax[1].legend()
        ax[0].title.set_text('Functional Lasers')
        ax[1].title.set_text('Broken Lasers')
        ax[0].set_ylabel('Intensity')
        ax[1].set_ylabel('Intensity')
        ax[1].set_xlabel('time [in seconds]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.fig_loc, 'summary_statistics_vizualisation.png'))


    def visualize_raw_data(self, n_obs):
        '''
        here we want to create a function, that visualizes the data with their labels
        :return:
        '''
        # print(np.concatenate((self.X, self.y), axis=1))
        fig, ax = plt.subplots(n_obs,1, sharex=True,figsize = (11,7))
        for i in range(n_obs):
            if self.y[i] == 1:
                ax[i].plot(self.X[i], color = 'green', alpha = 1, label = 'functional')
            else:
                ax[i].plot(self.X[i], color='red', alpha = 1, label = 'broken')
            ax[i].legend()
            ax[i].set_ylabel('intensity')
            ax[i].set_yticklabels([])
            ax[i].set_yticks([])
        ax[n_obs-1].set_xlabel('time [in seconds]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.fig_loc, 'sample_data_vizualisation.png'))
        plt.close()


    def get_train_test_sets(self, train_ratio:float, normalizing:bool, feature_engineering:bool, forward_diff = None):
        '''
        This function performs the train test split.
        There is also Code for local normalization (normalizing by each observation, and not the whole dataset)
        and for Min-Max Normalization, which is commented currently

        :param train_ratio: ratio of the train instances \in (0,1)
        :param normalizing: Boolean whether Z-score normalization should be applied
        :param feature_engineering: Boolean whether Feature Engineering should be applied
        :param forward_diff: only used if feature_engineering==True, then it defines the difference
        :return: 4 matrices/vectors: Training Data, X and y, testing data X and y
        '''
        random.seed(42)
        self.scaled_X = self.X.copy()

        if normalizing:
            # local normalizing (for every observation) (results in worse precision, sometimes better recall):
            # for i in range(self.X.shape[0]):
            #     self.scaled_X[i] = (self.X[i] - self.X.mean(axis = 1)[i]) / self.X.std(axis = 1)[i]
            #     self.scaled_X[i] = (self.X[i] - self.X.min(axis = 1)[i]) / (self.X.max(axis = 1)[i] - self.X.min(axis = 1)[i]) * (1 - 0) + 0

            # global normalizing:
            # Min-Max Normalization:
            # self.scaled_X = (self.X - self.X.min()) / (self.X.max() - self.X.min()) * (1 - 0) + 0
            # Z-Score Normalization:
            self.scaled_X = (self.X - self.X.mean()) / self.X.std()


        if feature_engineering:
            self.engineered_scaled_X = np.empty(shape = (self.scaled_X.shape[0], (self.scaled_X.shape[1] + 2)))
            for i in range(self.X.shape[0]):
                self.engineered_scaled_X[i] = np.concatenate((self.scaled_X[i], [np.max(self.X[i][0:len(self.X[i]) - forward_diff] - self.X[i][forward_diff:]), np.std(self.X[i])]), axis = 0)
            # this adds all diffs, not only the max of the diffs
            # self.engineered_scaled_X = np.empty(shape = (self.scaled_X.shape[0], (self.scaled_X.shape[1] * 2) - forward_diff))
            # for i in range(self.X.shape[0]):
            #     self.engineered_scaled_X[i] = np.concatenate((self.scaled_X[i], self.scaled_X[i][0:len(self.scaled_X[i]) - forward_diff] - self.scaled_X[i][forward_diff:]), axis=0)
            self.scaled_X = self.engineered_scaled_X.copy()

        indexes = sorted(random.sample(range(1, self.N_obs), int(self.N_obs * train_ratio)))
        indexes_test = sorted([i for i in range(1, self.N_obs) if i not in indexes])
        self.X_train = self.scaled_X[indexes]
        self.X_test = self.scaled_X[indexes_test]
        self.y_train = self.y[indexes]
        self.y_test = self.y[indexes_test]

        return self.X_train, self.X_test, self.y_train, self.y_test
