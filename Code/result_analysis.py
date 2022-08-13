import numpy as np


class ModelAnalysis():
    def __init__(self, X_test, y_test, model):
        '''
        :param X_test: Input Dataset
        :param y_test: True Labels of this dataset
        :param model: Model, which maps X_test onto y_test. it has to have the predict function
        '''
        self.X_test = X_test
        self.y_test = y_test

        self.predictions = model.predict(self.X_test)
        self.model = model
        self.model_name = model.name
        model.accuracy = self.get_accuracy()
        model.precision = self.get_precision()
        model.recall = self.get_recall()

    def get_accuracy(self):
        '''
        This function calculated the Accuracy of the model on the test set
        :return: accuracy
        '''
        true_pos = ((self.y_test == 1) & (self.predictions.reshape(self.predictions.shape[0],1) == 1)).sum()
        true_neg = ((self.y_test == -1) & (self.predictions.reshape(self.predictions.shape[0],1) == -1)).sum()
        false_neg = ((self.y_test == 1) & (self.predictions.reshape(self.predictions.shape[0],1) == -1)).sum()
        false_pos = ((self.y_test == -1) & (self.predictions.reshape(self.predictions.shape[0],1) == 1)).sum()
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    def get_precision(self):
        '''
        This function calculated the Accuracy of the model on the test set
        :return: precision
        '''
        true_pos = ((self.y_test == 1) & (self.predictions.reshape(self.predictions.shape[0],1) == 1)).sum()
        false_pos = ((self.y_test == -1) & (self.predictions.reshape(self.predictions.shape[0],1) == 1)).sum()
        return true_pos / (true_pos + false_pos)

    def get_recall(self):
        '''
        This function calculated the Accuracy of the model on the test set
        :return: recall
        '''
        true_pos = ((self.y_test == 1) & (self.predictions.reshape(self.predictions.shape[0],1) == 1)).sum()
        false_neg = ((self.y_test == 1) & (self.predictions.reshape(self.predictions.shape[0],1) == -1)).sum()
        return true_pos / (true_pos + false_neg)

    def make_whole_report(self):
        '''
        This function prints the three most important Evaluation metrics
        :return:
        '''
        accuracy = self.get_accuracy()
        precision = self.get_precision()
        recall = self.get_recall()
        print()
        print(f'The Precision of the model: {self.model_name} is: {np.round(precision * 100,2)}%')
        print(f'The Recall of the model: {self.model_name} is: {np.round(recall * 100,2)}%')
        print(f'The Accuracy of the model: {self.model_name} is: {np.round(accuracy * 100,2)}%')



    def make_interpretation_report(self):
        self.model.explain_model()
