'''
This scripts is used for loading data from packages or from files

'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class iris_data_loader:

    def get_iris_data(self):
        '''
        Function to load and return iris dataset from sklearn package
        '''
        iris = load_iris()
        features = iris.data
        target = iris.target
        return features, target

    def train_test_splitter_iris(self):
        '''
        Function to return Train-Test-Data for Iris data
        '''
        features, target = self.get_iris_data()
        X_train, X_test, y_train, y_test = \
        train_test_split(features, target, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test
