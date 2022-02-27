'''
This script contains customized functions and classes around
decision trees and random forests
'''

from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class CreateForest:

    def __init__(self, forest: list, \
                features, target, k_fold: int) -> None:
        self.forest = forest
        self.importance = []
        self.features = features
        self.target = target
        self.k_fold = k_fold

    def check_models(self) -> None:
        for rf_model in self.forest:
            idx = np.arange(self.features.shape[0])
            np.random.shuffle(idx)
            #K-fold cross validation
            slice_size = self.features.shape[0] / self.k_fold
            score_sum = 0.0
            for i in range(0, self.k_fold):
                start = int(i * slice_size)
                end = int((i + 1) * slice_size)
                X_train = self.features[np.append(idx[0:start], idx[end:])]
                y_train = self.target[np.append(idx[0:start], idx[end:])]
                X_test = self.features[idx[start:end]]
                y_test = self.target[idx[start:end]]
                clf = rf_model.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                score_sum += score
                self.importance.append(rf_model.feature_importances_)
            print("Model: " + str(rf_model))
            print("Average CV Score " + str(score_sum/float(self.k_fold)))
            print("-------------------------------------")

    def plot_importance(self, importances: list) -> None:
        for importance in importances:
            plt.bar(importance)
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.show()

    def random_forst_scores(self, model) -> float:
        clf = model.fit(self.X_train, self.y_train)
        score = clf.score(self.X_test, self.y_test)
        return score

    def create_scores(self) -> None:
        for trees in self.forest:
            model = RandomForestClassifier(n_estimators = trees)
            self.score.append(self.random_forst_scores(model))

    def create_accuracy_plot(self) -> None:
        plt.plot(self.forest, self.score)
        plt.xlabel("Trees")
        plt.ylabel("Accuracy")
        plt.show()

    def create_plot(self):
        tree_plot = CreateForest(self.forest,
            self.X_train, self.X_test, self.y_train, self.y_test)
        tree_plot.create_scores()
        tree_plot.create_accuracy_plot()
        