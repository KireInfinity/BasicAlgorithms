'''
This script triggers decision tree and random forest functions
'''

from sklearn.ensemble import RandomForestClassifier
from CustomDataLoader import iris_data_loader
from treeForest import CreateForest


def entry_point() -> None:
    iris_data = iris_data_loader()
    features, target = iris_data.get_iris_data()

    model_benchmark = RandomForestClassifier()
    model_min_split = RandomForestClassifier(min_samples_split = 6)
    model_min_sample = RandomForestClassifier(min_samples_leaf = 3)
    model_max_leafs = RandomForestClassifier(max_leaf_nodes = 20)

    random_forest_models = [
        model_benchmark,
        model_min_split,
        model_min_sample,
        model_max_leafs
    ]

    forests = CreateForest(random_forest_models, features, target, 20)
    forests.check_models()
    print(forests.importance)
    forests.plot_importance(forests.importance)

entry_point()
