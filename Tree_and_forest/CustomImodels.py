'''
This script is an initla playground for https://github.com/csinva/imodels
Related to XAI-topic

Further models:
from imodels  import BoostedRulesClassifier, \
              FIGSClassifier, SkopeRulesClassifier
from imodels  import RuleFitRegressor, HSTreeRegressorCV, SLIMRegressor
'''

from imodels import DecisionTreeClassifier
from CustomDataLoader import iris_data_loader

iris_data = iris_data_loader()
X_train, X_test, y_train, y_test = iris_data.train_test_splitter_iris()

model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)
print(model)
print(preds)
print(preds_proba)
