## Big Data Analytics Homework-2
####  M10502282 ¼B«³¨ä

------------
###XGBoost Article
#####Import Libraries:
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
```
#####Load Data:
```python
data = pd.read_csv('Dataset/train/LargeTrain.csv')
train = pd.DataFrame(data)
target = 'Class'
```
#####Define a function for modeling and cross-validation
```python
def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Class'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Class'], cv=cv_folds)
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Class'].values, dtrain_predictions)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
```
####Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
```python
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=10,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)
```
![](https://github.com/Liu-Yi-Chi/Big_Data_Analytics_HW2/blob/master/img/Default.PNG)

------------
####Step 2: Tune max_depth and min_child_weight
```python
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=5,min_child_weight=1,
 gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax', scale_pos_weight=1, seed=27),
 param_grid = param_test1 , n_jobs=4 , iid=False , cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
The ideal values are 9 for max_depth and 1 for min_child_weight

------------
####Step 3: Tune gamma
```python
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, 
  max_depth=9,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
  objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
  param_grid = param_test3, n_jobs=4 , iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```
 0 is the optimum one

------------
####Step 4: Tune subsample and colsample_bytree
```python
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
 param_grid = param_test4, n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```
The ideal values are 0.9 for subsample and 0.6 for colsample_bytree

------------
####Step 5: Tuning Regularization Parameters
```python
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.9 , colsample_bytree=0.6 ,
 objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
 param_grid = param_test6 ,n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
```
we got a better reg_alpha=0.1
#####Now we can apply this regularization in the model and look at the impact:
```python
xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=10,
 max_depth=9,
 min_child_weight=1,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.6,
 reg_alpha=0.1,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictors)
```
![](https://github.com/Liu-Yi-Chi/Big_Data_Analytics_HW2/blob/master/img/Tune.PNG)

------------
###GBM Parameters
#####Lets start by importing the required libraries and loading the data:
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

data = pd.read_csv('Dataset/train/LargeTrain.csv')
train = pd.DataFrame(data)
target = 'Class'
```
####Step 1- Find the number of estimators for a high learning rate
```python
predictors = [x for x in train.columns if x not in [target]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
 min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
 param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
The ideal values are 80 for n_estimators

------------
####Step 2- Tune tree-specific parameters
- Tune max_depth and num_samples_split


```python
#Grid seach on subsample and max_features
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
 max_features='sqrt', subsample=0.8,random_state=10), 
 param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```
The ideal values are 11 for max_depth and 200 for min_samples_split

- Tune min_samples_leaf

```python
#Grid seach on subsample and max_features
param_test3 = {'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
 max_depth=11,max_features='sqrt', subsample=0.8, random_state=10), 
 param_grid = param_test3,n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```
The ideal values are 40 for min_samples_leaf

- Tune max_features

```python
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=11, 
 min_samples_split=1600, min_samples_leaf=40, subsample=0.8, random_state=10),
 param_grid = param_test4,n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```
The ideal values are 19 for max_features

------------
####Step3- Tune Subsample and Lower Learning Rate

```python
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
 max_depth=11,min_samples_split=1600, min_samples_leaf=40, subsample=0.8, random_state=10,max_features=19),
 param_grid = param_test5,n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
```
The ideal values are 0.9 for subsample

----------------