#!!! baseline classifiers on MINST dataset
import os
import numpy as np
import utils
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import copy

# working dir
workingDir = os.getcwd()
setDit = os.path.join(workingDir, "mnist")
# load data
train = np.load(os.path.join(setDit, "train.npy"), encoding='latin1')
valid = np.load(os.path.join(setDit, "validation.npy"), encoding='latin1')
test = np.load(os.path.join(setDit, "test.npy"), encoding='latin1')
# load labels
yTtrain = np.load(os.path.join(setDit, "train_labels.npy"), encoding='latin1')
yValid = np.load(os.path.join(setDit, "validation_labels.npy"), encoding='latin1')
yTest = np.load(os.path.join(setDit, "test_labels.npy"), encoding='latin1')
# convert images to B/W
xTtrain = utils.toBW(train)
xValid = utils.toBW(valid)
xTest = utils.toBW(test)
# combibe training and validation sets
x = np.concatenate((xTtrain, xValid), axis=0)
y = np.concatenate((yTtrain, yValid))
nClasses = len(np.unique(y))

#x, y = utils.cutSet(x, y, 10000) # for testing

# Takes 8 hours
print("RBF SVC")
classifierSVC = svm.SVC(C=5, kernel='rbf', gamma=0.05,\
    coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200,\
    class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',\
    random_state=None)
classifierSVC.fit(x, y)            
yForecastSVC = classifierSVC.predict(xTest) 
yForecastProbaSVC = classifierSVC.predict_proba(xTest)
scoreSVC = utils.getScore(yTest, yForecastSVC)
print('SVC RBF accuracy: ', scoreSVC['Accuracy'])

print("Random Forest") 
classifierRF = RandomForestClassifier(n_estimators=10, criterion='entropy',\
    max_depth=None, min_samples_split=17, min_samples_leaf=1,\
    min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None,\
    min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,\
    oob_score=False, warm_start=False, class_weight=None, n_jobs=4)
classifierRF.fit(x, y)            
yForecastRF = classifierRF.predict(xTest) 
yForecastProbaRF = classifierRF.predict_proba(xTest)
scoreRF = utils.getScore(yTest, yForecastRF)
print('RandomForestClassifier accuracy: ', scoreRF['Accuracy'])

print("MultinomialNB")
classifierNB = MultinomialNB(alpha=1.0)
classifierNB.fit(x, y)            
yForecastNB = classifierNB.predict(xTest) 
yForecastProbaNB = classifierNB.predict_proba(xTest)
scoreNB = utils.getScore(yTest, yForecastNB)
print('MultinomialNB accuracy: ', scoreNB['Accuracy'])

print("XGBoost") 
classifierXG = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=10,\
    silent=1, objective='multi:softprob', booster='gbtree', num_class=10, eval_metric='merror', \
    n_jobs=4, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,\
    colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,\
    scale_pos_weight=1, base_score=0.5, seed=None, missing=None)
classifierXG.fit(x, y)      
yForecastXG = classifierXG.predict(xTest) 
yForecastProbaXG = classifierXG.predict_proba(xTest)
scoreXG = utils.getScore(yTest, yForecastXG)
print('XGBClassifier accuracy: ', scoreXG['Accuracy'])

print("LogisticRegression") 
classifierLR = LogisticRegression(penalty='l2', dual=False, tol=0.0001,\
    C=1.0, fit_intercept=False, intercept_scaling=1, class_weight=None,\
    random_state=None, solver='sag', max_iter=1000, multi_class='multinomial',\
    verbose=0, warm_start=False, n_jobs=4)
classifierLR.fit(x, y)     
yForecastLR = classifierLR.predict(xTest) 
yForecastProbaLR = classifierLR.predict_proba(xTest)
scoreLR = utils.getScore(yTest, yForecastLR)
print('LogisticRegression accuracy: ', scoreLR['Accuracy'])

print("KNeighborsClassifier") 
classifierKNN = KNeighborsClassifier(n_neighbors=5, weights='uniform',\
    algorithm='auto', leaf_size=30, p=2, metric='minkowski')
classifierKNN.fit(x, y)       
yForecastKNN = classifierKNN.predict(xTest) 
yForecastProbaKNN = classifierKNN.predict_proba(xTest)
scoreKNN = utils.getScore(yTest, yForecastKNN)
print('scoreKNN accuracy: ', scoreKNN['Accuracy'])

# Ensemble
yForecastProbaSVC = yForecastProbaSVC * scoreSVC['Accuracy']
yForecastProbaRF = yForecastProbaRF * scoreRF['Accuracy']
yForecastProbaNB = yForecastProbaNB * scoreNB['Accuracy']
yForecastProbaXG = yForecastProbaXG * scoreXG['Accuracy']
yForecastProbaLR = yForecastProbaLR * scoreLR['Accuracy']
yForecastProbaKNN = yForecastProbaKNN * scoreKNN['Accuracy']

# select 3 best classifiers for ensemble model
yForecastProba = copy.deepcopy(yForecastProbaKNN)
yForecastProba = np.add(yForecastProba, yForecastProbaSVC)
yForecastProba = np.add(yForecastProba, yForecastProbaRF)
#yForecastProba = np.add(yForecastProba, yForecastProbaXG)
#yForecastProba = np.add(yForecastProba, yForecastProbaNB)
#yForecastProba = np.add(yForecastProba, yForecastProbaLR)

yForecast = np.argmax(yForecastProba, axis=1)
score = utils.getScore(yTest, yForecast)
print('Ensemble accuracy: ', score['Accuracy'])

