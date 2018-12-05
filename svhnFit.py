import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from datetime import datetime
from sklearn import svm
import pandas as pd
import seaborn as sns; sns.set()
from numpy import unravel_index

# Load data
workingDir = os.getcwd()
setDit = os.path.join(workingDir, 'SVHN')
shape=(32, 32)
xHogTrainFull, yHogTrainFull = utils.rgbToHOG(os.path.join(setDit, 'train_32x32.mat'), pixelsPerCell=(8, 8))
xHogTest, yHogTest = utils.rgbToHOG(os.path.join(setDit, 'test_32x32.mat'), pixelsPerCell=(8, 8))

xHogTrain = xHogTrainFull
yHogTrain = yHogTrainFull
xHogTrain, yHogTrain = utils.cutSet(xHogTrainFull, yHogTrainFull, 10000) # for testing

###############################################################################
# take random observations to reduce data size
x, y = utils.cutSet(xHogTrainFull, yHogTrainFull, 1000)

print("Manual grid search RBF SVC")
cRange = np.linspace(1, 100, 10, dtype=int)
gammaRange = np.round(np.logspace(-4, -1, 10), decimals=5)
StartTime = datetime.now()
# store results for grid search in data frame
results = pd.DataFrame(np.zeros(shape=(len(cRange), len(gammaRange)),\
    dtype=float), index=cRange, columns=gammaRange)
for c in cRange:
    for gamma in gammaRange:        
        classifierSVC = svm.SVC(C=c, kernel='rbf', gamma=gamma,\
            coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,\
            class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',\
            random_state=None)
        classifierSVC.fit(x, y)            
        yForecastSVC = classifierSVC.predict(xHogTest) 
        scoreSVC = utils.getScore(yHogTest, yForecastSVC)
        results.loc[c, gamma] = scoreSVC['Accuracy']
        print('{} {} {} {} {} {}'.format('C=', c, 'Gamma=', gamma, 'Accuracy=', scoreSVC['Accuracy']))
        
EndTime = datetime.now()
ElapsedTime = EndTime - StartTime
print("Manual grid search RBF SVC worked: ", ElapsedTime)

# Plot heat map
fig, ax = plt.subplots(figsize=(19, 10))
bx = sns.heatmap(ax=ax, data=results, annot=True, fmt="f")
bx.set(xlabel='Gamma', ylabel='C')
title = 'SVHN. Classifier svc RBF. Ggrid search heat map'
bx.set_title(title)
bx.figure.savefig('{}{}'.format(title, '.png'))

# Get best hyperparameters for SVC
a = results.values
row, col = unravel_index(a.argmax(), a.shape)
cBest = cRange[row]
gammaBest = gammaRange[col]
print('SVC best parameters:')
print('C=', cBest, 'Gamma=', gammaBest)

# Get best fit with best hyperparameters
print('Apply best hyperparameter to full dataset')
classifierSVC = svm.SVC(C=cBest, kernel='rbf', gamma=gammaBest,\
    coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,\
    class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',\
    random_state=None)
classifierSVC.fit(xHogTrain, yHogTrain)            
yForecastSVC = classifierSVC.predict(xHogTest) 
scoreSVC = utils.getScore(yHogTest, yForecastSVC)
print('SVC best accuracy: ', scoreSVC['Accuracy'])

###################################################################
# take random observations to reduce data size
x, y = utils.cutSet(xHogTrainFull, yHogTrainFull, 10000)

print("Random Forest") 
space = np.linspace(2, 20, 19, dtype=int)
accuracy = []
for i in space:
    classifierRF = RandomForestClassifier(n_estimators=10, criterion='entropy',\
        max_depth=None, min_samples_split=i, min_samples_leaf=1,\
        min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None,\
        min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,\
        oob_score=False, warm_start=False, class_weight=None, n_jobs=4)
    classifierRF.fit(x, y)            
    yForecastRF = classifierRF.predict(xHogTest) 
    scoreRF = utils.getScore(yHogTest, yForecastRF)
    accuracy.append(scoreRF['Accuracy'])
    print('{} {} {} {}'.format('min_samples_split=', i, 'Accuracy=', scoreRF['Accuracy']))

print('Best min_samples_split = ', space[np.argmax(accuracy)])

best_index = np.argmax(accuracy)
best_score = accuracy[best_index]

# Plot grid search progress
plt.figure(figsize=(10, 10))
title = "SVHN. RandomForestClassifier Grid search evaluating"
plt.title(title, fontsize=16)
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
ax = plt.gca()
ax.plot(space, accuracy, color='g', alpha=1, label="Test set")
ax.plot([space[best_index], ] * 2, [0, best_score],
            linestyle='-.', color='k', marker='x', markeredgewidth=3, ms=8)
ax.annotate("%0.2f" % best_score,
                (space[best_index], best_score + 0.005))
plt.legend(loc='lower left')
plt.savefig('{}{}'.format(title, '.png'))

print('Apply best hyperparameter to full dataset')
classifierRF = RandomForestClassifier(n_estimators=10, criterion='entropy',\
    max_depth=None, min_samples_split=space[best_index], min_samples_leaf=1,\
    min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None,\
    min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,\
    oob_score=False, warm_start=False, class_weight=None, n_jobs=4)
classifierRF.fit(xHogTrain, yHogTrain)            
yForecastRF = classifierRF.predict(xHogTest) 
scoreRF = utils.getScore(yHogTest, yForecastRF)
print('RandomForestClassifier best accuracy: ', scoreRF['Accuracy'])

###############################################################################

x, y = utils.cutSet(xHogTrainFull, yHogTrainFull, 10000) # reduce set size

print("KNeighborsClassifier") 
space = np.linspace(1, 100, 20, dtype=int)
accuracy = []
for i in space:
    classifierKNN = KNeighborsClassifier(n_neighbors=i, weights='uniform',\
        algorithm='auto', leaf_size=30, p=2, metric='minkowski')
    classifierKNN.fit(x, y)       
    yForecastKNN = classifierKNN.predict(xHogTest) 
    scoreKNN = utils.getScore(yHogTest, yForecastKNN)
    accuracy.append(scoreKNN['Accuracy'])
    print('{} {} {} {}'.format('n_neighbors=', i, 'Accuracy=', scoreKNN['Accuracy']))

print('Best n_neighbors = ', space[np.argmax(accuracy)])

best_index = np.argmax(accuracy)
best_score = accuracy[best_index]

# Plot grid search progress
plt.figure(figsize=(10, 10))
title = "SVHN. KNeighborsClassifier Grid search evaluating"
plt.title(title, fontsize=16)
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
ax = plt.gca()
ax.plot(space, accuracy, color='g', alpha=1, label="Test set")
ax.plot([space[best_index], ] * 2, [0, best_score],
            linestyle='-.', color='k', marker='x', markeredgewidth=3, ms=8)
ax.annotate("%0.2f" % best_score,
                (space[best_index], best_score + 0.005))
plt.legend(loc='lower left')
plt.savefig('{}{}'.format(title, '.png'))

print("KNeighborsClassifier") 
classifierKNN = KNeighborsClassifier(n_neighbors=space[best_index], weights='uniform',\
    algorithm='auto', leaf_size=30, p=2, metric='minkowski')
classifierKNN.fit(xHogTrain, yHogTrain)       
yForecastKNN = classifierKNN.predict(xHogTest) 
scoreKNN = utils.getScore(yHogTest, yForecastKNN)
print('KNeighborsClassifier best accuracy: ', scoreKNN['Accuracy'])

###############################################################################
print("XGBoost") 
classifierXG = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=10,\
    silent=1, objective='multi:softprob', booster='gbtree', num_class=10, eval_metric='merror', \
    n_jobs=4, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,\
    colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,\
    scale_pos_weight=1, base_score=0.5, seed=None, missing=None)
classifierXG.fit(xHogTrain, yHogTrain)      
yForecastXG = classifierXG.predict(xHogTest) 
yForecastProbaXG = classifierXG.predict_proba(xHogTest)
scoreXG = utils.getScore(yHogTest, yForecastXG)
print('XGBClassifier accuracy: ', scoreXG['Accuracy'])

