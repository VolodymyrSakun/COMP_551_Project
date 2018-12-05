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
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from numpy import unravel_index

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
xTrainFull = np.concatenate((xTtrain, xValid), axis=0)
yTrainFull = np.concatenate((yTtrain, yValid))
nClasses = len(np.unique(yTrainFull))

xTrain = xTrainFull
yTrain = yTrainFull
#xTtrain, yTtrain = utils.cutSet(xTrainFull, yTrainFull, 5000) # for testing

###############################################################################
x, y = utils.cutSet(xTrainFull, yTrainFull, 1000) # for testing

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
        yForecastSVC = classifierSVC.predict(xTest) 
        scoreSVC = utils.getScore(yTest, yForecastSVC)
        results.loc[c, gamma] = scoreSVC['Accuracy']
        print('{} {} {} {} {} {}'.format('C=', c, 'Gamma=', gamma, 'Accuracy=', scoreSVC['Accuracy']))
        
EndTime = datetime.now()
ElapsedTime = EndTime - StartTime
print("Manual grid search RBF SVC worked: ", ElapsedTime)

# Plot heat map
fig, ax = plt.subplots(figsize=(19, 10))
bx = sns.heatmap(ax=ax, data=results, annot=True, fmt="f")
bx.set(xlabel='Gamma', ylabel='C')
title = 'MNIST. Classifier svc RBF. Ggrid search heat map'
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
    coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200,\
    class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',\
    random_state=None)
classifierSVC.fit(xTrain, yTrain)            
yForecastSVC = classifierSVC.predict(xTest) 
yForecastProbaSVC = classifierSVC.predict_proba(xTest)
scoreSVC = utils.getScore(yTest, yForecastSVC)
print('SVC best accuracy: ', scoreSVC['Accuracy'])

###############################################################################
# take random observations to reduce data size
x, y = utils.cutSet(xTrainFull, yTrainFull, 1000)

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
    yForecastRF = classifierRF.predict(xTest) 
    scoreRF = utils.getScore(yTest, yForecastRF)
    accuracy.append(scoreRF['Accuracy'])
    print('{} {} {} {}'.format('min_samples_split=', i, 'Accuracy=', scoreRF['Accuracy']))

print('Best min_samples_split = ', space[np.argmax(accuracy)])

best_index = np.argmax(accuracy)
best_score = accuracy[best_index]

# Plot grid search progress
plt.figure(figsize=(10, 10))
title = "MNIST. RandomForestClassifier Grid search evaluating"
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
classifierRF.fit(xTrain, yTrain)            
yForecastRF = classifierRF.predict(xTest) 
yForecastProbaRF = classifierRF.predict_proba(xTest)
scoreRF = utils.getScore(yTest, yForecastRF)
print('RandomForestClassifier best accuracy: ', scoreRF['Accuracy'])

###############################################################################
print("MultinomialNB")
classifierNB = MultinomialNB(alpha=1.0)
classifierNB.fit(xTrainFull, yTrainFull)            
yForecastNB = classifierNB.predict(xTest) 
yForecastProbaNB = classifierNB.predict_proba(xTest)
scoreNB = utils.getScore(yTest, yForecastNB)
print('MultinomialNB accuracy: ', scoreNB['Accuracy'])

###############################################################################
print("XGBoost") 
classifierXG = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=10,\
    silent=1, objective='multi:softprob', booster='gbtree', num_class=nClasses, eval_metric='merror', \
    n_jobs=4, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,\
    colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,\
    scale_pos_weight=1, base_score=0.5, seed=None, missing=None)
classifierXG.fit(xTrain, yTrain)      
yForecastXG = classifierXG.predict(xTest) 
yForecastProbaXG = classifierXG.predict_proba(xTest)
scoreXG = utils.getScore(yTest, yForecastXG)
print('XGBClassifier accuracy: ', scoreXG['Accuracy'])

###############################################################################

# take random observations to reduce data size
x, y = utils.cutSet(xTrainFull, yTrainFull, 1000)

print("LogisticRegression") 
space = np.linspace(1, 100, 20, dtype=int)
accuracy = []
for i in space:
    classifierLR = LogisticRegression(penalty='l2', dual=False, tol=0.0001,\
        C=i, fit_intercept=False, intercept_scaling=1, class_weight=None,\
        random_state=None, solver='sag', max_iter=10000, multi_class='multinomial',\
        verbose=0, warm_start=False, n_jobs=4)
    classifierLR.fit(x, y)     
    yForecastLR = classifierLR.predict(xTest) 
    scoreLR = utils.getScore(yTest, yForecastLR)
    accuracy.append(scoreLR['Accuracy'])
    print('{} {} {} {}'.format('C=', i, 'Accuracy=', scoreLR['Accuracy']))

print('Best C = ', space[np.argmax(accuracy)])

best_index = np.argmax(accuracy)
best_score = accuracy[best_index]

# Plot grid search progress
plt.figure(figsize=(10, 10))
title = "MNIST. LogisticRegression Grid search evaluating"
plt.title(title, fontsize=16)
plt.xlabel("C")
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
classifierLR = LogisticRegression(penalty='l2', dual=False, tol=0.0001,\
    C=space[best_index], fit_intercept=False, intercept_scaling=1, class_weight=None,\
    random_state=None, solver='sag', max_iter=1000, multi_class='multinomial',\
    verbose=0, warm_start=False, n_jobs=4)
classifierLR.fit(xTrain, yTrain)     
yForecastLR = classifierLR.predict(xTest) 
yForecastProbaLR = classifierLR.predict_proba(xTest)
scoreLR = utils.getScore(yTest, yForecastLR)
print('LogisticRegression best accuracy: ', scoreLR['Accuracy'])

###############################################################################

# take random observations to reduce data size
x, y = utils.cutSet(xTrainFull, yTrainFull, 1000)

print("KNeighborsClassifier") 
space = np.linspace(1, 20, 20, dtype=int)
accuracy = []
for i in space:
    classifierKNN = KNeighborsClassifier(n_neighbors=i, weights='uniform',\
        algorithm='auto', leaf_size=30, p=2, metric='minkowski')
    classifierKNN.fit(x, y)       
    yForecastKNN = classifierKNN.predict(xTest) 
    scoreKNN = utils.getScore(yTest, yForecastKNN)
    accuracy.append(scoreKNN['Accuracy'])
    print('{} {} {} {}'.format('C=', i, 'Accuracy=', scoreKNN['Accuracy']))

print('Best n_neighbors = ', space[np.argmax(accuracy)])

best_index = np.argmax(accuracy)
best_score = accuracy[best_index]

# Plot grid search progress
plt.figure(figsize=(10, 10))
title = "MNIST. KNeighborsClassifier Grid search evaluating"
plt.title(title, fontsize=16)
plt.xlabel("n_neighbors")
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
classifierKNN.fit(xTrain, yTrain)       
yForecastKNN = classifierKNN.predict(xTest) 
yForecastProbaKNN = classifierKNN.predict_proba(xTest)
scoreKNN = utils.getScore(yTest, yForecastKNN)
print('KNeighborsClassifier best accuracy: ', scoreKNN['Accuracy'])


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

