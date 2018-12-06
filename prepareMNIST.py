import numpy as np
import utils
import os

train, valid, test = utils.unpickle('mnist.pkl')

xTtrain, yTtrain = train[0], train[1]
xValid, yValid = valid[0], valid[1]
xTest, yTest = test[0], test[1]

workingDir = os.getcwd()
setDit = os.path.join(workingDir, "mnist")
try:
    os.mkdir(setDit)   
except:
    pass

np.save(os.path.join(setDit, "train.npy"), xTtrain)
valid = np.save(os.path.join(setDit, "validation.npy"), xValid)
test = np.save(os.path.join(setDit, "test.npy"), xTest)

yTtrain = np.save(os.path.join(setDit, "train_labels.npy"), yTtrain)
yValid = np.save(os.path.join(setDit, "validation_labels.npy"), yValid)
yTest = np.save(os.path.join(setDit, "test_labels.npy"), yTest)

