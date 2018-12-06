To get dataset extract mnist.pkl from mnist.pkl.gzip
Put it into working directory
run file prepareMNIST.py
SubDirectory "mnist" with all data inside will be there

mnistFit.py

Uses several baselines to fit the MNIST dataset 

Classifiers used:

svm with RBF kernel, MultinomialNB, XGBClassifier, LogisticRegression, KNeighborsClassifier, RandomForestClassifier

uses functions from utils.py

Uses train data to train each classifier
Uses validation data to find best hyperparameters for each classifier
Uses grid search to find best parameters
Plots heat maps is two hyperparameters were in grid search
Plots accuracy versus hyperparameter if only one hyperparameter used

When best hyperparameters were found use train + validation set to create model
 
Uses test data to show goodness of fit

svhnFit.py

datasets can be downloaded from here:
http://ufldl.stanford.edu/housenumbers/

Required files should be in current dir/SVHN 

extra_32x32.mat
test_32x32.mat
train_32x32.mat

Uses several baselines to fit the MNIST dataset 
Uses histogram of oriented gradients (HOG) to create features from images

Classifiers used:

svm with RBF kernel, XGBClassifier, KNeighborsClassifier, RandomForestClassifier

Similar procedure to mnistFit.py
