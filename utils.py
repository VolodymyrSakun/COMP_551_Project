import numpy as np
import pandas as pd
import copy
#import h2o
import scipy.io as sio
import pickle 
from PIL import Image
from skimage.feature import hog

def toBW(images):
    x = np.zeros(shape=images.shape, dtype=np.uint8)
    for i in range(0, len(images), 1):
        a = images[i]
        x[i] = np.where(a == 0, 0, 1).astype(np.uint8)
    return x

def getScore(y_true, y_pred):
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return None
    y = np.concatenate((y_true, y_pred))
    classLabels = np.unique(y) # sorted array
    index = []
    columns = []
    for label in classLabels:
        index.append('{}{}'.format('Predicted ', label))
        columns.append('{}{}'.format('Actual ', label))
    cm = np.zeros(shape=(len(classLabels), len(classLabels)), dtype=int)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 2:
        y_true = y_true.reshape(-1)
    if y_pred.ndim == 2:
        y_pred = y_pred.reshape(-1)
    if len(y_true) != len(y_pred):
        return False
    nTotal = len(y_true)
    for i in range(0, len(y_true), 1):
        y_trueValue = y_true[i]
        y_predValue = y_pred[i]
        row = None
        column = None
        for idx, label in enumerate(classLabels, start=0):# tuples[index, label]
            if label == y_trueValue:
                column = idx
            if label == y_predValue:
                row = idx
        if row is None or column is None:
            print("Crash getScore. Class does not contain label")
            return False
        cm[row, column] += 1 
                    
    cmFrame = pd.DataFrame(cm, index=index, columns=columns, dtype=int)
# sum of all non-diagonal cells of cm / nTotal        
    misclassification = 0
    accuracy = 0
    for i in range(0, len(classLabels), 1):
        for j in range(0, len(classLabels), 1):
            if i == j:
                accuracy += cm[i, j]
                continue
            misclassification += cm[i, j]
    misclassification /= nTotal  
    accuracy /= nTotal 
    out = {}
    for i in range(0, len(classLabels), 1):
        out[classLabels[i]] = {}
        tp_fp = np.sum(cm[i, :])
        tp_fn = np.sum(cm[:, i])
        if tp_fp != 0:
            out[classLabels[i]]['Precision'] = cm[i, i] / tp_fp # tp / (tp + fp)            
        else:
            out[classLabels[i]]['Precision'] = 0
        if tp_fn != 0:            
            out[classLabels[i]]['Recall'] = cm[i, i] / tp_fn  # tp / (tp + fn)    
        else:
            out[classLabels[i]]['Recall'] = 0
        if (out[classLabels[i]]['Precision'] + out[classLabels[i]]['Recall']) != 0:
            out[classLabels[i]]['F1'] = 2 * (out[classLabels[i]]['Precision'] * \
                out[classLabels[i]]['Recall']) / (out[classLabels[i]]['Precision'] + \
                out[classLabels[i]]['Recall'])
        else:
            out[classLabels[i]]['F1'] = 0

    return {'Accuracy': accuracy, 'Misclassification': misclassification,\
            'Confusion_Matrix': cmFrame, 'Labels': out}

def validateFit(estimator, data, columnsX, columnY, nFolds=5, Labels=None):     

    setPure = data.copy(deep=True)
    setPure.reset_index(drop=True, inplace=True) # reset index
    setPure = setPure.reindex(np.random.permutation(setPure.index)) # shuffle 
    setPure.sort_index(inplace=True)
        
    size = setPure.shape[0]
    if size < nFolds:
        return None# too few observations
        
    binSize = int(size / nFolds)
    lo = 0
    hi = binSize-1
    intervals = []
    i = 1
    while True:
        i += 1
        intervals.append((lo, hi))
        if hi == (size-1):
            break
        lo += binSize
        hi += binSize
        if i == nFolds:
            hi = size-1        
    
    scoresValid = []
    for i in range(0, len(intervals), 1):   
        print("Split ", i)
        lo = intervals[i][0]
        hi = intervals[i][1]
            
        setValid = setPure.loc[lo:hi, :]
        intervalsTrain = copy.deepcopy(intervals)
        del(intervalsTrain[i]) # remove interval currently using for test set
        setTrain = None
        # train test split        
        for j in range(0, len(intervalsTrain), 1):   
            low = intervalsTrain[j][0]
            high = intervalsTrain[j][1]
            if setTrain is None:
                setTrain = setPure.loc[low:high, :].copy(deep=True)
            else:
                new = setPure.loc[low:high, :].copy(deep=True)
                setTrain = pd.concat([setTrain, new], axis=0)
        
        scoreValid = None
#        if estimator.__class__.__name__ == 'H2ODeepLearningEstimator': # check it
#            
#            trainH2O = h2o.H2OFrame(setTrain)
#            validH2O = h2o.H2OFrame(setValid)
#    
#            estimator.train(x = columnsX, y = columnY, training_frame = trainH2O)
#                            
#            y_predH2O = estimator.predict(validH2O)
#            y_pred = h2o.as_list(y_predH2O)
#            y_pred = np.where(y_pred['predict'] == 'Yes', 1, 0)
#            y_true_valid = np.where(setValid[columnY] == 'Yes', 1, 0)
#            scoreValid = getScore(y_true_valid, y_pred)
                          
        if estimator.__class__.__name__ == 'RandomForestClassifier' or \
            estimator.__class__.__name__ == 'XGBClassifier' or \
            estimator.__class__.__name__ == 'SVC' or \
            estimator.__class__.__name__ == 'LogisticRegression' or \
            estimator.__class__.__name__ == 'MultinomialNB':
                
                
            y_true_train = setTrain[columnY].values
            estimator.fit(setTrain.loc[:, columnsX].values, y_true_train)
            
            y_true_valid = setValid[columnY].values
            y_pred = estimator.predict(setValid.loc[:, columnsX].values)            
            scoreValid = getScore(y_true_valid, y_pred)
            
        scoresValid.append(scoreValid)
        
    if len(scoresValid) > 0:
        accuracy = 0
        misclassification = 0
        F1 = {}
        for score in scoresValid:
            accuracy += score['Accuracy']
            misclassification += score['Misclassification']
            F1 = {}
            for key, value in score['Labels'].items():
                if key in F1.keys():
                    F1[key] += score['Labels'][key]['F1']
                else:
                    F1[key] = score['Labels'][key]['F1']
        accuracy /= len(scoresValid)
        misclassification /= len(scoresValid)
    # average F1        
    F1Avg = 0
    for f1 in F1.values():
        F1Avg += f1
    F1Avg /= len(F1)

    return {'Accuracy': accuracy, 'Misclassification': misclassification,\
            'F1': F1, 'F1_Average': F1Avg, 'Folds': scoresValid}

def fitPredict(estimator, dataTrain, dataForecast, columnsX, columnY):
    yTrain = dataTrain[columnY].values
    estimator.fit(dataTrain.loc[:, columnsX].values, yTrain)            
    yForecast = estimator.predict(dataForecast.loc[:, columnsX].values)            
    return yForecast

# load cifar-10
# Loaded in this way, each of the batch files contains a dictionary with the 
# following elements:
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 
# 32x32 colour image. The first 1024 entries contain the red channel values,
# the next 1024 the green, and the final 1024 the blue. The image is stored 
# in row-major order, so that the first 32 entries of the array are the red
# channel values of the first row of the image.
#labels -- a list of 10000 numbers in the range 0-9. The number at index i 
# indicates the label of the ith image in the array data.

#The dataset contains another file, called batches.meta. It too contains a 
# Python dictionary object. It has the following entries:
#label_names -- a 10-element list which gives meaningful names to the numeric
# labels in the labels array described above. For example, 
# label_names[0] == "airplane", label_names[1] == "automobile", etc.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def loadSVHN(fileName):
    """
    load SVYN images
    """
    train_data = sio.loadmat(fileName)
    x = train_data['X']
    y = train_data['y']
    return x, y

def cutSet(x, y, N):
    """
    return random N samples from observations
    """
    nFeatures = x.shape[1]
    Set = pd.DataFrame(np.concatenate((x, y.reshape(-1, 1)), axis=1), dtype=x.dtype)
    Set = Set.reindex(np.random.permutation(Set.index)) # shuffle 
    Set.reset_index(drop=True, inplace=True) # reset index  
    x = Set.iloc[0:N, 0:nFeatures].values
    y = Set.iloc[0:N, nFeatures].values
    return x, y

def saveObject(fileName, obj):
    f = open(fileName, "wb")
    pickle.dump(obj, f)
    f.close()
    return

def loadObject(fileName):
    f = open(fileName, "rb")
    obj = pickle.load(f)
    f.close()
    return obj

def normalizeVector(vector):
    """
    substruct mean and divide by standard deviation
    """
    mean = np.mean(vector)
    std = np.std(vector)
    b = vector - mean
    b /= std
    return b

def rgbToBW(fileIn, shape, normalize=True):
    """
    load SVHN images and return in grayscale format
    """
    dataX, dataY = loadSVHN(fileIn)
    if normalize:
        dtype = float
    else:
        dtype = np.uint8
    x = np.zeros(shape=(len(dataY), shape[0] * shape[1]), dtype=dtype)
    for i in range(0, len(dataY), 1):
        a = dataX[:, :, :, i]
        img = Image.fromarray(a)
        gray = img.convert('L')
        arr = np.asarray(gray.getdata(), dtype=dtype)
        if normalize:
            arr = normalizeVector(arr)
        x[i, :] = arr[:]
    return x, dataY

def rgbToHOG(fileIn, pixelsPerCell=(8, 8)):
    """
    read SVHN images and return Histogram of Oriented Gradients
    """
    dataX, dataY = loadSVHN(fileIn)
    hogFeatures = []
    for i in range(0, len(dataY), 1):
        imageRGB = dataX[:, :, :, i]
        hogFeature = hog(imageRGB, orientations=8, pixels_per_cell=pixelsPerCell, block_norm='L2-Hys',
            visualize=False, feature_vector=True, cells_per_block=(1, 1), multichannel=True)
        hogFeatures.append(hogFeature)

    return np.array(hogFeatures), dataY.reshape(-1)
