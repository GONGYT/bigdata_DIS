# -*- coding: utf-8 -*-
'''
Author:       Yutao Gong
Date:         May 1, 2019
Organization: DIS Computational Analysis of Big Data
'''

'''
Import the packages needed for classification
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import skimage.filters as filters
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import os
from scipy import ndimage
plt.close()

'''
Set directory parameters
'''
# Set the directories for the data and the CSV files that contain ids/labels
dir_train_images  = 'C:\\Users\\gongy\\Downloads\\classification\\train'
dir_test_images   = 'C:\\Users\\gongy\\Downloads\\classification\\test'
dir_train_labels  = 'C:\\Users\\gongy\\Downloads\\classification\\train.csv'

'''
Include the functions used for loading, preprocessing, features extraction, 
classification, and performance evaluation
'''

def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory 

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids       = labels_pd.id.values
    data      = []
    
    for identifier in ids:
        fname     = os.path.join(dir_data, identifier)
        image     = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids
    

def preprocess_and_extract_features(data):
    '''Preprocess data and extract features
    
    Preprocess: normalize, scale, repair
    Extract features: transformations and dimensionality reduction
    '''
    
    vectorized_red_data = data[:, :, :, 0].reshape(data.shape[0],-1)
    vectorized_green_data = data[:, :, :, 0].reshape(data.shape[0],-1)
    vectorized_blue_data = data[:, :, :, 0].reshape(data.shape[0],-1)
    
    # Make the image grayscale
    datastd = np.std(data,axis = 3)
    data_sob = ndimage.sobel(data)
    data = np.mean(data, axis=3)
    
    # Vectorize the grayscale matrices
    vectorized_data = data.reshape(data.shape[0],-1)
    vectorized_datastd = datastd.reshape(datastd.shape[0],-1)
    vectorized_sob = data_sob.reshape(data_sob.shape[0],-1)
    
    
    # extract the mean and standard deviation of each sample as features
    feature_mean = np.mean(vectorized_data,axis=1)
    feature_std  = np.std(vectorized_data,axis=1)
    feature_min = np.min(vectorized_data,axis=1)
    feature_max = np.max(vectorized_data,axis=1)
    feature_smean = np.mean(vectorized_datastd,axis=1)
    sob_mean = np.mean(vectorized_sob,axis=1)
    red_mean = np.mean(vectorized_red_data,axis=1)
    green_mean = np.mean(vectorized_green_data,axis=1)
    blue_mean = np.mean(vectorized_blue_data,axis=1)
    sob_std = np.std(vectorized_sob,axis=1)

    
    # Combine the extracted features into a single feature vector
    features = np.stack((feature_mean,feature_std,feature_min,feature_max,feature_smean,
                         sob_mean,sob_std, red_mean, green_mean, blue_mean),axis=-1)
    
    return features

def set_classifier():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    return RandomForestClassifier(n_estimators=100)
    #return KNeighborsClassifier(n_neighbors=11)

def cv_performance_assessment(X,y,k,clf):
    '''Cross validated performance assessment
    
    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use
    
    Divide the training data into k folds of training and validation data. 
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are 
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.empty(y.shape[0],dtype='object')
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    i = 1
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val   = X[train_index], X[val_index]
        y_train          = y[train_index]
        
        # Train the classifier
        X_train_features = preprocess_and_extract_features(X_train)
        clf              = clf.fit(X_train_features,y_train)
        
        # Test the classifier on the validation data for this fold
        X_val_features   = preprocess_and_extract_features(X_val)
        cpred            = clf.predict_proba(X_val_features)
        
        # Save the predictions for this fold
        prediction_scores[val_index] = cpred[:,1]
        print('Training Fold {} of {} Complete'.format(i,k))
        i += 1
    return prediction_scores

def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)
   
    plt.plot([0,1],[0,1],'--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.axis('square')
    plt.legend()
    plt.tight_layout()


# Set parameters for the analysis
num_training_folds = 5

# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)
print('Data Loaded')

# Choose which classifier to use
clf = set_classifier()

# Perform cross validated performance assessment
prediction_scores = cv_performance_assessment(data,labels,num_training_folds,clf)

# Compute and plot the ROC curves
plot_roc(labels, prediction_scores)


'''
Sample script for producing a Kaggle submission
'''

produce_prediction = False # Switch this to True when you're ready to make a prediction
if produce_prediction:
    # Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    training_features              = preprocess_and_extract_features(training_data)
    clf                            = set_classifier()
    clf.fit(training_features,training_labels)

    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training=False)
    test_features  = preprocess_and_extract_features(test_data)
    test_scores    = clf.predict_proba(test_features)[:,1]

    submission_file = pd.DataFrame({'id':    ids,
                                   'score':  test_scores})
    
