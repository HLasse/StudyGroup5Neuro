# MEG classification of word face stimuli
import os
import pandas as pd

import numpy as np 

#from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import naive_bayes, model_selection
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Function to get 3 dimensional array to long format df
def unpack_data(array):
    out = pd.DataFrame()

    for trial in range(0, array.shape[0]):
        temp_df = pd.DataFrame()
        for sensor in range(0, array.shape[1]):
            temp_df[sensor] = array[trial, sensor, :]
        temp_df['trial_n'] = trial
        temp_df['label'] = labels[trial]
        out = out.append(temp_df)


    return out




# Loading data
labels = np.load('pos_neg_img_labels.npy')
mat = np.load('pos_neg_img_trials.npy')
 
""" 
IGNORE ALL THIS

# Reshaping data for use in classifier
#data = data.reshape(-1, data.shape[-2])

#labels2 = np.repeat(labels, 1501)

# Preparing for 10 fold cross-validation
n_trials = mat.shape[0]
n_samples = mat.shape[2]

# Unpacking array to dataframe
data = unpack_data(mat)
data = data.reset_index()
# Creating n folds
fold_n = 10
fold_size = int(n_trials / fold_n)

folds = list(range(0, fold_n))
# Adding a folds column
data['fold'] = np.repeat(folds, (n_samples*n_trials) / fold_n)


# Cross-validation
for fold in range(0, fold_n):
    # Creating training set
    test_data = data[data['fold'] == fold].iloc[:,1:307]
    test_labels = data[data['fold'] == fold]['label']

    train_data = data[data['fold'] != fold].iloc[:,1:307]
    train_labs = data[data['fold'] != fold]['label']

    clf = GaussianNB()
    model = clf.fit(train_data, train_labs)

    predictions = model.predict(test_data)
    correct = [1 if pred == lab else 0 for pred, lab in zip(predictions, test_labels)]


# Creating a model with all data, testing on each timepoint
results = pd.DataFrame()



n_folds = 10
fold_size = int(mat.shape[0] / n_folds)
groups = np.repeat(list(range(0,240)), 1501)

test1 = mat.reshape(-1, mat.shape[-2]) 
test2 = mat.transpose(2,1,0).reshape((mat.shape[0]*mat.shape[2], mat.shape[1]))

labels2 = np.repeat(labels, 1501)
time = np.array(range(mat.shape[0]))
time = np.repeat(time, mat.shape[2])
time = np.sort(time)



from sklearn.model_selection import GroupKFold

group_kfold = GroupKFold(n_splits=10)


for train_index, test_index in group_kfold.split(data, labels2, groups):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label2[train_index], labels2[test_index]

    clf = GaussianNB()

    


for fold in range(0, n_folds):
    test_data = data[fold:fold*size,:]
    test_labels = labels2[fold:fold*fold_size]

    train_data = data[]

# from face plot week 3 tute
n_samples = data.shape[2]
results = pd.DataFrame()

# Creating a model for each time point, testing at the point
for sample_n in range(n_samples):
    dat = data[:, :, sample_n]
    cv_score = cross_val_score(GaussianNB(), dat, labels, cv=10)
    mean_acc = np.mean(cv_score)
    results.at[sample_n, 'sample'] = sample_n
    results.at[sample_n, 'acc'] = mean_acc

    if sample_n % 20 == 0:
        print(sample_n)



# Reducing dimensionality to 2, so we can use it in the classifier
# Essentially, we're making it into long format
#test = trials.reshape(-1, trials.shape[-1])

 """

############-------
# This is the stuff
# -----------------
###########--------
# Pipeline
# linear SVM
svm = make_pipeline(StandardScaler(),  
    LinearSVC(random_state=0, max_iter=100000))

res_svm = [cross_validate(svm, mat[:,:,i], labels, cv = 3, 
    scoring = ['accuracy', 'f1', 'roc_auc'], return_train_score = True) for i in range(mat.shape[2])]

# Creating results dataframe
results = pd.DataFrame()

# The metrics we want to extract
cols = list(res_svm[0].keys())[-6:]

# Adding results to dataframe
for col in cols:
    results[col + '_svm'] = [np.mean(time_point[col]) for time_point in res_svm]
    results[col + '_svm_sd'] = [np.std(time_point[col]) for time_point in res_svm]

# We are using the image data, so time starts 500 ms before stimuli onset
results['time'] = np.arange(-500, 1001, 1)

results.to_csv('MEG_results_svm.csv', index = False, header = True)


# Naive bayes
nb = make_pipeline(StandardScaler(),  
    GaussianNB())

res_nb = [cross_validate(nb, mat[:,:,i], labels, cv = 3, 
    scoring = ['accuracy', 'f1', 'roc_auc'], return_train_score = True) for i in range(mat.shape[2])]






# Easy ugly plot



fig = plt.figure()
ax = plt.axes()
plt.plot(time_points[:10], test_acc)