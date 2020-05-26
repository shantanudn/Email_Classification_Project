# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:49:51 2020

@author: vw178e
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('C:/Users/vw178e/Excelr_Project/New_env/Deployment/emails1.csv')
df=dataset.iloc[:,3:] # removing first three columns

# removing duplicate data
df.drop_duplicates(subset='content', keep='first', inplace=True)
df['Class'].value_counts()
df.describe()

df1 = pd.read_csv('C:/Users/vw178e/Excelr_Project/New_env/Deployment/new_abusive_dataset.csv')

frames=[df, df1]

df_new= pd.concat(frames)

# creating new dataset
df_new.to_csv("email_new1.csv",encoding="utf-8")

df = pd.read_csv('email_new1.csv')

df=df.iloc[:, 1:]
df.drop_duplicates(subset='content', keep='first', inplace=True)
df['Class'].value_counts()
df.describe()




# =============================================================================
# Modelling
# =============================================================================

from sklearn.svm import LinearSVC

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split, cross_val_score

X = df['content']
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Variable shapes
X_train.shape, y_train.shape, X_test.shape, y_test.shape


#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range=[1,3])
X_train_transformed = vect.fit_transform(X_train)
X_test_transformed = vect.transform(X_test)
X_train_transformed.shape
X_test_transformed.shape


#Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


### SVM
svm_model = LinearSVC()
svm_model.fit(X_train_transformed,y_train)
train_pred_svc = svm_model.predict(X_train_transformed)
accuracy_train_svc = np.mean(train_pred_svc==y_train) 

y_preds = svm_model.predict(X_test_transformed)
accuracy_test_svc = np.mean(y_preds==y_test)

print(classification_report(y_test,y_preds))
pd.crosstab(y_test,y_preds)


# Import Seaborn
import seaborn as sns
sns.set(font_scale=1.5) # Increase font size

def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+0.5, top - 0.5)

plot_conf_mat(y_test, y_preds)    

#Plot ROC curve and calculate Auc metric
plot_roc_curve(svm_model,test_matrix,y_test)

# applying SMOTE to balance the data
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train1, y_train1 = sm.fit_sample(X_train_transformed, y_train)
y_train1.value_counts()
y_train.value_counts()

svm_model = LinearSVC()
svm_model.fit(X_train1,y_train1)
train_pred_svc = svm_model.predict(X_train1)
accuracy_train_svc = np.mean(train_pred_svc==y_train1) 
accuracy_train_svc
y_preds = svm_model.predict(X_test_transformed)
accuracy_test_svc = np.mean(y_preds==y_test)

print(classification_report(y_test,y_preds))
pd.crosstab(y_test,y_preds)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svm_model, X = X_train1, y = y_train1, cv = 10)
accuracies.mean()  # 0.9961787365177196 
accuracies.std() #0.0011791141121546213

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.1, 1, 10, 100], 'loss':['hinge', 'squared_hinge'], 'penalty': ['l1', 'l2']}]#{'C': [0.01, 0.1, 1], 'kernel': ['rbf'], 'gamma': [0.1, 1]}]
grid_search = GridSearchCV(estimator = svm_model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train1, y_train1)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# with tuned parameters
svm_model = LinearSVC(C=10, loss= 'squared_hinge', penalty= 'l2')
svm_model.fit(X_train1,y_train1)
train_pred_svc = svm_model.predict(X_train1)
accuracy_train_svc = np.mean(train_pred_svc==y_train1) 

y_preds = svm_model.predict(X_test_transformed)
accuracy_test_svc = np.mean(y_preds==y_test)

print(classification_report(y_test,y_preds))
pd.crosstab(y_test,y_preds)

# final model
from sklearn.feature_extraction.text import TfidfVectorizer
#vect = TfidfVectorizer(ngram_range=[1,3])
#
#X_transformed = vect.fit_transform(X)

X_transformed = TfidfVectorizer(ngram_range=[1,3]).fit(X)
# =============================================================================
# # Saving pickle transformer
# =============================================================================
import pickle
pickle.dump(X_transformed, open("transformer42.pkl", "wb"))


sm = SMOTE(random_state=42)
X2, y2 = sm.fit_sample(X1, y)
y.value_counts()
y2.value_counts()

# with tuned parameters
svm_model = LinearSVC(C=10, loss= 'squared_hinge', penalty= 'l2')
svm_model.fit(X_transformed,y)


pred_svc = svm_model.predict(X_transformed)
accuracy_svc = np.mean(pred_svc==y) #0.999935367114788

print(classification_report(y,pred_svc))
pd.crosstab(y,pred_svc)

# Saving model to disk
pickle.dump(svm_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict(vect.transform(['Kindly send mail today'])))


print(model.predict(vect.transform(['you fucker'])))

print(model.predict(vect.transform(['bitches please'])))

