# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:34:41 2021

@author: Kumar Rupesh
"""

import numpy as np
import pandas as pd


    
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def import_or_install(package):
    try:
        __import__(package)
        #print(package + " is already installed.")
    except Exception as e: 
        print(e)
        install(package)
        
import_or_install("sys")
import_or_install("os")
import_or_install("subprocess")

import_or_install("matplotlib")
import_or_install("datetime")
import_or_install("sklearn")
import_or_install("pandas_datareader")
import_or_install("matplotlib")
import_or_install("warnings")


# from pylab import plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# plt.style.use('seaborn')
#%matplotlib inline
#pip install pandas_datareader
import pandas_datareader
from pandas_datareader import data as web
import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold



params = {'axes.titlesize':'30',
          'xtick.labelsize':'24',
          'ytick.labelsize':'24'}
matplotlib.rcParams.update(params)

   



def fetch():
    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2018, 1, 1)
    input_features = pd.DataFrame(web.DataReader('GOOG','yahoo', start, end)['Close'])
    input_features.columns= ['prices']
    
    fig =input_features.plot(figsize=(6, 6),lw=2, colormap='jet', marker='.', markersize=5,  title='Historical Prices',fontsize=16).get_figure();
    fig.savefig('prices.png')
    
    input_features['returns'] = np.log(input_features / input_features.shift(1))
    input_features['momentum'] = input_features['prices'] - input_features['prices'].shift(1)
    input_features.dropna(inplace=True)
    input_features['returnsign'] = np.sign(input_features['returns'])
    input_features['returnsignY'] = np.sign(input_features['returns'].shift(-1))
    #input_features['returnsignY'] = int(input_features['returnsignY'])
    input_features['MovingAverage'] = input_features['prices'].rolling(window=20).mean()
    
    cols = [ 'momentum', 'MovingAverage', 'returns']
    lags =4
    for lag in range(1, lags+1):
        col = 'ret_lag_%d' % lag
        input_features[col] = input_features['returns'].shift(lag)
        cols.append(col)
        
    input_features.dropna(inplace=True)

    
    return input_features,cols
    

def train():
    #Train the model:
    X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(input_features[cols], input_features['returnsignY'], test_size=0.50, shuffle=True)
    
    logit_P= linear_model.LogisticRegression(max_iter=10000, C = 1e5,penalty='l2') #Re-estimate on Training Dataset
    logit_P.fit(X_Train, Y_Train)
   
    logit_P.fit(input_features[cols], input_features['returnsignY'])

    
    return X_Train,Y_Train, X_Test,Y_Test, logit_P
    
def CustomizedPredict(Y_Probab):
   
    Pred = Y_Probab[:,1]
   
    Pred[Pred >= 0.55] = 1
    Pred[Pred < 0.55] = -1
   
    return Pred
    
    
def plot_confusion(cm, target_names, title,    cmap=plt.cm.Blues):
    cm_norm = cm * 1. / cm.sum(axis=1)[:, np.newaxis] # standardize the confusion matrix
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ticks = np.arange(len(target_names))
    plt.xticks(ticks, target_names, rotation=45)
    plt.yticks(ticks, target_names)
    plt.xlabel('Predicted',fontsize=32)
    
    plt.ylabel('Actual',fontsize=32)
    plt.title(title, fontsize=40)
    plt.tight_layout()
    plt.savefig('confmatrix.png')
    
    

def plot_logistic(return_ind):
    
    return_ind_name = ['Negative ', 'Positive '] # This is in terms of label for Positives for Roc Curve
    
    # Plot training data only
    predict_prob = logit_P.predict_proba(X_Train)
    Y_Train_Pred = CustomizedPredict(predict_prob)
       
    logit_roc_aucTr = roc_auc_score(Y_Train,Y_Train_Pred ) #Here, logit_googl_Test can run on Test or repeated on Train
    fprTr, tprTr, thresholdsTr = roc_curve(Y_Train, logit_P.predict_proba(X_Train)[:,1], pos_label=return_ind)
    
    
    #Plot validation Data only
    predict_prob = logit_P.predict_proba(X_Test)
    Y_Test_Pred = CustomizedPredict(predict_prob)
       
    logit_roc_aucT = roc_auc_score(Y_Test,Y_Test_Pred ) #Here, logit_googl_Test can run on Test or repeated on Train
    fprT, tprT, thresholdsT = roc_curve(Y_Test, logit_P.predict_proba(X_Test)[:,1], pos_label=return_ind)
    
     #Plot completed population
    predict_prob = logit_P.predict_proba(input_features[cols])
    Y_Pred = CustomizedPredict(predict_prob)
    
    logit_roc_aucP = roc_auc_score(np.sign(input_features['returnsignY']), Y_Pred)
    fprP, tprP, thresholdsP = roc_curve(np.sign(input_features['returnsignY']), logit_P.predict_proba(input_features[cols])[:,1], pos_label=return_ind)
    #print(tprT)
    
    
    
    
    fig, ax = plt.subplots(figsize=(14,14)) #fig = plt.figure(figsize=(18,10))
    

    
    
    
# 1) Plot a diagnoal line of fully random classifier
    ax.plot([0, 1], [0, 1],'r--', label='Random Classifier')
# 1) Plot ROC Curve for the precictions on Test Dataset
    ax.plot(fprTr, tprTr, linewidth=10, label='Train data: Logistic Regression (area = %2.8f)' % logit_roc_aucTr)
    ax.plot(fprT, tprT, linewidth=10, label='Test data: Logistic Regression (area = %0.8f)' % logit_roc_aucT)
# 1) Plot ROC Curve for the Full Dataset (Population)
    ax.plot(fprP, tprP, linewidth=10, label='Population data: Logistic Regression (area = %02.8f)' % logit_roc_aucP)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    
    plt.xlabel('False ' + return_ind_name[return_ind] + ' Rate', fontsize=30)
    plt.ylabel('True ' + return_ind_name[return_ind] + ' Rate', fontsize=30)
    plt.title("Area Under ROC", fontsize=50)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig('roc.png')
    plt.show(block=False)
    return ax
    
    
def evaluate():
    #X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(input_features[cols], input_features['returnsign'], test_size=0.50, shuffle=True)
    
    #logit_googl_Test = linear_model.LogisticRegression(C=1e5) #Re-estimate on Training Dataset
    #logit_googl_Test.fit(X_Train, Y_Train) ###FITTING DONE HERE
#Y_Pred = logit_googl_Test.predict(X_Test)

    predict_prob = logit_P.predict_proba(X_Test)
    
    Y_Test_Pred=CustomizedPredict(predict_prob)
 
    conmatrix = confusion_matrix(Y_Test, Y_Test_Pred)
    print("Below is the confusion matrix of validation data using model fitt on training data")
    print(conmatrix)
    
    plt.clf
    plt.figure(figsize=(12, 12), facecolor='w')
    plt.subplot(111)
    plot_confusion(conmatrix, ['Negative move','Positive move'], 'Confusion Matrix - Test Data')
   
    plt.show(block=False)
 
    plot_logistic(1)
    print("\nFigures have been saved in the script directory..... ")
    
    

def build_paper():
    print("Generating PDF reports")
    os.system("pdflatex Card.tex")
    os.system("pdflatex Paper.tex")
    
    print("\nThe PDFs Card and Paper have been saved in the script directory..... ")
    
    
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    
    
print("######################################################################")

param = sys.argv[1]
   
if param == "fetch":
    print("Fetching dataset for training.........\n\n")
    input_features,cols =fetch()
    print("Below are top 10 records of fetched dataset\n\n")
    print(input_features.head(10))
      
print("\n######################################################################")
      
if param == "train":
    print("Running training............\n\n")
    input_features,cols =fetch()
    X_Train,Y_Train,X_Test,Y_Test,logit_P = train()
    print("Below are the fitted logistic regression coefficients \n")
    print(logit_P.coef_)
    print("\n\n")
    print("Below are the fitted logistic regression intercept  \n")
    print(logit_P.intercept_)
       
      
      
if param == "evaluate":
    print("Running evaluation............\n\n")
    input_features,cols =fetch()
    X_Train,Y_Train,X_Test,Y_Test,logit_P = train()
    evaluate()
    
if param == "build_paper":
    os.system("pdflatex Card.tex")
    os.system("pdflatex paper.tex")
      
      

      
      
      
      
      #input_features = pd.DataFrame()
      
    
    
    





