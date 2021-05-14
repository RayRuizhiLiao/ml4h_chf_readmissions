import pandas as pd
import nltk
nltk.download('punkt')
import re
import numpy as np
from nltk import sentiment
from nltk.sentiment import SentimentIntensityAnalyzer

reports_train = []
labels_train = []
reports_test = []
labels_test = []
x = pd.read_excel("adm_with_cxr.xlsx")
training = set()
#load data
#ensure that the same subject_id does not appear in the training and testing

most_pos = 0
report_most_pos = 0
for i in range(len(x["path"])):
    try:
        f = open(x["path"][i], "r")
        if x["subject_id"][i] not in training:
          training.add(x["subject_id"][i])
          rreport = f.read()
          #define sentiment analysis for each report
          sia = SentimentIntensityAnalyzer()
          sentiment = sia.polarity_scores(rreport)
          if sentiment['pos'] > most_pos:
            most_pos = sentiment['pos']
            report_mos_pos = rreport
          reports_train.append(list(sentiment.values())[:3])
          labels_train.append([x["48h_hf"][i],x["14d_hf"][i],x["30d_hf"][i],x["er_hf"][i],x["48h"][i],x["14d"][i],x["30d"][i],x["er"][i]])
        else:
          rreport = f.read()
          sia = SentimentIntensityAnalyzer()
          sentiment = sia.polarity_scores(rreport)
          reports_test.append(list(sentiment.values())[:3])
          labels_test.append([x["48h_hf"][i],x["14d_hf"][i],x["30d_hf"][i],x["er_hf"][i],x["48h"][i],x["14d"][i],x["30d"][i],x["er"][i]])
    except:
        continue
  
#print most positive report
print(most_pos)
print(report_mos_pos)
      
#split data

cutoff = len(reports_train)
cutoff_val = int(7/8*len(reports_train))
reports = np.array(reports_train + reports_test)

reports_train = reports[0:cutoff_val]
reports_val = reports[cutoff_val:cutoff]
reports_test = reports[cutoff:]

cutoff_val = int(7/8*len(labels_train))
l = labels_train
labels_train = np.array(l[0:cutoff_val])
labels_val   = np.array(l[cutoff_val:])
labels_test  = np.array(labels_test)

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

#ensure sizes are consistent with split
print('Sizes')
print(reports_train.shape)
print(reports_val.shape)
print(reports_test.shape)

# iterate through each outcome for hyperparameter optimization, maximizing F1 score
def model(i):
  wmax = 0
  fmax = 0
  for w in range(1,250,1):
    clf = RandomForestClassifier(class_weight = {0: 1, 1: w})
    clf.fit(reports_train,labels_train[:,i])
    y_pred = clf.predict(reports_val)
    p, r, f, s = precision_recall_fscore_support(labels_val[:,i],y_pred=y_pred,labels=(0,1))
    if f[1] > fmax: 
      wmax = w
      fmax = f[1]
  clf = RandomForestClassifier()
  clf.fit(reports_train,labels_train[:,i])
  y_pred = clf.predict(reports_test)
  p, r, f, s = precision_recall_fscore_support(labels_test[:,i],y_pred=y_pred,labels=(0,1))
  auc = sklearn.metrics.roc_auc_score(labels_test[:,i],clf.predict_proba(reports_test)[:,1],labels=(0,1))
  return [p[1],r[1],f[1],auc]

iterate through eachmodel, print out final scores
score = []
for i in range(8):
  score.append(model(i))
print(score)
