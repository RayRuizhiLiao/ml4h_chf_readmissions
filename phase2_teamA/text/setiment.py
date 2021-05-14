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
#ensure that the same subject_id does not appear in the training and testing
for i in range(len(x["path"])):
    try:
        f = open(x["path"][i], "r")
        if x["subject_id"][i] not in training:
          training.add(x["subject_id"][i])
          rreport = f.read()
          sia = SentimentIntensityAnalyzer()
          sentiment = sia.polarity_scores(rreport)
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

print('Sizes')
print(reports_train.shape)
print(reports_val.shape)
print(reports_test.shape)

def oversample(reports_train, labels_train):
  x = reports_train
  y = labels_train[:,7]
  y = labels_val[:,7]
  y = labels_test[:,7]

def model(i):
  wmax = 0
  fmax = 0
  for w in range(2,250,1):
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

score = []
for i in range(8):
  score.append(model(i))
print(score)