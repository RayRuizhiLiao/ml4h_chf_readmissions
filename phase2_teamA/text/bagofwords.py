import pandas as pd
import nltk
nltk.download('punkt')
import re
import numpy as np

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
          reports_train.append(f.read())
          labels_train.append([x["48h_hf"][i],x["14d_hf"][i],x["30d_hf"][i],x["er_hf"][i],x["48h"][i],x["14d"][i],x["30d"][i],x["er"][i]])
        else:
          reports_test.append(f.read())
          labels_test.append([x["48h_hf"][i],x["14d_hf"][i],x["30d_hf"][i],x["er_hf"][i],x["48h"][i],x["14d"][i],x["30d"][i],x["er"][i]])
    except:
        continue

def ngram(reports, n):
  #convert text to lower case, remove non-word characters, remove punctuation
  for i in range(len(reports)):
      reports[i] = reports[i].lower()
      reports[i] = re.sub(r'\W', ' ', reports[i])
      reports[i] = re.sub(r'\s+', ' ', reports[i])
  
  word2count = {}
  for data in reports:
      words = nltk.word_tokenize(data)
      for i in range(len(words) - (n - 1)):
        word = ""
        for j in range(i,i+n):
          word += words[j] + " "
        word = word[:-1]
        if word not in word2count.keys():
          word2count[word] = 1
        else:
          word2count[word] += 1
  
  import heapq #take the top n words
  freq_words = heapq.nlargest(300, word2count, key=word2count.get)
  # print("Unigram Word Frequency: ", freq_words)
  # print(freq_words)
  
  final_vec = []
  for data in reports:
      vector = []
      for word in freq_words:
          if word in nltk.word_tokenize(data):
              vector.append(word2count[word]) # to indicate the presence or absense of a word, the code can read instead: vector.append(1)
          else:
              vector.append(0)
      final_vec.append(vector)
  final_vec = np.asarray(final_vec) # returns a 2D array #reports x vector
  return final_vec, freq_words

cutoff = len(reports_train)
cutoff_val = int(7/8*len(reports_train))
reports,freq_words = ngram(reports_train + reports_test,1)

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
  
oversample(reports_train,labels_train)

##def model(i):
##  wmax = 0
##  fmax = 0
##  for w in range(2,250,1):
##    clf = RandomForestClassifier(class_weight = {0: 1, 1: w})
##    clf.fit(reports_train,labels_train[:,i])
##    y_pred = clf.predict(reports_val)
##    p, r, f, s = precision_recall_fscore_support(labels_val[:,i],y_pred=y_pred,labels=(0,1))
##    if f[1] > fmax: 
##      wmax = w
##      fmax = f[1]
##  clf = RandomForestClassifier(class_weight = {0: 1, 1: wmax})
##  clf.fit(reports_train,labels_train[:,i])
##  y_pred = clf.predict(reports_test)
##  # print(confusion_matrix(y_true = labels_test[:,i], y_pred = y_pred, labels=(0,1)))
##  p, r, f, s = precision_recall_fscore_support(labels_test[:,i],y_pred=y_pred,labels=(0,1))
##  auc = sklearn.metrics.roc_auc_score(labels_test[:,i],clf.predict_proba(reports_test)[:,1],labels=(0,1))
##  return [p[1],r[1],f[1],auc,clf.feature_importances_]
## 
##score = [] 
##print(freq_words)
##for i in range(8):
##  score.append(model(i))
##print(score)

clf = RandomForestClassifier()
clf.fit(reports_train,labels_train[:,7])
y_pred = clf.predict(reports_test)
p, r, f, s = precision_recall_fscore_support(labels_test[:,7],y_pred=y_pred,labels=(0,1))
auc = sklearn.metrics.roc_auc_score(labels_test[:,7],clf.predict_proba(reports_test)[:,1],labels=(0,1))
print(freq_words)