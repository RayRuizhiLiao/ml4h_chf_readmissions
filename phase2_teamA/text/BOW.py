import pandas as pd

labels = pd.read_excel("adm_with_cxr.xlsx")

import nltk
import re
import numpy as np

# reports is a list of radiology reports

##reports = []
### nltk.download('punkt')
### use this function to parse by sentence: dataset = nltk.sent_tokenize(text)
##
###convert text to lower case, remove non-word characters, remove punctuation
##for i in range(len(reports)):
##    reports[i] = reports[i].lower()
##    reports[i] = re.sub(r'\W', ' ', reports[i])
##    reports[i] = re.sub(r'\s+', ' ', reports[i])
##
##word2count = {}
##for data in reports:
##    words = nltk.word_tokenize(data)
##    for word in words:
##        if word not in word2count.keys():
##            word2count[word] = 1
##        else:
##            word2count[word] += 1
##
##import heapq
###take the top n words, here n = 100
##freq_words = heapq.nlargest(100, word2count, key=word2count.get)
##print(freq_words)
##
##final_vec = []
##for data in reports:
##    vector = []
##    for word in freq_words:
##        if word in nltk.word_tokenize(data):
##            vector.append(word2count[word]) # to indicate the presence or absense of a word, the code can read instead: vector.append(1)
##        else:
##            vector.append(0)
##    final_vec.append(vector)
##final_vec = np.asarray(final_vec)
###returns a 2D array #reports x vector
##print(final_vec)
