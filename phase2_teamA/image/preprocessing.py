#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from dateutil import parser


# ## Load data

# In[100]:


base_dir = '/data/vision/polina/projects/chestxray/'


# In[102]:


metadata = pd.read_csv('mimic-cxr-2.0.0-metadata.gz')
print(metadata.head())


# In[104]:


#admissions = pd.read_csv(base_dir+'/mimic-iv-1.0/core/admissions.csv', low_memory=False)
#admissions.head()


# In[118]:


admissions = pd.read_csv(base_dir+'/kamineni/ml4h_chf_readmissions/phase1_teamB/final_cohort_with_outcome_labels.csv')
print(admissions.head())


# ## Preprocessing

# ### Extract cxr timestamps

# In[110]:

input(metadata.columns)
input(metadata.ViewPosition.value_counts())
input(metadata.ViewCodeSequence_CodeMeaning.value_counts())
#antero-posterior         146448
#postero-anterior          95858

print(metadata.shape)
metadata = metadata[metadata.ViewCodeSequence_CodeMeaning.isin(['antero-posterior', 'postero-anterior'])]
print(metadata.shape)

# convert StudyDate + StudyTime to datetime objects in 'StudyDateTime'
metadata['StudyTime'] = metadata['StudyTime'].astype(str)
metadata['StudyTime'] = metadata['StudyTime'].apply(lambda t: '0'*(6-t.index('.')) + t)

metadata['StudyDateTime'] = metadata['StudyDate'].astype(str) + 'T' + metadata['StudyTime'].astype(str)
metadata['StudyDateTime'] = metadata['StudyDateTime'].apply(lambda date: parser.parse(date))
print("study time modified")

# ### Extract admissions timestamps 

# In[106]:


# Convert date strings to datetime objects
admissions['admittime'] = admissions['admittime'].map(parser.parse)
admissions['dischtime'] = admissions['dischtime'].map(parser.parse)


# ### Link cxr to hadm (only last study captured for each hadm)

# In[119]:


# isolate cohort admissions
#admissions = admissions[admissions['subject_id'].isin(cohort['subject_id'].unique())]


# In[121]:


len(admissions)


# In[155]:


admissions['last_study_id'] = None
admissions['last_dicom_id'] = None
admissions['last_study_time'] = None


# In[156]:


# add last study_id/dicom_id to corresponding hadm entries
print(len(metadata.index))

for entry in metadata.index:
    meta_row = metadata.loc[entry]
    subject_id = meta_row['subject_id']
    time = meta_row['StudyDateTime']
    subject_adm = admissions[admissions['subject_id'] == subject_id]
    
    for adm_entry in subject_adm.index:
        adm_row = subject_adm.loc[adm_entry]
        if adm_row['admittime'] < time < adm_row['dischtime']:
            if adm_row['last_study_time'] == None or adm_row['last_study_time'] < time:
                admissions.at[adm_entry,'last_study_id'] = meta_row['study_id']
                admissions.at[adm_entry,'last_dicom_id'] = meta_row['dicom_id']
                admissions.at[adm_entry,'last_study_time'] = time

# drop entries without corresponding dicom/study entries
adm_with_cxr = admissions.drop(admissions[admissions['last_study_id'].isnull()].index)
print(adm_with_cxr.columns)

# drop unnecessary columns
adm_with_cxr.drop(columns=['admission_type', 'admission_location', 'discharge_location', 'insurance', 'language', 'marital_status', 'ethnicity', 'hospital_expire_flag'], inplace=True)

adm_with_cxr.to_csv('final_cohort_with_imageids.csv')
len(adm_with_cxr)
print(adm_with_cxr.head())


