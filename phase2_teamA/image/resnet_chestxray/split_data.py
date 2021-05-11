import os
import pandas as pd
import numpy as np
import random
import cv2

from resnet_chestxray import model_utils


current_dir = os.path.dirname(__file__)

correct_mimicids = pd.read_csv('data/final_cohort_with_mimicids.csv')
cohort = pd.read_csv('../final_cohort_with_imageids.csv')
#cohort['mimic_id'] = cohort.apply(lambda row: str(row['subject_id'])+'_'+str(row['last_study_id'])+'_'+str(row['last_dicom_id']), axis = 1)
#print(cohort.mimic_id.value_counts())
#cohort = cohort[cohort.mimic_id.isin(correct_mimicids.mimic_id)]
#cohort = cohort.drop('mimic_id', axis = 1)
#print(cohort.shape)

label = '30d_hf'
labels = cohort[label]
input(cohort[label].value_counts())
n=len(cohort)
trains = [1]*int(n*0.8)+[0]*(n- int(0.8*n))
#random.Random(4).shuffle(trains)
cohort['train'] = trains
print(cohort.columns)
cohort = cohort.rename({"last_study_id":"study_id", "last_dicom_id": "dicom_id"},axis = 1)


print(cohort.columns)
cohort[cohort.train == 1][['subject_id', 'study_id', 'dicom_id']].to_csv('data/train_metadata.csv', index = False)
cohort[cohort.train == 1][['study_id',label]].to_csv('data/train_'+label+'_labels.csv', index =False)
cohort[cohort.train == 0][['subject_id', 'study_id', 'dicom_id']].to_csv('data/test_metadata.csv', index = False)
cohort[cohort.train == 0][['study_id',label]].to_csv('data/test_'+label+'_labels.csv', index = False)

train_metadata = os.path.join(current_dir, 'data/train_metadata.csv')
save_path = os.path.join(current_dir, 'data/training_'+label+'.csv')
train_labels = os.path.join(current_dir, 'data/train_'+label+'_labels.csv')

model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=train_metadata,
													label_metadata=train_labels,
													data_key='study_id',
												        label_key=[label], 
                                                                                                        mimiccxr_selection = None,
                                                                                                        save_path=save_path)
data = pd.read_csv(save_path)
data = data[data.mimic_id.isin(correct_mimicids.mimic_id)]
print(data.shape)
print(data.head())
data.to_csv(save_path)

'''
data = pd.read_csv(save_path)
default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/'
counter_None = 0
other_counter = 0
input(len(data.mimic_id))
bad_mimic_ids_train = []
for img_id in data.mimic_id:
    png_path = os.path.join(default, f'{img_id}.png')
    print(png_path)
    img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
    print(img)
    if img is None:
        counter_None += 1
        bad_mimic_ids_train.append(img_id)
    else:
        other_counter += 1
print(counter_None)
print(other_counter)
data = data[~data.mimic_id.isin(bad_mimic_ids_train)]
print(data.shape)
print(data.head())
data.to_csv(save_path)
'''

test_metadata = os.path.join(current_dir, 'data/test_metadata.csv')
save_path = os.path.join(current_dir, 'data/test_'+label+'.csv')
test_labels = os.path.join(current_dir, 'data/test_'+label+'_labels.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=test_metadata,
													label_metadata=test_labels,
													data_key='study_id',
                                                                                                        label_key=[label],
                                                                                                        mimiccxr_selection = None,
													save_path=save_path)

data = pd.read_csv(save_path)
data = data[data.mimic_id.isin(correct_mimicids.mimic_id)]
print(data.shape)
print(data.head())
data.to_csv(save_path)

'''
ONLY RUN ONCE
data = pd.read_csv(save_path)
default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/'
counter_None = 0
other_counter = 0
input(len(data.mimic_id))
bad_mimic_ids_test = []
for img_id in data.mimic_id:
    png_path = os.path.join(default, f'{img_id}.png')
    print(png_path)
    img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
    print(img)
    if img is None:
        counter_None += 1
        bad_mimic_ids_test.append(img_id)
    else:
        other_counter += 1
print(counter_None)
print(other_counter)
data = data[~data.mimic_id.isin(bad_mimic_ids_test)]
print(data.shape)
print(data.head())
data.to_csv(save_path)

cohort = pd.read_csv('data/final_cohort_with_mimicids.csv')
cohort = cohort[~cohort.mimic_id.isin(bad_mimic_ids_test+bad_mimic_ids_train)]
cohort.to_csv('data/final_cohort_with_mimicids.csv')
'''
