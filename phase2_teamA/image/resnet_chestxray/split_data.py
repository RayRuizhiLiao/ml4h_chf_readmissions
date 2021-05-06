import os
import pandas as pd
import numpy as np
import random

from resnet_chestxray import model_utils


current_dir = os.path.dirname(__file__)

cohort = pd.read_csv('data/final_cohort_with_imageids.csv')
label = '30d_hf'
labels = cohort[label]
n=len(cohort)
trains = [0]*int(n*0.7)+[1]*(n- int(0.7*n))
random.shuffle(trains)
cohort['train'] = trains
print(cohort.columns)
cohort = cohort.rename({"last_study_id":"study_id", "last_dicom_id": "dicom_id"},axis = 1)


print(cohort.columns)
cohort[cohort.train == 1][['subject_id', 'study_id', 'dicom_id']].to_csv('data/train_metadata.csv')
cohort[cohort.train == 1][['study_id',label]].to_csv('data/train_labels.csv')
cohort[cohort.train == 0][['subject_id', 'study_id', 'dicom_id']].to_csv('data/test_metadata.csv')
cohort[cohort.train == 0][['study_id',label]].to_csv('data/test_labels.csv')

train_metadata = os.path.join(current_dir, 'data/train_metadata.csv')
save_path = os.path.join(current_dir, 'data/training_readmission.csv')
train_labels = os.path.join(current_dir, 'data/train_labels.csv')

model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=train_metadata,
													label_metadata=train_labels,
													data_key='study_id',
												        label_key=[label], 
                                                                                                        mimiccxr_selection = None,
                                                                                                        save_path=save_path)

test_metadata = os.path.join(current_dir, 'data/test_metadata.csv')
save_path = os.path.join(current_dir, 'data/test_readmission.csv')
test_labels = os.path.join(current_dir, 'data/test_labels.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=test_metadata,
													label_metadata=test_labels,
													data_key='study_id',
                                                                                                        label_key=[label],
                                                                                                        mimiccxr_selection = None,
													save_path=save_path)

assert 1==2
'''
Split data using both chexpert labels and edema severity labels
'''
chexpert_labels = os.path.join(current_dir, 'mimic_cxr_labels/mimic-cxr-2.0.0-chexpert.csv')
test_edema = os.path.join(current_dir, 'data/test_edema.csv')
train_edema = os.path.join(current_dir, 'data/training_edema.csv')
chexpert_split =  os.path.join(current_dir, 'mimic_cxr_labels/mimic-cxr-2.0.0-split.csv')

test_edema = pd.read_csv(test_edema)
train_edema = pd.read_csv(train_edema)
test_edema_subjectid = [id[1:9] for id in test_edema.mimic_id.tolist()]
train_edema_subjectid = [id[1:9] for id in train_edema.mimic_id.tolist()]

chexpert_split = pd.read_csv(chexpert_split)
test_chexpert_split = chexpert_split[chexpert_split['split'].isin(['test'])].reset_index(drop=True)
train_chexpert_split = chexpert_split[chexpert_split['split'].isin(['train'])].reset_index(drop=True)

test_chexpert_split_edema = test_chexpert_split[~test_chexpert_split['subject_id'].apply(str).isin(train_edema_subjectid)].reset_index(drop=True)
test_chexpert_split_edema_subjectid = test_chexpert_split_edema['subject_id'].apply(str).tolist()

test_final_subjectid = test_chexpert_split_edema_subjectid+test_edema_subjectid

chexpert_labels = pd.read_csv(chexpert_labels)
chexpert_labels = chexpert_labels.replace(np.nan, -2)
test_chexpert = chexpert_labels[chexpert_labels['subject_id'].apply(str).isin(test_final_subjectid)].reset_index(drop=True)
test_chexpert.to_csv(os.path.join(current_dir, 'data/test_chexpert_tmp.csv'), index=False)
train_chexpert = chexpert_labels[~chexpert_labels['subject_id'].apply(str).isin(test_final_subjectid)].reset_index(drop=True)
train_chexpert.to_csv(os.path.join(current_dir, 'data/training_chexpert_tmp.csv'), index=False)

chexpert_keys = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',\
				 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',\
				 'Pneumothorax', 'Support Devices']
save_path = os.path.join(current_dir, 'data/training_chexpert.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=os.path.join(current_dir, 'data/training_chexpert_tmp.csv'),
													data_key='study_id',
													label_key=chexpert_keys,
													save_path=save_path)
save_path = os.path.join(current_dir, 'data/test_chexpert.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=os.path.join(current_dir, 'data/test_chexpert_tmp.csv'),
													data_key='study_id',
													label_key=chexpert_keys,
													save_path=save_path)
