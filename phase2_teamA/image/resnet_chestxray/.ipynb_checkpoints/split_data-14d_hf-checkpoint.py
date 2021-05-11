import os
import pandas as pd
import numpy as np
import random
import cv2
from resnet_chestxray import model_utils


current_dir = os.path.dirname(__file__)

# cohort = pd.read_csv('data/final_cohort_with_imageids.csv')
#cohort = pd.read_csv(os.path.join(current_dir,'data/final_cohort_with_imageids.csv'))
correct_mimicids = pd.read_csv('data/final_cohort_with_mimicids.csv')
cohort = pd.read_csv('../final_cohort_with_imageids.csv')
label = 'er_hf'
labels = cohort[label]
n=len(cohort)
trains = [1]*int(n*0.8)+[0]*(n- int(0.8*n))
#random.Random(4).shuffle(trains)
cohort['train'] = trains
print(cohort.columns)
cohort = cohort.rename({"last_study_id":"study_id", "last_dicom_id": "dicom_id"},axis = 1)

cohort[cohort.train == 1][['subject_id', 'study_id', 'dicom_id']].to_csv(f"data/train_metadata-{label}.csv")
cohort[cohort.train == 1][['study_id',label]].to_csv(f"data/train_labels-{label}.csv")
cohort[cohort.train == 0][['subject_id', 'study_id', 'dicom_id']].to_csv(f"data/test_metadata-{label}.csv")
cohort[cohort.train == 0][['study_id',label]].to_csv(f"data/test_labels-{label}.csv")

train_metadata = os.path.join(current_dir, f"data/train_metadata-{label}.csv")
save_path = os.path.join(current_dir, f"data/train_readmission-{label}.csv")
train_labels = os.path.join(current_dir, f"data/train_labels-{label}.csv")

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
## TODO remove images which cant be found 
# data = pd.read_csv(save_path)
# default='physionet.org/files/mimic-cxr-jpg/2.0.0/files/images/'
# counter_None = 0
# other_counter = 0
# input(len(data.mimic_id))
# bad_mimic_ids = []
# for img_id in data.mimic_id:
#     png_path = os.path.join(default, f'{img_id}.png')
#     print(png_path)
#     img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
#     print(img)
#     if img is None:
#         counter_None += 1
#         bad_mimic_ids.append(img_id)
#     else:
#         other_counter += 1
# print(counter_None)
# print(other_counter)
# data = data[~data.mimic_id.isin(bad_mimic_ids)]
# print(data.shape)
# print(data.head())
# data.to_csv(save_path)


test_metadata = os.path.join(current_dir, f"data/test_metadata-{label}.csv")
save_path = os.path.join(current_dir, f"data/test_readmission-{label}.csv")
test_labels = os.path.join(current_dir, f"data/test_labels-{label}.csv")
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
# assert 1==2
'''
Split data using both chexpert labels and edema severity labels
'''
# chexpert_labels = os.path.join(current_dir, 'mimic_cxr_labels/mimic-cxr-2.0.0-chexpert.csv')
# test_edema = os.path.join(current_dir, 'data/test_edema.csv')
# train_edema = os.path.join(current_dir, 'data/training_edema.csv')
# chexpert_split =  os.path.join(current_dir, 'mimic_cxr_labels/mimic-cxr-2.0.0-split.csv')

# test_edema = pd.read_csv(test_edema)
# train_edema = pd.read_csv(train_edema)
# test_edema_subjectid = [id[1:9] for id in test_edema.mimic_id.tolist()]
# train_edema_subjectid = [id[1:9] for id in train_edema.mimic_id.tolist()]

# chexpert_split = pd.read_csv(chexpert_split)
# test_chexpert_split = chexpert_split[chexpert_split['split'].isin(['test'])].reset_index(drop=True)
# train_chexpert_split = chexpert_split[chexpert_split['split'].isin(['train'])].reset_index(drop=True)

# test_chexpert_split_edema = test_chexpert_split[~test_chexpert_split['subject_id'].apply(str).isin(train_edema_subjectid)].reset_index(drop=True)
# test_chexpert_split_edema_subjectid = test_chexpert_split_edema['subject_id'].apply(str).tolist()

# test_final_subjectid = test_chexpert_split_edema_subjectid+test_edema_subjectid

# chexpert_labels = pd.read_csv(chexpert_labels)
# chexpert_labels = chexpert_labels.replace(np.nan, -2)
# test_chexpert = chexpert_labels[chexpert_labels['subject_id'].apply(str).isin(test_final_subjectid)].reset_index(drop=True)
# test_chexpert.to_csv(os.path.join(current_dir, 'data/test_chexpert_tmp.csv'), index=False)
# train_chexpert = chexpert_labels[~chexpert_labels['subject_id'].apply(str).isin(test_final_subjectid)].reset_index(drop=True)
# train_chexpert.to_csv(os.path.join(current_dir, 'data/training_chexpert_tmp.csv'), index=False)

# chexpert_keys = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',\
# 				 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',\
# 				 'Pneumothorax', 'Support Devices']
# save_path = os.path.join(current_dir, 'data/training_chexpert.csv')
# model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
# 													label_metadata=os.path.join(current_dir, 'data/training_chexpert_tmp.csv'),
# 													data_key='study_id',
# 													label_key=chexpert_keys,
# 													save_path=save_path)
# save_path = os.path.join(current_dir, 'data/test_chexpert.csv')
# model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
# 													label_metadata=os.path.join(current_dir, 'data/test_chexpert_tmp.csv'),
# 													data_key='study_id',
# 													label_key=chexpert_keys,
# 													save_path=save_path)
