import os
import pandas as pd
import numpy as np
import random
import cv2

from resnet_chestxray import model_utils


current_dir = os.path.dirname(__file__)

cohort = pd.read_csv('data/final_cohort_with_imageids.csv')
label = '30d_hf'
labels = cohort[label]
n=len(cohort)
<<<<<<< HEAD
<<<<<<< HEAD
trains = [0]*int(n*0.7)+[1]*(n- int(0.7*n))
=======
trains = [1]*int(n*0.7)+[0]*(n- int(0.7*n))
>>>>>>> f132e89ef91fdc37183c3fd536be7759ce7e8690
=======
trains = [1]*int(n*0.7)+[0]*(n- int(0.7*n))
>>>>>>> f132e89ef91fdc37183c3fd536be7759ce7e8690
random.Random(4).shuffle(trains)
cohort['train'] = trains
print(cohort.columns)
cohort = cohort.rename({"last_study_id":"study_id", "last_dicom_id": "dicom_id"},axis = 1)


print(cohort.columns)
cohort[cohort.train == 1][['subject_id', 'study_id', 'dicom_id']].to_csv('data/train_metadata.csv')
cohort[cohort.train == 1][['study_id',label]].to_csv('data/train_'+label+'_labels.csv')
cohort[cohort.train == 0][['subject_id', 'study_id', 'dicom_id']].to_csv('data/test_metadata.csv')
cohort[cohort.train == 0][['study_id',label]].to_csv('data/test_'+label+'_labels.csv')

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
default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/'
counter_None = 0
other_counter = 0
input(len(data.mimic_id))
bad_mimic_ids = []
for img_id in data.mimic_id:
    png_path = os.path.join(default, f'{img_id}.png')
    print(png_path)
    img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
    print(img)
    if img is None:
        counter_None += 1
        bad_mimic_ids.append(img_id)
    else:
        other_counter += 1
print(counter_None)
print(other_counter)
data = data[~data.mimic_id.isin(bad_mimic_ids)]
print(data.shape)
print(data.head())
data.to_csv(save_path)


test_metadata = os.path.join(current_dir, 'data/test_metadata.csv')
save_path = os.path.join(current_dir, 'data/test_'+label+'.csv')
test_labels = os.path.join(current_dir, 'data/test_'+label+'_labels.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=test_metadata,
													label_metadata=test_labels,
													data_key='study_id',
                                                                                                        label_key=[label],
                                                                                                        mimiccxr_selection = None,
													save_path=save_path)
