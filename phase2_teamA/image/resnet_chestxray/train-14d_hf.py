'''
Author: Ruizhi Liao

Main script to run training
'''

import os
import argparse
import logging

import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from resnet_chestxray.main_utils_14d_hf import ModelManager

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int,
					help='Mini-batch size')
parser.add_argument('--num_train_epochs', default=400, type=int,
                    help='Number of training epochs')
parser.add_argument('--loss_method', type=str,
                    default='CrossEntropyLoss',#
                   # default='BCEWithLogitsLoss',
                    # default='BCELoss',
                    help='Loss function for model training')
parser.add_argument('--init_lr', default=5e-4, type=float, 
                    help='Intial learning rate')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=2, type=int,
                    help='The number of ouput channels')
parser.add_argument('--model_architecture', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used')

parser.add_argument('--data_dir', type=str,
					default='physionet.org/files/mimic-cxr-jpg/2.0.0/files/images/',
					help='The image data directory')
parser.add_argument('--dataset_metadata', type=str,
					default=os.path.join(current_dir, 'data/train_readmission-14d_hf.csv'),
					help='The metadata for the model training ')
parser.add_argument('--save_dir', type=str,
					default='physionet.org/files/mimic-cxr-jpg/2.0.0/files/experiments/')
parser.add_argument('--label_key', type=str,
                    default='14d_hf',
                    help='The supervised task (the key of the corresponding label column)')


def train():
	args = parser.parse_args()

	print(args)

	'''
	Check cuda
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

	'''
	Create a sub-directory under save_dir 
	based on the label key
	'''
	args.save_dir = os.path.join(args.save_dir, 
								 args.model_architecture+'_'+args.label_key)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# Configure the log file
	log_path = os.path.join(args.save_dir, 'training.log')
	logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', 
						format='%(asctime)s - %(name)s %(message)s', 
						datefmt='%m-%d %H:%M')

	model_manager = ModelManager(model_name=args.model_architecture, 
								 img_size=args.img_size,
								 output_channels=args.output_channels)

	model_manager.train(device=device,
						args=args)

train()