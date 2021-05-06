'''
Author: Ruizhi Liao

Main script to run evaluation
'''

import os
import argparse
import logging
import json

import torch

from resnet_chestxray.main_utils import ModelManager, build_model

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int,
					help='Mini-batch size')

parser.add_argument('--label_key', default='Edema', type=str,
					help='The label key/classification task')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=1, type=int,
                    help='The number of ouput channels')
parser.add_argument('--model_architecture', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used')

parser.add_argument('--data_dir', type=str,
					default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/',
					help='The image data directory')
parser.add_argument('--dataset_metadata', type=str,
					default=os.path.join(current_dir, 'data/test_chexpert.csv'),
					help='The metadata for the model training ')
parser.add_argument('--save_dir', type=str,
					default='/data/vision/polina/scratch/ruizhi/chestxray/experiments/supervised_image/'\
					'tmp_postmiccai_v2/')
parser.add_argument('--checkpoint_name', type=str,
					default='pytorch_model_epoch300.bin')


def eval(all_epochs=-1):
	args = parser.parse_args()

	print(args)

	'''
	Check cuda
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

	'''
	Create a sub-directory under save_dir 
	based on the label key
	'''
	args.save_dir = os.path.join(args.save_dir, args.model_architecture+'_'+args.label_key)
	
	checkpoint_path = os.path.join(args.save_dir, args.checkpoint_name)

	model_manager = ModelManager(model_name=args.model_architecture, 
								 img_size=args.img_size,
								 output_channels=args.output_channels)
	inference_results, eval_results= model_manager.eval(device=device,
														args=args,
														checkpoint_path=checkpoint_path)

	print(f"{checkpoint_path} evaluation results: {eval_results}")

	'''
	Evaluate on all epochs if all_epochs>0
	'''
	if all_epochs>0:
		aucs_all_epochs = []
		for epoch in range(all_epochs):
			args.checkpoint_name = f'pytorch_model_epoch{epoch+1}.bin'
			checkpoint_path = os.path.join(args.save_dir, args.checkpoint_name)
			model_manager = ModelManager(model_name=args.model_architecture,
										 img_size=args.img_size,
										 output_channels=args.output_channels)
			inference_results, eval_results= model_manager.eval(device=device,
																args=args,
																checkpoint_path=checkpoint_path)
			if args.label_key == 'edema_severity':
				aucs_all_epochs.append(eval_results['ordinal_aucs'])
			else:
				aucs_all_epochs.append(eval_results['aucs'][0])

		print(f"All epochs AUCs: {aucs_all_epochs}")

		eval_results_all={}
		eval_results_all['ordinal_aucs']=aucs_all_epochs
		results_path = os.path.join(args.save_dir, 'eval_results_all.json')
		with open(results_path, 'w') as fp:
			json.dump(eval_results_all, fp)

eval(all_epochs=-1)