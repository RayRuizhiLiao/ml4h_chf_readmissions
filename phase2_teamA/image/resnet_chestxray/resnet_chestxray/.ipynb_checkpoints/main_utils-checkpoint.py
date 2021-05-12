'''
Author: Ruizhi Liao

Main_utils script to run training and evaluation 
of a residual network model on chest x-ray images
'''

import os
from tqdm import tqdm, trange
import logging
from scipy.stats import logistic
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
import csv
from scipy.special import softmax
from scipy.special import expit
import time
import cv2

import torch
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .model import build_resnet256_6_2_1, build_resnet512_6_2_1
from .model import build_resnet1024_7_2_1, build_resnet2048_7_2_1
from .model_utils import CXRImageDataset, convert_to_onehot
import eval_metrics
from PIL import Image

# label = '14d_hf'
def build_training_dataset(data_dir, img_size: int, dataset_metadata='../data/train_readmission-er_hf.csv',
        random_degrees=[-20,20], random_translate=[0.1,0.1], label_key='er_hf'):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomAffine(degrees=random_degrees, translate=random_translate),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.Lambda(
            lambda img: np.array(img).astype(np.float32)),
        torchvision.transforms.Lambda(
            lambda img: img / img.max())
        ])
    training_dataset = CXRImageDataset(data_dir=data_dir, 
            dataset_metadata=dataset_metadata, 
            transform=transform,
            label_key=label_key)

    return training_dataset

def build_evaluation_dataset(data_dir, img_size: int, dataset_metadata='../data/test_readmission-er_hf.csv', label_key='er_hf'):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.Lambda(
            lambda img: np.array(img).astype(np.float32)),
        torchvision.transforms.Lambda(
            lambda img: img / img.max())
        ])
    evaluation_dataset = CXRImageDataset(data_dir=data_dir, 
            dataset_metadata=dataset_metadata, 
            transform=transform,
            label_key=label_key)

    return evaluation_dataset

def build_model(model_name, checkpoint_path=None, output_channels=4):
    if checkpoint_path == None:
        if model_name == 'resnet256_6_2_1':
            model = build_resnet256_6_2_1(output_channels=output_channels)
        if model_name == 'resnet512_6_2_1':
            model = build_resnet512_6_2_1(output_channels=output_channels)
        if model_name == 'resnet1024_7_2_1':
            model = build_resnet1024_7_2_1(output_channels=output_channels)
        if model_name == 'resnet2048_7_2_1':
            model = build_resnet2048_7_2_1(output_channels=output_channels)
    else:
        if model_name == 'resnet256_6_2_1':
            model = build_resnet256_6_2_1(pretrained=True,
                    pretrained_model_path=checkpoint_path,
                    output_channels=output_channels)
        if model_name == 'resnet512_6_2_1':
            model = build_resnet512_6_2_1(pretrained=True,
                    pretrained_model_path=checkpoint_path,
                    output_channels=output_channels)
        if model_name == 'resnet1024_7_2_1':
            model = build_resnet1024_7_2_1(pretrained=True,
                    pretrained_model_path=checkpoint_path,
                    output_channels=output_channels)
        if model_name == 'resnet2048_7_2_1':
            model = build_resnet2048_7_2_1(pretrained=True,
                    pretrained_model_path=checkpoint_path,
                    output_channels=output_channels)	
    return model


class ModelManager:

    def __init__(self, model_name, img_size, output_channels=4):
        self.model_name = model_name
        self.output_channels = output_channels
        self.model = build_model(self.model_name, output_channels=self.output_channels)
        self.img_size = img_size
        self.logger = logging.getLogger(__name__)

    def train(self, device, args):
        # data_dir, dataset_metadata, save_dir,
        # 	  batch_size=64, num_train_epochs=300, 
        # 	  device='cuda', init_lr=5e-4, logging_steps=50,
        # 	  label_key='edema_severity', loss_method='CrossEntropyLoss'):
        '''
        Create a logger for logging model training
        '''
        logger = logging.getLogger(__name__)

        '''
        Create an instance of traning data loader
        '''
        print('***** Instantiate a data loader *****')
        dataset = build_training_dataset(data_dir=args.data_dir,
                                                                         img_size=self.img_size,
                                                                         dataset_metadata=args.dataset_metadata,
                                                                         label_key=args.label_key)
        data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=8,
                                                         pin_memory=True)
        print(f'Total number of training images: {len(dataset)}')

        ''' 
        Create an instance of loss
        '''
        print('***** Instantiate the training loss *****')
        if args.loss_method == 'CrossEntropyLoss':
                loss_criterion = CrossEntropyLoss().to(device)
        elif args.loss_method == 'BCEWithLogitsLoss':
                loss_criterion = BCEWithLogitsLoss().to(device)
        elif args.loss_method == 'BCELoss':
                loss_criterion = BCELoss().to(device)
        '''
        Create an instance of optimizer and learning rate scheduler
        '''
        print('***** Instantiate an optimizer *****')
        optimizer = optim.Adam(self.model.parameters(), lr=args.init_lr)

        '''
        Train the model
        '''
        print('***** Train the model *****')
        self.model = self.model.to(device)
        self.model.train()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
                start_time = time.time()
                epoch_loss = 0
                epoch_iterator = tqdm(data_loader, desc="Iteration")
                for i, batch in enumerate(epoch_iterator, 0):
                        # Parse the batch 
                        images, labels, image_ids = batch
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        # Zero out the parameter gradients
                        optimizer.zero_grad()

                        # Forward + backward + optimize
                        outputs = self.model(images)
                        pred_logits = outputs[-1]
                        if args.loss_method == 'BCEWithLogitsLoss':
                            labels = torch.reshape(labels, pred_logits.size())

                        # pred_logits[labels<0] = 0
                        # labels[labels<0] = 0.5
                        #labels = labels.type_as(outputs)
                        loss = loss_criterion(pred_logits.float(), labels.float())
             
                        loss.backward()
                        optimizer.step()

                        # Record training statistics
                        epoch_loss += loss.item()
                        print("Epoch loss %s" %str(epoch_loss))

                        if not loss.item()>0:
                                logger.info(f"loss: {loss.item()}")
                                logger.info(f"pred_logits: {pred_logits}")
                                logger.info(f"labels: {labels}")
                self.model.save_pretrained(args.save_dir, epoch=epoch + 1)
                interval = time.time() - start_time

                print(f'Epoch {epoch+1} finished! Epoch loss: {epoch_loss:.5f}')

                logger.info(f"  Epoch {epoch+1} loss = {epoch_loss:.5f}")
                logger.info(f"  Epoch {epoch+1} took {interval:.3f} s")

        return

    def eval(self, device, args, checkpoint_path):
            '''
            Load the checkpoint (essentially create a "different" model)
            '''
            self.model = build_model(model_name=self.model_name,
                                                             output_channels=self.output_channels,
                                                             checkpoint_path=checkpoint_path)

            '''
            Create an instance of evaluation data loader
            '''
            print('***** Instantiate a data loader *****')
            dataset = build_evaluation_dataset(data_dir=args.data_dir,
                                                                               img_size=self.img_size,
                                                                               dataset_metadata=args.dataset_metadata,
                                                                               label_key=args.label_key)
            data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=8,
                                                             pin_memory=True)
            print(f'Total number of evaluation images: {len(dataset)}')

            '''
            Evaluate the model
            '''
            print('***** Evaluate the model *****')
            self.model = self.model.to(device)
            self.model.eval()

            # For storing labels and model predictions
            all_preds_prob = []
            all_preds_logit = []
            all_labels = []

            epoch_iterator = tqdm(data_loader, desc="Iteration")
            for i, batch in enumerate(epoch_iterator, 0):
                    # Parse the batch 
                    images, labels, image_ids = batch
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.no_grad():
                            outputs = self.model(images)

                            preds_prob = outputs[0]
                            preds_logit = outputs[-1]
                            #print(outputs)
                            #if not args.label_key == 'edema_severity':
                                    #labels = torch.reshape(labels, preds_logit.size())

                            preds_prob = preds_prob.detach().cpu().numpy()
                            preds_logit = preds_logit.detach().cpu().numpy()
                            labels = labels.detach().cpu().numpy()

                            all_preds_prob += \
                                    [preds_prob[j] for j in range(len(labels))]
                            all_preds_logit += \
                                    [preds_logit[j] for j in range(len(labels))]
                            all_labels += \
                                    [labels[j] for j in range(len(labels))]

            all_preds_class = np.argmax(all_preds_prob, axis=1)
            inference_results = {'all_preds_prob': all_preds_prob,
                                                     'all_preds_class': all_preds_class,
                                                     'all_preds_logit': all_preds_logit,
                                                     'all_labels': all_labels}
            eval_results = {}

            #if args.label_key in ['14d_hf', '30d_hf', '48h_hf']:
            all_preds_prob = [1 / (1 + np.exp(-logit)) for logit in all_preds_logit]
            #all_preds_prob = all_preds_logit
            aucs = eval_metrics.compute_binary_auc(all_labels, all_preds_prob)
            eval_results['aucs'] = aucs
# 			all_onehot_labels = [convert_to_onehot(label) for label in all_labels]

# 			ordinal_aucs = eval_metrics.compute_ordinal_auc(all_onehot_labels, all_preds_prob)
# 			eval_results['ordinal_aucs'] = ordinal_aucs

# 			ordinal_acc_f1 = eval_metrics.compute_ordinal_acc_f1_metrics(all_onehot_labels, 
# 																	     all_preds_prob)
# 			eval_results.update(ordinal_acc_f1)

# 			eval_results['mse'] = eval_metrics.compute_mse(all_labels, all_preds_prob)

            results_acc_f1, _, _ = eval_metrics.compute_acc_f1_metrics(all_labels, all_preds_prob)
            eval_results.update(results_acc_f1)
#             else:
#                     all_preds_prob = [1 / (1 + np.exp(-logit)) for logit in all_preds_logit]
#                     all_preds_class = np.argmax(all_preds_prob, axis=1)
#                     aucs = eval_metrics.compute_multiclass_auc(all_labels, all_preds_prob)
#                     eval_results['aucs'] = aucs

            return inference_results, eval_results

    def infer(self, device, args, checkpoint_path, img_path):
            '''
            Load the checkpoint (essentially create a "different" model)
            '''
            self.model = build_model(model_name=self.model_name,
                                                             output_channels=self.output_channels,
                                                             checkpoint_path=checkpoint_path)

            '''
            Load and Preprocess the input image
            '''
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            #img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #img = Image.open(img_path)

            transform = torchvision.transforms.Compose([
                    torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.CenterCrop(args.img_size),
                    torchvision.transforms.Lambda(
                            lambda img: np.array(img).astype(np.float32)),
                    torchvision.transforms.Lambda(
                            lambda img: img / img.max())
            ])

            img = transform(img)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)

            '''
            Run model inference
            '''
            self.model = self.model.to(device)
            self.model.eval()

            img = torch.tensor(img)
            img = img.to(device, non_blocking=True)

            with torch.no_grad():
                    outputs = self.model(img)

                    pred_logit = outputs[-1]
                    pred_logit = pred_logit.detach().cpu().numpy()
                    if args.label_key == 'edema_severity':
                            pred_prob = outputs[0]
                            pred_prob = pred_prob.detach().cpu().numpy()
                    else:
                            pred_prob = expit(pred_logit)

            inference_results = {'pred_prob': pred_prob,
                                                     'pred_logit': pred_logit}

            return inference_results

# TODO: optimize this method and maybe the csv format
# def split_tr_eval(split_list_path, training_folds, evaluation_folds):
# 	"""
# 	Given a data split list (.csv), training folds and evaluation folds,
# 	return DICOM IDs and the associated labels for training and evaluation
# 	"""

# 	print('Data split list being used: ', split_list_path)

# 	train_labels = {}
# 	train_ids = {}
# 	eval_labels = {}
# 	eval_ids = {}
# 	count_labels = [0,0,0,0]

# 	with open(split_list_path, 'r') as train_label_file:
# 		train_label_file_reader = csv.reader(train_label_file)
# 		row = next(train_label_file_reader)
# 		for row in train_label_file_reader:
# 			if row[-1] != 'TEST':
# 				if int(row[-1]) in training_folds:
# 					train_labels[row[2]] = [float(row[3])]
# 					train_ids[row[2]] = row[1]
# 					count_labels[int(row[3])] += 1
# 				if int(row[-1]) in evaluation_folds:
# 					eval_labels[row[2]] = [float(row[3])]
# 					eval_ids[row[2]] = row[1]
# 			if row[-1] == 'TEST' and -1 in evaluation_folds:
# 					eval_labels[row[2]] = [float(row[3])]
# 					eval_ids[row[2]] = row[1]              

# 	class_reweights = np.array([float(sum(count_labels)/i) for i in count_labels])

# 	print("Training and evaluation folds: ", training_folds, evaluation_folds)
# 	print("Total number of training labels: ", len(train_labels))
# 	print("Total number of training DICOM IDs: ", len(train_ids))
# 	print("Total number of evaluation labels: ", len(eval_labels))
# 	print("Total number of evaluation DICOM IDs: ", len(eval_ids))
# 	print("Label distribution in the training data: {}".format(count_labels))
# 	print("Class reweights: {}".format(class_reweights))

# 	return train_labels, train_ids, eval_labels, eval_ids, class_reweights

# Model training function
def train(args, device, model):

        '''
        Create a logger for logging model training
        '''
        logger = logging.getLogger(__name__)

        '''
        Create an instance of traning data loader
        '''
        xray_transform = RandomTranslateCrop(2048)
        train_labels, train_dicom_ids, _, _, class_reweights = split_tr_eval(
                args.data_split_path, args.training_folds, args.evaluation_folds)
        cxr_dataset = CXRImageDataset(train_dicom_ids, train_labels, args.image_dir,
                                      transform=xray_transform, image_format=args.image_format)
        data_loader = DataLoader(cxr_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=8,
                                 pin_memory=True)
        print('Total number of training images: ', len(cxr_dataset))

        ''' 
        Create an instance of loss
        '''
        if args.loss == 'CE':
                loss_criterion = CrossEntropyLoss().to(device)
        if args.loss == 'reweighted_CE':
                class_reweights = torch.tensor(class_reweights, dtype=torch.float32)
                loss_criterion = CrossEntropyLoss(weight=class_reweights).to(device)

        '''
        Create an instance of optimizer and learning rate scheduler
        '''
        optimizer = optim.Adam(model.parameters(), 
                                                        lr=args.init_lr)
        if args.scheduler == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-6)

        '''
        Log training info
        '''
        logger.info("***** Training info *****")
        logger.info("  Model architecture: %s", args.model_architecture)
        logger.info("  Data split file: %s", args.data_split_path)
        logger.info("  Training folds: %s\t Evaluation folds: %s"%(args.training_folds, args.evaluation_folds))
        logger.info("  Number of training examples: %d", len(cxr_dataset))
        logger.info("  Number of epochs: %d", args.num_train_epochs)
        logger.info("  Batch size: %d", args.batch_size)
        logger.info("  Initial learning rate: %f", args.init_lr)
        logger.info("  Learning rate scheduler: %s", args.scheduler)
        logger.info("  Loss function: %s", args.loss)

        '''
        Create an instance of a tensorboard writer
        '''
        tsbd_writer = SummaryWriter(log_dir=args.tsbd_dir)
        # images, labels = next(iter(data_loader))
        # images = images.to(device)
        # labels = labels.to(device)
        # tsbd_writer.add_graph(model, images)

        '''
        Train the model
        '''
        model.train()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        global_step = 0
        running_loss = 0
        logger.info("***** Training the model *****")
        for epoch in train_iterator:
            logger.info("  Starting a new epoch: %d", epoch + 1)
            epoch_iterator = tqdm(data_loader, desc="Iteration")
            tr_loss = 0
            for i, batch in enumerate(epoch_iterator, 0):
                # Get the batch 
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                inputs, labels, labels_raw = batch

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = loss_criterion(outputs[-1], labels_raw)
                loss.backward()
                optimizer.step()

                # Print and record statistics
                running_loss += loss.item()
                tr_loss += loss.item()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    #grid = torchvision.utils.make_grid(inputs)
                    #tsbd_writer.add_image('images', grid, global_step)
                    tsbd_writer.add_scalar('loss/train', 
                                           running_loss / (args.logging_steps*args.batch_size), 
                                           global_step)
                    tsbd_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logger.info("  [%d, %5d, %5d] learning rate = %f"%\
                            (epoch + 1, i + 1, global_step, optimizer.param_groups[0]['lr']))
                    logger.info("  [%d, %5d, %5d] loss = %.5f"%\
                            (epoch + 1, i + 1, global_step, running_loss / (args.logging_steps*args.batch_size)))        
                    running_loss = 0
            logger.info("  Finished an epoch: %d", epoch + 1)
            logger.info("  Training loss of epoch %d = %.5f"%\
                    (epoch+1, tr_loss / (len(cxr_dataset)*args.batch_size)))
            model.save_pretrained(args.checkpoints_dir, epoch=epoch + 1)
            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(tr_loss)

        tsbd_writer.close()

# Model evaluation function
def evaluate(args, device, model):

        '''
        Create a logger for logging model evaluation results
        '''
        logger = logging.getLogger(__name__)

        '''
        Create an instance of evaluation data loader
        '''
        xray_transform = CenterCrop(2048)
        _, _, eval_labels, eval_dicom_ids, _ = split_tr_eval(args.data_split_path,
                                                                                                                 args.training_folds,
                                                                                                                 args.evaluation_folds)
        cxr_dataset = CXRImageDataset(eval_dicom_ids, eval_labels, args.image_dir,
                                                                  transform=xray_transform, 
                                                                  image_format=args.image_format)
        data_loader = DataLoader(cxr_dataset, batch_size=args.batch_size,
                                 num_workers=8, pin_memory=True)
        print('Total number of evaluation images: ', len(cxr_dataset))

        '''
        Log evaluation info
        '''
        logger.info("***** Evaluation info *****")
        logger.info("  Model architecture: %s", args.model_architecture)
        logger.info("  Data split file: %s", args.data_split_path)
        logger.info("  Training folds: %s\t Evaluation folds: %s"%(args.training_folds, args.evaluation_folds))
        logger.info("  Number of evaluation examples: %d", len(cxr_dataset))
        logger.info("  Number of epochs: %d", args.num_train_epochs)
        logger.info("  Batch size: %d", args.batch_size)
        logger.info("  Model checkpoint {}:".format(args.checkpoint_path))

        '''
        Evaluate the model
        '''

        logger.info("***** Evaluating the model *****")

        # For storing labels and model predictions
        preds = []
        labels = []
        embeddings = []

        model.eval()
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        for i, batch in enumerate(epoch_iterator, 0):
                # Get the batch; each batch is a list of [image, label]
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                image, label, _ = batch
                with torch.no_grad():
                        output, embedding, _ = model(image)
                        pred = output.detach().cpu().numpy()
                        embedding = embedding.detach().cpu().numpy()
                        label = label.detach().cpu().numpy()
                        for j in range(len(pred)):
                                preds.append(pred[j])
                                labels.append(label[j])
                                embeddings.append(embedding[j])

        labels_raw = np.argmax(labels, axis=1)
        eval_results = {}

        ordinal_aucs = eval_metrics.compute_ordinal_auc(labels, preds)
        eval_results['ordinal_aucs'] = ordinal_aucs

        pairwise_aucs = eval_metrics.compute_pairwise_auc(labels, preds)
        eval_results['pairwise_auc'] = pairwise_aucs

        multiclass_aucs = eval_metrics.compute_multiclass_auc(labels, preds)
        eval_results['multiclass_aucs'] = multiclass_aucs

        eval_results['mse'] = eval_metrics.compute_mse(labels_raw, preds)

        results_acc_f1, _, _ = eval_metrics.compute_acc_f1_metrics(labels_raw, preds)
        eval_results.update(results_acc_f1)

        logger.info("  AUC(0v123) = %4f", eval_results['ordinal_aucs'][0])
        logger.info("  AUC(01v23) = %4f", eval_results['ordinal_aucs'][1])
        logger.info("  AUC(012v3) = %4f", eval_results['ordinal_aucs'][2])

        logger.info("  AUC(0v1) = %4f", eval_results['pairwise_auc']['0v1'])
        logger.info("  AUC(0v2) = %4f", eval_results['pairwise_auc']['0v2'])
        logger.info("  AUC(0v3) = %4f", eval_results['pairwise_auc']['0v3'])
        logger.info("  AUC(1v2) = %4f", eval_results['pairwise_auc']['1v2'])
        logger.info("  AUC(1v3) = %4f", eval_results['pairwise_auc']['1v3'])
        logger.info("  AUC(2v3) = %4f", eval_results['pairwise_auc']['2v3'])

        logger.info("  AUC(0v123) = %4f", eval_results['multiclass_aucs'][0])
        logger.info("  AUC(1v023) = %4f", eval_results['multiclass_aucs'][1])
        logger.info("  AUC(2v013) = %4f", eval_results['multiclass_aucs'][2])
        logger.info("  AUC(3v012) = %4f", eval_results['multiclass_aucs'][3])

        logger.info("  MSE = %4f", eval_results['mse'])

        logger.info("  Macro_F1 = %4f", eval_results['macro_f1'])
        logger.info("  Accuracy = %4f", eval_results['accuracy'])

        return eval_results, embeddings, labels_raw

# Model inference given an image
def inference(model, image):

        xray_transform = CenterCrop(2048)

        image = xray_transform(image)
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        image = torch.tensor(image, dtype=torch.float32)

        with torch.no_grad():
                probs, _, _ = model(image)

        return probs

# Model inference with gradcam given an image
def inference_gradcam(model_gradcam, image, target_layer):

        xray_transform = CenterCrop(2048)

        image = xray_transform(image)
        input_img = image.reshape(1, 1, image.shape[0], image.shape[1])
        input_img = torch.tensor(input_img, dtype=torch.float32)

        probs = model_gradcam.forward(input_img)

        _, ids = probs.sort(dim=1, descending=True)
        predicted_classes = ids[:, [0]]
        model_gradcam.backward(ids=predicted_classes)
        regions = model_gradcam.generate(target_layer=target_layer)
        gcam_img = regions[0]

        return probs, gcam_img, input_img[0]
