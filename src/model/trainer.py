from collections import Counter, defaultdict
import datetime
from itertools import chain
import os
import shutil
import numpy as np
import torch
import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from model.classification import classifier

from .configurations import config

class Trainer():
    def __init__(self,
                 model,
                 tag_label_encoder,
                 tokenized_inputs_train,
                 tokenized_inputs_test,
                 tags_labels_train,
                 tags_labels_test
                ):
        # Set the device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store the input data and labels
        self.tokenized_inputs_train = tokenized_inputs_train
        self.tokenized_inputs_test = tokenized_inputs_test

        self.tags_labels_train = tags_labels_train
        self.tags_labels_test = tags_labels_test

        # Determine the number of classes for tags
        self.tags_num_classes = len(tags_labels_train[0])

        # Move the model to the specified device
        self.model = model.to(self.device)

        # Define Classifier Instance
        self.classifier_instance = classifier(self.model, self.device, self.tags_num_classes)

        # Retrieve configuration parameters
        self.batch_size = config['batchSize']
        self.num_epochs = config['numEpochs']

        self.accumulation_steps = config['gradient_accumulation_steps']
        self.max_grad_norm = config['max_grad_norm']

        self.tag_classes = tag_label_encoder.classes_

        self.save = config['save']

        # Initialize input data variables
        self.input_ids_train = self.tokenized_inputs_train['input_ids']
        self.attention_mask_train = self.tokenized_inputs_train['attention_mask']

        self.input_ids_test = tokenized_inputs_test['input_ids']
        self.attention_mask_test = tokenized_inputs_test['attention_mask']

        self.task = config['task']

