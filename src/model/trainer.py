from collections import Counter, defaultdict
from datetime import datetime
from itertools import chain
import os
import shutil
import numpy as np
import torch
import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from classification import classifier

from configurations import config

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

    def train(self):
        input_ids_train = self.input_ids_train
        attention_mask_train = self.attention_mask_train
        tags_labels_train = self.tags_labels_train

        input_ids_test = self.input_ids_test
        attention_mask_test = self.attention_mask_test
        tags_labels_test = self.tags_labels_test

        # Set the optimizer and learning rate
        optimizer = self.classifier_instance.optimizer
        # parameters = self.classifier_instance.parameters

        # Set the batch size
        batch_size = self.batch_size

        # Create a DataLoader for batching the data
        train_dataset = TensorDataset(input_ids_train, attention_mask_train, tags_labels_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

        valid_dataset = TensorDataset(input_ids_test, attention_mask_test, tags_labels_test)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # Set the number of training epochs
        num_epochs = self.num_epochs
        device = self.device

        model = self.model
        classifier_instance = self.classifier_instance

        # Training loop
        min_loss = 999999
        count = 0

        #epochs

        max_tag_acc_epochs = 0
        max_tag_f1_macro_epochs = 0
        max_tag_f1_micro_epochs = 0
        max_tag_f1_weighted_epochs = 0
        max_tag_f1_samples_epochs = 0
        max_tag_roc_auc_score_epochs = 0

        #score

        max_tag_acc = 0
        max_tag_f1_macro = 0
        max_tag_f1_micro = 0
        max_tag_f1_weighted = 0
        max_tag_f1_samples = 0
        max_tag_roc_auc_score = 0

        thresholds = [0.001] + [i * 0.01 for i in range(1, 101)]

        for epoch in range(num_epochs):
            # set early stopping
            if count > 8:
               break
            train_loss = 0.0
            valid_loss = 0.0

            tags_true = []
            tags_pred = defaultdict(list)
            tags_pred_proba = []

            count += 1

            # Training
            classifier_instance.train()
            # Zero the gradients
            optimizer.zero_grad()
            for batch in tqdm.tqdm(train_dataloader):
                # Unpack the batch
                input_ids, attention_mask, tags_labels = batch

                # Move the inputs and labels to the chosen device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                tags_labels = tags_labels.to(device)

                # Forward pass
                loss, _ = classifier_instance.forward(input_ids, attention_mask, tags_labels)
                loss /= self.accumulation_steps

                # Backward pass and optimization
                loss.backward()

                if epoch % self.accumulation_steps ==  0 or epoch == batch_size - 1 or self.accumulation_steps == 0:
                    if self.max_grad_norm > 0:

                        torch.nn.utils.clip_grad_norm_(chain(
                            model.parameters(),
                            classifier_instance.tags_classifier.parameters()
                        ), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item()

            # Validation
            #model.eval()
            classifier_instance.eval()
            with torch.no_grad():
                for batch in tqdm.tqdm(valid_dataloader):
                    # Unpack the batch
                    input_ids, attention_mask, tags_labels = batch

                    # Move the inputs and labels to the chosen device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    tags_labels = tags_labels.to(device)

                    # Forward pass
                    loss, output = classifier_instance.forward(input_ids, attention_mask, tags_labels)

                    valid_loss += loss.item()

                    tags_output = output

                    tags_pred_proba.extend(tags_output.detach().cpu().clone().tolist())

                    # Extract indices where the value is above the threshold.
                    for threshold in thresholds:
                        tags_pred[threshold].extend([(row >= threshold).nonzero().flatten().tolist() for row in tags_output.detach().cpu().clone()])

                    tags_true.extend([torch.nonzero(row).flatten().tolist() for row in tags_labels.detach().cpu().clone()])


            # Calculate average loss
            train_loss /= len(train_dataset)
            valid_loss /= len(valid_dataset)


            if epoch % self.accumulation_steps ==  0 or epoch == batch_size - 1 or self.accumulation_steps == 0:

                # Print the loss, F1 score, precision, and recall for monitoring
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

                tag_true = []

                for index_list in tags_true:
                    result_true = [0] * self.tags_num_classes  # Create a list of length num_classes.
                    for index in index_list:
                        result_true[index] = 1  # Fill the corresponding index with 1.

                    tag_true.append(result_true)
                    

                epoch_max_tag_acc = 0
                epoch_max_tag_f1_macro = 0
                epoch_max_tag_f1_micro = 0
                epoch_max_tag_f1_weighted = 0
                epoch_max_tag_f1_samples = 0

                epoch_max_tag_roc_auc_score = roc_auc_score(tag_true, tags_pred_proba)
                tag_true = np.array(tag_true)
                tags_pred_proba = np.array(tags_pred_proba)

                for threshold in thresholds:
                    tag_pred = []
                    for index_list in tags_pred[threshold]:
                        result_pred = [0] * self.tags_num_classes  # Create a list of length num_classes.
                        for index in index_list:
                            result_pred[index] = 1 # Fill the corresponding index with 1.

                        tag_pred.append(result_pred)


                    tag_acc = accuracy_score(tag_true, tag_pred)
                    tag_f1_macro = f1_score(tag_true, tag_pred, average='macro', zero_division=0)
                    tag_f1_micro = f1_score(tag_true, tag_pred, average='micro', zero_division=0)
                    tag_f1_weighted = f1_score(tag_true, tag_pred, average='weighted', zero_division=0)
                    tag_f1_samples = f1_score(tag_true, tag_pred, average='samples', zero_division=0)

                    epoch_max_tag_acc = max(epoch_max_tag_acc, tag_acc)
                    epoch_max_tag_f1_macro = max(epoch_max_tag_f1_macro, tag_f1_macro)
                    epoch_max_tag_f1_micro = max(epoch_max_tag_f1_micro, tag_f1_micro)
                    epoch_max_tag_f1_weighted = max(epoch_max_tag_f1_weighted, tag_f1_weighted)
                    epoch_max_tag_f1_samples = max(epoch_max_tag_f1_samples, tag_f1_samples)

                print("tag acc Max Score in this epoch:", epoch_max_tag_acc)
                print("tag valid Max F1 Score(macro) per class in this epoch:", epoch_max_tag_f1_macro)
                print("tag valid Max F1 Score(micro) per class in this epoch:", epoch_max_tag_f1_micro)
                print("tag valid Max F1 Score(weighted) per class in this epoch:", epoch_max_tag_f1_weighted)
                print("tag valid Max F1 Score(samples) per class in this epoch:", epoch_max_tag_f1_samples)
                print()
                print("tag valid Max roc_auc_score avg in this epoch:", epoch_max_tag_roc_auc_score)

                for num_classes in range(self.tags_num_classes):
                    score = roc_auc_score(tag_true[:, num_classes], tags_pred_proba[:, num_classes])
                    print(f"{self.tag_classes[num_classes]} : {score}")
                print()

                print(f"tag acc Max Score: {max_tag_acc} at {max_tag_acc_epochs}epochs")
                print(f"tag valid Max F1 Score(macro) per class: {max_tag_f1_macro} at {max_tag_f1_macro_epochs}epochs")
                print(f"tag valid Max F1 Score(micro) per class: {max_tag_f1_micro} at {max_tag_f1_micro_epochs}epochs")
                print(f"tag valid Max F1 Score(weighted) per class: {max_tag_f1_weighted} at {max_tag_f1_weighted_epochs}epochs")
                print(f"tag valid Max F1 Score(samples) per class: {max_tag_f1_samples} at {max_tag_f1_samples_epochs}epochs")
                print(f"tag valid Max roc_auc_score: {max_tag_roc_auc_score} at {max_tag_roc_auc_score_epochs}epochs")
                print()


                if epoch_max_tag_acc > max_tag_acc:
                    max_tag_acc_epochs = epoch
                    max_tag_acc = max(epoch_max_tag_acc, max_tag_acc)

                if epoch_max_tag_f1_macro > max_tag_f1_macro:
                    max_tag_f1_macro_epochs = epoch
                    max_tag_f1_macro = max(epoch_max_tag_f1_macro, max_tag_f1_macro)

                    if self.save:
                        self.save_checkpoint(epoch)
                    count = 0
                    print('Best Model Saved !')
                    print()

                if epoch_max_tag_f1_micro > max_tag_f1_micro:
                    max_tag_f1_micro_epochs = epoch
                    max_tag_f1_micro = max(epoch_max_tag_f1_micro, max_tag_f1_micro)

                if epoch_max_tag_f1_weighted > max_tag_f1_weighted:
                    max_tag_f1_weighted_epochs = epoch
                    max_tag_f1_weighted = max(epoch_max_tag_f1_weighted, max_tag_f1_weighted)

                if epoch_max_tag_f1_samples > max_tag_f1_samples:
                    max_tag_f1_samples_epochs = epoch
                    max_tag_f1_samples = max(epoch_max_tag_f1_samples, max_tag_f1_samples)

                if epoch_max_tag_roc_auc_score > max_tag_roc_auc_score:
                    max_tag_roc_auc_score_epochs = epoch
                    max_tag_roc_auc_score = max(epoch_max_tag_roc_auc_score, max_tag_roc_auc_score)

                print('----------------------------------------------------------------------------')
                print()

    def save_checkpoint(self, epoch, max_checkpoints=5):
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        checkpoint_filename = f"{now.strftime('%Y-%m-%d')}_{epoch + 1}"
        checkpoint_path = os.path.join(f"./models/{today}", checkpoint_filename)

        # If the directory does not exist, create it.
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Save the model state_dict
        torch.save(self.classifier_instance.state_dict(), os.path.join(checkpoint_path, f"model.pt"))
        checkpoint_files = sorted(os.listdir(f"./models/{today}"))
        # Delete oldest checkpoint if there are too many
        while len(checkpoint_files) > max_checkpoints + 1:
            checkpoint_files = sorted(os.listdir(f"./models/{today}"))
            oldest_checkpoint = os.path.join(f"./models/{today}", checkpoint_files[0])
            #os.remove(oldest_checkpoint)
            if os.path.exists(oldest_checkpoint) and os.path.isdir(oldest_checkpoint):
                # Check if the directory is empty, and if not, use shutil.rmtree() to recursively delete it.
                try:
                    shutil.rmtree(oldest_checkpoint)
                except Exception as e:
                    print(f"Error while deleting directory: {e}")