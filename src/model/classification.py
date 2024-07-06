import torch
import torch.nn as nn

from .configurations import config

class MultiLabelClassificationHead(nn.Module):
    def __init__(self, num_labels, hidden_size=768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size) # Fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Fully connected layer
        self.fc3 = nn.Linear(hidden_size, num_labels) # Fully connected layer
        self.sigmoid = nn.Sigmoid() # Sigmoid activation function

    def forward(self, x):
        x = self.fc1(x) # Apply the fully connected layer
        x = self.fc2(x) # Apply the fully connected layer
        x = self.fc3(x) # Apply the fully connected layer
        x = self.sigmoid(x) # Apply the sigmoid activation
        return x
    

# Define a classifier class
class classifier(nn.Module):
    def __init__(self, model, device, tags_num_classes):
        super().__init__()
        self.tags_num_classes = tags_num_classes  # Number of classes for tags

        # Set the device (GPU or CPU)
        self.device = device

        # Initialize multi-label
        self.tags_classifier = MultiLabelClassificationHead(num_labels=self.tags_num_classes).to(self.device)

        # Define loss functions for multi-label
        self.BCE = nn.BCELoss().to(self.device)  # Binary Cross Entropy loss for multi-label classification

        self.model = model
        self.lr = config['lr']

        self.parameters = [
                {'params': self.model.parameters()},
                {'params': self.tags_classifier.parameters()},
            ]

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=self.lr
        )

    def forward(self, input_ids, attention_mask, tags_labels):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # Pooled output from the model

        output = self.tags_classifier(pooled_output) # Predict tags using the tags classifier
        loss = self.BCE(output, tags_labels) # Calculate the loss for tags

        return loss, output