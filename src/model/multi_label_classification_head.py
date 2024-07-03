import torch.nn as nn

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