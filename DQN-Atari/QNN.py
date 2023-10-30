import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # input_dim: (84,84,4)
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(input_dim[2], 32, kernel_size=8, stride=4)
        
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the input size for the first fully connected layer
        conv_output_size = self._get_conv_output_size(input_dim)
        
        # Define the first fully connected layer
        self.fc1 = nn.Linear(conv_output_size, 512)
        
        # Define the second fully connected layer
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the output
        print(x.shape)
        x = torch.relu(self.fc1(x))
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return x
    
    def _get_conv_output_size(self, input_dim):
        # Helper function to calculate the output size of the convolutional layers
        x = torch.zeros(1, *input_dim)
        x = x.view(x.size(0), -1)
        conv_output_size = x.size(1)
        return conv_output_size