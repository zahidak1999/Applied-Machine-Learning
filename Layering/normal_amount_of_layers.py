import torch
import torch.nn as nn
import torchvision

# Load observations from the MNIST dataset
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # First Convolution Layer: 1 input channel, 32 output channels (filters), kernel size 5, padding 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # Max Pooling after first convolution: 32@28x28 -> 32@14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second Convolution Layer: 32 input channels, 64 output channels, kernel size 5, padding 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Max Pooling after second convolution: 64@14x14 -> 64@7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layer (dense layer): 64 * 7 * 7 -> 1024
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)  # 50% chance of dropping a unit
        # Final fully connected layer: 1024 -> 10 (10 classes for MNIST)
        self.fc2 = nn.Linear(1024, 10)

    def logits(self, x):
        # Apply first convolution + pooling
        x = self.pool1(self.conv1(x))
        # Apply second convolution + pooling
        x = self.pool2(self.conv2(x))

        # Flatten the output from convolutional layers for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)

        # Apply first dense layer with ReLU and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training

        # Apply second dense layer (final output)
        return self.fc2(x)

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss function
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test))