import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np


training_data = datasets.MNIST(
root='data',
download=True,
train=True,
transform=ToTensor())

test_data = datasets.MNIST(
root='data',
download=True,
train=False,
transform=ToTensor())


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def plot_line(input_data: list, title: str, legend: list, xlabel: str, ylabel: str, save: bool = False, savename: str = None, show: bool = False):
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(input_data)
    plt.title(title)
    if legend:
        plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    


def test_images():

    mapping = {training_data.classes.index(x):x[:1] for x in training_data.classes}


    train_features, train_labels = next(iter(train_dataloader))
    img = train_features[0].squeeze()
    label=train_labels[0]
    plt.imshow(img)
    plt.show()
    print(label.item())


    figure = plt.figure(figsize=(9,9))
    cols, rows = 5, 5
    for i in range(1, cols*rows+1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(mapping[label])
        plt.axis('off')
        plt.imshow(img.squeeze())
    plt.show()


    
class SimpleNet(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.fc3(x))
        x = self.softmax(x)
        return x
    

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*7*7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x
        

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(28*28).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 2
    best_weights = None
    best_accuracy = -np.inf
    accuracy_hist = []
    loss_hist = []

    print(f'Training for {n_epochs} epochs with model {model.__class__.__name__}')


    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        
        for train_features, train_labels in (bar:=tqdm.tqdm(train_dataloader, unit='batch')):
            bar.set_description(f'Epoch: {epoch+1}/{n_epochs}')
            train_features, train_labels = train_features.to(device), train_labels.to(device)
            output = model(train_features)

            loss = loss_fn(output, train_labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        
        model.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for test_features, test_labels in (test_bar:=tqdm.tqdm(test_dataloader)):
                test_bar.set_description(f'Testing for Epoch {epoch+1}')
                test_features, test_labels = test_features.to(device), test_labels.to(device)
                
                output = model(test_features)
                loss = loss_fn(output, test_labels)
                test_loss += loss.item()
                
                predicted = torch.argmax(output, dim=1)
                total += test_labels.shape[0]
                correct += (predicted == test_labels).sum().item()
                
        test_loss /= len(test_dataloader)
        loss_hist.append(test_loss)
        accuracy = 100 * correct/total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = copy.deepcopy(model.state_dict())
        accuracy_hist.append(accuracy)


    loss_graphic = plot_line(loss_hist, 'Loss over the Epochs', ['Loss'], 'Epoch', 'Cross Entropy Loss', True, 'loss_hist_mnist.png', True)

    accuracy_graphic = plot_line(accuracy_hist, 'Accuracy over the Epochs', ['Accuracy'], 'Epoch', 'Accuracy in Percent', True, 'accuracy_hist_mnist.png', True)
