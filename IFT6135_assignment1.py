import torch
import numpy as np
import torchvision
import torchvision.transforms
from NN import NN

## Sets hyper_param
batch_size = 1  # mini_batch size
num_epochs = 10  # number of training epochs
store_every = 1000
lr0 = 0.02
data_size = (1,28,28)

def load_data():
    ## Load Dataset and creates loaders
    ## mnist images are 1x28x28
    ## label is an int from 0 to 9

    mnist_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(
            root='./data', train=True,
            transform=mnist_transforms, download=True)
    mnist_test = torchvision.datasets.MNIST(
            root='./data', train=False,
            transform=mnist_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=batch_size, shuffle=True, num_workers=2)
    print(len(mnist_train), len(mnist_test))
    print(len(train_loader), len(test_loader))

    return train_loader, test_loader


def main():
    train_loader, test_loader = load_data()
    network = NN(input_size=784, hidden_dims=(512, 512), num_classes=10, init_dist='glorot', zero_bias=True, activation='relu')

    network.train(train_loader, num_epochs=num_epochs, learning_rate=lr0)

if __name__=="__main__":
    main()