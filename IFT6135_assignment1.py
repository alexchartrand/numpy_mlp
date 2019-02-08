import torch
import numpy as np

import torchvision
import torchvision.transforms

## Sets hyper_param
batch_size = 64  # mini_batch size
num_epochs = 50  # number of training epochs
cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available')
else:
    print('cuda is not available')

store_every = 1000
lr0 = 0.02


## Load Dataset and creates loaders
## mnist images are 1x28x28
## label is an int from 0 to 9

data_size = (1,28,28)
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

# Code inspired by my own repository created during my attending to the IFT603 course at University of Sherbrooke (private repo, ask if you want to have access)
# https://github.com/RaphaelRoyerRivard/ift603-tp4

class NN(object):

    def __init__(self, input_size=784, hidden_dims=(512, 512), num_classes=10, init_dist='glorot', zero_bias=False,
                 activation='relu'):
        self.num_classes = num_classes
        self.layers = []
        self.layers.append(DenseLayer(784, hidden_dims[0], init_dist, zero_bias, activation))
        for i, size in enumerate(hidden_dims):
            out = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else num_classes
            self.layers.append(DenseLayer(size, out, init_dist, zero_bias, activation))

    def forward(self, input, labels=None):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
        return x

    def loss(self, scores, y):
        loss = -self.logSoftmax(scores, y)
        loss = np.average(loss, axis=0)

        t = np.zeros(self.num_classes)
        t[y] = 1
        scores_gradient = self.softmax(scores) - t
        scores_gradient = np.average(scores_gradient, axis=0)

        return loss, scores_gradient

    def softmax(self, scores):
        exp = np.exp(scores)
        return exp / np.sum(exp)

    def logSoftmax(self, scores, y):
        return scores[y] - np.log(np.sum(np.exp(scores)))

    def backward(self, scores_gradient):
        current_layer_gradient = scores_gradient
        for layer in reversed(self.layers):
            current_layer_gradient = layer.backward(current_layer_gradient)

    def update(self, learning_rate):
        for i, layer in enumerate(self.layers):
            layer.W -= layer.dW * learning_rate

    def train(self, data, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            print("epoch", epoch + 1)
            total_epoch_loss = 0

            for batch_index, (batch_images, batch_labels) in enumerate(data):
                batch_images = self.reshapeInput(batch_images)
                batch_labels = batch_labels.numpy()

                # TODO REMOVE
                # batch_images = batch_images[0]
                # batch_labels = batch_labels[0]

                print("minibatch", batch_index + 1)

                for layer in self.layers:
                    layer.reset_gradient()

                print("input shape", batch_images.shape)
                scores = self.forward(batch_images)
                print("scores", scores.shape)
                loss, scores_gradient = self.loss(scores, batch_labels)
                print("loss", loss.shape)
                print("scores_gradient", scores_gradient.shape)
                self.backward(scores_gradient)
                self.update(learning_rate)
                total_epoch_loss += loss
                if batch_index == 10:
                    break
            print("average epoch loss", total_epoch_loss / batch_index)

    def reshapeInput(self, batch_images):
        batch_images = batch_images.numpy()
        return batch_images.reshape((batch_images.shape[0], 784))

    def test(self):
        pass


class DenseLayer(object):

    def __init__(self, input_size, output_size, init_dist, zero_bias, activation=None):
        self.activation = activation  # None, 'relu', or 'sigmoid'
        self.W = None
        self.dW = None
        self.input_size = input_size  # number of input neurons
        self.output_size = output_size  # number of output neurons
        self.reinit(init_dist, zero_bias)

        self.last_x = None
        self.last_activ = None

    def reinit(self, init_dist, zero_bias):
        if init_dist == 'normal':
            self.W = np.random.randn(self.input_size + 1, self.output_size)
        elif init_dist == 'glorot':
            dl = np.sqrt(6 / (self.input_size + self.output_size))
            self.W = np.ones((self.input_size + 1, self.output_size)) * np.random.uniform(-dl, dl, (
            self.input_size + 1, self.output_size))
        else:
            self.W = np.zeros((self.input_size + 1, self.output_size))
        self.W[-1, :] = 0 if zero_bias else 1  # Initialize biases
        self.dW = np.zeros_like(self.W)

    def reset_gradient(self):
        self.dW.fill(0.0)

    def forward(self, x):
        x = augment(x)
        if self.activation == 'relu':
            # f = np.maximum(0, np.dot(self.W.T, x))
            f = np.maximum(0, np.dot(x, self.W))
        else:
            print("Error: activation", self.activation, "not supported")
        self.last_x = x
        self.last_activ = f
        return f

    def backward(self, gradient):
        print("gradient", gradient.shape)  # shape(10,)
        if self.activation == 'relu':
            # Tentative infructueuse 1
            # print("activations^T", self.last_activ.T.shape)
            # nabla = np.dot(gradient, self.last_activ.T)
            # print("nabla", nabla.shape)

            # Tentative infructueuse 2
            # relu_activation = self.last_activ != 0.0
            # relu_gradient = gradient * relu_activation
            # error = self.W * relu_gradient
            # print("error", error.shape)

            # Version fonctionnelle (je crois) sans support de mini-batch
            dout_drelu = self.last_activ != 0.0
            print("dout_drelu", dout_drelu.shape)  # shape: (64, 10)

            dnext_drelu = gradient * dout_drelu  # shape: (64, 10)
            print("dnext_drelu", dnext_drelu.shape)

            print("a", self.last_x[:, np.newaxis].shape)  # shape: (64, 1, 513)
            print("b", dnext_drelu.shape)
            dnext_dW = self.last_x[:, np.newaxis] * dnext_drelu
            dnext_dX = dnext_drelu.dot(self.W.T)
        else:
            print("Error: activation", self.activation, "not supported")
        dnext_dX = dnext_dX[:-1]  # discard the gradient wrt the 1.0 of homogeneous coord

        self.dW += dnext_dW
        return dnext_dX

def augment(x):
  if len(x.shape) == 1:
    return np.concatenate([x, [1.0]])
  else:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1)

network = NN(input_size=784, hidden_dims=(512, 512), num_classes=10, init_dist='glorot', zero_bias=True, activation='relu')
# network.train(mnist_train, 10, 1e-2)
network.train(train_loader, num_epochs=10, learning_rate=1e-2)