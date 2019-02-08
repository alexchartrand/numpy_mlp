import numpy as np
import math

# Code inspired by my own repository created during my attending to the IFT603 course at University of Sherbrooke (private repo, ask if you want to have access)
# https://github.com/RaphaelRoyerRivard/ift603-tp4

class NN(object):

    def __init__(self, input_size=784, hidden_dims=(512, 512), num_classes=10, init_dist='normal', zero_bias=False,
                 activation='relu'):
        self.num_classes = num_classes
        self.layers = []
        self.layers.append(DenseLayer(input_size, hidden_dims[0], init_dist, zero_bias, activation))
        for i, size in enumerate(hidden_dims):

            if i + 1 < len(hidden_dims):
                dl = DenseLayer(size, hidden_dims[i + 1], init_dist, zero_bias, activation)
            else:
                # This is our last layer
                dl = DenseLayer(size, num_classes, init_dist, zero_bias, 'softmax') # 'softmax'

            self.layers.append(dl)

    def forward(self, input, labels=None):
        x=input
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

    @staticmethod
    def softmax(scores):
        exp = np.exp(scores)
        return exp / np.sum(exp)

    @staticmethod
    def logSoftmax(scores, y):
        return scores[y] - np.log(np.sum(np.exp(scores)))

    @staticmethod
    def crossEntropy(scores, y):
        m = y.shape[0]

        log_likelihood = -np.log(scores[range(m), y])

        return np.average(log_likelihood)

    @staticmethod
    def grad_cross_entropy(scores, y):
        m = y.shape[0]
        scores[range(m), y] -= 1
        scores = scores / m
        return scores

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
            number_of_batch = 0
            for batch_index, (batch_images, batch_labels) in enumerate(data):
                batch_images = self.reshapeInput(batch_images)
                batch_labels = batch_labels.numpy()

                #print("minibatch", batch_index + 1)

                for layer in self.layers:
                    layer.reset_gradient()

                scores = self.forward(batch_images)
                if np.isnan(scores).any():
                    print("problem")

                crossEntropy = self.crossEntropy(scores, batch_labels)

                scores_gradient = self.grad_cross_entropy(scores, batch_labels)

                #loss, scores_gradient = self.loss(scores, batch_labels)
                #print("loss", loss.shape)
                #print("scores_gradient", scores_gradient.shape)

                self.backward(scores_gradient)
                self.update(learning_rate)

                total_epoch_loss += crossEntropy
                number_of_batch += 1

            print("average epoch loss ", total_epoch_loss/number_of_batch )

    def reshapeInput(self, batch_images):
        batch_images = batch_images.numpy()
        return batch_images.reshape((batch_images.shape[0], 784))


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
        self.last_xDot = None

    def reinit(self, init_dist, zero_bias):
        if init_dist == 'normal':
            self.W = np.random.randn(self.input_size + 1, self.output_size)
        elif init_dist == 'glorot':
            dl = np.sqrt(6 / (self.input_size + self.output_size))
            self.W = np.ones((self.input_size + 1, self.output_size)) * np.random.uniform(-dl, dl, (self.input_size + 1, self.output_size))
        else:
            self.W = np.zeros((self.input_size + 1, self.output_size))
        self.W[-1, :] = 0 if zero_bias else 1  # Initialize biases
        self.dW = np.zeros_like(self.W)

    def reset_gradient(self):
        self.dW.fill(0.0)

    @staticmethod
    def softmax(X):
        xmax = np.max(X, axis=1, keepdims=True)
        exp = np.exp(X-xmax)
        sum_ax = np.sum(exp, axis=1)
        return exp / sum_ax[:,None]

    @staticmethod
    def grad_softmax(X):
        grad = X * (1. - X)
        return grad

    @staticmethod
    def relu(X):
        return np.maximum(0, X)

    @staticmethod
    def grad_relu(X):
        x_grad = np.copy(X)
        x_grad[x_grad <= 0] = 0
        x_grad[x_grad > 0] = 1
        return x_grad

    def forward(self, x):
        x = augment(x)

        x_dot = np.dot(x, self.W)
        if self.activation == 'relu':
            f = self.relu(x_dot)
        elif self.activation == 'softmax':
            f = self.softmax(x_dot)
        else:
            raise Exception("Error: activation " + self.activation + "not supported")

        self.last_x = x
        self.last_xDot = x_dot
        self.last_activ = f
        return f

    def backward(self, gradient):
        if self.activation == 'relu':
            dout = self.grad_relu(self.last_xDot)
        elif self.activation == "softmax":
            #dout = self.grad_softmax(self.last_xDot)
            dout = self.last_xDot
        else:
            raise Exception("Error: activation" + self.activation + "not supported")

        dnext = gradient * dout  # shape: (batch, 10)
        dnext_dW = dnext.T.dot(self.last_x).T
        dnext_dX = dnext.dot(self.W.T)
        dnext_dX = dnext_dX[:,:-1]

        self.dW += dnext_dW
        return dnext_dX

def augment(x):
  if len(x.shape) == 1:
    return np.concatenate([x, [1.0]])
  else:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1)