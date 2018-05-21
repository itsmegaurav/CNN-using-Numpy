import numpy as np
from deepnet.utils import accuracy, softmax
from deepnet.utils import one_hot_encode
		
class CNN:

    def __init__(self, layers, loss_func=SoftmaxLoss):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.loss_func = loss_func

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def predict(self, X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)

