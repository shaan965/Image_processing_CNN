import torch
from torch import nn

from torch.nn import (Module, Sequential, NLLLoss,
                      Flatten, Linear, ReLU, LogSoftmax, Conv2d, MaxPool2d)


from cnn_utils import get_MNIST, show_examples

class MLP1(Module):
   
    def __init__(self):
        super().__init__()
        # Input will be a tensor with dimension `(n_datapoints, n_channels, height, width)`
        # So `(64, 1, 30, 30)` in the case of MNIST dataset
        self.layers = Sequential(
            Flatten(),                                    # -> (64, 900)
            Linear(in_features=30*30, out_features=512),  # -> (64, 512)
            ReLU(),                                       # -> (64, 512)
            Linear(in_features=512, out_features=10),     # -> (64, 10)
            LogSoftmax(dim=1)                             # -> (64, 10)
            )

    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs

class MLP2(Module):
    """
    A densely-connected feedforward network with a two hidden layers of 128
    and 64 ReLU units respectively.  Output layer is 10 fully-connected log softmax
    units (i.e., the output will be a vector of log-probabilities).
    """
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Flatten(),
            # TODO
            )

    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs

class CNN(Module):
    """
    A convolutional neural network with the following layers:
    * A layer of 32 convolutional units with a kernel size of 5x5 and a stride of 1,1, with relu activation
    * A max-pooling layer with a pool size of 2x2 and a stride of 2,2.
    * A layer of 64 convolutional units with a kernel size of 5x5 and the default stride, with relu activation.
    * A max-pooling layer with a pool size of 2x2 and the default stride.
    * A `Flatten` layer (to reshape the image from a 2D matrix into a single long vector)
    * A layer of 512 fully-connected linear units with relu activation
    * A layer of 10 fully-connected linear units with log-softmax activation (the output layer)

    Output layer is 10 fully-connected log-softmax units (i.e., the output will
    be a vector of log-probabilities).
    """
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            # TODO
        )
    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs
    

def train_model(batches, model, num_epochs=5, verbose=True):
    """
    Train `model` using the training data in `batches` for `num_epochs`
    passes over the full training set, using the negative log likelihood loss.
    If `verbose` is True, print progress reports.
    """
    report_interval = len(batches)//10
    loss_fn = NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (X,y) in enumerate(batches):
            loss = loss_fn(model(X), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (batch_idx % report_interval == 0):
                print(f"epoch {epoch+1}/{num_epochs} "
                      f"batch {batch_idx+1:>3d}/{len(batches)} loss={loss.item()}")

def accuracy(batches, model):
    """
    Evaluate `model` on the data in `batches` and return the accuracy.
    """
    model.eval()
    num_correct = 0
    total_size = 0
    with torch.no_grad():
        for X,y in batches:
            pred = model(X)
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_size += y.size()[0]

    return float(num_correct) / float(total_size)

def main():
    trn1,tst1 = get_MNIST('top_left')
    trn2,tst2 = get_MNIST('bottom_right')

    # Left column is images from top_left dataset
    # Right column is corresponding images from bottom_right dataset
    show_examples(tst1, tst2, 'examples.png')

    mlp1 = MLP1().to('cpu')
    train_model(trn1, mlp1)
    print(f"*** MLP1 accuracies: "
          f"tst1={accuracy(tst1,mlp1)} tst2={accuracy(tst2,mlp1)}")

    mlp2 = MLP2().to('cpu')
    train_model(trn1, mlp2)
    print(f"*** MLP2 accuracies: "
          f"tst1={accuracy(tst1,mlp2)} tst2={accuracy(tst2,mlp2)}")

    cnn = CNN().to('cpu')
    train_model(trn1, cnn)
    print(f"*** CNN accuracies: "
          f"tst1={accuracy(tst1,cnn)} tst2={accuracy(tst2,cnn)}")


if __name__ == '__main__':
    main()
