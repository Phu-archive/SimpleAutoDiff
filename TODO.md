There are going to be 4 phases.

# First phase
Multiple scalar variables auto differentiation system. I don't think that we should start with vectorized data. So I think that I will create a single neuron that is able to predict some linear regression.

## More Details
These are the main features.
* Able to calculate(in order) - Plus, Minus, Multiply, Divide, Sigmoid, Natural Logarithm, Sin, Cos, Exponential, mean-loss
* Able to manipulate computational graph freely.  
* Able to calculate gradient of the variables.
* Able to visualize computational graph using graphviz.
* Able to train a linear regression and so on ...

# Second Phase
Start using a vectorized data (numpy). We can test it using our network that is trained based on MNIST dataset.

## More Details
These are the main features.
* Able to train a simple feedforward network.
* Able to visualize computational graph using graphviz.
* Adding more functions(in order) - softmax, cross-entropy loss, dropout, L1-Regularizer, L2-Regularizer
* Able to train MNIST, GAN (possibly) ...

## Third Phase
Moving from feedforward neural network to Convolutional neural network.

## More Details
These are the main features.
* Able to train a CNN.
* Able to visualize computational graph using graphviz.
* Adding more functions(in order) - Convolution, Pooling, Flattening
* Able to train MNIST, GAN, CIFAR-10, Autoencoders, VAE ...

## Forth Phase
Moving from Convolutional neural network to Recurrent network

## More Details
These are the main features.
* Able to train a RNN.
* Able to visualize computational graph using graphviz.
* Adding more functions(in order) - Backprop through time, LSTM, GRU, attention, ...  
* Able to train Language modeling model, Simple Translation (maybe) ...

## Last Phase
Creating more example projects - Neural Turing Machine and many mores.
