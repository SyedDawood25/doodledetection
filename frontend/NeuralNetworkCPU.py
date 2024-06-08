import numpy as np
import matplotlib.pyplot as plt

class HiddenLayer:

    def __init__(self, noOfInputs, noOfNeurons, l2_regularizer_w=0, l2_regularizer_b=0):
        self.weights = 0.01 * np.random.randn(noOfInputs, noOfNeurons)
        self.biases = np.zeros((1, noOfNeurons))
        self.l2_regularizer_w = l2_regularizer_w
        self.l2_regularizer_b = l2_regularizer_b
    
    def forwardProp(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
    
    def backwardProp(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.l2_regularizer_w > 0:
            self.dweights += 2 * self.l2_regularizer_w * self.weights
        
        if self.l2_regularizer_b > 0:
            self.dbiases += 2 * self.l2_regularizer_b * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:

    def forwardProp(self, layerOutputs):
        self.inputs = layerOutputs
        self.outputs = np.maximum(0, layerOutputs)
    
    def backwardProp(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftmaxActivation:

    def calculate(self, layerOutputs):
        exponents = np.exp(layerOutputs - np.max(layerOutputs, axis=1, keepdims=True))
        return exponents / np.sum(exponents, axis=1, keepdims=True)

class CategoricalCrossEntropy:

    def calculate(self, softmaxOutputs, y_true):
        samples = len(softmaxOutputs)
        softmaxOutputsClipped = np.clip(softmaxOutputs, 1e-7, 1-1e-7)
        correctProbabilities = softmaxOutputsClipped[range(samples), y_true]
        return np.mean(-np.log(correctProbabilities))
    
class OutputActivationAndLoss:

    def __init__(self):
        self.activation = SoftmaxActivation()
        self.lossFunction = CategoricalCrossEntropy()

    def forwardProp(self, layerOutputs, y_true=None, calculateLoss=True):
        self.outputs = self.activation.calculate(layerOutputs)
        if calculateLoss:
            self.loss = self.lossFunction.calculate(self.outputs, y_true)
    
    def backwardProp(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
    
    def regularizationLoss(self, layer):

        regularization_loss = 0

        if layer.l2_regularizer_w > 0:
            regularization_loss += layer.l2_regularizer_w * np.sum(layer.weights ** 2)
        
        if layer.l2_regularizer_b > 0:
            regularization_loss += layer.l2_regularizer_b * np.sum(layer.biases ** 2)
        
        return regularization_loss

# Vanilla SGD with momentum and decay
class OptimizerSGD:

    def __init__(self, learning_rate = 1, lr_decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_lr = self.learning_rate
        self.lr_decay = lr_decay
        self.iterations = 0
        self.momentum = momentum
    
    def preUpdate(self):
        if self.lr_decay:
            self.current_lr = self.learning_rate * (1 / 1 + self.lr_decay * self.iterations)
    
    def postUpdate(self):
        self.iterations += 1

    def updateParams(self, layer):

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        
        else:
            weight_updates = -self.current_lr * layer.dweights
            bias_updates = -self.current_lr * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

# AdaGrad
class OptimizerAdaGrad:

    def __init__(self, learning_rate = 1, lr_decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_lr = self.learning_rate
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
    
    def preUpdate(self):
        if self.lr_decay:
            self.current_lr = self.learning_rate * (1 / 1 + self.lr_decay * self.iterations)
    
    def postUpdate(self):
        self.iterations += 1

    def updateParams(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

# RMSProp
class OptimizerRMSProp:

    def __init__(self, learning_rate = 1, lr_decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_lr = self.learning_rate
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def preUpdate(self):
        if self.lr_decay:
            self.current_lr = self.learning_rate * (1 / 1 + self.lr_decay * self.iterations)
    
    def postUpdate(self):
        self.iterations += 1

    def updateParams(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache += self.rho * layer.weight_cache + (1 - self.rho) * layer.dbiases ** 2
        
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

# Adam
class OptimizerAdam:

    def __init__(self, learning_rate = 1, lr_decay=0, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_lr = self.learning_rate
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def preUpdate(self):
        if self.lr_decay:
            self.current_lr = self.learning_rate * (1 / 1 + self.lr_decay * self.iterations)
    
    def postUpdate(self):
        self.iterations += 1

    def updateParams(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update Momentums with current gradients
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        # Corrected Momentums
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        # Update Caches with Squared current gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases ** 2

        # Corrected Caches
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
        
        # Update Rule
        layer.weights += -self.current_lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

class DropoutLayer:
  
  def __init__(self, dropout_rate):
    self.dropout_rate = 1 - dropout_rate
  
  def forwardProp(self, inputs):
    self.inputs = inputs

    self.binaryMask = np.random.binomial(1, self.dropout_rate, size=self.inputs.shape) / self.dropout_rate

    self.outputs = self.inputs * self.binaryMask
  
  def backwardProp(self, dvalues):
    self.dinputs = dvalues * self.binaryMask

class NN:

    def __init__(self, layer1, layer1_a, outputLayer, output_a, dropout_rate=0.5):
        self.layer1 = layer1
        self.layer1_a = layer1_a
        self.dropOut = DropoutLayer(dropout_rate)
        self.outputLayer = outputLayer
        self.output_a = output_a

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true) * 100

    def train(self, X, y, optimizer, val_x=None, val_y=None, epochs=1001, epsilon=0, printAfter=100, validation=False, plotGraph=True):

        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []

        for i in range(epochs):

            # Validation Test
            val_acc = val_loss = None
            if validation:
                self.layer1.forwardProp(val_x)
                self.layer1_a.forwardProp(self.layer1.outputs)
                self.dropOut.forwardProp(self.layer1_a.outputs)
                self.outputLayer.forwardProp(self.dropOut.outputs)
                self.output_a.forwardProp(self.outputLayer.outputs, val_y)

                val_acc = self.accuracy(np.argmax(self.output_a.outputs, axis=1), val_y)
                val_loss = self.output_a.loss
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

            # Forward Propagation Train
            self.layer1.forwardProp(X)
            self.layer1_a.forwardProp(self.layer1.outputs)
            self.dropOut.forwardProp(self.layer1_a.outputs)
            self.outputLayer.forwardProp(self.dropOut.outputs)
            self.output_a.forwardProp(self.outputLayer.outputs, y)
            acc = self.accuracy(np.argmax(self.output_a.outputs, axis=1), y)

            loss = self.output_a.loss
            reg_loss = self.output_a.regularizationLoss(self.layer1) + self.output_a.regularizationLoss(self.outputLayer)

            totalLoss = loss + reg_loss

            if i > 0 and epsilon > 0 and abs(totalLoss - losses[i-1]) <= epsilon:
                break

            losses.append(totalLoss)
            accuracies.append(acc)

            if i % printAfter == 0:
                print(f'Epoch: {i}, Train Loss: {losses[len(losses)-1]}, Train Accuracy: {acc}')
                if val_acc and val_loss is not None:
                    print(f'          Validation Loss: {val_losses[len(val_losses)-1]}, Validation Accuracy: {val_acc}')    

            # Bacward Propagation
            self.output_a.backwardProp(self.output_a.outputs, y)
            self.outputLayer.backwardProp(self.output_a.dinputs)
            self.dropOut.backwardProp(self.outputLayer.dinputs)
            self.layer1_a.backwardProp(self.dropOut.dinputs)
            self.layer1.backwardProp(self.layer1_a.dinputs)

            # Update Rule
            optimizer.preUpdate()
            optimizer.updateParams(self.layer1)
            optimizer.updateParams(self.outputLayer)
            optimizer.postUpdate()
        
        if plotGraph:
            plt.figure(figsize=(12,6))
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(len(losses)), losses)
            plt.plot(np.arange(len(val_losses)), val_losses)
            plt.ylabel('Loss')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(len(accuracies)), accuracies)
            plt.plot(np.arange(len(val_accuracies)), val_accuracies)
            plt.ylabel('Accuracy')
            plt.show()

    def test(self, X_test, y_test):
        self.layer1.forwardProp(X_test)
        self.layer1_a.forwardProp(self.layer1.outputs)
        self.outputLayer.forwardProp(self.layer1_a.outputs)
        self.output_a.forwardProp(self.outputLayer.outputs, y_test)
        acc = self.accuracy(np.argmax(self.output_a.outputs, axis=1), y_test)

        return acc, self.output_a.loss

    def predict(self, X):
        self.layer1.forwardProp(X)
        self.layer1_a.forwardProp(self.layer1.outputs)
        self.outputLayer.forwardProp(self.layer1_a.outputs)
        self.output_a.forwardProp(self.outputLayer.outputs, calculateLoss=False)

        return self.output_a.outputs, np.argmax(self.output_a.outputs, axis=1)