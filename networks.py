import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1 = 0, weight_regularizer_L2 = 0, bias_regularizer_L1 = 0, bias_regularizer_L2 = 0):
        #initialize all the required parameters
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs, training):
        #Output of a neuron is the multiplication of all inputs and weights, summed, and added with bias
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #taking the gradient with respect to each 
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #taking the gradient on regularization
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.weights += self.weight_regularizer_L1*dL1

        if self.weight_regularizer_L2 > 0:
            self.dweights += 2*self.weight_regularizer_L2 * self.weights
        
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1

        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2*self.bias_regularizer_L2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases
    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

'''PURPOSE: Dropout randomly zeros the outputs of neurons to prevent overfitting by limiting the model's reliance on certain neurons'''
class Layer_Dropout:
    def __init__(self, rate):
        #invert to get the success rate
        self.rate = 1-rate

    def forward(self, inputs, training):
        self.inputs = inputs
        # If not in the training mode - return values as we do not want to apply dropout to our validation data
        if not training:
            self.output = inputs.copy()
            return
        # using the binomial distr, randomly assign which neurons to be on and the other as having zero on the output
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.binary_mask

'''PURPOSE: Applied to the hidden layers of the neural network to ensure the output values do not go below 0 '''
class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self,dvalues):
        #make a copy od the dvalues first 
        self.dinputs = dvalues.copy()
        #the gradient is zero wherever the inputs values are less than or equal to zero
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


'''PURPOSE: Applied to the output layer to produce confidence scores
ex: [0.1,0.7,0.2] Adds up to one resembling a probability distribution'''
class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        #unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize the probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)
        self.output = probabilities

    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten the output array
            single_output = single_output.reshape(-1,1)
        #calculate the jacobian matrix
        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        #calculate sample-wise gradient 
        self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

'''PURPOSE: Applied to the output layer to produce score in range of 0-1 with values closer to 1 suggesting most likely
ex: [0.8,0.1,0.6] note not mutually exclusive'''
class Activation_Sigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        #sigmoid function
        self.output = 1/(1+np.exp(-inputs))

    def backward(self, dvalues):
        #Derivative of the Sigmoid Function
        self.dinputs = dvalues * (1-self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1



'''PURPOSE: Stochastic Gradient Descent is an optmization method used to find the global minimum of the loss function 
in other words, the weights and biases that will lead to the minimum loss'''
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay = 0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        if self.momentum:

            #if the momentum does not have initialized weight and biase then create them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            #Update the weight by multiplying the momentum with the current layer momentum and subtract the retain factor multiplied by the current gradient
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            #no momentum here, just update using the retain factor and gradient 
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        #now update the actual weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

"""PURPOSE: Short for Adaptive Momentum, Adam is most widely used optimizer"""
class Optimizer_Adam:
    
    def __init__(self, learning_rate = 0.001, decay = 0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0/(1.0 + self.decay * self.iterations))
    
    def update_params(self, layer):
        #if the layer does not have cashe arrays, create them filled with zeros 
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Unlike SGD, Adam optimizer adds a bias correction where both the momentum and 
        # cache are divided by beta_1 and also beta_2 
        
        
        #update momentum with gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights

        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

        #Corrected momentum 
        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations+1))
        
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations+1))

        #update cache with squared current gradients 
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2

        #Corrected Cache
        weight_cache_corrected = layer.weight_cache / (1-self.beta_2 ** self.iterations+1)

        bias_cache_corrected = layer.bias_cache / (1-self.beta_2 ** self.iterations + 1)

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

'''Common Loss class all Loss types use which is to calculate the overall loss'''
class Loss:

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            # L1 regularization - weights
            if layer.weight_regularizer_L1 > 0:
                regularization_loss += layer.weight_regularizer_L1 * \
                                       np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_regularizer_L2 > 0:
                regularization_loss += layer.weight_regularizer_L2 * \
                                       np.sum(layer.weights * \
                                              layer.weights)
            # L1 regularization - biases
            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 * \
                                       np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 * \
                                       np.sum(layer.biases * \
                                              layer.biases)
        return regularization_loss
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization = False):

        sample_losses = self.forward(output, y)

        #Calculate the mean loss
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization = False):
        
        data_loss = self.accumulated_sum/self.accumulated_count

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def new_pass(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0

'''Most commonly used loss function alongside Softmax Activation
It is simply the -predicted_probability * log(true_probability)
'''
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        #Ensures no division by 0 and also doesn't drag the mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #Probabilities for target values if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]

        #Mask values only if one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        
        #Number of Samples
        samples = len(dvalues)
        #Number of labels in every sample
        labels = len(dvalues[0])

        #If Catgorical labels then turn into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #Calc gradient
        self.dinputs = -y_true / dvalues
        #Normalize the gradient
        self.dinputs = self.dinputs / samples

'''PURPOSE: Combining Softmax and Cross Entropy Loss Activation yields a faster backward step'''
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        #If labels are one hot encoded then turn them into categorical labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #normalize gradient
        self.dinputs = self.dinputs / samples
    
"""Similar to Categorical Cross-Entropy Loss but Binary Cross Entropy includes the -log of the incorrect class outputs also"""
class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        #Again to prevent division by zero and not drag the mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped)) 
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        #Calculate gradient
        self.dinputs = -(y_true/clipped_dvalues-(1-y_true) / (1-clipped_dvalues)) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples

'''Along the loss, accuracy is used to measure how often the largest confidence score is the correct class
use argmax of the softmax output to find the predicted class of max output'''
class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    # Reset variables for next pass
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

'''Same accuracy calculation but this one needs the ground truth values to be a 2D array not one hot-encoded'''
class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

class Model:

    def __init__(self):
        #empty list of layers
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)
    
    #sets loss, optimizer, and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):

        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            #if first layer then the prev is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            #for the layers in between, prev is just the layer that comes before
            elif i < layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            #the output layer involves the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation  =self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

        #Call Activation_Softmax_loss_categoricalCrossentropy instead for a quicker backward step
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    def train(self, X, y, *, epochs = 1, batch_size = None, print_every = 1, validation_data = None):

        self.accuracy.init(y)

        train_steps = 1

        if validation_data is not None:
            validation_steps = 1

            X_val, y_val = validation_data

            if batch_size is not None:
                train_steps = len(X)//batch_size
                if train_steps * batch_size < len(X):
                    train_steps += 1

                if validation_data is not None:
                    validation_steps = len(X_val) // batch_size
                    if validation_steps * batch_size < len(X_val):
                            validation_steps += 1


        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                #If not batch_size, just test on the whole dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                #Forward pass
                output = self.forward(batch_X, training=True)
                #Calc Loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                #Total Loss
                loss = data_loss + regularization_loss
                # Calc Acciracy
                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)
                #Backward pass
                self.backward(output, batch_y)
                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                #Summary at each step
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            #Total training summary
            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')
            # If there is the validation data
            if validation_data is not None:
                # Reset accuracy and loss 
                self.loss.new_pass()
                self.accuracy.new_pass()
                for step in range(validation_steps):
                    #If batch_size is None, then train on full_dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                        batch_y = y_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                    # Perform the forward pass note on training = False meaning it will not apply Dropout on validation data
                    output = self.forward(batch_X, training=False)
                    #Calc Loss
                    self.loss.calculate(output, batch_y)
                    #Calc Accuracy
                    predictions = self.output_layer_activation.predictions(
                                      output)
                    self.accuracy.calculate(predictions, batch_y)
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                #Validation summary
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')
                    
    def forward(self, X, training):
        
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
                    
        return layer.output
    
    def backward(self, output, y):
        
        #Only call if Activation Function being used Softmax and loss is Categorical Cross Entropy
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            #call the backward step for all previous layers excluding the last in reversed order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return 
        

        self.loss.backward(output, y)


        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):

        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
           
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        #Reset loss and accuracy
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
                  
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[
                    step*batch_size:(step+1)*batch_size
                ]
                batch_y = y_val[
                    step*batch_size:(step+1)*batch_size
                ]
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(
                              output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

