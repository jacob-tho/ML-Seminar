import numpy as np
import random
import gzip
import pickle
import pdb
import matplotlib.pyplot as plt



class Dense_Layer:
    def __init__(self, n_inputs: int, neurons: int): #Gewicht und Bias am Anfang initialisieren
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_inputs, neurons)) * 0.01
        self.biases = np.zeros((1, neurons)) #Wird durch Backpropagation eh geändert


    def forward(self, inputs: int) -> np.array:
        #Forward-Pass: Daten gehen von Anfang bis Ende/links nach rechts (Keine Schleifen oder 0 weights)
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases #Die Wichtigste Zeile - inputs und weights tauschen? weights transponieren?

    def backward(self, dvalues: np.array): #Erhält output von activation backward
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

#Aktivierungsfunktion sigmoid
class Activation_Sigmoid:

    def forward(self, inputs) -> np.array: #.output nach Aktivierungsfunktion für jeden Input
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues): #dvalues ist output von loss backward
        self.dinputs = (1 - self.output) * self.output * dvalues

    def predictions(self, outputs):
        return (outputs > 0.5) *1


#Find minimum, then shuffle training data and repeat
class Optimizer_SGD:
     def __init__(self, learning_rate=0.5):
         self.learning_rate = learning_rate

     def update_params(self, layer):
         layer.weights += -(self.learning_rate) * layer.dweights
         layer.biases += -(self.learning_rate) * layer.dbiases


class Loss_MeanSquared():
    def forward(self, true: np.array, predict: np.array):
        sample_loss = np.mean(((true - predict.argmax(axis=1))**2), axis=-1)
        return sample_loss

    def backward(self, dvalues, true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (true - dvalues) / outputs
        #Normalisieren, da MEAN square root
        self.dinputs = self.dinputs / samples #-> Kommt in backpass von activation

#'''
with gzip.open('fashion-mnist.pickled.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
'''
with gzip.open('mnist.pickled.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
'''
train_x, train_y = train_set #Numpy-arrays
test_x, test_y = test_set

# Evaluation function
def evaluate_model(test_x, test_y):
    # Forward pass
    input_layer.forward(test_x)
    activation_input.forward(input_layer.output)
    hidden_layer.forward(activation_input.output)
    activation_hidden.forward(hidden_layer.output)

    # loss und accuracy
    data_loss = loss_activation.forward(test_y, activation_hidden.output)
    prediction = activation_hidden.output.argmax(axis=1)
    accuracy = np.mean(prediction == test_y)

    print(f'Test Loss: {data_loss} - Test Accuracy: {accuracy}')
    return data_loss, accuracy

if __name__ == '__main__':

    np.random.seed(8)

    start_neurons = 784
    hidden_neurons = 30
    output_neurons = 10

    input_layer = Dense_Layer(start_neurons, hidden_neurons)
    activation_input = Activation_Sigmoid()

    hidden_layer = Dense_Layer(hidden_neurons, output_neurons)
    activation_hidden = Activation_Sigmoid()

    loss_activation = Loss_MeanSquared()
    optimizer = Optimizer_SGD()

    epoch = 30
    batch_size = 10

    verlust = []
    genauigkeit = []

    for loop in range(epoch):
        indices = np.arange(train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        epoch_loss = 0
        epoch_accuracy = 0
        for start in range(0,train_x.shape[0], batch_size):
            end = start + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
    #Feedforward input layer -> hidden

            input_layer.forward(batch_x) #Da kommen die Trainigsdaten rein
            activation_input.forward(input_layer.output)

    #Feedforward hidden layer -> output

            hidden_layer.forward(activation_input.output)
            activation_hidden.forward(hidden_layer.output)

    #Loss und Accuracy
            data_loss = loss_activation.forward(batch_y,activation_hidden.output)
            epoch_loss += data_loss
            prediction = activation_hidden.output.argmax(axis=1)
            accuracy = np.mean(prediction == batch_y)
            epoch_accuracy += accuracy

    #Backpropagation
            loss_activation.backward(activation_hidden.output, batch_y)
            activation_hidden.backward(loss_activation.dinputs)
            hidden_layer.backward(activation_hidden.dinputs)
            activation_input.backward(hidden_layer.dinputs)
            input_layer.backward(activation_input.dinputs)


    #SGD Optimierung
            optimizer.update_params(input_layer)
            optimizer.update_params(hidden_layer)

    #Evalutation
        print(f" Epoch: {loop} Data-Loss: {data_loss:.3f}, Accuracy: {accuracy:.3f}")
        verlust.append(epoch_loss / (train_x.shape[0] // batch_size))
        genauigkeit.append(epoch_accuracy / (train_x.shape[0] // batch_size))
        print(f'Epoch {loop+1}/{epoch} - Loss: {verlust[-1]} - Accuracy: {genauigkeit[-1]}')
        test_loss, test_accuracy = evaluate_model(test_x, test_y)

    #Visualisierung der Werte
    y_axis = verlust
    y2 = genauigkeit
    x_axis = range(epoch)
    fig, axs = plt.subplots(2)
    axs[0].plot(x_axis, y_axis)
    axs[1].plot(x_axis, y2)
    plt.show()
