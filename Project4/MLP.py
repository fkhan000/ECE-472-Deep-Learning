import tensorflow as tf
from Linear import Linear


############### MLP CLASS ###############

#This class implements the multilayer perceptron. It takes
#in the number of inputs (ie the # of nodes for the first layer),
#number of hidden layers to be used, how many nodes each hidden layer should have,
#and the activation functions used by the model


class MLP(tf.Module):

    def __init__(self, num_inputs,
                 num_outputs,
                 num_hidden_layers,
                 hidden_layer_width,
                 hidden_activation=tf.identity,
                 output_activation=tf.identity,
                 ):

        #we represent the MLP as a list of instances from our linear class
        
        self.layers = [Linear(num_inputs, hidden_layer_width)]

        for i in range(num_hidden_layers):
            self.layers.append(Linear(hidden_layer_width, hidden_layer_width))

        self.layers.append(Linear(hidden_layer_width, num_outputs))

        self.layers[-1].w.assign(tf.zeros(shape = self.layers[-1].w.shape))
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    #here we produce the output of our model
    def __call__(self, x):

        z = x
        
        #we pass the input from the previous layer to the current one and then apply
        #the current layer's activation function
        
        for index in range(len(self.layers) - 1):

            z = self.hidden_activation(self.layers[index](z))
           
        return self.output_activation(self.layers[-1](z))


