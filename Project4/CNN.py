import tensorflow as tf
from MLP import MLP
import math
from Layers import Norm_Layer, Residual_Layer, Conv_Layer

class CNN(tf.Module):

    
    def __init__(self, input_size, #tuple of ints giving dimension of image (length, width)
                 input_depth, #int number of channels for input
                 num_classes, #int, number of classes
                 skip, #int, length of skip connection
                 conv_activation = tf.identity, #activation function for each convolutional layer
                 output_activation = tf.identity #output activation function
                 ):
        
        self.input_size = input_size
        
        self.input_depth = input_depth
        
        self.num_classes = num_classes
        
        self.skip = skip
        
        self.conv_output_size = input_size

        self.layers = []
        
        self.conv_layers = []

        self.residual_blocks = []

        self.norm_layers = []

        self.perceptron = None

        self.conv_activation = conv_activation

        self.output_activation = output_activation


    #Convolutional_Layer class method allows user to add convolutional layer
    #to their CNN model. 
    def Convolutional_Layer(self, kernel_size,
                            stride, layer_depth):

        
        depth = self.input_depth

        if len(self.conv_layers) != 0:

            depth = self.conv_layers[-1].kernel.shape[-1]

        #we add the convolutional layer to our list
        self.conv_layers.append(Conv_Layer(kernel_size[0], kernel_size[1],
                                           stride, depth,
                                           layer_depth,
                                           input_size = self.conv_output_size))

        
        if len(self.conv_layers) >= self.skip:

            #and if we already have more than skip conv layers, then we would be adding a residual layer
            #going from the current layer to the one skip layers ago
            self.residual_blocks.append(Residual_Layer(self.conv_layers[-1*self.skip], self.conv_layers[-1]))

        self.layers.append("Convolutional Layer")


        
    #Our user can add a batch norm layer
    def Group_Norm_Layer(self, G):

        self.norm_layers.append(Norm_Layer(G))

        self.layers.append("Batch Norm Layer")

    #as well a single MLP at the very end of the CNN
    def Dense_Layers(self,
                    num_hidden_layers,
                    hidden_layer_width,
                    hidden_activation = tf.identity,
                    output_activation = tf.identity):

        
        self.perceptron = MLP(self.conv_output_size[0]*self.conv_output_size[1]*self.conv_layers[-1].kernel.shape[-1],
                               self.num_classes,
                               num_hidden_layers = num_hidden_layers,
                               hidden_layer_width = hidden_layer_width,
                               output_activation = output_activation,
                              )

        self.layers.append("Dense Layers")
    
        
    #This function calculates the output of our CNN when given a NxIxJ
    #image tensor, x where N is the batch size, and I, J the dimensions of the image
    def __call__(self, x):
        

        z = tf.reshape(x, [x.shape[0], self.input_size[0], self.input_size[1], self.input_depth])

        #we will keep track of what type of layer we're at in this list
        indices = [0, 0]

        #a circular array containing the outputs from previous layers in the model
        past_z = [0 for i in range(self.skip)]

        #for each layer
        for layer in self.layers:
            
            if layer == "Convolutional Layer":

                #we pass our input through the convolutional input
                z = self.conv_layers[indices[0]](z)


                if indices[0] >= self.skip:

                    #and also add the output of the residual layer directly to z
                    z += self.residual_blocks[indices[0] - self.skip](past_z[(indices[0] - self.skip)%self.skip])

                past_z[indices[0] % self.skip] = z
                
                z = self.conv_activation(z)
                
                indices[0] += 1


            elif layer == "Batch Norm Layer":
                
                z = self.norm_layers[indices[1]](z)
                indices[1] += 1

                
            else:
                
                z = tf.reshape(z, [x.shape[0], -1])
                
                z = self.perceptron(z)
                
        
        return z
