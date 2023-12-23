import tensorflow as tf
from MLP import MLP


########CONV_LAYER CLASS########

#This class implements a single convolution layer in a CNN
#It holds the kernel for that layer and convolves the filter
#with a given image

class Conv_2D(tf.Module):

    #This constructor takes in the length and width of the
    #kernel as well as its stride
    def __init__(self, length, width, stride, input_depth, layer_depth):

        rng = tf.random.get_global_generator()

        #to ensure that the output doesn't from this layer
        #doesn't get saturated by the activation function,
        #we set the standard deviation to be inversely proportional
        #to the size of the kernel
        stddev = tf.math.sqrt(2 / (length+ width))

        #we initialize the weights of the kernel by sampling
        #from a normal distribution
        self.kernel = tf.Variable(
            rng.normal(shape = (length, width, input_depth, layer_depth), stddev = stddev),
            trainable = True,
            name = "Conv/kernel"
            )
        
        self.stride = stride

        

    #Here we convolve the kernel with the input x.
    #x is a tensor of size N x I x J x 1
    #where N is our batch size, I, J are the
    #length and width of our image 
    def __call__(self, x):

        rng = tf.random.get_global_generator()

        
        return tf.nn.conv2d(x,
                            self.kernel,
                            self.stride,
                            padding = "VALID"
                            )

            

########CLASSIFIER CLASS########

#The classifier class implements our CNN
#It consists of a series of a convolutional layers
#and at the last convolutional layer, the output
#gets flattened and then fed into an MLP
#with the output layer consisting of N nodes
#where N is the number of classes.
    
class Classifier(tf.Module):

    
    def __init__(self, input_size, #tuple of ints giving dimension of image
                 num_classes, #int, number of classes
                 input_depth, #int number of channels for input
                 kernel_sizes, #list of tuples of ints giving dimensions of kernels used
                 strides, #list of ints from [1,2,4]
                 layer_depths, #list of ints
                 conv_activation = tf.identity, #activation function for each convolutional layer
                 output_activation = tf.identity
                 ):
        
        self.input_size = input_size
        self.input_depth = input_depth
        
        self.conv_layers = []

        self.conv_output_size = input_size


        #for each kernel
        for index in range(len(kernel_sizes)):

            #we append a new convolutional layer to our list of layers
            #with the corresponding kernel size and stride
        
            if index == 0:
                self.conv_layers.append(Conv_2D(kernel_sizes[index][0],
                                          kernel_sizes[index][1],
                                          strides[index], input_depth,
                                            layer_depths[index]
                                     ))
            else:
                self.conv_layers.append(Conv_2D(kernel_sizes[index][0],
                                              kernel_sizes[index][1],
                                              strides[index], layer_depths[index-1],
                                                layer_depths[index]
                                         ))
                
            #I also calculated the size of the output at the final convolution layer
            #by finding the size of the output for each convolution layer using this
            #that I found on stackoverflow: https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
            
            self.conv_output_size = [ (self.conv_output_size[i] - kernel_sizes[index][i])//strides[index][i] + 1 for i in range(len(kernel_sizes[index]))]

        self.conv_output_size.append(layer_depths[-1])

        #if either of the dimensions of the output is negative,
        #then our kernels were too large so we raise an exception to tell
        #the user that we were unable to construct the model

        
            
        self.conv_activation = conv_activation

        #and finally we construct the MLP. Since the input to the MLP
        #is the flattened output of the last convolutional layer, we
        #make the width of the input layer the number of elements in the output
        #from the convolutional layer

        self.perceptron = MLP(self.conv_output_size[0]*self.conv_output_size[1]*self.conv_output_size[2],
                               num_classes,
                               num_hidden_layers = 1,
                               hidden_layer_width = 100,
                               output_activation = output_activation
                              )
        
    #This function calculates the output of our CNN when given a NxIxJ
    #image tensor, x where N is the batch size, and I, J the dimensions of the image
    def __call__(self, x):

        #We reshape the input tensor into the format that our convolutional layer
        #can accept
        z = tf.reshape(x, [x.shape[0], self.input_size[0], self.input_size[1], self.input_depth])

        for layer in self.conv_layers:

            #then we pass our input through each of the convolution layers
            z = self.conv_activation(layer(z))

        #and then pass it through our MLP

        z = tf.reshape(z, [x.shape[0], -1])
        z = self.perceptron(z)


        #Finally we reshape the output so that it matches the labels
        #used in our dataset

        return z
