import tensorflow as tf
from MLP import MLP
import math


#The code for this comes from the group norm paper
#that Prof Curro showed us: https://arxiv.org/abs/1803.08494
class Norm_Layer(tf.Module):

    def __init__(self, G):

        rng = tf.random.get_global_generator()

        #we make gamma and beta trainable and initialize them
        #by drawing them from a uniform distribution
        self.gamma = tf.Variable(rng.uniform(shape = [1], maxval = 1),
                                 trainable = True,
                                 name = "norm/gamma")
        
        self.beta = tf.Variable(rng.uniform(shape = [1], maxval = 1),
                                 trainable = True,
                                 name = "norm/beta")

        self.G = G

                              
    def __call__(self, x):

        N, H, W, C = x.shape

        eps = 1e-5
        
        x = tf.reshape(x, [N, H, W, self.G, C // self.G])

        
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims = True)

        x = (x-mean) / tf.sqrt(var + eps)

        x = tf.reshape(x, [N, H, W, C])

        return x*self.gamma + self.beta


#The residual layer takes in two layers in a network and creates a skip connection
#between them so that the output of layer1 is transformed and becomes input to
#layer 2

    #good unit test check to see if vanishing gradient problem
    #is less significant in resnets
class Residual_Layer(tf.Module):

    def __init__(self, layer1, layer2, layer_depth = 32):
        
        stride1 = 1
        length = 3
        width = 3

        self.padding = "SAME"

        #our residual block consists fo two convolutional layers
        #as well as two batch normalization layers
        
        self.conv_1 = Conv_Layer(length, width, stride1,
                                 layer1.kernel.shape[-1],
                                 layer_depth)
        

        self.conv_2 = Conv_Layer(length, width, 1,
                                 layer_depth,
                                 layer2.kernel.shape[-1])

        self.GN1 = Norm_Layer(16)

        
        self.GN2 = Norm_Layer(16)

        
    def __call__(self, x):

        #we order the layers of our residual block as described
        #in "Identity Mappings in Deep Residual Networks"
        
        z = tf.nn.relu(self.GN1(x))

        z = self.GN2(self.conv_1(z, padding = self.padding))

        return self.conv_2(tf.nn.relu(z),
                           padding = "SAME")
        
     

        

    
########CONV_LAYER CLASS########

#This class implements a single convolution layer in a CNN
#It holds the kernel for that layer and convolves the filter
#with a given image

class Conv_Layer(tf.Module):

    #This constructor takes in the length and width of the
    #kernel as well as its stride
    def __init__(self, length, width, stride, input_depth, layer_depth, input_size = None):

        rng = tf.random.get_global_generator()

        #to ensure that the output doesn't from this layer
        #doesn't get saturated by the activation function,
        #we set the standard deviation to be inversely proportional
        #to the size of the kernel
        stddev = tf.cast(tf.math.sqrt(2 / (length*width)), tf.float32)

        #we initialize the weights of the kernel by sampling
        #from a normal distribution
        self.kernel = tf.Variable(
            rng.normal(shape = (length, width, input_depth, layer_depth), stddev = stddev),
            trainable = True,
            name = "Conv/kernel"
            )
        
        self.stride = stride

        self.input_size = input_size

        

        

    #Here we convolve the kernel with the input x.
    #x is a tensor of size N x I x J x 1
    #where N is our batch size, I, J are the
    #length and width of our image 
    def __call__(self, x, padding = "SAME"):

        rng = tf.random.get_global_generator()

        
        return tf.nn.conv2d(x,
                            self.kernel,
                            self.stride,
                            padding = padding
                            )




