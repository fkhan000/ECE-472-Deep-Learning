import tensorflow as tf


#########LINEAR CLASS#########

#This class represents a single layer of the MLP class

class Linear(tf.Module):

    #The constructor takes in number of inputs and outputs
    #for the layer and a boolean that tells us whether or not
    #the layer should have a bias term
    
    def __init__(self, num_inputs, num_outputs,  bias=True):
        rng = tf.random.get_global_generator()

        #we set our standard deviation
        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        #and hold the weight matrix for the layer in a tensorflow variable
        self.w = tf.Variable(
            #we initialize the weights by drawing it from a normal distribution
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),

            trainable=True,
            name="Linear/w"
        )

        self.bias = bias

        #and if we want to have a bias term
        if self.bias:
            #we draw the bias from a normal distribution as well
            self.b = tf.Variable(
                rng.normal(
                    shape=[1, num_outputs], stddev=stddev
                ),
                trainable=True,
                name="Linear/b"
            )
    #This function generates the output of the layer by multiplying the input
    #by the layer's weight matrix and adding the bias term
            
    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z
