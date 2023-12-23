import tensorflow as tf


#########LINEAR CLASS#########

#This class represents a single layer of the MLP class

class Linear(tf.Module):

    #The constructor takes in number of inputs and outputs
    #for the layer and a boolean that tells us whether or not
    #the layer should have a bias term
    
    def __init__(self, num_inputs, num_outputs, bias=True, first = False):
        rng = tf.random.get_global_generator()
        bound = tf.math.sqrt(6/(num_inputs))
        
        #and hold the weight matrix for the layer in a tensorflow variable
        self.w = tf.Variable(
            #we initialize the weights by drawing it from a uniform distribution

            rng.uniform(shape=[num_inputs, num_outputs], minval=-1*bound, maxval = bound),

            trainable=True,
            name= "Linear/w"
        )


        self.bias = bias

        #and if we want to have a bias term
        if self.bias:
            #we draw the bias from a uniform distribution as well
            self.b = tf.Variable(
                rng.uniform(
                    shape=[1, num_outputs], minval = -1*bound, maxval = bound,
                ),
                trainable=True,
                name= "Linear/b"
            )
        
        #and if this is the first layer, we multiply the weights by 30 so that the model can better capture high
        #frequency details in the image
        if first:
            self.w = tf.math.multiply(self.w, 30)
            self.b = tf.math.multiply(self.b, 30)
    #by the layer's weight matrix and adding the bias term
            
    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z