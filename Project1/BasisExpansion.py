import tensorflow as tf



#BASIS CLASS: This class holds the means and standard deviations
#for the set of gaussian functions used

class Basis(tf.Module):

    #We initialize the means and standard deviations by sampling them
    #from a uniform distribution from 0 to 1. Both means and sdevs
    #are of size Nx1 where N is the number of basis functions used.
    
    def __init__(self, numFunc):
        rng = tf.random.get_global_generator()

        self.means = tf.Variable(
            rng.uniform(shape = [numFunc, 1]),
            trainable = True,
            name = "Means")


        self.sdevs = tf.Variable(
            rng.uniform(shape = [numFunc, 1]),
            trainable = True,
            name = "Stdevs")
    
    #the evaluate function gives us the value of each of the gaussian
    #functions at a single point x where x can be a 1x1 tensor or a float.
    #The function returns an Nx1 tensor where N is the number of basis functions
    def evaluate(self, x):

        return tf.math.exp(-1*tf.math.square( (x-self.means)/self.sdevs))
