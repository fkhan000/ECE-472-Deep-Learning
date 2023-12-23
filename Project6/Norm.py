import tensorflow as tf

#The code for this comes from the group norm paper
#that Prof Curro showed us: https://arxiv.org/abs/1803.08494
class Norm_Layer(tf.Module):

    def __init__(self):

        rng = tf.random.get_global_generator()

        #we make gamma and beta trainable and initialize them
        #by drawing them from a uniform distribution
        self.gamma = tf.Variable(rng.uniform(shape = [1], maxval = 1),
                                 trainable = True,
                                 name = "norm/gamma")
        
        self.beta = tf.Variable(rng.uniform(shape = [1], maxval = 1),
                                 trainable = True,
                                 name = "norm/beta")

                              
    def __call__(self, x):

        eps = 1e-5
        
        mean, var = tf.nn.moments(x, [1], keepdims = True)

        x = (x-mean) / tf.sqrt(var + eps)

        return x*self.gamma + self.beta
