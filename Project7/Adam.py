import tensorflow as tf

#This class implements our Adam optimizer
class Adam():

    #constructor takes in 4 parameters, variables the list of the model's
    #trainable variables, and then the beta1, beta2, and epsilon parameters
    def __init__(self, variables, beta1, beta2, eps):

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        #we store the first and second momentum estimates in a list
        #with each element being the corresponding estimate for that trainable
        #variable
        self.m = []
        self.v = []
        self.iter = 0

        for var in variables:
            #we initialize the moment estimates to 0
            self.m.append(tf.Variable(tf.zeros(tf.shape(var)), trainable = False))
            self.v.append(tf.Variable(tf.zeros(tf.shape(var)), trainable = False))

        
    #This function uses the calculated gradient(g) and updates the
    #model's weights (v) and alpha is the step size
    def __call__(self, g, v, alpha):

        #we increment out iteration count
        self.iter += 1

        for index in range(len(g)):

            #we update our estimates of the first and second momentum
            self.m[index].assign(self.beta1*self.m[index] + (1-self.beta1)*g[index])
            self.v[index].assign(self.beta2*self.v[index] + (1-self.beta2)*tf.math.square(g[index]))

            #and then adjust it so as to correct for the bias
            m_hat = self.m[index]/(1-self.beta1**self.iter)
            v_hat = self.v[index]/(1-self.beta2**self.iter)
            
            #we obtain our update
            update = -1*alpha*m_hat/(tf.math.sqrt(v_hat) + self.eps)
            #and then update the corresponding trainable variable in the model
            v[index].assign_add(update)
            
            

            
            
    
        

