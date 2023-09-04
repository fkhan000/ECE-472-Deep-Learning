import BasisExpansion as BE
import math
import tensorflow as tf
import matplotlib.pyplot as plt

#LINEAR CLASS: This class holds the weights and bias terms for the
#linear regression model.
class Linear(tf.Module):

    #we initialize the model by drawing the weights and bias
    #terms from a normal distribution with mean 0 and variance 1
    #We make the weights and bias tensor flow variables so that we can
    #more easily perform gradient descent with them using the tf library

    #Since for this assignment we're only trying to predict a single value and we're using SGD, I decided to fix
    #the number of outputs for the model to 1. As a result, the dimensions of the weights
    #and bias variale are 1xN and 1x1 respectively.
    
    def __init__(self, num_weights):

        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2/(num_weights + 1))
        self.weights = tf.Variable(rng.normal(shape = [1, num_weights], stddev = stddev),
                                   trainable = True,
                                   name = "Weights")
        
        self.bias = tf.Variable(rng.normal(shape = [1, 1], stddev = stddev),
                                trainable = True,
                                name = "Bias")

    #this function takes in an Nx1 input vector and multiplies it by the weights matrix and adds
    #the bias term to give our 1x1 output.
    def __call__(self, x):

        return self.weights @ x + self.bias


#This function updates our variables by moving
#them a step in the direction opposite to the gradient
#of the loss function. It takes in alpha which is our step
#size, variables which is a list of the trainable variables that we
#want to optimize, and gradients which represents our gradient
def step(alpha, variables, gradients):

    #for each variable and correspoding component in the gradient vector
    for var, grad in zip(variables, gradients):

        #we update the variable by moving it in the direction opposite
        #to the partial derivative
        var.assign_sub(alpha*grad)


#Here we use the linear and BE module to approximate a noisy sine wave
#using linear combinations of gaussian functions. The gaussian functions
#themselves aren't fixed either however. We're also allowed to optimize
#the parameters of the gaussian functions (mean and variance) as well.

if __name__ == "__main__":

    #Here we set some initial parameters for our experiment.
    num_func = 5
    num_samples = 50
    step_size = 0.01
    num_iters = 250

    #We fix the seed for our global generator so that we always
    #get the same initial values for our variables as well as the same
    #sample points which ensures that our code is reproducible.
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    #we sample the sine wave at random points from 0 to 1
    #(before I was actually sampling x from a normal instead of a uniform
    #distribution which made me wonder for a while why my fit wasn't that good)
    #We also add some 0 mean guassian noise with standard deviation 0.1 to the sine wave
    
    x = rng.uniform(shape=(num_samples, 1))
    noise = rng.normal(shape=(num_samples, 1), stddev = 0.1)
    
    y =  tf.math.sin(2*math.pi*x) + noise

    #We instantiate our linear regression model as well as our set of guassian functions
    linear = Linear(num_func)

    basis = BE.Basis(num_func)

    #Since we're performing SGD, for each iteration of gradient descent
    for itr in range(num_iters):

        #we go through each point in our training set
        for idx in range(len(x)):

            #and compute the gradient of the squared error of our model
            with tf.GradientTape() as tape:

                x_batch = tf.gather(x, idx)
                y_batch = tf.gather(y, idx)

                #to produce our estimate, we use the evaluate class function for basis
                #to get am Nx1 vector holding the value of each of the gaussian functions in our set
                #at the point x. Then we call on the linear class function to multiply those values
                #by the weights of our model plus the bias term. 
                y_hat = linear(basis.evaluate(x_batch))
                
                loss = 0.5*(y_batch - y_hat)**2

            #we get a list of our trainable variables by adding the list of trainable variables
            #from the Linear and Basis class.
            parameters = linear.trainable_variables + basis.trainable_variables
            grads = tape.gradient(loss, parameters)

            step(step_size, parameters, grads)

    #Since the input values were taken by sampling from a uniform distribution, they aren't sorted
    #which means that if we wanted to plot the fitted curve we made as it is right now,
    #the curve would be going back and forth and it would look like a random mess because the plt.plot function
    #connections based on if they're right next to one another in the list. So instead we can zip the two lists to get a list
    #of two tuples and then use the sorted function to sort the list of tuples by the first element in each tuple.
    #Then to unzip the object we can do zip(*..) to get the two lists back
            
    xs, ys = zip(*sorted(zip(x.numpy().squeeze(), [linear(basis.evaluate(tf.gather(x, idx)))[0][0].numpy() for idx in range(len(x))])))

    plt.figure()
    
    plt.plot(xs, ys)

    #Along with the plot of the fitted curve, we also add a scatterplot of the actual data points to the figure
    plt.scatter(x.numpy().squeeze(), y.numpy().squeeze())

    #and then we save the figure as a pdf file
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Fit of Noisy Sine Wave Using Gaussian Functions")
    plt.savefig("Khan_F_DL_Assignment1_LinFit.pdf")


    plt.figure()

    #We also need to make a plot of the basis functions used by our model

    #So we will sample each of the basis functions at a 1000 points equally spaced from -4 to 4 
    xdat = [-4 + 0.008*index for index in range(1000)]

    ydat = [basis.evaluate(xi) for xi in xdat]


    #Then we plot each basis function
    for funcN in range(num_func):
        #since each list in ydat holds the values of the basis functions at a single point
        #we need to go through ydat to get the funcN-th value in each list in ydat
        
        plt.plot(xdat,  [ydat[index][funcN][0].numpy() for index in range(len(ydat))])


    #and save the plot as a pdf
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian Basis Functions")
    plt.savefig("Khan_F_DL_Assignment1_BasisFunctions.pdf")
