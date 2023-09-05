from Linear import Linear
from BasisExpansion import Basis
import tensorflow as tf
import pytest

#This module performs unit and integration testing to ensure
#that the program works as intended

##LINEAR MODULE UNIT TESTS

#To make the unit tests for the Linear module I modified the tests
#that Professor Curro made for his linear regression class:
#https://gist.github.com/ccurro/491d3e888e06f446ec1ede559adfd47d

#When the bias for our linear model is 0 then we should expect it to have
#properties like linearity and homogeneity.

#To test for additivity

def test_additivity():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    num_inputs = 10

    #a linear model was instantiated with 10 inputs
    linear = Linear(num_inputs)
    
    #and the bias set to 0
    linear.bias.assign([[0]])

    #Then we created two 10x1 vectors whose components
    #were sampled from a normal distribution
    a = rng.normal(shape=[num_inputs, 1])
    b = rng.normal(shape=[num_inputs, 1])

    #finally we checked to see if linear(a+b) = linear(a) + linear(b).
    #If it did, then that meant that the model was associative
    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b), summarize=2)


#Then to check for homogeneity
def test_homogeneity():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    #we created a linear model with 0 bias
    linear = Linear(num_inputs)
    linear.bias.assign([[0]])

    #a 10x1 vector a and a random constant
    a = rng.normal(shape=[num_inputs, 1])
    
    #Since my linear class only gives outputs of size 1,
    #b would always be a 1x1 tensor
    b = rng.normal(shape=[1, 1])

    #We then checked to see if linear(a*b) = linear(a)*b
    tf.debugging.assert_near(linear(a * b), linear(a) * b, summarize=2)

#Then we want to make sure that the output of our model is always
#giving a 1x1 tensor.
    
def test_lin_dimen():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    #So we create a random input
    num_inputs = 10
    a = rng.normal(shape=[num_inputs, 1])

    linear = Linear(num_inputs)
    output = linear(a)

    #and check to see if the output of the model
    #would be of size 1.
    tf.assert_equal(tf.shape(output), 1)


#To ensure that the model's variables
#are trainable, that their partial derivative wrt
#to the loss function is never 0 or undefined,

def test_trainable_linear():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    num_inputs = 10

    #we createa linear model and get a random input
    linear = Linear(num_inputs)

    a = rng.normal(shape=[num_inputs, 1])

    #then we compute the gradient of the loss function wrt the weights
    #and bias
    with tf.GradientTape() as tape:
        z = linear(a)
        loss = tf.math.reduce_mean(z**2)
    
    grads = tape.gradient(loss, linear.trainable_variables)

    #we then check to make sure that each of the gradients are defined
    #and that they're never non zero
    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    #Finally we check to make sure that the number of gradients computed
    #is equal to the number of trainable variables whcih should be 2
    assert len(grads) == len(linear.trainable_variables) == 2

#Finally because the output of our linear model takes the sum of all of these weights
#the output is a random variable that's normally distributed with variance N*c^2 where
#c is the standard deviation of a single weight/bias. If the number of inputs is too large
#we could cause our inputs to be very spread out which might make it take longer for our model
#to converge. So we want linear models with larger number of inputs to have a smaller standard deviation.

#To test this,

@pytest.mark.parametrize(
    "input_a, input_b",
    [(1000, 500), (400, 300), (30, 3), (600, 200)]
    )
def test_init_properties(input_a, input_b):

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    #we create two linear models with different input sizes (with input_a > input_b)
    linear_a = Linear(input_a)
    linear_b = Linear(input_b)

    #measure their standard deviations
    std_a = tf.math.reduce_std(linear_a.weights)
    std_b = tf.math.reduce_std(linear_b.weights)
    
    #and then check to see if std_a < std_b
    tf.debugging.assert_less(std_a, std_b)


##BASIS EXPANSION MODULE UNIT TESTS

#For the basis expansion class I wanted to make sure that
#the dimension of the output of the evaluate function was always
#an Nx1 tensor where N is the number of basis functions
    
def test_basis_dimen():
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    numFunc = 10
    #we create an instance of the Basis class
    basis = Basis(numFunc)

    #and sample an input from a normal distribution
    x = rng.normal(shape = [1, 1])
    output = basis.evaluate(x)

    #We then check to make sure that the output of the evaluate function
    #is equal to numFunc.
    tf.assert_equal(tf.shape(output)[0], numFunc)


#This function tests to see if when we make a few adjustments to one of the basis functions
#that we can get the normal pdf and get it to integrate to 1.
def test_prob_dist():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    #we make only one gaussian in our basis
    numFunc = 1

    basis = Basis(numFunc)
    
    
    #we're gonna integrate over a range that's 60 standard deviations across centered at the mean
    #In the normal pdf, there's a factor of 1/2 in the exponent so we need to scale down the inputs to the gaussian
    #by a factor of sqrt(0.5) so that we effectively get that 0.5 coefficient in the exponent
    xdat = [basis.means[0].numpy() - (30 - 0.001*index)*(basis.sdevs[0].numpy()/(2**0.5)) for index in range(60000)]

    #evaluate the basis function at these points
    ydat = basis.evaluate(xdat)

    
    dx = 0.001*basis.sdevs[0].numpy()

    #then we can just do a left hand Riemann sum by multiplying our evaluations by dx and adding them up
    #After that, we have to scale what we have by the coefficient that's in the normal pdf
    integ = tf.math.reduce_sum(ydat*dx)/(basis.sdevs[0].numpy()*(2*math.pi)**0.5)

    #If the basis function is actually a guassian then our integral should be very close to 1
    tf.assert_less(1 - integ, 0.01)

##INTEGRATION TESTING

#One of the things I wanted to test here was that
#the combined ouput of the linear model and the
#basis functions was always a 1x1 tensor.
def test_comb_dimen():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    x = rng.normal(shape = [1,1])

    linear = Linear(num_inputs)

    basis = Basis(num_inputs)

    output = linear(basis.evaluate(x))

    tf.assert_equal(tf.shape(output), 1)

#I also tested to see if all of the variables, especially the means and variances
#were trainable when the combined output is used in the loss function.
    
def test_all_trainable():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    #we instantiate the linear model and the basis functions
    num_inputs = 10
    linear = Linear(num_inputs)

    basis = Basis(num_inputs)

    #get a random input
    a = rng.normal(shape=[num_inputs, 1])

    #and then compute the gradient of the loss function that uses the combined output
    #wrt all 4 of our variables
    with tf.GradientTape() as tape:
        z = linear(basis.evaluate(a))
        loss = tf.math.reduce_mean(z**2)
    
    grads = tape.gradient(loss, linear.trainable_variables + basis.trainable_variables)

    #we then check to make sure that each of the gradients are defined
    #and that they're never non zero
    
    for grad, var in zip(grads, linear.trainable_variables + basis.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    #Finally, we check to make sure that the number of gradients computed is 4
    assert len(grads) == len(linear.trainable_variables + basis.trainable_variables) == 4



#The last property I wanted to test was whether or not the gradient was being calculated properly
#I got the idea for this test from the paper that we were assigned to read for the week: "Stochastic
#Gradient Descent Tricks". To test if the gradient is correct, you can check if it can be used to approximate
#the loss function if all of the variables are shifted by some value epsilon.
    
@pytest.mark.parametrize(
    "num_inputs",
    [(100), (50), (10), (20), (1000), (5000), (10000)]
    )

def test_correct_gradient(num_inputs):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)


    #We create a vector that we will use to make a very small shift to the variables
    #The components are drawn from a uniform distribution from 0 to 1e-5
    epsilon = rng.uniform(shape =  [num_inputs*3 + 1, 1],
                          minval = 0,
                          maxval = 1e-5)

    #we instantiate our model and the basis functions
    basis = Basis(num_inputs)
    linear = Linear(num_inputs)

    #randomly choose our initial point
    a = rng.normal(shape=[num_inputs, 1])

    #Next we compute the gradient of our loss function at point a
    with tf.GradientTape() as tape:
        z = linear(basis.evaluate(a))
        loss1 = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss1, linear.trainable_variables + basis.trainable_variables)

    #we then shift our variables by the epsilon vector
    linear.weights.assign_add(tf.transpose(epsilon[0:num_inputs]))
    linear.bias.assign_add(epsilon[num_inputs:num_inputs + 1])
    basis.means.assign_add(epsilon[num_inputs + 1:num_inputs*2 + 1])
    basis.sdevs.assign_add(epsilon[num_inputs*2 + 1:])

    #we calculate the new loss at this point (a stays fixed)
    z = linear(basis.evaluate(a))
    loss2 = tf.math.reduce_mean(z**2)


    #and now we use the gradient to make a linear approximation for loss2
    #we take the dot product between the gradient and the epsilon vector
    delta_y = tf.matmul(grads[1], epsilon[0:num_inputs]) + tf.matmul(grads[0], epsilon[num_inputs:num_inputs + 1]) + tf.matmul(tf.transpose(grads[2]), epsilon[num_inputs + 1:num_inputs*2 + 1]) + tf.matmul(tf.transpose(grads[3]), epsilon[num_inputs*2 + 1:])

    #and then we check to see if our approximation and the actual loss at this new point are close to one another. 
    tf.debugging.assert_less(abs(loss2-  (loss1 + delta_y)), 0.01)
