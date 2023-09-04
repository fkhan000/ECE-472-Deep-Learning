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
