import pytest
import tensorflow as tf
from CNN import CNN
from Layers import Norm_Layer, Residual_Layer, Conv_Layer

#Integration Test: Does model overfit quickly on small training set
#Integration Test: Check to see if CNN is less prone to vanishing gradient
#problem with skip connections

#########MODULE TESTS##########


##Norm Layer

#This function tests to make sure that norm layer is outputting the correct
#dimensions which should be the same as its input
def test_norm_dimensionality():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    a = rng.normal(shape = [128, 32, 32, 256])
    
    layer = Norm_Layer(16)

    
    z = layer(a)

    tf.assert_equal(z.shape, a.shape)
    

#This function makes sure that the norm layer is setting the mean
#and variance properly
def test_moments():
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    N = 128
    H = 32
    W = 32
    C = 256
    G = 16
    
    a = rng.normal(shape = [N, H, W, C])

    b = rng.normal(shape = [N, H, W, C])

    layer = Norm_Layer(G)

    #we get our mean and standard deviation
    mean = layer.beta

    sdev = layer.gamma

    z = layer(a)

    z = tf.reshape(z, [N, H, W, G, C // G])

    #we run it two inputs through the layers and check to see if they both
    #have the same mean and variance
    mean_a, var_a = tf.nn.moments(z, [1, 2, 4], keepdims = True)

    z = layer(b)

    z = tf.reshape(z, [N, H, W, G, C // G])

    mean_b, var_b = tf.nn.moments(z, [1, 2, 4], keepdims = True)

    tf.debugging.assert_near(mean_b, mean_a)
    tf.debugging.assert_near(mean, mean_a)

    tf.debugging.assert_near(var_a, var_b)
    tf.debugging.assert_near(tf.math.square(sdev), var_a, atol = 1e-3)



###########RESIDUAL LAYER###########


#this checks to see if the residual block is outptutting the correct
#dimensions
def test_res_dimensionality():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    #we create two convolution layers
    layer_1 = Conv_Layer(4, 2, 1, 32, 16)

    layer_2 = Conv_Layer(7, 2, 3, 16, 256)

    res_layer = Residual_Layer(layer_1, layer_2)


    #then we run it through the layer 1 and then through the residual block
    a = rng.normal(shape = [100, 28, 28, 32])

    z = layer_1(a)

    z_res = res_layer(z)

    #and check to see if the dimensions match up
    tf.debugging.assert_equal(z_res.shape[-1], layer_2.kernel.shape[-1])


############INTEGRATION TESTS###############


#This class checks to see if our classifier is
#outputting the correct dimension
def test_dimensionality_classifier():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_size = (28, 28)
    num_classes = 10
    input_depth = 3
    skip = 2
    
    model = CNN(input_size,
                       input_depth,
                       num_classes,
                       skip)
    

    for i in range(6):
        model.Convolutional_Layer( (3,3), 1, 64)
        model.Group_Norm_Layer(16)
        
    model.Dense_Layers(2, 100)
    
    #our input will be a 4 dimensional tensor
    #with the first dimension being the batch size,
    #the next two dimensions being the length and width of each
    #"image" and how many channels our image has
    a = rng.normal(shape = [100, 28, 28, input_depth])

    z = model(a)

    #if we did everything right we should have a 100x10 tensor
    #with each row corresponding to a vector that gives something
    #related to the probability that the training point is the ith class
    output_size = (100, 10)
    tf.assert_equal(tf.shape(z), (output_size))

#Here we check to see if all of the variables
#are trainable
def test_trainable_classifier():
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_size = (32, 32)
    num_classes = 10
    input_depth = 3
    skip = 2
    
    model = CNN(input_size,
                       input_depth,
                       num_classes,
                       skip,
                conv_activation = tf.nn.relu)
    

    for i in range(4):
        model.Convolutional_Layer( (3,3), 1, 64)
        model.Group_Norm_Layer(16)
    
    model.Dense_Layers(1, 100)

    a = rng.normal(shape = [100, 32, 32, input_depth])
    with tf.GradientTape() as tape:
        z = model(a)
        
        loss = tf.math.square(z)

    
    grads = tape.gradient(loss, model.trainable_variables)

    #With our architecture, we would have 4 variables from the convolution layers (one for each kernel),
    #2 variables from each group norm layer (4 layers), and then in each residual block,
    #we have 2 convolution layers and 2 group norm layers. Finally, we have an ML with 3 layers
    #so it has 6 variables. So altogether we should have 36 trainable variables
    assert len(grads) == len(model.trainable_variables) == 4 + 2*4 + 3*(2 + 2*2) + 3*2

#This is a helper function for our test_less_vanishing function
#It calculates the fraction of variables in the model with 0 gradients
    
def get_frac_gradients(a, resid = True):
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_size = (50, 50)
    num_classes = 10
    input_depth = 3
    
    if resid:
        skip = 2
    else:
        #to make a CNN without a residual block, I just made the skip attribute
    #50
        skip = 50
    
    model = CNN(input_size,
                       input_depth,
                       num_classes,
                       skip,
                conv_activation = tf.nn.relu)

    # we add in 20 convolution layers and 20 group norm layers
    for i in range(20):
        model.Convolutional_Layer( (3,3), 1, 64)
        model.Group_Norm_Layer(16)

    #and then feed it into an MLP with 4 layers
    model.Dense_Layers(2, 100)
    
    with tf.GradientTape() as tape:

        z = model(a)

        loss = tf.math.reduce_mean(tf.math.square(z))

    grads = tape.gradient(loss, model.trainable_variables)


    count = 0

    #and now we count how many of our gradients are 0. For the
    #nn's with like 500 or a 1000 hidden layers, nearly all of the
    #gradients are 0 but for the much smaller ones with only 25-50
    #layers there aren't as many zero gradients
    
    for grad in grads:
        #we calculate the sum of the squares of all of the gradients of the weights
        #in each layer. And then check if that's equal to 0. If it's 0 then that
        #means that that entire layer won't change during this step. So when
        #you have a really large neural network, if the vanishing gradient problem
        #isn't handled correctly, training becomes very inefficient as very few of your weights
        #actually get updated per step
        
        if grad == None or tf.math.reduce_sum(tf.math.square(grad)) == 0:
            count += 1

    return count/len(grads)

    

#This function tests to see if our residual blocks
#help lessen the vanishing gradient problem for models
#with many layers. 
    
def test_less_vanishing():

    input_depth = 3
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    a = rng.normal(shape = [100, 50, 50, input_depth])

    #we then use the get_frac_gradients function to get the
    #percent of gradients in our model whose gradients are 0
    frac_grad0_resid = get_frac_gradients(a, resid = False)

    
    
    frac_grad0_no_resid = get_frac_gradients(a)

    #If we have implemented and integrated the residual block class correctly
    #then we should have a greater number of gradients as a percent of the total number of variables in our model
    
    tf.debugging.assert_less_equal(frac_grad0_resid, frac_grad0_no_resid)

    
