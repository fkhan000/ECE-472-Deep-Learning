import pytest
import tensorflow as tf
from CNN import Conv_2D, Classifier
import copy
########MODULE TESTS FOR CONV_2D CLASS########


#This function tests to see if the dimensions of the output
#of the convolution layer is what we expect it to be

def test_dimensionality():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    input_depth = 2
    layer_depth = 3


    #we create a random input to the layer
    a = rng.normal(shape = [100, 28, 28, input_depth])

    
    #create our layer
    layer = Conv_2D(3, 2,
                    (2, 1),
                    input_depth,
                    layer_depth
                    )

    z = layer(a)

    #The size of the output from a kernel is given by the formula:
    #floor((W-K+2*P)/S) + 1 where W is the size of our input (length and width)
    #K is the size of the kernel (again length and width) P is the padding
    #but since we had set PADDING to be VALID, P would just be 0 here and
    #finally S is the stride.
    output_size = ( 100, (28 - 3)//2 + 1, (28-2)//1 + 1, layer_depth)

    #If we implemented this class correctly the output size should be exactly
    #the same as what we just calculated it to be
    tf.assert_equal(tf.shape(z), (output_size))


#In this test, we check to see that the initialization
#of the convolutional layer is such that the standard deviation
#of each of the weights in the kernel decreases with the size of the kernel.
#This helps prevent early saturation of our nodes
@pytest.mark.parametrize(
    "size1, size2",
    [([100, 30], [50, 10]), ([45, 300], [30, 40]), ([30, 3], [6, 2])]
    )

def test_init_properties(size1, size2):
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    #we create two conv layers with sizes such that
    #kerenl_a is larger than kernel_b
    layer_a = Conv_2D(size1[0], size1[1],
                      2, 3, 2)

    layer_b = Conv_2D(size2[0], size2[1],
                      2, 3, 2)

    #Then we calculate the standard deviation of the weights of the kernels
    std_a = tf.math.reduce_std(layer_a.kernel)

    std_b = tf.math.reduce_std(layer_b.kernel)

    #and if we did this part correctly, std_a should be less than std_b
    tf.debugging.assert_less(std_a, std_b)


#Here we test to see if all of the variables in our conv_2D are trainable
def test_trainable_layer():
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    input_depth = 2
    layer_depth = 3
    length = 3
    width = 2
    
    layer = Conv_2D(length, width,
                    (2, 1),
                    input_depth,
                    layer_depth
                    )
    
    a = rng.normal(shape = [100, 28, 28, input_depth])

    

    with tf.GradientTape() as tape:
        z = layer(a)

        loss =  tf.math.reduce_mean(z**2)

    #we calculate the gradient
    grads = tape.gradient(loss, layer.trainable_variables)

    for grad, var in zip(grads, layer.trainable_variables):

        #and then check to see how many of the variables have nonzero gradients
        #and are defined
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    #then we check to see if all of our variables are trainable
    assert len(grads) == len(layer.trainable_variables) == 1    
    
########MODULE TESTS FOR CLASSIFIER CLASS########

#This class checks to see if our classifier is
#outputting the correct dimension
def test_dimensionality_classifier():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_size = (28, 28)
    num_classes = 10
    input_depth = 3
    kernel_sizes = [(1, 2), (2,2)]


    strides = [(1,2), (2, 3)]

    layer_depths = [64, 32]

    
    model = Classifier(input_size,
                       num_classes,
                       input_depth,
                       kernel_sizes,
                       strides,
                       layer_depths)
    
    
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

    input_size = (28, 28)
    num_classes = 10
    input_depth = 3
    kernel_sizes = [(1, 2), (2,2)]


    strides = [(1,2), (2, 3)]

    layer_depths = [64, 32]


    model = Classifier(input_size,
                       num_classes,
                       input_depth,
                       kernel_sizes,
                       strides,
                       layer_depths)
    
    
    
    a = rng.normal(shape = [100, 28, 28, input_depth])


    with tf.GradientTape() as tape:
        z = model(a)

        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, model.trainable_variables)

    for grad, var in zip(grads, model.trainable_variables):
        
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    #since our CNN is attached to an MLP with one hidden layer, the MLP
    #would add 6 trainable variables (two for each layer)
    assert len(grads) == len(model.trainable_variables) == len(kernel_sizes) +  6

#Here we check to make sure that we are doing drop out regularization correctly
def test_drop_out():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    
    input_size = (28, 28)
    num_classes = 10
    input_depth = 3
    kernel_sizes = [(1, 2), (2,2)]

    strides = [(1,2), (2, 3)]

    layer_depths = [64, 32]

    #we will zero out 20% of the weights in our model
    drop_rate = 0.2

    #we create our classifier
    model = Classifier(input_size,
                       num_classes,
                       input_depth,
                       kernel_sizes,
                       strides,
                       layer_depths)

    #create a random input to the classifier
    a = rng.normal(shape = [100, 28, 28, input_depth])

    drop_outs = []

    recov = []
    

    #we will use model2 as a way to see what changes have been made to model.
    #At the end of one training step, we should see all of weights that were dropped
    #out to have the same values as before and the weights that weren't dropped out
    #should be different
    model2 = copy.deepcopy(model)

    #for each variable in our model
    for variable in model.trainable_variables:

        #we create a tensor of 1s and take 20% of the entries to be 0
        drop_out = tf.nn.dropout(tf.ones(variable.shape), rate = drop_rate)*(1-drop_rate)

        #then before removing them we save all of the weights about to be dropped out
        #into reco
        reco = tf.math.subtract(variable, variable*drop_out)

        
        
        recov.append(reco)

        
        drop_outs.append(drop_out)

        #then we update our variable
        variable.assign(variable*drop_out)

    
    #we calculate the gradient of the "loss"
    with tf.GradientTape() as tape:

        z = model(a)

        loss = tf.math.reduce_mean(z**2)

    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = .1)
    grads = list(tape.gradient(loss, model.trainable_variables))
    
    #before updating, we zero out the entries of the gradients
    #corresponding to the weights that were dropped out. Since they
    #were removed from the model, they shouldn't be updated for this iteration
    for index in range(len(grads)):
        grads[index] *= drop_outs[index]

    #we apply the gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #after updating, we add back in the dropped out weights
    for var, reco in zip(model.trainable_variables, recov):
        var.assign_add(reco)

    #then for each variable in the unaltered and altered model
    for var1, var2, drop in zip(model.trainable_variables, model2.trainable_variables, drop_outs):

        #we check to see if the weights that were dropped out are still the same
        tf.assert_equal(tf.math.reduce_min(tf.cast(tf.math.equal(var1*(tf.math.subtract(tf.ones(drop.shape), drop)),
                       var2*(tf.math.subtract(tf.ones(drop.shape), drop))), "float32")), tf.constant(1.0))

        #and then we check to see if the weights that weren't dropped out are now different
        tf.assert_equal(
            tf.math.reduce_max(tf.cast(tf.math.equal(
                var1,
                var2*drop), "float32")), tf.constant(0.0))
