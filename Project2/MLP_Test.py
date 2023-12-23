from MLP import MLP
import tensorflow as tf
import pytest



#if the activation functions are just identity and we set the bias
#of each layer to 0 then our MLP should have linearity so it should
#have properties like additivity and homogeneity


def test_additivity():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    num_inputs = 10
    #we create an MLP
    model = MLP(num_inputs, 10, 5, 15)


    #and assign the bias of each layer to 0
    for layer in model.layers:

        layer.b.assign(tf.zeros_like(layer.b))

    #we create two random inputs to our model
    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    #and then we check if the output of our model distributes over addition
    tf.debugging.assert_near(model(a+b), model(a) + model(b), summarize = 2)


def test_homogeneity():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    num_inputs = 10
    num_test_cases = 100

    #we create our model
    model = MLP(num_inputs, 10, 5, 15)

    #assign the bias of each layer to 0
    for layer in model.layers:

        layer.b.assign(tf.zeros_like(layer.b))

    #create a random vector to be our input and another random vector
    #to be our "scalars"
    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    #we then check if model(a*b) = model(a)*b
    tf.debugging.assert_near(model(a * b), model(a) * b, summarize=2)


#here we test to make sure that the dimensionality of the output of our MLP
#is correct
    
def test_dimensionality():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    num_inputs = 10
    num_outputs = 10
    model = MLP(num_inputs, num_outputs, 5, 15)

    #we create a random input for our MLP
    a = rng.normal(shape=[1, num_inputs])

    #pass it in to the model
    z = model(a)

    #and then check if the output is num_outputs long
    tf.assert_equal(tf.shape(z)[-1], num_outputs)
    

#here we check if the variables of our MLP are trainable
def test_trainable():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    
    num_inputs = 10
    num_hidden_layers = 5

    #we create our model
    model = MLP(num_inputs, 10, num_hidden_layers, 15,
                hidden_activation = tf.nn.elu,
                output_activation = tf.math.sigmoid
                )
    #and a random input
    a = rng.normal(shape=[1, num_inputs])


    #get the output and the "loss"
    with tf.GradientTape() as tape:
        z = model(a)
        loss = tf.math.reduce_mean(z**2)

    #and then we get the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    #and check if all of the gradients are finite and nonzero
    for grad, var in zip(grads, model.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    #we then also check to see if the gradients for all of our trainable variables
    #were calculated
        
    assert len(grads) == len(model.trainable_variables) == 2*(num_hidden_layers + 2)


#this test checks to see if the vanishing gradient becomes more
#prevelant for larger neural networks. 
@pytest.mark.parametrize(
    "num_h_layer1, num_h_layer2",
    [(1000, 500), (500, 250), (200, 100), (50, 25)]
    )
def test_vanishing_grad(num_h_layer1, num_h_layer2):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    #We create two models with different number of layers
    model1 = MLP(num_inputs, 10, num_h_layer1, 15,
                hidden_activation = tf.nn.elu,
                output_activation = tf.math.sigmoid
                )

    model2 = MLP(num_inputs, 10, num_h_layer2, 15,
                hidden_activation = tf.nn.elu,
                output_activation = tf.math.sigmoid
                )

    #we create a random input
    a = rng.normal(shape=[1, num_inputs])

    counts = []

    #for each model we calculate the gradient 
    for model in [model1, model2]:
        
        with tf.GradientTape() as tape:
            z = model(a)
            loss = tf.math.reduce_mean(z**2)

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
            if tf.math.reduce_sum(tf.math.square(grad)) == 0:
                count += 1
        counts.append(count)
        

    #we then check to see if the number of nonzero gradients for the larger network
    #is greater than that of the smaller one
    tf.debugging.assert_greater_equal(counts[0], counts[1])
