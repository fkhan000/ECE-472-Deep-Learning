import tensorflow as tf
import pytest
from Attention import MultiHeadAttention, Transformer
import numpy as np

#here we test the dimensionality of our attention block
def test_dimensionality():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    num_inputs = 50
    num_heads = 5

    
    block = MultiHeadAttention(num_inputs, num_heads)

    a = rng.normal(shape=[1, 10, 50])

    z = block(a)

    #and we should expect the same output size
    tf.debugging.assert_equal(z.shape, (1, 10, 50))



#this function tests to see if our model is trainable
#by checking if the derivatives of each of the weights
#in the model exist and are nonzero
def test_trainable():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 50
    num_heads = 5

    block = MultiHeadAttention(num_inputs, num_heads)

    a = rng.normal(shape=[1, 10, 50])

    
    with tf.GradientTape() as tape:
        z = block(a)

        loss = tf.math.reduce_mean(z**2)
    #we calculate the gradient 
    grads = tape.gradient(loss, block.trainable_variables)
    
    for grad, var in zip(grads, block.trainable_variables):
        #and then check to see if they all exist and are nonzero
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(block.trainable_variables) == 5*3*1 + 3*2 + 1*1

#this function tests to see if our MHA is causal
#it does by checking to see if the derivatives of the outputs
#are independent of "future inputs"
def test_causal():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_heads = 1
    num_blocks = 6

    block = MultiHeadAttention(num_inputs, num_heads, causal_mask = True)

    
    with tf.GradientTape() as tape:
        a = rng.normal(shape=[1, num_inputs, num_inputs])
        tape.watch(a)
        z = block(a)
        #since our output and input are both matrices, the jacobian is a little bit weirdly formatted
        jacobian = tape.batch_jacobian(z,  a)

    jacobian = jacobian.numpy()

    #the jacobian is stored in a 1x2x2x2x2 tensor which is just a 2x2 matrix where each entry is itself
    #a 2x2 matrix (i don't even wanna think about the hessian matrix would look like)
    #each entry in the larger 2x2 matrix can be thought of as a mini-jacobian for the corresponding output. So for the
    #first entry stores the derivatives of the first output element wrt to the input elements and so on.
    #Since we have two input tokens, we expect that the the first output shouldn't depend on the 2nd input token
    #So the bottom row of the first 2 "mini jacobians" should be 0 and the rest nonzero (see the README for a better
    #explanation
    assert jacobian[0][0][0][1][0] == jacobian[0][0][0][1][1] == jacobian[0][0][1][1][0] == jacobian[0][0][1][1][1] == 0
    assert np.count_nonzero(jacobian) == 12


@pytest.mark.parametrize(
    "causal_mask",
    [(True), (False)]
    )
#and for the transformer block the output should
#have the same size as the input
def test_dimensionality_transformer(causal_mask):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    num_inputs = 50
    sequence_length = 10
    num_heads = 5
    num_blocks = 6

    model = Transformer(num_inputs, num_heads, num_blocks, causal_mask = causal_mask)

    a = rng.normal(shape=[10, 15, 50])

    z = model(a)

    tf.debugging.assert_equal(z.shape, (10, 15, 50))


@pytest.mark.parametrize(
    "causal_mask",
    [(True), (False)]
    )
#and then we check to see if the gradients of our transformer
#exist and are nonzero
def test_trainable_transformer(causal_mask):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 50
    num_heads = 5
    num_blocks = 6

    model = Transformer(num_inputs, num_heads, num_blocks, causal_mask = causal_mask)

    a = rng.normal(shape=[15, 10, num_inputs])


    with tf.GradientTape() as tape:
        z = model(a)

        loss = tf.math.reduce_mean(z**2)
        
    grads = tape.gradient(loss, model.trainable_variables)

    for grad, var in zip(grads, model.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")

    assert len(grads) == len(model.trainable_variables) #== 22*6 + 2*(6+ 1)+ 6*1




    
