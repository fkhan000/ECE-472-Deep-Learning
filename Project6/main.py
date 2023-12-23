from Attention import Transformer
import tensorflow as tf
import numpy as np
from Linear import Linear
from Adam import Adam

#This module tests to see if the transformer block can perform
#autoregressive prediction. To do this, I created a sample
#sentence and had a transformer overfit on it and then showed
#that it was able to reconstruct the sentence from just the <START>
#token

max_iter = 500
#since our sentence has 8 tokens, we set d_model to be 8
d_model = 8
warmup_steps = max_iter


sentence = "<START> Hi my name is Bart Simpson <END>"
words = sentence.split(" ")

#for encoding the tokens, I had each word be a row in an 8x8 identity matrix
#such that the ith token in the sentence corresponds to the ith row in the matrix
tokenized = np.reshape(np.identity(8, dtype = "float32"), (1,8,8))

#since we want our model to predict the next token, we want our label to be shifted
#over by one row so that the first row of label is the second row of our input matrix
#in order to make the matrix have the right size I added a padding token (a row of 0s) to the
#label matrix
label = np.delete(tokenized, 0, 1)
label = np.append(label, np.array([0 for i in range(8)])).reshape(1, 8, 8)

label = label.astype("float32")

model = Transformer(8, 2, 2)
#we use our implementation of the Adam algorithm with the parameters
#used in the All You Need is Attention paper
optimizer = Adam(model.trainable_variables, .9, .98, 1e-9)

iteration = 1

while(iteration < max_iter):

    #additonally we also use the learning schedule that was recommended by them
    step_size = (d_model**(-0.5))*min(iteration**(-0.5), iteration*1/(warmup_steps**1.5))

    iteration += 1
    
    with tf.GradientTape() as tape:

        #we feed in the input matrix to our model
        predicted = model(tokenized)
        #and compute the cross entropy loss between the predicted and label matrices
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = label,
            logits = predicted))

    #and then we update our model weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer(grads, model.trainable_variables, step_size)


#for inference, we start by feeding in an 8x8 matrix
#that only has the <START> token encoding as its first row
#the rest of the rows are paddding tokens
l = [ [0 for i in range(8)] for i in range(8)]

inp = np.reshape(l, (1, 8, 8)).astype("float32")
inp[0][0][0] = 1
print(loss)

pred = "<START>"
index = 0

#while we haven't reached the end of our sentence
while(pred != "<END>"):
    #we print out our prediction
    print(pred)
    #and look at which word has the highest chance of coming next after the previous words
    word_idx = tf.math.argmax(model(inp), axis = 2).numpy()[0][index]
    #we get our prediction
    pred = words[word_idx]
    index += 1
    #and then update the input matrix with our model's prediction
    inp[0][index][index] = 1

print("<END>")
    


