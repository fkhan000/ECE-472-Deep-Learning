import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import struct
import Gen_Tune
from CNN import Classifier
import copy
import matplotlib.pyplot as plt


#Given the filename for the training or test set, this function
#extracts the data from the file and places it into a tensor
def extract(filename):

    #we open up the file
    with open(filename, "rb") as file:

        #and use the first 4 bytes to determine if this file contains images or labels
        if struct.unpack(">I", file.read(4))[0] == 0x00000803:

            #if the former we read the next 12 bytes to get the dimensions (size of each image
            #and number of images)
            shape = [struct.unpack(">I", file.read(4))[0] for index in range(3)]
            
        
        else:
            #else we only need to read the next 4 bytes to get the dimensions of our dataset
            #which is just going to be the number of labels in the file
            shape = [struct.unpack(">I", file.read(4))[0]]

        #we then read in the entire file 8 bytes at a time
        data = np.fromfile(file, dtype = np.dtype(np.uint8).newbyteorder(">"))


    #and reshape the tensor into the correct shape
    data = data.reshape(shape)


    return data


#we create an 80, 20 split between the training and validation set
#and shuffle the dataset
train_x, val_x, train_y, val_y = train_test_split(extract("MNIST_Database/train-images.idx3-ubyte"),
                                                      extract("MNIST_Database/train-labels.idx1-ubyte"),
                                                      test_size = 0.2,
                                                      random_state = 0x3e9c
                                                      )

test_x, test_y = (extract("MNIST_Database/t10k-images.idx3-ubyte"), extract("MNIST_Database/t10k-labels.idx1-ubyte"))

#conver the datasets into floats 
train_x, val_x, test_x = map(lambda x: tf.convert_to_tensor(x, dtype = np.float32), [train_x, val_x, test_x])
train_y, val_y, test_y = map(lambda x: tf.convert_to_tensor(x, dtype = np.int32), [train_y, val_y, test_y])



#This function trains the model on training set using given hyper parameters
#if export is False, the function just returns the model else it returns
#a list of the training and validation errors as well as the testing accuracy
#as a function of the number of training steps

def train(par,  export = False):

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    #we create our classifier
    model = Classifier( (28, 28), 10, 1,
                        [(2,2), (2,2), (2,2)],
                        [(2,2), (2,2), (2,2)],
                        [32, 32, 32],
                        conv_activation = tf.nn.relu,
                        output_activation = tf.nn.sigmoid
                        )

    
    decay_rate = .9999

    iteration = 1
    batch_size = 100

    #we will store the losses on the training, validation and test sets in these lists
    iterations = []
    training_losses = []
    validation_losses = []
    test_losses = []

    
    while( iteration < 20000):

        #every 50 iterations
        if iteration % 50 == 0:

            #we measure the performance of our model on each of the datasets
            cur_loss = evaluate(model)
            cur_loss += par[2]*tf.math.reduce_sum([ tf.math.reduce_sum(tf.math.square(x)) for x in model.trainable_variables])

            if export:
                iterations.append(iteration)
                validation_losses.append(cur_loss.numpy())
                training_losses.append(loss.numpy())
                test_acc = evaluate(model, val = False)
                test_losses.append(test_acc.numpy())

                print(test_acc)
                

        #we perform dropout regularization (see Module Test for a more detailed explanation)
               
        drop_outs = []
        recov = []
        for variable in model.trainable_variables:

            #we create the drop out matrix
            drop_out = tf.nn.dropout(tf.ones(variable.shape), rate = par[0])*(1-par[0])


            #save the weights in our model that will be dropped out
            reco = tf.math.subtract(variable, variable*drop_out)
            recov.append(reco)


            
            drop_outs.append(drop_out)

            #and then zero out some of the weights in our model
            variable.assign(variable*drop_out)
                
                
        
        #we select our batch
        batch_indices = rng.uniform(
            shape=[batch_size], maxval= train_x.shape[0], dtype=tf.int32
            )
        
        with tf.GradientTape() as tape:

                
            x_batch = tf.gather(train_x, batch_indices)
            y_batch = tf.gather(train_y, batch_indices)

            #we get the output from our model
            y_hat = model(x_batch)

            #and calculate the softmax cross entropy loss
            loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = y_batch,
                logits = y_hat))

            loss += par[2]*tf.math.reduce_sum([ tf.math.reduce_sum(tf.math.square(x)) for x in model.trainable_variables])

        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = par[1])

        #we then calculate the gradients
        grads = list(tape.gradient(loss, model.trainable_variables))

        #and to make sure we don't update the zeroed out weights, we zero out the corresponding
        #entries in our gradients
        for index in range(len(grads)):
            grads[index] *= drop_outs[index]

        #then we update our weights    
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        
        par[2] *= decay_rate
        iteration += 1

        
        for var, reco in zip(model.trainable_variables, recov):
            #finally we get back the zeroed out weights by adding them back in from reco
            var.assign_add(reco)

    if export:
        return model, iterations, training_losses, validation_losses, test_losses

    return model

#evaluates model on either validation or test set
def evaluate(model, val = True):

    #if we are tuning our hyperparameters
    if val:
        #then we evaluate the cross entropy loss on the validation set
        return tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = val_y,
                logits = model(val_x)))
    else:

        #else we want to calculate the accuracy of the model on the test set
        pred = model(test_x)

        #for each prediction, we get the column # of the largest value in that row
        max_val = tf.argmax(model(test_x), axis = 1)

        #then we convert predictions and labels into a one hot vector
        predictions = tf.one_hot(max_val, 10, dtype = tf.int32).numpy()

        labels = tf.one_hot(test_y, 10, dtype = tf.int32).numpy()

        #and then compare each of the rows between the predictions and labels
        #to determine the accuracy
        result = tf.math.equal(predictions, labels)
        result = tf.reduce_all(result, axis = 1)

        return tf.reduce_mean(tf.cast(result, 'float32'))
        
    
    
        
if __name__ == "__main__":

    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    #we call on the train function to get our model and the losses on each of the datasets
    model, iterations, train_losses, val_losses, test_losses = train([0, .001, .003], export = True)

    #and then plot the losses as a function of number of training steps
    plt.plot(iterations, train_losses, "b", label = "Training Error")
    plt.plot(iterations, val_losses, "r", label = "Validation Error")

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(loc = "upper right")
    plt.title("Training and Validation Error over Time") 

    plt.savefig("Training_Validation_Error_Over_Time.pdf")
    plt.clf()
    plt.plot(iterations, test_losses)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy as a Function of Time")

    plt.savefig("Test_Accuracy_Over_Time.pdf")
    
    
    accuracy = evaluate(model, val = False)

    print("Accuracy Achieved on Test Set: " + str(accuracy))

        

