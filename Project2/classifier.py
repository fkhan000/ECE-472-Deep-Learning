import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from MLP import MLP
from sklearn.inspection import DecisionBoundaryDisplay

#This step function goes through each of the variables in our model
#and updates them by 

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


#Here we use the MLP class to classify points on a Cartesian plane drawn from 2
#archimedean spirals
if __name__ == "__main__":

    #we set our train size to be 10000
    train_size = 10000
    rng = np.random.default_rng()

    #since there is no explicit equation for this curve in cartesian form
    #it's a lot easier to use the polar form to produce the curves

    #we sample theta from a uniform distribution
    theta = rng.uniform(np.pi/8, 4*np.pi, train_size)

    noise = rng.normal(0, 0.05, size = train_size)
    phi = 3*np.pi

    #We convert the polar coordinates to x and y cooridnates. We assign the label 0 to all points on curve 1
    curve1 = np.column_stack(((theta*np.cos(theta), theta*np.sin(theta) + noise, np.zeros(train_size))))

    #and the label 1 to all points on the second curve
    curve2 = np.column_stack( (theta*np.cos(theta + phi), theta*np.sin(theta + phi) + noise, np.ones(train_size)))

    #then we concatenate both set of labelled points and shuffle them
    train_set = np.float32(np.vstack((curve1, curve2)))
    rng.shuffle(train_set)
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    #set our training parameters
    num_iters = 10000
    step_size = .15
    batch_size = round(0.1*(2*train_size))
    decay_rate = .9999
    lamb = 0.0001

    #initialize our model
    model = MLP(2, 1, 10, 15,
                hidden_activation = tf.nn.elu,
                output_activation = tf.math.sigmoid
                )

    #for each iteration
    for itera in range(num_iters):

        #we sample from the training set
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=train_size*2, dtype=tf.int32
            )
        
        with tf.GradientTape() as tape:

            #get the coordinates and labels for each point
            x_batch = tf.gather(train_set, batch_indices)[:,:2]
            
            y_batch = tf.reshape(tf.gather(train_set, batch_indices)[:,-1], (batch_size, 1))

            #pass the coordinates in to our model
            y_hat = model(x_batch)

            #and then calculate the loss using the binary cross entropy loss function
            #we also add a complexity penalty so that we can make sure 
            loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels = y_batch,
                logits = y_hat
                )) + lamb*tf.math.reduce_sum([ tf.math.reduce_sum(tf.math.square(x)) for x in model.trainable_variables])

        
        #we calculate the gradient with respect to our model's parameters
        grads = tape.gradient(loss, model.trainable_variables)

        #update our parameters
        step(step_size, model.trainable_variables, grads)

        #and slightly decrease our step size
        step_size *= decay_rate


    #to plot the decision boundary, we first create a grid of points that we will evaluate our model on
        
    feature_1, feature_2 = np.meshgrid(
        np.linspace(train_set[:,0].min(), train_set[:,0].max()),
        np.linspace(train_set[:,1].min(), train_set[:,1].max())
        )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

    #we evaluate the model on these points
    y_pred = np.reshape(model(grid), feature_1.shape)

    rng = np.random.default_rng()

    #and we then create a plot of the decision boundary and save it as a pdf
    display = DecisionBoundaryDisplay(
        xx0 = feature_1,
        xx1 = feature_2,
        response = y_pred)

    display.plot()
    display.ax_.scatter(train_set[:,0], train_set[:,1], c=train_set[:,2], edgecolor="black")
    

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Decision Boundary of Perceptron Classifier on Spiral Dataset")
    plt.savefig("Khan_F_DL_Assignment2_DecisionBoundary.pdf")

    

    
