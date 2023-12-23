import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import tensorflow as tf
import Gen_Tune
from CNN import CNN


train_x = train_y = val_x = val_y = test_x = test_y = None


def load_data_10(fname, training = True):

    with open(fname, "rb") as f:

        di = pickle.load(f, encoding='bytes')

    data = np.reshape(di[b"data"], (len(data), 3, 32, 32)).astype("float32").transpose(0, 2, 3, 1)/255.0
    
    if training:

        data = tf.image.random_crop(data, size = (len(data), 24, 24, 3))
        
        data = tf.image.resize_with_crop_or_pad(data, 32, 32)
        altered = tf.image.random_flip_left_right(tf.image.random_brightness(data, max_delta=0.5, seed = 0xe134))

        data = tf.concat([data, altered], axis = 0)
    
        labels = tf.concat([di[b"fine_labels"], di[b"fine_labels"]], axis = 0)

    return np.asarray(data), np.asarray(labels)


def load_data_100(fname, training = True):

    with open(fname, "rb") as f:

        di = pickle.load(f, encoding='bytes')
    data = di[b"data"]

    data = data.reshape(len(data), 3, 32, 32).astype("float32").transpose(0, 2, 3, 1)/255.0

    labels = di[b"fine_labels"]
    if training:
        
        altered = tf.image.random_flip_left_right(tf.image.random_brightness(data, max_delta=0.5, seed = 0xe134))

        data = tf.concat([data, altered], axis = 0)
    
        labels = tf.concat([di[b"fine_labels"], di[b"fine_labels"]], axis = 0)

    return np.asarray(data), np.asarray(labels)


def train(num_classes, parameters,  export = False):

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    #we create our classifier
    model = CNN( (32, 32),
                 3, num_classes, 2,
                 conv_activation = tf.nn.relu)

    for i in range(8):
        
        model.Convolutional_Layer((3,3), 1, 64)
        model.Group_Norm_Layer(16)

    model.Dense_Layers(num_hidden_layers = 2,
                       hidden_layer_width = 100,
                       hidden_activation = tf.nn.relu,
                       output_activation = tf.identity)

    
    drop_rate =parameters[0]
    lamb = parameters[1]
    step_size = parameters[2]

    
    print("num_params", tf.math.add_n([tf.math.reduce_prod(var.shape) for var in model.trainable_variables]))
    decay_rate = .9999

    batch_size = 128

    #we will store the losses on the training, validation and test sets in these lists
    iterations = []
    training_losses = []
    validation_losses = []
    test_losses = []

    max_iter = 50

    if export:
        max_iter = 2000

    optimizer = tf.keras.optimizers.AdamW(learning_rate = step_size)

    for iteration in range(1, max_iter):               

        #we perform dropout regularization (see Module Test for a more detailed explanation)
               
        drop_outs = []
        recov = []
        for variable in model.trainable_variables:

            if len(variable.shape) == 1:
                drop_outs.append(tf.constant([1.0]))
                recov.append(tf.constant([0.0]))
                continue

            #we create the drop out matrix
            drop_out = tf.nn.dropout(tf.ones(variable.shape), rate = drop_rate)*(1-drop_rate)


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
            y_hat = model(x_batch)/(1-drop_rate)

            #and calculate the softmax cross entropy loss
            loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = y_batch,
                logits = y_hat))

            loss += lamb*tf.math.reduce_sum([ tf.math.reduce_sum(tf.math.square(x)) for x in model.trainable_variables])

        #we then calculate the gradients
        grads = list(tape.gradient(loss, model.trainable_variables))

        #and to make sure we don't update the zeroed out weights, we zero out the corresponding
        #entries in ourå gradients
        for index in range(len(grads)):
            grads[index] *= drop_outs[index]

        #then we update our weights

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        
        step_size *= decay_rate
        
        for var, reco in zip(model.trainable_variables, recov):
            #finally we get back the zeroed out weights by adding them back in from reco
            var.assign_add(reco)

    #metrics = [iterations, training_losses, validation_losses, test_losses]

    test_acc = evaluate(num_classes, model, test_x, test_y, top_5 = (num_classes == 100))
    val_acc = evaluate(num_classes, model, val_x, val_y, top_5 = (num_classes == 100))

    if export:
        if num_classes == 100:
            return val_acc, test_acc
        else:
            return val_acc.numpy(), test_acc.numpy()

    return model



#computes average cross entropy loss on validation set
def eval_loss(model):
    return tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = val_y,
                logits = model(val_x)))

#evaluates model on either validation or test set
def evaluate(num_classes, model, X, Y, top_5 = False):

    #else we want to calculate the accuracy of the model on the test/validation set
    pred = model(X)

    if top_5:
        return top_k_accuracy_score(test_y, pred, k = 5)
        

    #for each prediction, we get the column # of the largest value in that row
    max_val = tf.argmax(model(X), axis = 1)

    #then we convert predictions and labels into a one hot vector
    predictions = tf.one_hot(max_val, num_classes, dtype = tf.int32).numpy()

    labels = tf.one_hot(Y, num_classes, dtype = tf.int32).numpy()

    
    #and then compare each of the rows between the predictions and labels
    #to determine the accuracy
    result = tf.math.equal(predictions, labels)
    result = tf.reduce_all(result, axis = 1)

    return tf.reduce_mean(tf.cast(result, 'float32'))



def validation(num_classes):

    global train_x, train_y
    parameters = np.vstack([[0, 0, len(DROP_OUT_RATE) -1, 1],
                                [1, 0, 0.2, 1],
                                [1, 0, 0.1, 1]])

    temp_x  = train_x
    temp_y = train_y

    indices = tf.range(tf.shape(train_x)[0])
    samp_indices = tf.random.shuffle(indices)[:50]
    train_x = tf.gather(train_x, samp_indices)

    train_y = tf.gather(train_y, samp_indices)
    model = Gen_Tune.tuning(5, 3, lambda x: eval_loss(train(num_classes, x)), parameters)

    train_x = temp_x
    train_y = temp_y
    return model


def get_datasets(num_classes):

    global train_x, train_y, val_x, val_y, test_x, test_y

    if num_classes == 100:

        train_x, train_y = load_data_100("cifar-100-python/train")
        test_x, test_y  = load_data_100("cifar-100-python/test", training = False)

    else:
    
        batches = [load_data_10("cifar-" + str(num_classes) + "-batches-py/data_batch_" + str(index)) for index in range(1, 6)]

        train_x  = np.concatenate([batch[0] for batch in batches], axis = 0)

        train_y = np.concatenate([batch[1] for batch in batches])

        del batches

        test_x, test_y = load_data_10("cifar-" + str(num_classes) + "-batches-py/test_batch", training = False )



    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                      test_size = 0.1,
                                                      random_state = 0x3e9c
                                                 )
    
def driver(cifar_10 = True):
    
    num_classes = 10
    
    if not cifar_10:
        num_classes = 100

    get_datasets(num_classes)

    parameters = [0, .00001, .001]

    return train(num_classes, parameters, export = True)
