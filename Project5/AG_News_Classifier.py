import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_text as text
from matplotlib import pyplot as plt


#we load in the AG News dataset using tfds and split it so that 10% of the training set
#is allocated for validation and we use the entire test set to evaluate our model at the end
train_data, val_data, test_data = tfds.load('ag_news_subset:1.0.0',
                                         split = ["train[:90%]", "train[90%:]", "test"],
                                         as_supervised=True)



batch_size = 64

#we shuffle the training set
train_data = train_data.shuffle(train_data.cardinality(), seed = 0x34b21a3)

#reates batches from our training data. The prefetch part makes it so that while
#the GPU is performing backwards propagation, the CPU is fetching the next batch of data
#This ensures that the GPU is always being used when training
train_data = train_data.batch(batch_size).prefetch(1)

val_data = val_data.batch(batch_size).prefetch(1)

test_data = test_data.batch(batch_size).prefetch(1)

#for my pretrained model I decided to use the BERT model 
bert_handle = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2'
preprocessing_model = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

#Before we can input to BERT, we need to preprocess the text so we need to create a preprocessing layer
#preprocess_layer = hub.KerasLayer(preprocessing_model)

bert_model = hub.KerasLayer(bert_handle)


input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Input')

#Before we can input to BERT, we need to preprocess the text so we need to create a preprocessing layer
preprocessing_layer = hub.KerasLayer(preprocessing_model, name='preprocessing_layer')

bert_input = preprocessing_layer(input_text)

#then our bert layer creates the encoding of our text for us
bert_encoder = hub.KerasLayer(bert_handle, trainable=True, name='bert_encoder')

bert_outputs = bert_encoder(bert_input)

# Since we are classifying text, we want to use the BERT's pooled output. This basically pools
#all of the embeddings of the tokens in our input text into a single embedding.
pooled_bert_output = bert_outputs['pooled_output']

#After this, I added two dense layers with the first being 32 nodes wide and the second being 4 nodes wide
dense_net = tf.keras.layers.Dense(32, activation='relu', name='fully_connected_layer')(pooled_bert_output)

#dense_net = tf.keras.layers.Dropout(0.3)(dense_net)

#The final dense layer is 4 nodes wide since we have four classes. 
final_output = tf.keras.layers.Dense(4, activation='softmax', name='classifier')(dense_net)


news_classifier = tf.keras.Model(input_text, final_output)

train_steps = len(train_data)//batch_size

val_steps = train_steps//8


##This function trains and evaluates our model on the validation set
#We will use this function to determine what would be a good value for our
#step size

def eval_val(alpha):

    #In order to reset the model, I had to rebuild it
    dense_net = tf.keras.layers.Dense(32, activation='relu', name='fully_connected_layer')(pooled_bert_output)

    final_output = tf.keras.layers.Dense(4, activation='softmax', name='classifier')(dense_net)

    news_classifier = tf.keras.Model(input_text, final_output)

    #we compile our classifier
    news_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

    #and fit it on the training set
    history = news_classifier.fit(train_data, 
                        epochs=2,
                        #verbose = 0,
                        validation_data=val_data,
                        steps_per_epoch= train_steps,
                        validation_steps=val_steps)

    #and finally we get the cross entropy loss on the validation set
    return history.history["val_loss"][-1]



#Here we try out multiple step size values
#and see which ones give us the best performance
#on the validation set

best_param = 1e-5
min_loss = float("inf")

for alpha in np.arange(1e-5, 1e-4, 1e-5):

    #This clears our model which makes sure that we aren't
    #picking up where the last iteration left and training the
    #model for more than 2 epochs
    tf.keras.backend.clear_session()
    loss = eval_val(alpha)

    
    if loss < min_loss:
        min_loss = loss
        best_param = alpha


tf.keras.backend.clear_session()


news_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_param), 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])

#Finally with the optimal alpha value found, we train our model
#for 3 epochs and evaluate on the validation and test set at the end of each
#epoch

#now that we have finished tuning our model, we merge our training
#set with our validation set
train_data =  train_data.concatenate(val_data)

#and train it for 8 epochs
history = news_classifier.fit(train_data,
                            epochs=8,
                            steps_per_epoch=train_steps,
                            validation_data=test_data)


#we then plot the accuracy of the model on the training and test
#sets as a function of the number of epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel("epochs")
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("Khan_F_DL_Assignment5.pdf")
plt.show()


