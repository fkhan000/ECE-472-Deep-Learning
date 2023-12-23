import tensorflow as tf
from MLP import MLP
from Adam import Adam
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

###############IMAGE INPAINTING###############

#we apply a 10% drop rate for our mask
DROP_RATE = 0.1
MAX_ITER = 5000
rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

image = Image.open("Test_Card_F.png")
SIZE = image.size

#we normalize the pixel values of the image by using min max scaling
labels = np.asarray(image)/255.0

#then our inputs into our model are the coordinates in the image (x,y)
x_train = np.indices((SIZE[1], SIZE[0])).transpose((1,2, 0)).reshape(SIZE[0]*SIZE[1], 2).astype("float32")

#normalize the inputs
x_train =  x_train /(x_train.max(axis = 0))

y_train = labels.astype("float32")

#reshape to get the labels that our  model will try to predict
y_train = y_train.reshape((SIZE[0]*SIZE[1], 3))

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)

#then in our training image, we randomly remove 10% of the pixels. Our model should be able to extrapolate
#on the training image and be able to recover the missing pixel values
drop_indices = indices[: int(len(indices)*DROP_RATE)]
indices = indices[int(len(indices)*DROP_RATE):]

#for visualization, we show what this mask would look like by setting those pixel values to 1
masked_img = y_train
masked_img[drop_indices] = np.array([1.0, 1.0, 1.0])

l_train = masked_img.reshape(SIZE[1], SIZE[0], 3)

plt.imshow(masked_img)
plt.imsave("masked_img.png", l_train)

x_train = x_train[indices]
y_train = y_train[indices]


x_test = np.indices((SIZE[1], SIZE[0])).transpose((1,2, 0)).reshape(SIZE[0]*SIZE[1], 2).astype("float32")
x_test  = x_test/(x_test.max(axis = 0))

y_test = labels.astype("float32")
y_test = y_test.reshape((SIZE[0]*SIZE[1], 3))

#we initialize our model with sine activation functions
model = MLP(2, 3, 5, 300, 
            hidden_activation = tf.math.sin, 
            output_activation = tf.math.sin)


optimizer = Adam(model.trainable_variables, .9, .999, 1e-7)

for iter in (pbar := tqdm(range(MAX_ITER))):

    with tf.GradientTape() as tape:
        #I didn't use batching here because I found that it made the gradient very noisy and the model
        #would have a hard time converging. So instead for each iteration, the model evalues the entire
        #training set
        y_hat = model(x_train)

        loss = tf.reduce_mean(tf.math.square(tf.math.subtract(y_hat, y_train)))


    grads = list(tape.gradient(loss, model.trainable_variables))

    optimizer(grads, model.trainable_variables, 1e-4)

    pbar.set_description("loss = " + str(loss))

#we evaluate on the test set
y_pred = model(x_test).numpy()
y_pred = y_pred.reshape(SIZE[1], SIZE[0], 3)

#clip the outputs
y_pred[y_pred < 0] = 0.0
y_pred[y_pred > 1] = 1.0

#and then export the extrapolated image
plt.imshow(y_pred)
plt.imsave("extrap_img.png", y_pred)

###############UPSAMPLING###############

#One cool application is that by predicting the pixel values of points in between the points in the training image
#we can increase the resolution of the training image. In this example, the image resolution is increased by a factor of 4
#but it can probably be increased by much more
x_upsmpl = np.indices((SIZE[1]*2, SIZE[0]*2)).transpose((1,2, 0)).reshape(SIZE[0]*SIZE[1]*4, 2).astype("float32")

#also due to a spelling error I ended up not normalizing the upsampled coordinates and got a really cool image that 
#I'm using as my laptop background
x_upsmpl  = x_upsmpl/(x_upsmpl.max(axis =0))

#we evaluate on the upsampled grid
y_upsmpl = model(x_upsmpl).numpy()


y_upsmpl = y_upsmpl.reshape(SIZE[1]*2, SIZE[0]*2, 3)

#and then export the upsampled image
y_upsmpl[y_upsmpl < 0] = 0.0
y_upsmpl[y_upsmpl > 1] = 1.0
plt.imshow(y_upsmpl)
plt.imsave("upsampled_img.png", y_upsmpl)