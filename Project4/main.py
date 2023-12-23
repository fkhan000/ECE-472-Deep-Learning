from CIFAR_Classifier import driver
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":

    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    
    for cifar_10 in [False, True]:

        val, test = driver(cifar_10 = cifar_10)

        if cifar_10:
            print("Accuracy on Validation Set: " + str(val))
            print("Accuracy on Test Set: " + str(test))

        else:
            print("Top-5 Accuracy on Validation Set: " + str(val))
            print("Top-5 Accuracy on Test Set: " + str(test))
