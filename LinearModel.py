#Implemented with reference to tensorflow tutorials
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


#Importing and reading the dataset

data = input_data.read_data_sets("data/MNIST/", one_hot=True)

data.test.cls = np.array([label.argmax() for label in data.test.labels])

#Preprocessing
img_height = 28
img_width = 28

img_size_flat = img_height * img_width #Flattening image to one dimensional array

img_shape = (img_height, img_width)

num_classes = 10

#Defining Placeholders for Input
x = tf.placeholder(tf.float32, [None, img_size_flat]) #Contains arbitrary number of images and flattened image size

y_true = tf.placeholder(tf.float32, [None, num_classes]) #contains arbitrary amount of true labels in the shape of num_classes

y_true_cls = tf.placeholder(tf.int64, [None]) #contain true classes of training expression


#Defining Model Parameters
W = tf.Variable(tf.zeros([img_size_flat, num_classes])) #Initializing weights using Variable method, shape = 784, 10

b = tf.Variable(tf.zeros([num_classes])) #biases shape = 1-0

logits = tf.matmul(x, W) + b #shape = [num_images, num_labels] (x, 784) and (784,10) + (10) formula = sum(W*x) + b

#Training Parameters
y_pred = tf.nn.softmax(logits) #Applying softmax to logits expression so as to get the probability distribution, in order to predict

y_pred_cls = tf.argmax(y_pred, dimension=1) #Getting the highest valued prediction

#Optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true) #Using Cross-Entropy optimizer in order to predict the class

#Cost to be minimized
cost = tf.reduce_mean(cross_entropy) #Defines the training "Cost" which the optimizer seeks to reduce

#Gradient Descent Definition
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.4).minimize(cost) #Gradient Descent optimizer with learning rate

#Parameters for Evaluation
correct_prediction = tf.equal(y_pred_cls, y_true_cls) #Defining criteria for correct prediction, returns truth value

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #computes the mean value, and casts it to float datatype

sess = tf.Session() #Initialize Tensorflow session, builds graph

sess.run(tf.global_variables_initializer()) #Initializes Variables

##Implementation of Stochastic Gradient Descent
batch_size = 100 #Batch of data taken from dataset at a time
#Testing Dictionary Input
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
def print_accuracy():

    acc = sess.run(accuracy, feed_dict=feed_dict_test) #Defining Test dicitionary for testing.
    print("Accuracy on test-set : {0:.1%}".format(acc))

def optimize(num_iterations):
    for i in range(num_iterations):

        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true : y_true_batch} #Input dictionary, data to be used for training
        sess.run(optimizer, feed_dict=feed_dict_train)
        print("Iteration:", i)
        print_accuracy()





optimize(1700)
#print_accuracy()
