#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:13:43 2019

@author: omprakash
"""

import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
dataset = input_data.read_data_sets("mnist/data/", one_hot=True)

dataset.train.images.shape  #55000 image,784 pixel per iamge
dataset.test.images.shape

dataset.train.images[1].shape
sample_data= dataset.train.images[1].reshape(28,28) # try other no inplace 1
plt.imshow(sample_data,cmap='gist_gray')
#  one_hot encoder = True it makes that index position to 1
dataset.train.labels[1]
plt.imshow(dataset.train.images[1].reshape(784,1),
           cmap='gist_gray',aspect=0.02)

#here None means - no batch we defined but we sent 784 pixel at a time
x = tf.placeholder(tf.float32,shape=[None,784])
# 10 beacuse 0-9 = 10 value (check datset.train.label[1])
y = tf.placeholder(tf.float32,[None,10])

#x*w+ b adding weight and bias add variable
#matrix multiplication (1,784) * (784,10) = (1,10)
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y_calculate = tf.matmul(x,w) + b 

#gradient discent and optimiser 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                   labels= y,logits= y_calculate))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.5)
train = optimizer.minimize(cross_entropy)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)   
    #1000 times train data using 100 batch
    for step in range(1000):
        batch_x,batch_y = dataset.train.next_batch(100)
        #passes the place holder to the batch x,y place holder
        sess.run(train,feed_dict={x: batch_x,y:batch_y})
    #find where index position is 1 and compare the result
    matches = tf.equal(tf.argmax(y_calculate,1),tf.argmax(y,1))   
    #tensor obj to convert float  cast it
    accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))
    # find the accuracy on test set
    print(sess.run(accuracy,feed_dict={x:dataset.test.images,y:dataset.test.labels}))
    







"""
learning_rate = 0.001
training_epochs = 15
batch_size = 100
n_classes = 10 # MNIST total classes (0-9 digits)
n_samples = dataset.train.num_examples
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
    #x : Place Holder for Data Input,weights,biases : Dictionary,
    #x*w+ b adding weight and bias
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Second Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Last Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#tf.random_normal() passing a matrix of size 784*256 of random value in layer1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
#weight 2d because matrix multiplication biases 1d as we are adding
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)

#cost optimisation using Adamoptimiser
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#train the model (1 sample of training data)
t =dataset.train.next_batch(1)
len(t[1])
x_sample,y_sample = t    #tupple unpacking 
plt.imshow(x_sample.reshape(28,28),cmap='Greys')
print(y_sample)   # label 
init = tf.global_variables_initializer()

#with tf.Session() as sess:
sess = tf.InteractiveSession()
sess.run(init)

# run upto 15 times
for epoch in range(training_epochs):
    avg_cost = 0.0 # make float number
    total_batch = int(n_samples/batch_size)  #55000/100
    for i in range(total_batch):  #550
        # Grab the next batch of training data and labels
        batch_x,batch_y = dataset.train.next_batch(batch_size) #100
        # Feed dictionary for optimization and loss value
        # tupple unpacking when u don't need a value use _(underscore)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        #avg cost
        avg_cost += c/total_batch
    print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))
print ("model completed training epochs {} ".format(training_epochs))

#model evaluation
#find where index position is 1 and compare the result
correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#tensor obj to convert float 
correct_predictions = tf.cast(correct_predictions,"float")
print (correct_predictions[0])

accuracy = tf.reduce_mean(correct_predictions)

dataset.test.labels
dataset.test.images[1]

#use eval method to find the accuracy 
accuracy.eval({x:dataset.test.images,y:dataset.test.labels})
"""



