import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import my_lib as my

sess = tf.InteractiveSession()

ROOT_PATH = "GTSRB"
train_data_dir = os.path.join(ROOT_PATH, "Final_Training/Images/")
test_data_dir = os.path.join(ROOT_PATH, "Final_Test/")
images, labels = my.load_train_data(train_data_dir)
images_test = my.load_test_data(test_data_dir)
print("Unique Labels: {0}\nTotal Images:{1}".format(len(set(labels)), len(images)))
print("Total Test Images:{0}".format(len(images_test)))
my.display_images_and_labels(images, labels)
my.display_label_images(images, labels, 32)

for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

images28 = [skimage.transform.resize(image, (28, 28))
           for image in images]
my.display_images_and_labels(images28, labels)

images28_test = [skimage.transform.resize(image_test, (28, 28))
           for image_test in images_test]

for image in images28[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

labels_a = np.array(labels)
images_a = np.array(images28)

images_test_a = np.array(images28_test)
print("labels: ", labels_a.shape)
print('images: ', images_a.shape)
print('test images: ', images_test_a.shape)

buff = np.arange(0, 39209)
np.random.shuffle(buff)
labels_shuffled = labels_a
images_shuffled = images_a

for i in range(39209):
    j = buff[i]
    labels_shuffled[i] = labels_a[j]
    images_shuffled[i] = images_a[j]

labels_onehot = np.zeros((39209, 43))
labels_onehot[np.arange(39209), labels_a] = 1
print("labels one hot: ", labels_onehot[10000:10005])

labels_shuffled_onehot = np.zeros((39209, 43))
labels_shuffled_onehot[np.arange(39209), labels_shuffled] = 1

batch_images = np.zeros((35, 800, 28, 28, 3))
batch_labels = np.zeros((35, 800, 43))
for i in range(35):
    batch_images[i] = images_shuffled[800*i:800*i+800]
    batch_labels[i] = labels_shuffled_onehot[800*i:800*i+800]

print("batch_images: ", batch_images[8].shape)
print("label_images: ", batch_labels[8].shape)

loss_buffer = np.zeros(1000)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 28, 28, 3])
y_ = tf.placeholder(tf.int32, [None, 43])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 43])
b_fc2 = bias_variable([43])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predicted_labels = tf.argmax(y_conv, 1)
sess.run(tf.global_variables_initializer())

for i in range(10000):
  _, loss_value = sess.run([train_step, cross_entropy],feed_dict={x: batch_images[i % 35], y_: batch_labels[i % 35], keep_prob: 0.5})
  if i % 10 == 0:
    print i
    print("Loss: ", loss_value)
    loss_buffer[i/10] = loss_value

print("cross validation accuracy %g"%accuracy.eval(feed_dict={
    x: images_shuffled[28000:39209], y_: labels_shuffled_onehot[28000:39209], keep_prob: 1.0}))

result = predicted_labels.eval(feed_dict={x: images_test_a[0:12630], keep_prob: 1.0})

np.savetxt('result.csv', result, delimiter=',')
np.savetxt('loss.csv', loss_buffer, delimiter=',')
