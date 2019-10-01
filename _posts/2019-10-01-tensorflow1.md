---
layout: post
title: " Udacity: tensorflow (1): 가장 기초적인 dense layer 학습 "

---

### 1. The basics: Training first model

섭씨온도를 화씨온도로 변환시키는 모델을 만들어보자.

~~~

# import modules
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# set up training data

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
  

# create the model
## build a layer I0 : Dense network
## input_shape=[1] 는 layer의 인풋으로 dimension이 1인, 즉 스칼라 값이 들어간다는 의미
## units=1 는 layer의 hidden unit이 1개 -> w11,b1
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# assemble layers into the model
## input으로는 list of layers를 받는다.
model= tf.keras.Sequential([I0])

# complie the model, with loss and optimizer functions

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
              
# train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

# display training statistics
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

# use to predict values
print(model.predict([100]))

# looking at the layer weights
print("These are the layer variables: {}".format(l0.get_weights()))

# little experiment

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))


~~~

### 2. Fashion MNIST Dataset
* Fasion MNIST 데이터셋은 28 * 28 픽셀사이즈의 gray-scale images of clothing 입니다. 따라서 한 사진 당 784 바이트를 필요로 합니다. NN의 input은 784 pixels를 담아야 합니다.
* 티셔츠가 top, 샌달이 middle, 앵클부츠가 bottom에 있습니다. 
* 총 레이블은 0~9, 클래스는 각 래이블마다 clothing의 종류가 매칭되어있습니다. (Ex. 레이블 0= 티셔츠 , 1= trouser, 2= pullover ... )
* 총 70000장의 사진 중 6만장을 training에, 만장을 test에 쓸 것입니다.
*  NN의 input은 벡터형이 되어야 하므로, 2차원 데이터인 사진은 stretched(flatten) 되어 1-dim인 784by 1 vector로 변경됩니다. 즉, 784 units이 input으로 들어갑니다. 
*  layer는 dense로써 128 unit으로 구성하고 activation은 relu를 사용합니다.
*  output은 레이블 수인 10개입니다. 각 10개의 숫자는 확률로 표현이 됩니다. 10개의 확률 중 가장 큰 확률을 가진 클래스가 모델이 예측한 사진의 클래스가 됩니다. 


~~~

# pip tensorflow_datasets
!pip install -U tensorflow_datasets

from __future__ import absolute_import, division, print_function, unicode_literals


# Import TensorFlow and TensorFlow Datasets
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

# This will go away in the future.
# If this gives an error, you might be running TensorFlow 2 or above
# If so, then just comment out this line and run this cell again
tf.enable_eager_execution()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# import datasets
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# class label defined
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
               
               
               
               
# explore the data
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# preprocess the data
def normalize(images, labels):
  images = tf.cast(images, tf.float32) # cast every pixel value to the float type
  images /= 255 # pixel value to 0~1 instead of 0~255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)


# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# display first 25 imgs from the training set 
plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()


# building the models
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])

# complie the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train the model
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

# make predictions and explore
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)
#
predictions.shape # 32개 사진에 대하여 10개의 숫자 반환 (행기준 1사진)

predictions[0] # prediction for the first img
np.argmax(predictions[0]) # 6
test_labes[0] # same as above. correct! 

# graph this to look at the full set of 10 class predictions
def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# look at 0th img,predictions, prediction array

i = 0 # 1st img
# probability distribution across the classes in this bar chart. 
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)


# for 13th img
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)

# several imgs with their predictions
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3 # print 15 imgs, blue= correct classification by model
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  
# use the trained model to make a prediction about a single img
# Grab an 1st image from the test dataset
img = test_images[0]
print(img.shape)
# Add the image to a batch where it's the only member.
img = np.array([img])

print(img.shape) # 처음 1은 create a one batch / one image라는 뜻. 

# predict img
predictions_single = model.predict(img)
print(predictions_single)

lot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0]) # 6 index


~~~




