import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from os.path import join
from keras import models, layers, backend
from keras.api.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mnist_data_loader import MnistDataloader

def make_model(name, cnn_layers_number):
  """ Create the CNN model with provided
  name and number of layers.
  """
  name = str(name)

  # instantiate model:
  cnn = models.Sequential(name=name)

  # our convolutional layer has 32 filters and a kernel size of 3x3
  # the input shape must be 28x28x1, as we have 28x28 pixel images one channel
  # max pooling is used to reduce dimensionality
  # this process is repeated for cnn_layers_number times

  for i in range(cnn_layers_number):
    if (i == 0):
      cnn.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1), name = f'Conv{i+1}'))
      cnn.add(layers.MaxPooling2D((2,2), name = f'MaxPool{i+1}'))
    else:
      cnn.add(layers.Conv2D(64,(3,3), activation = 'relu', name = f'Conv{i+1}'))
      cnn.add(layers.MaxPooling2D((2,2), name = f'MaxPool{i+1}'))

  # need to add the fully connected layers so that the feature vector we extract can be used to classify images into categories

  cnn.add(layers.Flatten(name = 'Flattening'))                          # convert to vector for FCN
  cnn.add(layers.Dense(100, activation = 'relu', name = 'Dense'))       # fully connected dense layer for classification
  cnn.add(layers.Dense(10, activation = 'softmax', name = 'Softmax'))   # using softmax to convert from logits to probability

  return cnn

if __name__ == '__main__':
  # Local training and testing data
  base_path = 'data'
  train_images_path = join(base_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
  train_labels_path = join(base_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
  test_images_path = join(base_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
  test_labels_path = join(base_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

  # Load MINST dataset
  mnist_dataloader = MnistDataloader(train_images_path, 
                                    train_labels_path, 
                                    test_images_path, 
                                    test_labels_path)

  (x_trainval, y_trainval), (x_test, y_test) = mnist_dataloader.load_data()

  x_trainval_dim = x_trainval.shape
  y_trainval_dim = y_trainval.shape

  x_test_dim = x_test.shape
  y_test_dim = y_test.shape

  NUM_CLASS = len(np.unique(y_trainval))
  print(f"There are {NUM_CLASS} unique classes")

  print(f"There are {x_trainval_dim[0]} samples in the training dataset, each of size {x_trainval_dim[1], x_trainval_dim[2]}\n"
        f"There are {x_test_dim[0]} samples in the testing dataset ")

  print(f"The training label set has dimension of {y_trainval_dim}")

  # Randomly show some images
  for i in range(9):
      index = np.random.randint(0, x_trainval_dim[0])
      plt.subplot(3, 3, i+1)
      plt.title(str(y_trainval[index]))
      plt.imshow(x_trainval[index], cmap='gray')
      plt.axis('off')

  plt.show()

  # Scale values to within normalized range
  x_trainval, x_test = x_trainval / 255.0, x_test / 255.0

  cnn = make_model('CNN1', 3)
  cnn.summary()

  # First compile the model
  backend.clear_session()
  cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)

  # plot training and validation history:
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.85, 1])
  plt.legend(loc='lower right')
  plt.show()