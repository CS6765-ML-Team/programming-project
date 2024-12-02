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

def define_seq_model(name, input_shape, n_layers, n_filters, kernel_size, pool_size, use_max_pool=True):
  """ Create the CNN model with provided
  name and number of layers.
  """
  # Instantiate a sequential model:
  cnn = models.Sequential(name=str(name))

  n_filters_2 = 2*n_filters

  # Define 2*n_layers + 1 layers by defining
  # a convolutional layer and pooling layer 
  # n_layers times, in addition to a single
  # Input layer. The parameters of these layers,
  # the input shape, number of filters, kernel size,
  # pool size, and whether to use max or average pooling,
  # can all be configured through the parameters of this function.
  cnn.add(layers.Input(shape=input_shape))
  cnn.add(layers.Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', name='Conv0'))
  pool_layer = layers.MaxPooling2D(pool_size, name='MaxPool0') if use_max_pool else layers.AveragePooling2D(pool_size, name='AvgPool0')
  cnn.add(pool_layer)
  for i in range(n_layers-1):
    cnn.add(layers.Conv2D(filters=n_filters_2, kernel_size=kernel_size, activation='relu', name=f'Conv{i+1}'))
    pool_layer = layers.MaxPooling2D(pool_size, name=f'MaxPool{i+1}') if use_max_pool else layers.AveragePooling2D(pool_size, name=f'AvgPool{i+1}')
    cnn.add(pool_layer)

  # Need to add the fully connected layers so that 
  # the feature vector we extract can be used 
  # to classify images into categories

  cnn.add(layers.Flatten(name='Flattening'))                      # convert to vector for FCN
  cnn.add(layers.Dense(100, activation='relu', name='Dense'))     # fully connected dense layer for classification
  cnn.add(layers.Dense(10, activation='softmax', name='Softmax')) # using softmax to convert from logits to probability

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

  cnn = define_seq_model(
    name="CNN1",
    input_shape=(28, 28, 1),
    n_layers=3,
    n_filters=32,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    use_max_pool=True
  )
  cnn.summary()

  # First compile the model
  backend.clear_session(free_memory=True)
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