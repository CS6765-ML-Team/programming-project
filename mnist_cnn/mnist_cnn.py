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
  
  # Dropout and batch normalization layers.

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
  mnist_dataloader = MnistDataloader(
    train_images_path=train_images_path, 
    train_labels_path=train_labels_path, 
    test_images_path=test_images_path, 
    test_labels_path=test_labels_path
  )

  (x_trainval, y_trainval), (x_test, y_test) = mnist_dataloader.load_data()

  x_trainval_dim = x_trainval.shape
  y_trainval_dim = y_trainval.shape

  x_test_dim = x_test.shape
  y_test_dim = y_test.shape

  print(f"There are {len(np.unique(y_trainval))} unique classes")
  print(f"There are {x_trainval_dim[0]} samples in the training dataset, each of size {x_trainval_dim[1], x_trainval_dim[2]}\n"
        f"There are {x_test_dim[0]} samples in the testing dataset ")
  print(f"The training label set has dimension of {y_trainval_dim}")

  # Randomly show some images
  # for i in range(9):
  #     index = np.random.randint(0, x_trainval_dim[0])
  #     plt.subplot(3, 3, i+1)
  #     plt.title(str(y_trainval[index]))
  #     plt.imshow(x_trainval[index], cmap='gray')
  #     plt.axis('off')

  # plt.show()

  # Scale values to within normalized range
  x_trainval, x_test = x_trainval / 255.0, x_test / 255.0

  # Perform tests that use the same model with different training 
  # parameters and structures. The following will be tested:
  #   - The number of filters
  #   - Learning rate using SGD as an optimizer
  #   - Batch sizes
  filters = [16, 32, 64]
  filters_results = [None for _ in range(len(filters))]
  for idx, f in enumerate(filters):
    backend.clear_session()
    cnn = define_seq_model(
      name=f'CNN_LR',
      input_shape=(28, 28, 1),
      n_layers=3,
      n_filters=f,
      kernel_size=(3, 3),
      pool_size=(2, 2),
      use_max_pool=True
    )
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)
    filters_results[idx] = history

  # Learning rate using SGD as an optimizer tests.
  learning_rates = [0.1, 0.01, 0.001]
  learning_rates_results = [None for _ in range(len(learning_rates))]
  for idx, lr in enumerate(learning_rates):
    backend.clear_session()
    cnn = define_seq_model(
      name=f'CNN_LR',
      input_shape=(28, 28, 1),
      n_layers=3,
      n_filters=32,
      kernel_size=(3, 3),
      pool_size=(2, 2),
      use_max_pool=True
    )
    opt = SGD(learning_rate=lr, momentum=0.9)
    cnn.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)
    learning_rates_results[idx] = history

  # Batch size tests.
  batch_sizes = [128, 256, 512]
  batch_sizes_results = [None for _ in range(len(batch_sizes))]
  for idx, bs in enumerate(batch_sizes):
    backend.clear_session()
    cnn = define_seq_model(
      name=f'CNN_BS',
      input_shape=(28, 28, 1),
      n_layers=3,
      n_filters=32,
      kernel_size=(3, 3),
      pool_size=(2, 2),
      use_max_pool=True
    )
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=bs, validation_data=(x_test, y_test), verbose=0)
    batch_sizes_results[idx] = history

  # Plot the results from the above tests 
  plt.figure(figsize=(15, 15))
  
  # plot filters 
  for i in range(3):
    plt.subplot(3, 3, i+1)
    plt.plot(filters_results[i].history['accuracy'], label='Training Accuracy')
    plt.plot(filters_results[i].history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Number of Filters: {filters[i]}')
    plt.legend(loc='lower right')
    plt.grid(True)
    
  # plot learning rate:
  for i in range(3):
    plt.subplot(3, 3, i+4)
    plt.plot(learning_rates_results[i].history['accuracy'], label='Training Accuracy')
    plt.plot(learning_rates_results[i].history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Rate: {learning_rates[i]}')
    plt.legend(loc='lower right')
    plt.grid(True)

  # plot batch size:
  for i in range(3):
    plt.subplot(3, 3, i+7)
    plt.plot(batch_sizes_results[i].history['accuracy'], label='Training Accuracy')
    plt.plot(batch_sizes_results[i].history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Batch Size: {batch_sizes[i]}')
    plt.legend(loc='lower right')
    plt.grid(True)

  plt.tight_layout()
  plt.show()