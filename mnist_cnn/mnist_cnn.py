import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from keras import models, layers, backend
from keras.api.optimizers import SGD

from mnist_data_loader import MnistDataloader

def plot_performance_plot(history, title=None, ylim=None):
  plt.plot(history['accuracy'], label='Training Accuracy')
  plt.plot(history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  if (ylim is not None):
    plt.ylim([0.8, 1])
  if (title is not None):
    plt.title(title)
  plt.legend(loc='lower right')
  plt.grid(True)
  
def plot_performance_subplot(subplot_size, subplot_idx, history, title=None, ylim=None):
  plt.subplot(subplot_size[0], subplot_size[1], subplot_idx)
  plot_performance_plot(history, title, ylim)

def define_seq_model(name, input_shape, n_layers, n_filters, kernel_size, pool_size, use_dropout=False, dropout_rate=0.2, use_batch_norm=False, use_max_pool=True):
  """ Create the CNN model with provided
  name and number of layers.
  """
  # Instantiate a sequential model:
  cnn = models.Sequential(name=str(name))

  n_filters_2 = 2*n_filters
  
  # Dropout and batch normalization layers.

  # Define the layers of the CNN returned from 
  # this function. These layers will include
  # n_layers worth of Convolution and Max/Avg 
  # pooling layers. Additional layers, such
  # as Dropout layers and 
  # Batch Normalization layers can be added with
  # the parameter flags. The parameters of the 
  # convolution layers, the input shape, 
  # number of filters, kernel size, pool size, 
  # and whether to use max or average pooling,
  # can all be configured through the parameters of this function.
  cnn.add(layers.Input(shape=input_shape))
  cnn.add(layers.Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', name='Conv0'))
  if (use_dropout):
    cnn.add(layers.Dropout(dropout_rate))
  if (use_batch_norm):
    cnn.add(layers.BatchNormalization())
  pool_layer = layers.MaxPooling2D(pool_size, name='MaxPool0') if use_max_pool else layers.AveragePooling2D(pool_size, name='AvgPool0')
  cnn.add(pool_layer)

  # Repeat the above process n_layers-1 times, except for the Input layer
  # and with double the number of filters. 
  for i in range(n_layers-1):
    cnn.add(layers.Conv2D(filters=n_filters_2, kernel_size=kernel_size, activation='relu', name=f'Conv{i+1}'))
    if (use_dropout):
      cnn.add(layers.Dropout(dropout_rate))
    if (use_batch_norm):
      cnn.add(layers.BatchNormalization())
    pool_layer = layers.MaxPooling2D(pool_size, name=f'MaxPool{i+1}') if use_max_pool else layers.AveragePooling2D(pool_size, name=f'AvgPool{i+1}')
    cnn.add(pool_layer)

  # Need to add the fully connected layers so that 
  # the feature vector we extract can be used 
  # to classify images into categories

  # Convert to vector for FCN
  cnn.add(layers.Flatten(name='Flattening'))                      

  # Fully connected dense layer for classification
  cnn.add(layers.Dense(100, activation='relu', name='Dense'))     

  # Batch norm if requested 
  if (use_batch_norm):
    cnn.add(layers.BatchNormalization())

  # using softmax to convert from logits to probability
  cnn.add(layers.Dense(10, activation='softmax', name='Softmax')) 

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

  # Scale values to within normalized range
  x_trainval, x_test = x_trainval / 255.0, x_test / 255.0

  # Perform a control test where we use the following default
  # CNN structure parameters:
  #   - Number of layers: 3
  #   - Number of filters: 32
  #   - Size of filters: (3, 3)
  #   - No Drop-Out layers
  #   - No Batch Normalization layers
  #   - Max Pooling
  #   - Pooling size: (2, 2)
  backend.clear_session()
  control_cnn = define_seq_model(
    name=f'CNN_CTRL',
    input_shape=(28, 28, 1),
    n_layers=3,
    n_filters=32,
    kernel_size=(3, 3),
    pool_size=(2, 2),
  )
  control_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  control_history = control_cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)

  # Plot the control CNN
  plot_performance_plot(control_history.history, None, [0.8, 1])
  plt.tight_layout()
  plt.show()

  # Perform tests that use different model structures:
  #   - Different number of filters
  #   - Different number of layers
  #   - Addition of drop-out layers with different drop-out rates
  #   - Addition of batch normalization layers
  #   - Using Average Pooling instead of Max Pooling
  
  # Different number of filters tests
  filters = [16, 32, 64]
  filters_results = [None for _ in range(len(filters))]
  for idx, f in enumerate(filters):
    backend.clear_session()
    cnn = define_seq_model(
      name=f'CNN_FI{idx}',
      input_shape=(28, 28, 1),
      n_layers=3,
      n_filters=f,
      kernel_size=(3, 3),
      pool_size=(2, 2),
    )
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)
    filters_results[idx] = history

  # Different number of layers
  n_layers = [1, 2, 3]
  layers_results = [None for _ in range(len(n_layers))]
  for idx, l in enumerate(n_layers):
    backend.clear_session()
    cnn = define_seq_model(
      name=f'CNN_LY{idx}',
      input_shape=(28, 28, 1),
      n_layers=l,
      n_filters=32,
      kernel_size=(3, 3),
      pool_size=(2, 2),
    )
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)
    layers_results[idx] = history
  
  # Plot result of changing the number of filters 
  for i in range(3):
    plot_performance_subplot((2, 3), i+1, filters_results[i].history, f'Number of Filters: {filters[i]}')
  
  # Plot result of changing the number of layers 
  for i in range(3):
    plot_performance_subplot((2, 3), i+4, layers_results[i].history, f'Number of Layers: {n_layers[i]}')

  # Show the above plots
  plt.tight_layout()
  plt.show()

  #######

  # Addition of Dropout layers with varying dropout rates
  dropout_rates = [0.1, 0.2, 0.5]
  dropout_rates_results = [None for _ in range(len(dropout_rates))]
  for idx, rate in enumerate(dropout_rates):
    backend.clear_session()
    cnn = define_seq_model(
      name=f'CNN_DO{idx}',
      input_shape=(28, 28, 1),
      n_layers=3,
      n_filters=32,
      kernel_size=(3, 3),
      pool_size=(2, 2),
      use_dropout=True,
      dropout_rate=rate
    )
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)
    dropout_rates_results[idx] = history

  # Plot result of adding Dropout layers with different dropout rates, compared
  # to the control CNN that was tested with no dropout layers
  plot_performance_subplot((2, 2), 1, control_history.history, "No Drop Out")
  for i in range(3):
    plot_performance_subplot((2, 2), i+2, dropout_rates_results[i].history, f'Rate of Dropout: {dropout_rates[i]}')

  # Show the above plot
  plt.tight_layout()
  plt.show()

  #######

  # Addition of Batch Normalization Layers 
  backend.clear_session()
  batch_norm_cnn = define_seq_model(
    name=f'CNN_NORM',
    input_shape=(28, 28, 1),
    n_layers=3,
    n_filters=32,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    use_batch_norm=True
  )
  batch_norm_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  batch_norm_history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)
    
  # Plot result of adding Batch Normalization layers, compared
  # to the control CNN that was tested with no Batch Normalization.
  plot_performance_subplot((2, 1), 1, control_history.history, "No Batch Normalization")
  plot_performance_subplot((2, 1), 2, batch_norm_history.history, "With Batch Normalization")

  # Show the above plot
  plt.tight_layout()
  plt.show()

  #######

  # Compare using Max Pooling to using Avg Pooling. Only need to calculate
  # Average Pooling model here since control model uses Max Pooling.
  backend.clear_session()
  avg_pool_cnn = define_seq_model(
    name=f'CNN_AVG',
    input_shape=(28, 28, 1),
    n_layers=3,
    n_filters=32,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    use_max_pool=False
  )
  avg_pool_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  avg_pool_history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=0)

  # Plot the results of using Max vs. Average pooling
  plot_performance_subplot((2, 1), 1, control_history.history, "Max Pooling")
  plot_performance_subplot((2, 1), 2, avg_pool_history.history, "Average Pooling")

  # Show the above plot
  plt.tight_layout()
  plt.show()

  # Perform tests that use the same model with different training 
  # parameters and structures. The following will be tested:
  #   - Learning rate using SGD as an optimizer
  #   - Batch sizes
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
    )
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(x_trainval, y_trainval, epochs=10, batch_size=bs, validation_data=(x_test, y_test), verbose=0)
    batch_sizes_results[idx] = history

  # plot learning rate:
  for i in range(3):
    plot_performance_subplot((2, 3), i+1, learning_rates_results[i].history, f'Learning Rate: {learning_rates[i]}')

  # plot batch size:
  for i in range(3):
    plot_performance_subplot((2, 3), i+4, batch_sizes_results[i].history, f'Batch Size: {batch_sizes[i]}')

  plt.tight_layout()
  plt.show()