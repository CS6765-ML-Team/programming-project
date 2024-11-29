import numpy as np
import struct
from array import array

class MnistDataloader(object):
  """ Load MNIST data set images and labels
  from local files.
  """
  def __init__(self, train_images_path, train_labels_path, 
               test_images_path, test_labels_path):
    self.train_images_path = train_images_path
    self.train_labels_path = train_labels_path
    self.test_images_path = test_images_path
    self.test_labels_path = test_labels_path
  
  # Function to read the labels and load
  def read_images_labels(self, images_path, labels_path):        
    labels = []
    with open(labels_path, 'rb') as file:
      magic, size = struct.unpack(">II", file.read(8))
      if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
      labels = array("B", file.read())        
    
    with open(images_path, 'rb') as file:
      magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
      if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
      image_data = array("B", file.read())        

    images = [[0] * rows * cols for _ in range(size)]
    for i in range(size):
      img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
      img = img.reshape(28, 28)
      images[i][:] = img            
    
    return np.array(images), np.array(labels)
          
  def load_data(self):
    x_train, y_train = self.read_images_labels(self.train_images_path, self.train_labels_path)
    x_test, y_test = self.read_images_labels(self.test_images_path, self.test_labels_path)
    return (x_train, y_train), (x_test, y_test)      