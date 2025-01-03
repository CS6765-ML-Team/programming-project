import numpy as np                      # Access to math operators
from id3 import ID3Classifier           # ID3 Algorithm implementation

class AdaBoost:
  """ An implementation of the Adaptive Boost
  (Adaboost) boosting classifier that uses
  a custom implementation of the ID3 Decision Tree
  algorithm as a weak base learner. Constructor 
  parameters include T, the number of estimators
  to use when training, and the optional depth parameter,
  which controls the maximum depth of each decision
  tree (default = +infinity).
  """
  def __init__(self, T, depth=np.inf):
    self.T = T
    self.depth = depth
    self.models = [None for _ in range(T)]
    self.alphas = [0.0 for _ in range(T)]

  def train(self, examples, targets):
    """ Use the AdaBoost ensemble learning method
    with ID3 as a base learner to learn a strong 
    hypothesis for the provided data set X_train.
    """
    # How many training examples we are training on
    n_examples = examples.shape[0]

    # Extract the name of the target value we 
    # are trying to classify in y_train
    target_name = targets.columns.values[0]

    # Before we add the weight column, extract the 
    # names of each feature in X_train
    features = examples.columns.values

    # Assign an initial weight to all training examples
    examples["weight"] = [1/n_examples] * n_examples

    # We now have to train T different decision trees 
    # using the provided data. After each tree is trained,
    # we must determine the total error of that decision 
    # tree (sum of weights of incorrectly classified examples)
    # and use the total error to assign a weight to
    # the decision tree.
    self.models = [ID3Classifier(self.depth) for _ in range(self.T)]
    for t in range(self.T):
      # Obtain a weak hypothesis using the ID3 base learner implementation
      self.models[t].train(examples, features, targets)

      # Create an array of prediction results where 1 represents a correct
      # prediction and -1 represents an incorrect prediction
      prediction_booleans = self.models[t].predict(examples) == targets[target_name]
      prediction_results = [1 if pred else -1 for pred in prediction_booleans.values]

      # Calculate the error from the incorrect predictions
      dt_error = examples.loc[prediction_booleans[prediction_booleans == False].index]["weight"].to_numpy().sum()

      # Determine the weight of this tree based on the total error of the tree    
      if (dt_error == 0):
        self.alphas[t] = 1
      #elif (dt_error >= 0.5):
      #  break
      else:
        EPS = 1e-10 # Account for divide by zero
        self.alphas[t] = 0.5 * np.log((1 - dt_error) / (dt_error + EPS)) 

      # Update example weights based on if they were correctly
      # or incorrectly classified with following formulas:
      #   (correctly classified): example weight * e^(-tree weight) 
      #   (incorrectly classified): example weight * e^(tree weight)
      # we levereage the prediction_results array here since it contains
      # 1 for each correct classification and -1 for each incorrect classification
      examples["weight"] *= np.exp(-self.alphas[t] * np.array(prediction_results))

      # Normalize the new sample weights so they add to 1
      examples["weight"] /= examples["weight"].to_numpy().sum()

  def predict(self, examples):
    """ Use list of weak hypothesis and alphas trained
    with the train method to build a strong hypothesis 
    for each training example providede. The classification
    of each provided training example is returned in an array.
    """
    n_examples = examples.shape[0]

    # Populate a 2D array of all predictions made by each model for each example
    predictions = np.array([[None for _ in range(n_examples)] for _ in range(self.T)])
    for t in range(self.T):
      predictions[t] = self.models[t].predict(examples)

    # If only 1 tree was trained, just return the predictions
    if (self.T == 1):
      return predictions[0]

    # Iterate through each prediction and compare them to
    # predictions made by other trees based off weight
    # to determine the 'true' prediction
    true_predictions = ['' for _ in range(n_examples)]
    for i in range(n_examples):
      example_predictions = predictions[:, i]
      prediction_weights = {} 
      for j, pred in enumerate(example_predictions):
        if pred in prediction_weights:
          prediction_weights[pred] += self.alphas[j]
        else:
          prediction_weights[pred] = self.alphas[j]

      top_prediction = example_predictions[0] 
      top_weight = -1
      for pred, weight in prediction_weights.items():
        if weight >= top_weight:
          top_weight, top_prediction = weight, pred

      true_predictions[i] = top_prediction

    return true_predictions