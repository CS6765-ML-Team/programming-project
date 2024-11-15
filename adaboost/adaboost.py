import numpy as np                      # Access to math operators
import pandas as pd                     # Data manipulation
import random                           # Random number generator
from id3 import id3_train, id3_predict  # ID3 training algorithm implementation

def adaboost(X_train, y_train, X_test, T):
  """ Use the AdaBoost ensemble learning method
  with ID3 as a base learner to learn a strong 
  hypothesis for the provided data set X_train.
  """
  # How many training examples we are training on
  n_examples = X_train.shape[0]

  # Extract the name of the target value we 
  # are trying to classify in y_train
  target_name = y_train.columns.values[0]

  # Before we add the weight column, extract the 
  # names of each feature in X_train
  feature_names = X_train.columns.values

  # Assign an initial weight to all training examples
  X_train["weight"] = [1/n_examples] * n_examples

  # We now have to train T different decision trees 
  # using the provided data. After each tree is trained,
  # we must determine the total error of that decision 
  # tree (sum of weights of incorrectly classified examples)
  # and use the total error to assign a weight to
  # the decision tree.
  trees = []
  tree_weights = []
  features = X_train.copy(deep=True)
  targets = y_train.copy(deep=True)
  for _ in range(T):
    # Obtain a weak hypothesis using the ID3 base learner implementation
    dt = id3_train(features, feature_names, targets, target_name)
    trees.append(dt)

    # Create an array of prediction results where 1 represents a correct
    # prediction and -1 represents an incorrect prediction
    prediction_booleans = id3_predict(features, dt) == targets[target_name]
    prediction_results = [1 if pred else -1 for pred in prediction_booleans.values]

    # Calculate the error from the incorrect predictions
    dt_error = features.loc[prediction_booleans[prediction_booleans == False].index]["weight"].to_numpy().sum()
    print(dt_error)

    # Determine the weight of this tree based on the total
    # error of the tree    
    EPS = 1e-10 # Account for divide by zero
    dt_weight = 0.5 * np.log((1 - dt_error) / (dt_error + EPS)) 
    tree_weights.append(dt_weight)

    # Update example weights based on if they were correctly
    # or incorrectly classified with following formulas:
    #   (correctly classified): example weight * e^(-tree weight) 
    #   (incorrectly classified): example weight * e^(tree weight)
    # we levereage the prediction_results array here since it contains
    # 1 for each correct classification and -1 for each incorrect classification
    features["weight"] = features["weight"] * np.exp(-dt_weight * np.array(prediction_results))

    # Normalize the new sample weights so they add to 1
    features["weight"] /= features["weight"].to_numpy().sum()

    # Accumulate list of feature weights
    weights_nparray = features["weight"].tolist()
    acc_weights = [sum(weights_nparray[:y]) for y in range(1, len(weights_nparray) + 1)]

    # Use the accumulated weights as a distribution to resample the features and targets
    old_features = features.copy(deep=True)
    old_targets = targets.copy(deep=True)
    resampled_indicies = []
    for i in range(n_examples):
      x = random.random() + acc_weights[0]  # Add smallest weight so we never have zero matches
      sample = [w[0] for w in zip(features.index.tolist(), acc_weights) if w[1] <= x][-1]
      resampled_indicies.append(sample)

    # Build the re-sampled features and target dataframes 
    # in a way that re-indexes the re-sampled data
    features = pd.DataFrame(data=old_features.loc[resampled_indicies].to_numpy(), columns=np.append(feature_names, "weight"))
    targets = pd.DataFrame(data=old_targets.loc[resampled_indicies].to_numpy(), columns=old_targets.columns.values)
    
    # Initialize all of the example weights in the newly
    # sampled data to the same value before
    # we continue training
    features["weight"] = [1/n_examples] * n_examples

  # Create a dataframe of the trained trees and their weights
  trained_trees = pd.DataFrame({'tree': trees, 'tree_weight': tree_weights})

  # TODO: Revisit the prediction step

  # List of weak hypothesis and weights for each of these
  # weak hypothesis that will allow us to create a strong
  # hypothesis. Classify each test example with each weak
  # hypothesis. When there are differing classifications,
  # the classification with the largest sum of weights is 
  # the classification returned for that particular example.
  predictions = []
  for _, row in X_test.iterrows():
    current_predictions = []
    current_weights = []
    for _, tree in trained_trees.iterrows():
      current_predictions.append(tree["tree"].traverse(row))
      current_weights.append(tree["tree_weight"])

    # Accumulate the predictions and their weights into a dataframe, sort by unique
    # entries for prediction, and sum the total weights for each unique prediction to 
    # get our final prediction
    prediction_df = pd.DataFrame({"prediction": current_predictions, "weight": current_weights})
    
    final_prediction = None
    final_weight = 0
    for prediction in prediction_df["prediction"].unique():
      weight = prediction_df[prediction_df["prediction"] == prediction]["weight"].to_numpy().sum()
      if (weight >= final_weight):
        final_weight, final_prediction = weight, prediction
    
    predictions.append(final_prediction)

  return predictions