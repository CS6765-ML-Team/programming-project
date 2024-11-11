import pandas as pd                         # For data manipulation
import numpy as np                          # For data calculation
from tree import DecisionTree, DecisionLeaf # Tree implementation
from util import calc_info_gain             # Function for calculating information gain of attribute

def id3_train(features, feature_names, descriminatory_feature, descriminatory_value, targets, target_name):
  """ Implementation of the ID3 decision tree
  induction algorithm used for classification.
  This is a recursive algorithm that choses the next
  feature to evaluate based on each feature's descriminatory
  power, or which feature will result in the most information
  gain about the data. This will continue until 1 of the 3
  following base cases is met:
    1.  Every training instance remaining has the same
        target feature value. In this case, return the
        target feature value. This enables ID3 to return
        the shortest (shallowest) possible trees.

    2.  There are no descriptive features remaining
        to divide the data based on. In this case, select
        the target feature value of majority in the data.

    3.  There are no training instances remaining in the
        data set. In this case, take the target feature value
        of majority in the parent (caller). This demonstrates
        the generalization power of ID3.
  """
  target_counts = targets.value_counts()

  # Note, the third
  # base case explained above is handled in the recursive
  # call section as we check for an empty new feature list
  # before we even issue the recursive call.
  if (len(feature_names) == 0 or target_counts.size == 1):
    # Return the target value of majority in current data set if
    # there are no features left to partition by, or return
    # the only remaining target value if there is only 1 remaining.
    # return target_counts.first()
    return DecisionLeaf(target_counts.head(1))
  
  # Examine the information gain of dividing the 
  # data based on each remaining feature greedily.
  # Recurse using the feature with the highest information gain.
  best_feature = ''
  best_info_gain = 0 
  for feature in feature_names:
    info_gain = calc_info_gain(features, feature, targets, target_name)
    if (info_gain >= best_info_gain):
      best_feature, best_info_gain = feature, info_gain

  # Remove the chosen best feature from the list of feature names
  # pruned_feature_names = feature_names.remove(best_feature)
  pruned_feature_names = np.delete(feature_names, np.where(feature_names == best_feature))

  # TODO: Will we need to control the size of this split?
  # Split the data based off the best feature that was found.
  feature_split = features[best_feature].value_counts()

  # Create a new DecisionTree to be returned
  dt = DecisionTree(descriminatory_feature, descriminatory_value)

  # Spawn a new recursive call for each feature split.
  for split in feature_split.index:
    new_features = features[features[best_feature] == split]
    if (new_features.empty):
      # Base case where new_features is empty, don't even issue the
      # recursive call, just return the current target value of majority.
      return target_counts.head(1)

    new_targets = targets.loc[new_features.index]

    dt.add_child(id3_train(new_features, pruned_feature_names, best_feature, split, new_targets, target_name))
  
  return dt

def id3_fit():
  """ Fit the provided data to a decision tree that has
  been trained using the id3_train method.
  """
  pass