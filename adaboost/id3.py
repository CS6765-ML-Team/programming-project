import numpy as np                      # For data calculation
from tree import BinaryDecisionTree     # Tree implementation
from util import calc_split_attribute   # Function for calculating information gain of attribute

def id3_train(features, feature_names, targets, target_name):
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
    return BinaryDecisionTree(target_name, target_counts.index.to_numpy()[0][0])

  # Determine the optimal feature and feature value to partition
  # the dataset into true and false branches
  split_feature, split_value = calc_split_attribute(features, targets, target_name) 

  # Remove the chosen best feature from the list of feature names
  # pruned_feature_names = feature_names.remove(best_feature)
  pruned_feature_names = np.delete(feature_names, np.where(feature_names == split_feature))

  # Create a new DecisionTree to be returned
  dt = BinaryDecisionTree(split_feature, split_value)

  # Create the true and false branches based on split feature
  false_branch = features[features[split_feature] <= split_value]
  true_branch = features[features[split_feature] > split_value]

  # Spawn recursive calls for true and false branches of decision tree, checking
  # for base case first where there are no features left in these trees so we just
  # return the current target value of majority as a leaf
  if (false_branch.empty):
    print("general")
    dt.add_false(BinaryDecisionTree(target_name, target_counts.index.to_numpy()[0][0]))
  else:
    dt.add_false(id3_train(false_branch, pruned_feature_names, targets.loc[false_branch.index], target_name))

  if (true_branch.empty):
    print("general")
    dt.add_true(BinaryDecisionTree(target_name, target_counts.index.to_numpy()[0][0]))
  else:
    dt.add_true(id3_train(true_branch, pruned_feature_names, targets.loc[true_branch.index], target_name))

  return dt

def id3_predict(features, dt):
  """ Predict the target values given testing examples
  and an already trained BinaryDecisionTree.
  """
  predictions = []
  for _, row in features.iterrows():
    predictions.append(dt.traverse(row))

  return predictions
