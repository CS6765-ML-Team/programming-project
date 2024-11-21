import numpy as np                      # For data calculation
from tree import BinaryDecisionTree     # Tree implementation
from util import calc_split_attribute   # Function for calculating information gain of attribute

class ID3Classifier:
  """
  """
  def __init__(self, depth=np.inf):
    self.depth = depth
    self.tree = None

  def train(self, examples, features, targets, target_name):
    """ Public method for training a decision tree using the 
    ID3 algorithm implemented in _train. This public facing
    method sets the tree member of the ID3Classifier class.
    """
    self.tree = None
    self.tree = self._train(examples, features, targets, target_name, 1) 

  def _train(self, examples, features, targets, target_name, depth):
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
    majority_target = target_counts.index.to_numpy()[0][0]

    # Note, the third
    # base case explained above is handled in the recursive
    # call section as we check for an empty new feature list
    # before we even issue the recursive call. Also adding a check
    # here for if we've reached our max tree depth.
    if (len(features) == 0 or target_counts.size == 1 or depth >= self.depth):
      # Return the target value of majority in current data set if
      # there are no features left to partition by, or return
      # the only remaining target value if there is only 1 remaining.
      return BinaryDecisionTree(0, majority_target)

    # Determine the optimal feature and feature value to partition
    # the dataset into true and false branches
    split_feature, split_value = calc_split_attribute(examples, features, targets, target_name) 
    # Determine the index of the split feature from the columns
    # of the entire dataset, i.e., all features
    split_feature_idx = examples.columns.get_loc(split_feature)

    # Remove the chosen best feature from the list of feature names
    pruned_features = np.delete(features, np.where(features == split_feature))

    # Create a new DecisionTree to be returned
    dt = BinaryDecisionTree(split_feature_idx, split_value)

    # Create the true and false branches based on split feature
    false_branch = examples[examples[split_feature] <= split_value]
    true_branch = examples[examples[split_feature] > split_value]

    # Spawn recursive calls for true and false branches of decision tree, checking
    # for base case first where there are no features left in these trees so we just
    # return the current target value of majority as a leaf
    if (false_branch.empty):
      dt.add_false(BinaryDecisionTree(0, majority_target))
    else:
      dt.add_false(self._train(false_branch, pruned_features, targets.loc[false_branch.index], target_name, depth + 1))

    if (true_branch.empty):
      dt.add_true(BinaryDecisionTree(0, majority_target))
    else:
      dt.add_true(self._train(true_branch, pruned_features, targets.loc[true_branch.index], target_name, depth + 1))

    return dt


  def predict(self, examples):
    """ Predict the target values given testing examples
    and a tree that has already been trained by the train method.
    """
    if (self.tree is None):
      print("Must train model using train() method first.")
      return []

    predictions = [None for _ in range(examples.shape[0])]
    for i, row in enumerate(examples.to_numpy()):
      predictions[i] = self.tree.traverse(row) 

    return predictions