import numpy as np                      # For data calculation
from itertools import starmap           # For quickly iterating over features
from tree import BinaryDecisionTree     # Tree implementation

class ID3Classifier:
  """ An implmentation of the Iterative Dichotomiser 3
  (ID3) decision tree algorithm. Constructor parameters
  include an optional depth paramter that controls the maximum
  depth of generated decision trees (default = +infinity).
  This ID3 implementation uses a weighted information gain calculation
  to determine the best feature to split on, and the best value
  of that feature to split on. The decision trees produced are 
  binary decision trees, where each node posesses a split feature
  and split feature value, and the two child branches are 
  designated for examples with feature values less than or
  equal to the split feature value (false branch), and
  values greater than the split feature value (true branch).
  This tree implementation is found in tree.py.
  """
  def __init__(self, depth=np.inf):
    self.depth = depth
    self.tree = None

  def train(self, examples, features, targets):
    """ Public method for training a decision tree using the 
    ID3 algorithm implemented in _train. This public facing
    method sets the tree member of the ID3Classifier class.
    """
    self.tree = None
    self.tree = self._train(examples, features, targets, 1) 

  def _train(self, examples, features, targets, depth):
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
    split_feature, split_value = ID3Classifier._calc_split_attribute(examples, features, targets) 
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
      dt.add_false(self._train(false_branch, pruned_features, targets.loc[false_branch.index], depth + 1))

    if (true_branch.empty):
      dt.add_true(BinaryDecisionTree(0, majority_target))
    else:
      dt.add_true(self._train(true_branch, pruned_features, targets.loc[true_branch.index], depth + 1))

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
  
  @staticmethod
  def _calc_entropy(targets):
    """ Calculate the entropy for a sample
    of training examples based on Shanon's 
    Entropy Model. The first column in the passed
    targets Dataframe is taken as the target
    to calculate the entropy against.
    """
    target_name = targets.columns.values[0]
    entropy = 0
    n_targets = targets.shape[0]
    for frequency in targets[target_name].value_counts():
      probability = frequency / n_targets 
      entropy -= probability * np.log2(probability)

    return entropy

  @staticmethod
  def _calc_attribute_info_gain(examples, n_samples, targets, total_entropy, feature):
    # Iterate through each unique value of this feature
    # and calculate the information gain of having this unique
    # value be the split value for this feature 
    unique_values = examples[feature].unique()

    # Calculate the less than or equal to and the greater than
    # splits of the data set using the unique values for this feature.
    # This creates an array of Dataframes of length len(unique_values)
    less_subsets = [examples.loc[examples[feature] <= value] for value in unique_values] 
    greater_subsets = [examples.loc[examples[feature] > value] for value in unique_values]

    # Calculate the entropy for each of the less than or equal to, and greater than, subsets 
    # of the data set for all unique values of this feature.
    less_entropy = np.array([(subset["weight"].to_numpy().sum() / n_samples) * ID3Classifier._calc_entropy(targets.loc[subset.index]) for subset in less_subsets])
    greater_entropy = np.array([(subset["weight"].to_numpy().sum() / n_samples) * ID3Classifier._calc_entropy(targets.loc[subset.index]) for subset in greater_subsets])

    split_info_gain = np.add(less_entropy, greater_entropy)

    info_gains = [total_entropy - split_gains for split_gains in split_info_gain]

    # Find the maximum info gain and its index
    max_info_gain_idx = np.argmax(info_gains)
    max_info_gain = info_gains[max_info_gain_idx]

    return max_info_gain, unique_values[max_info_gain_idx], feature

  @staticmethod
  def _calc_split_attribute(S, feature_names, targets):
    """ Given the dataset S, determine which attribute
    of the dataset and value of that attribute is optimal 
    to partition the data set. The attribute and split value
    is calculated based on the information gain of splitting
    each unique value for a given attribute and taking the 
    largest gain.
    """
    # The number of entries in dataset S
    S_size = S.shape[0]

    # Calculate the entropy for the entire dataset
    e = ID3Classifier._calc_entropy(targets)

    # Build the argument list, which is a list of tuples
    # where each tuple holds the 5 arguments to the function
    n_features = len(feature_names)
    arg_list = zip([S for _ in range(n_features)],
                   [S_size for _ in range(n_features)],
                   [targets for _ in range(n_features)],
                   [e for _ in range(n_features)],
                   feature_names)
    
    # Execute the multi-processing *map operation
    attribute_split_info_gains = starmap(ID3Classifier._calc_attribute_info_gain, arg_list)

    # Pull out the optimal split value and feature based on the maximum info gain
    _, split_value, split_feature = max(list(attribute_split_info_gains), key=lambda item: item[0])

    return split_feature, split_value