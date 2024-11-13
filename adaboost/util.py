import numpy as np    # For numerical operations

def calc_entropy(targets, target_name):
  """ Calculate the entropy for a sample
  of training examples based on Shanon's 
  Entropy Model. 
  """
  entropy = 0
  total_count = targets[target_name].count()
  for frequency in targets[target_name].value_counts():
    probability = frequency / total_count
    entropy += -1 * probability * np.log2(probability)

  return entropy

#
#  entropy_sum = 0
#  for value in unique_vals.index:
#    sv = S[S[A] == value]
#    sv_targets = targets.loc[sv.index]
#    sv_entropy = (unique_vals[value] / total_count) * calc_entropy(sv_targets, target_name)
#    print("Subset when {} = {}.\tSV Entropy = {}.".format(A, value, sv_entropy))
#    entropy_sum += sv_entropy 

  return info_gain, split_value 

def calc_split_attribute(S, targets, target_name):
  """ Given the dataset S, determine which attribute
  of the dataset and value of that attribute is optimal 
  to partition the data set. The attribute and split value
  is calculated based on the information gain of splitting
  each unique value for a given attribute and taking the 
  largest gain.
  """
  # The number of entries in dataset S
  S_size = len(S.index)

  # Calculate the entropy for the entire dataset
  e = calc_entropy(targets, target_name)

  # Retrieve a list of all features
  features = S.columns.values

  # Keep track of the highest information gain calculated
  # as well as what feature that information gain was for
  best_info_gain = 0
  best_feature = None
  best_split_value = 0

  # Iterate through each feature of dataset S
  for feature in features:
    # Determine the unique values of feature
    unique_vals = S[feature].value_counts()

    # Iterate through each unique value of this feature
    # and calculate the information gain of having this unique
    # value be the split value for this feature 
    split_value = 0
    feature_info_gain = 0
    for value in unique_vals.index:
      threshold = value
      sv_less = S[S[feature] <= threshold]
      sv_less_entropy = calc_entropy(targets.loc[sv_less.index], target_name)
      sv_great = S[S[feature] > threshold]
      sv_great_entropy = calc_entropy(targets.loc[sv_great.index], target_name)

      info_gain = e - ((len(sv_less.index) / S_size) * sv_less_entropy + (len(sv_great.index) / S_size) * sv_great_entropy)
      if (info_gain > feature_info_gain):
        feature_info_gain, split_value = info_gain, value

    if (feature_info_gain >= best_info_gain):
      best_info_gain, best_feature, best_split_value = feature_info_gain, feature, split_value

  return best_feature, best_split_value

    