import numpy as np                       # For numerical operations

def calc_entropy(targets, target_name):
  """ Calculate the entropy for a sample
  of training examples based on Shanon's 
  Entropy Model. 
  """
  entropy = 0
  n_targets = targets.shape[0]
  for frequency in targets[target_name].value_counts():
    probability = frequency / n_targets 
    entropy -= probability * np.log2(probability)

  return entropy

def calc_split_attribute(S, feature_names, targets, target_name):
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
  e = calc_entropy(targets, target_name)

  # Keep track of the highest information gain calculated
  # as well as what feature that information gain was for
  best_info_gain = -1 
  best_feature = None
  best_split_value = 0

  # Iterate through each feature of dataset S
  for feature in feature_names:
    # Iterate through each unique value of this feature
    # and calculate the information gain of having this unique
    # value be the split value for this feature 
    split_value = 0 
    feature_info_gain = -1 
    for threshold in S[feature].unique():
      sv_less = S[S[feature] <= threshold]
      sv_great = S[S[feature] > threshold]
      sv_size = sv_less.shape[0] / S_size

      # sv_less_weight = sv_less["weight"].to_numpy().sum()
      sv_less_entropy = calc_entropy(targets.loc[sv_less.index], target_name)
      # sv_great_weight = sv_great["weight"].to_numpy().sum()
      sv_great_entropy = calc_entropy(targets.loc[sv_great.index], target_name)

      info_gain = e - (sv_size * sv_less_entropy + (1 - sv_size) * sv_great_entropy)
      if (info_gain >= feature_info_gain):
        feature_info_gain, split_value = info_gain, threshold 

    if (feature_info_gain >= best_info_gain):
      best_info_gain, best_feature, best_split_value = feature_info_gain, feature, split_value

  return best_feature, best_split_value

    