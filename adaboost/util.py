import numpy as np                    # For numerical operations

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


def calc_info_gain(S, A, targets, target_name):
  """ Calculate the estimated information gain
  as the reduction of entropy in a sample of
  training examples by sorting on a specific feature.
  """
  e = calc_entropy(targets, target_name)

  # Split the data set S based on feature A, where
  # Sv becomes the subset for each value of A that S
  # has been split by. We must calculate the entropy
  # for each Sv. How do we split based on A if
  # A is an integer?

  total_count = S[A].count()
  unique_vals = S[A].value_counts()
  entropy_sum = 0
  for value in unique_vals.index:
    sv = S[S[A] == value]
    sv_targets = targets.loc[sv.index]
    #print("Subset when {} = {}.\tNumber of occurences = {}.".format(A, value, unique_vals[value]))
    entropy_sum += (unique_vals[value] / total_count) * calc_entropy(sv_targets, target_name)

  return e - entropy_sum