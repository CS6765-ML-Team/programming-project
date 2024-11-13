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


  # For each value of feature A in dataset S, split
  # S into subsets greater than or equal to and less 
  # than each value of feature A. For each of these splits
  # determine the total information gain.

  total_count = S[A].count()
  unique_vals = S[A].value_counts()



  split_value = 0
  max_info_gain = 0
  for value in unique_vals.index:
    sv_false = S[S[A] < value]
    sv_false_entropy = calc_entropy(targets.loc[sv_false.index], target_name)
    sv_true = S[S[A] >= value]
    sv_true_entropy = calc_entropy(targets.loc[sv_true.index], target_name)

    info_gain = e - (len(sv_true.index) / total_count) * sv_true_entropy + (len(sv_false.index) / total_count) * sv_false_entropy
    if (info_gain >= max_info_gain):
      max_info_gain, split_value = info_gain, value 



#
#  entropy_sum = 0
#  for value in unique_vals.index:
#    sv = S[S[A] == value]
#    sv_targets = targets.loc[sv.index]
#    sv_entropy = (unique_vals[value] / total_count) * calc_entropy(sv_targets, target_name)
#    print("Subset when {} = {}.\tSV Entropy = {}.".format(A, value, sv_entropy))
#    entropy_sum += sv_entropy 

  return info_gain, split_value 