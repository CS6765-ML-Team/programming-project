import pandas as pd               # For data manipulation and analysis
import time                       # For execution timing
import random                     # Random number generation

import matplotlib.pyplot as plt                               # For visualization
import seaborn as sns                                         # Advanced visualizations
from sklearn.metrics import accuracy_score, confusion_matrix  # For model evaluation
from sklearn.model_selection import train_test_split          # For splitting the data
from ucimlrepo import fetch_ucirepo                           # Letter recognition data set

from adaboost import AdaBoost

def estimator_benchmark(features, targets, T=50):
  """ Benchmark the Adaboost implementation
  found in Adaboost.py by running training/testing
  sessions on the provided features and targets but
  varying the number of estimators, or weak hypotheses
  learned by the Adaboost algorithm during each session.
  Returns a Dataframe containing the following:
    - Test data 'random state' integer (for reproducibility).
    - Accuracy of training.
    - Depth of learned tree.
    - Number of estimators
    - Time taken training model.
    - Time taken classifying.
  """
  # Fetch the dataset and sort into features/targets.
  letter_recognition = fetch_ucirepo(id=59) 
  features = letter_recognition.data.features
  targets = letter_recognition.data.targets
  
  # Depth of learned binary decision tree for each test
  depth = len(features.columns) 

  # Number of estimators to test during each session
  estimator_tests = range(1, T)
  n_estimator_tests = len(estimator_tests)

  # Choose a random state to split the data on
  state = random.randint(1, 2**30)

  # Map of returned data.
  results = {'State': [state for _ in range(n_estimator_tests)],
             'Accuracy': [0.0 for _ in range(n_estimator_tests)],
             'Depth': [depth for _ in range(n_estimator_tests)],
             'Estimators': estimator_tests,
             'Train Time': [0.0 for _ in range(n_estimator_tests)],
             'Classify Time': [0.0 for _ in range(n_estimator_tests)]
             }
  
  # Split using a random state.
  X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=state)

  # Perform n tests.
  for n in range(n_estimator_tests):
    # Number of estimators to use for current test
    estimators = estimator_tests[n]

    # Copy the dataframes into this iteration's datasets
    training_examples = X_train.copy(deep=True)
    training_targets = y_train.copy(deep=True)

    # Test the AdaBoost algorithm for 1 to 100 estimators.
    ada = AdaBoost(estimators, depth) 

    # Train the AdaBoost classifier and time how long it takes.
    start_train_time = time.time()
    ada.train(training_examples, training_targets)
    results['Train Time'][n] = time.time() - start_train_time 

    # Predict testing data
    start_classify_time = time.time()
    y_pred = ada.predict(X_test)
    results['Classify Time'][n] = time.time() - start_classify_time

    results['Accuracy'][n] = accuracy_score(y_test, y_pred)

  return pd.DataFrame(results)



def depth_benchmark(features, targets):
  """ Benchmark the Adaboost implementation
  found in Adaboost.py by running training/testing
  sessions on the provided features and targets but
  varying the depth of the learned trees on each session.
  Returns a Dataframe containing the following:
    - Test data 'random state' integer (for reproducibility).
    - Accuracy of training.
    - Depth of learned tree.
    - Number of estimators
    - Time taken training model.
    - Time taken classifying.
  """
  # Fetch the dataset and sort into features/targets.
  letter_recognition = fetch_ucirepo(id=59) 
  features = letter_recognition.data.features
  targets = letter_recognition.data.targets

  # Test from depth 1 to the number of features + 5
  depths = range(1, len(features.columns) + 5)
  n_depths = len(depths)

  # Choose a randome state to split the data on
  state = random.randint(1, 2**30)

  # The number of estimators for training
  t = 1 

  # Map of returned data.
  results = {'State': [state for _ in range(n_depths)],
             'Accuracy': [0.0 for _ in range(n_depths)],
             'Depth': depths,
             'Estimators': [t for _ in range(n_depths)],
             'Train Time': [0.0 for _ in range(n_depths)],
             'Classify Time': [0.0 for _ in range(n_depths)]
             }
  
  # Split using a random state.
  X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=state)

  # Perform n tests.
  for n in range(n_depths):
    # Depth of learned decision tree for current test
    depth = depths[n]

    # Copy the dataframes into this iteration's datasets
    training_examples = X_train.copy(deep=True)
    training_targets = y_train.copy(deep=True)

    # Test the AdaBoost algorithm for 1 to 100 estimators.
    ada = AdaBoost(t, depth) 

    # Train the AdaBoost classifier and time how long it takes.
    start_train_time = time.time()
    ada.train(training_examples, training_targets)
    results['Train Time'][n] = time.time() - start_train_time 

    # Predict testing data
    start_classify_time = time.time()
    y_pred = ada.predict(X_test)
    results['Classify Time'][n] = time.time() - start_classify_time

    results['Accuracy'][n] = accuracy_score(y_test, y_pred)

  return pd.DataFrame(results)


def adaboost_benchmark(features, targets, N=10, display_confusion_matrix=False):
  """ Benchmark the Adaboost implementation
  found in Adaboost.py by running N training/testing
  sessions on the provided features and targets. 
  Returns a Dataframe containing the following:
    - Test data 'random_state" integer (for reproducibility).
    - Accuracy of training.
    - Depth of learned tree.
    - Number of estimators
    - Time taken training model.
    - Time taken classifying.
  """
  # Fetch the dataset and sort into features/targets.
  letter_recognition = fetch_ucirepo(id=59) 
  features = letter_recognition.data.features
  targets = letter_recognition.data.targets

  # Depth of learned binary decision tree for each test
  depth = len(features.columns) 

  # Number of estimators 
  t = 14 

  # Map of returned data.
  results = {'State': random.sample(range(1, 2**30), N),
             'Accuracy': [0.0 for _ in range(N)],
             'Depth': [depth for _ in range(N)],
             'Estimators': [t for _ in range(N)],
             'Train Time': [0.0 for _ in range(N)],
             'Classify Time': [0.0 for _ in range(N)]
             }
  
  # Perform n tests.
  for n in range(N):
    # Split using a random state.
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=results['State'][n])

    # Copy the dataframes into this iteration's datasets
    training_examples = X_train.copy(deep=True)
    training_targets = y_train.copy(deep=True)

    # Test the AdaBoost algorithm for 1 to 100 estimators.
    ada = AdaBoost(t, depth) 

    # Train the AdaBoost classifier and time how long it takes.
    start_train_time = time.time()
    ada.train(training_examples, training_targets)
    results['Train Time'][n] = time.time() - start_train_time 

    # Predict testing data
    start_classify_time = time.time()
    y_pred = ada.predict(X_test)
    results['Classify Time'][n] = time.time() - start_classify_time

    results['Accuracy'][n] = accuracy_score(y_test, y_pred) * 100.00

    if (display_confusion_matrix):
      conf_matrix = confusion_matrix(y_test, y_pred)
      plt.figure(figsize=(6, 4))
      sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
      plt.title("Confusion Matrix Heatmap")
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.show()

  return pd.DataFrame(results)


if __name__ == '__main__':
  # fetch dataset 
  letter_recognition = fetch_ucirepo(id=59) 

  # Store the features and targets
  features = letter_recognition.data.features
  targets = letter_recognition.data.targets

  # Store a combination of features and targets
  total = pd.DataFrame(data=letter_recognition.data.original, columns=letter_recognition.data.headers)

  # Check for any missing values
  print(total.isnull().sum())

  # Get summary statistics of the data
  print(total.describe())

  # Check the distribution of target labels
  for label in targets.columns.values:
    print(total[label].value_counts())

  benchmark_results = adaboost_benchmark(features, targets, N=15, display_confusion_matrix=False)
  # benchmark_results = depth_benchmark(features, targets)
  # benchmark_results = estimator_benchmark(features, targets, 20)
  print(benchmark_results)

  # Save to csv
  benchmark_results.to_csv('results/benchmark.csv', index=False)


