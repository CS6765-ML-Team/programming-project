import pandas as pd                   # For data manipulation and analysis
from sklearn.metrics import accuracy_score, confusion_matrix  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting the data
from ucimlrepo import fetch_ucirepo   # Letter recognition data set
import time                               # For execution timing
from adaboost import AdaBoost

if __name__ == '__main__':
  # fetch dataset 
  letter_recognition = fetch_ucirepo(id=59) 

  # Store the features and targets
  features = letter_recognition.data.features
  targets = letter_recognition.data.targets

  # Store a combination of features and targets
  total = pd.DataFrame(data=letter_recognition.data.original, columns=letter_recognition.data.headers)

  # Check for any missing values
  # print(total.isnull().sum())

  # Get summary statistics of the data
  # print(total.describe())

  # Check the distribution of target labels
  # for label in targets.columns.values:
  #   print(total[label].value_counts())

  # Split the data into training and testing sets (80% train, 20% test)
  # Tests Run:
  #   - state: 42   accuracy: 87.65%
  #   - state: 21   accuracy: 85.42%
  #   - state: 99   accuracy: 86.88% 
  #   - state: 67   accuracy: 84.70%
  X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=67)

  # Optimal Parameters:
  #   - Number of estimators: 12
  #   - Depth of decision trees: number of features + 1
  t = 12
  d = len(features.columns) + 1

  # Copy the dataframes into this iteration's datasets
  training_examples = X_train.copy(deep=True)
  training_targets = y_train.copy(deep=True)

  # Test the AdaBoost algorithm for 1 to 100 estimators.
  ada = AdaBoost(t, d) 

  # Train the AdaBoost classifier and time how long it takes.
  start_time = time.time()
  ada.train(training_examples, training_targets)
  train_time = time.time() - start_time

  # Predict testing data
  y_pred = ada.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)

  # Print results to console
  print(f"Accuracy: {accuracy * 100:.2f}%")
 
  # Generate a confusion matrix
  print("Confusion Matrix:")
  print(conf_matrix)