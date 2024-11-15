import pandas as pd                   # For data manipulation and analysis
from sklearn.metrics import accuracy_score, confusion_matrix  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting the data
from ucimlrepo import fetch_ucirepo   # Letter recognition data set
import time                               # For execution timing
from adaboost import adaboost
import random
import numpy as np

if __name__ == '__main__':
  # fetch dataset 
  letter_recognition = fetch_ucirepo(id=59) 

  # Store the features and targets
  features = letter_recognition.data.features
  feature_names = features.columns.values
  targets = letter_recognition.data.targets
  target_names = targets.columns.values 

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
  X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

  start_time = time.time()
  y_pred = adaboost(X_train, y_train, X_test, 1)
  print(f"--- Model Trained on {len(X_train.index)} Examples in {time.time() - start_time} seconds ---")

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy * 100:.2f}%")
 
  # Generate a confusion matrix
  conf_matrix = confusion_matrix(y_test, y_pred)
  print("Confusion Matrix:")
  print(conf_matrix)