import numpy as np                    # For numerical operations
import pandas as pd                   # For data manipulation and analysis
from sklearn.metrics import accuracy_score, confusion_matrix  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting the data
from ucimlrepo import fetch_ucirepo   # Letter recognition data set

# fetch dataset 
letter_recognition = fetch_ucirepo(id=59) 

# Store the features and targets
features = letter_recognition.data
targets = letter_recognition.data

# Store a combination of features and targets
total = pd.DataFrame(data=letter_recognition.data.original, columns=letter_recognition.data.headers)

# Check for any missing values
print(total.isnull().sum())

# Get summary statistics of the data
print(total.describe())

# Check the distribution of target labels
for label in targets.columns.values:
  print(total[label].value_counts())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)