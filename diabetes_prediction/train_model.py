import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load dataset
dataset = pd.read_csv("E:/New folder/diabetes.csv")

# Scale the numeric columns
numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset[numeric_columns])

dataset_scaled = pd.DataFrame(dataset_scaled, columns=numeric_columns)

# Define features and target variable
X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome']
)

# Define the parameter distribution
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, Y_train)

# Get the best model
best_model_random = random_search.best_estimator_


# Save the model to a file
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(best_model_random, file)

print("Model training completed and saved as 'diabetes_model.pkl'.")
