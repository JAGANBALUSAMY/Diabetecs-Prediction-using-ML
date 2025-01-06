import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv(r"E:\New folder\diabetes.csv")

# Replace 0s with NaN for selected columns and fill with mean
dataset[["Glucose", "Insulin", "BMI"]] = dataset[["Glucose", "Insulin", "BMI"]].replace(0, np.nan)
dataset.fillna(dataset.mean(), inplace=True)

# Feature scaling
features = ["Glucose", "Insulin", "BMI", "Age"]
target = "Outcome"
scaler = StandardScaler()
X = scaler.fit_transform(dataset[features])
Y = dataset[target]

# Save the scaler for consistent scaling
with open(r'E:\diabetes_prediction\scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Train and save models
models = {
    "Random Forest Tree": RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions={
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        n_iter=50,
        cv=5,
        scoring='accuracy',
        random_state=42
    ),
    "Logistic Regression": RandomizedSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_distributions={
            'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear', 'lbfgs'],
        },
        n_iter=50,
        cv=5,
        scoring='accuracy',
        random_state=42
    ),
    "K-Nearest Neighbour": RandomizedSearchCV(
        estimator=KNeighborsClassifier(),
        param_distributions={
            'n_neighbors': range(1, 31),
            'weights': ['uniform', 'distance'],
        },
        n_iter=50,
        cv=5,
        scoring='accuracy',
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions={
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
        },
        n_iter=50,
        cv=5,
        scoring='accuracy',
        random_state=42
    ),
}

for name, model in models.items():
    if isinstance(model, RandomizedSearchCV):
        model.fit(X_train, Y_train)
        best_model = model.best_estimator_
    else:
        model.fit(X_train, Y_train)
        best_model = model

    with open(f"E:\\diabetes_prediction\\diabetes_{name.lower().replace(' ', '_')}_model.pkl", 'wb') as file:
        pickle.dump(best_model, file)

print("Model training completed.")
