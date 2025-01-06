from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Paths to model and scaler files
model_paths = {
    'Random Forest Tree': r'E:\diabetes_prediction\diabetes_random_forest_tree_model.pkl',
    'Logistic Regression': r'E:\diabetes_prediction\diabetes_logistic_regression_model.pkl',
    'K-Nearest Neighbour': r'E:\diabetes_prediction\diabetes_k-nearest_neighbour_model.pkl',
    'Naive Bayes': r'E:\diabetes_prediction\diabetes_naive_bayes_model.pkl',
    'Decision Tree': r'E:\diabetes_prediction\diabetes_decision_tree_model.pkl'
}
scaler_path = r'E:\diabetes_prediction\scaler.pkl'

# Load models and scaler
models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items()}
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_model = None

    if request.method == 'POST':  # Ensure the form is submitted via POST
        # Getting the form input values
        glucose = float(request.form['glucose'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])
        selected_model = request.form['model']

        # Load the selected model
        model = models.get(selected_model)
        if model is None:
            result = "Invalid model selected"
        else:
            # Prepare input features and scale them
            features = np.array([[glucose, insulin, bmi, age]])
            features_scaled = scaler.transform(features)

            # Predict using the selected model
            prediction = model.predict(features_scaled)
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    return render_template(
        'index.html', result=result, models=list(model_paths.keys()), selected_model=selected_model
    )

if __name__ == '__main__':
    app.run(debug=True)
