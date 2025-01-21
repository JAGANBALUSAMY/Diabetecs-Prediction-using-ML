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
    result = "No input is specified"
    selected_model = None
    form_values = {"glucose": "", "insulin": "", "bmi": "", "age": ""}

    if request.method == 'POST':
        try:
            # Extract input values
            glucose = request.form.get('glucose', "").strip()
            insulin = request.form.get('insulin', "").strip()
            bmi = request.form.get('bmi', "").strip()
            age = request.form.get('age', "").strip()
            selected_model = request.form.get('model', None)

            # Preserve form values for re-rendering
            form_values = {"glucose": glucose, "insulin": insulin, "bmi": bmi, "age": age}

            # Validate inputs
            if all([glucose, insulin, bmi, age, selected_model]):
                glucose = float(glucose)
                insulin = float(insulin)
                bmi = float(bmi)
                age = int(age)

                # Retrieve selected model
                model = models.get(selected_model)
                if model:
                    # Scale input and predict
                    features = np.array([[glucose, insulin, bmi, age]])
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)
                    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
                else:
                    result = "Invalid model selected."
            else:
                result = "No input is specified. Please fill out all fields."

        except ValueError:
            # Handle invalid numeric inputs
            result = "Invalid input. Please enter valid numeric values."

    return render_template(
        'index.html',
        result=result,
        models=list(model_paths.keys()),
        selected_model=selected_model,
        form_values=form_values
    )

if __name__ == '__main__':
    app.run(debug=True)
