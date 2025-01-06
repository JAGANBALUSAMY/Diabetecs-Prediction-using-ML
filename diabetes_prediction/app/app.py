from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

app = Flask(__name__)

# Corrected file path to avoid escape sequence warning
model_path = r'E:\diabetes_prediction\diabetes_model.pkl'

# Load the pre-trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':  # Ensure the form is submitted via POST
        # Getting the form input values
        glucose = float(request.form['glucose'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])

        # Feature scaling (you should use the same scaler used during training)
        scaler = StandardScaler()
        features = np.array([[glucose, insulin, bmi, age]])
        features_scaled = scaler.fit_transform(features)

        # Predict the result using the loaded model
        prediction = model.predict(features_scaled)
        
        # Convert prediction to a readable result
        if prediction[0] == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"

    # Render the template with the result
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
