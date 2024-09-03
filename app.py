import logging
import os

import pandas as pd
from flask import Flask, render_template, request, jsonify

from svm_classification import load_model

app = Flask(__name__)

logging.basicConfig(filename='app.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Try to load the trained machine learning model
try:
    model_path = 'models/trained_svm_model.pkl'
    if os.path.isfile(model_path):
        model = load_model(model_path)
        logging.info("Model loaded successfully")
    else:
        logging.warning("Model not loaded.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # If an error occurs during model loading, print the error and set model to None
    model = None


@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the main page and handles prediction requests from the form.

    Returns:
        Flask response: The rendered HTML template for the index page.
     """

    if request.method == 'POST':
        try:
            # Get input values from the form
            age = int(request.form['age'])
            salary = int(request.form['salary'])
            job = request.form['job']
            logging.info(f"Received prediction request: age = {age}, salary = {salary}, job = {job}")

            # Create a DataFrame from the input features
            features = pd.DataFrame({'age': [age], 'salary': [salary], 'job': [job]})

            # Make a prediction only if the model was loaded successfully
            if model is not None:
                prediction = model.predict(features)[0]
                logging.info(f"Prediction successful: {prediction}")
            else:
                prediction = "Model not loaded"
                logging.warning("Prediction attempted with unloaded model.")

            # Render the template with the prediction result
            return render_template('index.html', prediction=prediction)

        except Exception as e:
            # Handle errors during prediction
            logging.error(f"Prediction error: {e}")
            return render_template('index.html', error=f"Prediction Error: {str(e)}"), 500

    else:
        # If the request method is GET, render the template without a prediction
        return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests submitted as JSON data.

    Returns:
        Flask response: A JSON response containing the prediction or an error message.
    """
    logging.info("'/predict' endpoint accessed.")
    try:
        # Get JSON data from the request
        features = request.get_json()
        logging.info(f"Received JSON data: {features}")
        # Create a DataFrame from received features
        features_df = pd.DataFrame([features])

        # Make a prediction if the model is loaded
        if model is not None:
            prediction = model.predict(features_df)[0]
            logging.info(f"Prediction successful: {prediction}")
            # Return the prediction as a JSON response
            return jsonify({'prediction': prediction})
        else:
            # Return an error message if the model is not loaded
            logging.warning("Prediction attempted with unloaded model.")
            return jsonify({'error': 'Model not loaded'}), 500

    except Exception as e:
        # Handle general exceptions during the prediction process
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
