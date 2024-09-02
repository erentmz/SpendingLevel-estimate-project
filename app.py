from flask import Flask, render_template, request, jsonify
import pandas as pd

from svm_classification import load_model

app = Flask(__name__)

model = load_model('models/trained_svm_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = int(request.form['age'])
        salary = int(request.form['salary'])
        job = request.form['job']

        features = pd.DataFrame({'age': [age], 'salary': [salary], 'job': [job]})

        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.get_json()
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
