
from flask import Flask, request, jsonify
from pycaret.classification import load_model
import numpy as np

app = Flask(__name__)
model = load_model('pycaret_iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json.get('features')
    pred = model.predict([features])[0]
    return jsonify({'prediction': str(pred)})

if __name__ == '__main__':
    app.run(debug=True)
