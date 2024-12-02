from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

CORS(app, origins=[os.getenv('CORS_ORIGIN')])  # More specific CORS configuration

#Load the trained model and vectorizer
vectorizer = joblib.load("./models/tfidf_vectorizer.joblib")

#Load the models
models = {
    'model1' : joblib.load('./models/rf_model.joblib'),
    'model2' : joblib.load('./models/lr_model.joblib'),
    'model3' : joblib.load('./models/svm_model.joblib'),
    'model4' : joblib.load('./models/nb_model.joblib')   
}

@app.route('/detect_spam', methods=['POST'])

def detect_spam():
    data = request.json
    message = data.get('message', '').strip()
    selected_model = data.get('model', 'model1')  # Default to model1

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Check if the selected model is valid
    if selected_model not in models:
        return jsonify({'error': 'Invalid model selected'}), 400

    model = models[selected_model]

    # Transform the message using the vectorizer
    try:
        transformed_message = vectorizer.transform([message])
    except Exception as e:
        return jsonify({'error': 'Error processing message'}), 500

    # Convert to dense if the selected model is SVM
    if selected_model == 'model3':  # Assuming model3 is the SVM
        transformed_message = transformed_message.toarray()

    # Predict using the selected model
    prediction = model.predict(transformed_message)[0]
    return jsonify({'is_spam': bool(prediction)})

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    app.run(host=os.getenv('HOST', '0.0.0.0'), port=port, debug=os.getenv('FLASK_DEBUG', '0') == '1')