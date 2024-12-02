# Spam Detection API

A robust Flask-based REST API for spam detection using multiple machine learning models. This project provides a flexible backend service that can classify text messages as spam or non-spam using different classification algorithms.

## Features

- Multiple ML models support (Random Forest, Logistic Regression, SVM, Naive Bayes)
- RESTful API endpoint for spam detection
- Cross-Origin Resource Sharing (CORS) support
- Environment variable configuration
- Production-ready with Gunicorn support

## Tech Stack

- Python
- Flask
- Scikit-learn
- Joblib
- Flask-CORS
- Gunicorn (Production server)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aakash768/Spam-Email-Backend.git
cd Spam-Email-Backend
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```env
CORS_ORIGIN=your-frontend-origin
PORT=10000
HOST=0.0.0.0
FLASK_DEBUG=0
```

## Usage

### Running the Development Server

```bash
python app.py
```

### Running with Gunicorn (Production)

```bash
gunicorn app:app
```

## API Endpoints

### POST /detect_spam

Detects if a given message is spam or not.

**Request Body:**
```json
{
    "message": "Your text message here",
    "model": "model1"  // Optional: model1, model2, model3, or model4
}
```

**Response:**
```json
{
    "is_spam": true/false
}
```

**Models Available:**
- model1: Random Forest
- model2: Logistic Regression
- model3: Support Vector Machine (SVM)
- model4: Naive Bayes

## Project Structure

```
├── app.py              # Main application file
├── models/             # Directory containing ML models
│   ├── rf_model.joblib        # Random Forest model
│   ├── lr_model.joblib        # Logistic Regression model
│   ├── svm_model.joblib       # SVM model
│   ├── nb_model.joblib        # Naive Bayes model
│   └── tfidf_vectorizer.joblib # TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── gunicorn.conf.py   # Gunicorn configuration
```

## Error Handling

The API includes proper error handling for:
- Empty messages
- Invalid model selection
- Processing errors

