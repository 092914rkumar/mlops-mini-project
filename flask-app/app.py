from flask import Flask, render_template, request
import mlflow
import os
import dagshub
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing_utility import normalize_text
import pickle

app = Flask(__name__)

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT") # ← FIXED: proper variable name
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "092914rkumar"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token



dagshub_url = "https://dagshub.com"
repo_owner = "092914rkumar"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))
# load model from model registry

model_name = "my_model"
model_version = 2

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result=result[0])

app.run(debug=True)

