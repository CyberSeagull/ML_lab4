import requests
from io import BytesIO
from joblib import load


def download_model(url):
    response = requests.get(url)
    model_file = BytesIO(response.content)
    model = load(model_file)
    return model


def preprocess_input(text):
    processed_text = text.preprocess_data
    return processed_text


def predict_text(text, model):
    processed_text = preprocess_input(text)
    prediction = model.predict([processed_text])
    return prediction


model_url = (
    'https://github.com/CyberSeagull/ML_lab4/blob/main/src/models/'
    'random_forest_model_weights.joblib'
)


rand_forest_model = download_model(model_url)


text = input("Enter your text: ")
prediction = predict_text(text, rand_forest_model)
print("Prediction:", prediction)
