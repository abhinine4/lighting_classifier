import fitz
import json
import os
import requests
import pickle
import nltk
import re
import argparse
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier


def chek_nltk():
    nltk_path = nltk.data.path[0]
    stopword_path = os.path.join(nltk_path, 'corpora/stopwords.zip')
    wordnet_path = os.path.join(nltk_path, 'corpora/wordnet.zip')
    if not os.path.exists(stopword_path):
        nltk.download('stopwords')

    if not os.path.exists(wordnet_path):
        nltk.download('wordnet')

class Light():
    def __init__(self, model_path, cv_path):
        self.model, self.cv = self.load_models(model_path, cv_path)

    def load_models(self, model_p, cv_p):
        with open(model_p, 'rb') as model_file:
            lr_model = pickle.load(model_file)

        with open(cv_p, 'rb') as cv_file:
            cv = pickle.load(cv_file)
        return lr_model, cv

    def text_extractor(self, file_path):
        path = file_path

        if file_path.startswith(('http://', 'https://')):
            response = requests.get(file_path)
            print(f'HTTP response status code: {response.status_code}')
            if response.status_code == 200:
                d_filepath = os.path.join(os.getcwd(),'temp_data/temp.pdf')
                with open(d_filepath, 'wb') as pdf_object:
                    pdf_object.write(response.content)
                    print('file was successfully downloaded!')
                path = d_filepath
            else:
                return None
        
        doc = fitz.open(path)
        extracted_text = [page.get_text() for page in doc]
        text = ''
        for page in extracted_text:
            text += '\n\nPAGE' + page 
        doc.close() 
        return text

    def process_text_inf(self, doc_text):
        text = re.sub('[^a-zA-Z]', ' ', doc_text)
        text = text.lower()
        text = text.split()
        text = [word for word in text if word not in stopwords.words('english')]
        text = [lemmatizer.lemmatize(word) for word in text]
        text = ' '.join(text)
        return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)

    current_path = os.path.dirname(__file__)

    # model_path = os.path.join(os.getcwd(),'model/logistic_regression_model.pkl')
    model_path = os.path.join(current_path,'model/random_forest_model.pkl')
    cv_path = os.path.join(current_path,'model/count_vectorizer.pkl')

    chek_nltk()
    lemmatizer = WordNetLemmatizer()

    light = Light(model_path, cv_path)

    args = parser.parse_args()
    pdf_path = args.filepath

    doc_text = light.text_extractor(pdf_path)
    if doc_text:
        processed_text = light.process_text_inf(doc_text)
        text_cv = light.cv.transform(np.array([processed_text]))
        pred = light.model.predict(text_cv)
        probs = light.model.predict_proba(text_cv)[0]

        label = "lighting product" if pred[0] else "not lighting"
        score = probs[1] if pred[0] else probs[0]
        result = {
            'output' : pred[0],
            'label' : label,
            'score' : score 
        }
        print(result)

    else:
        print("Could not download pdf file from URL. Processing stopped.\n Try again or provide local path.")
