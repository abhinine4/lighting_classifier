#  Text processing

import os
import fitz
import json
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk_path = nltk.data.path[0]
stopword_path = os.path.join(nltk_path, 'corpora/stopwords.zip')
wordnet_path = os.path.join(nltk_path, 'corpora/wordnet.zip')
if not os.path.exists(stopword_path):
    nltk.download('stopwords')

if not os.path.exists(wordnet_path):
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def process_text(df):
    text = list(df['doc_text'])
    corpus = []
    for i in range(len(text)):
        r = re.sub('[^a-zA-Z]', ' ', text[i])
        r = r.lower()
        r = r.split()
        r = [word for word in r if word not in stopwords.words('english')]
        r = [lemmatizer.lemmatize(word) for word in r]
        r = ' '.join(r)
        corpus.append(r)
    df['doc_text'] = corpus
    return df

def text_extractor(file_path, csv_path):
    df = pd.read_csv(csv_path)
    dir_list = os.listdir(file_path)
    data = []

    for file in dir_list:
        id = file[:-4]
        cls = df.loc[df['ID'] == id]['Is lighting product?'].item()

        if type(cls)== str:
            label = 0
            if cls == 'Yes':
                label = 1
        else:
            label = cls
        
        print(label)

        doc = fitz.open(file_path+'/'+file)
        extracted_text = [page.get_text() for page in doc]
        text = ''
        for page in extracted_text:
            text += '\n\nPAGE' + page 

        doc.close()
        data.append({
            "id" : id,
            "doc_text" : text,
            "label" : label
        })
    df = pd.DataFrame(data)
    processed_text = process_text(df)
    return processed_text

if __name__ == "__main__":
    parspec_data = ['train_data','test_data']
    currentPath = os.path.dirname(__file__)
    for data_type in parspec_data:
        file_path = os.path.join(currentPath, 'data', data_type)
        csv_path = os.path.join(currentPath, 'data', f'parspec_{data_type}.csv') 
        # print(file_path)
        # print(csv_path)
        df_data = text_extractor(file_path, csv_path)
        df_data.to_json(currentPath+f'/data/{data_type}_processed.json', orient = 'records', lines = 'true')
