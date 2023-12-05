import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier


def eval():
    current_path = os.path.dirname(__file__)
  
    model_path = os.path.join(current_path,'model/random_forest_model.pkl')
    cv_path = os.path.join(current_path,'model/count_vectorizer.pkl')

    with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

    with open(cv_path, 'rb') as cv_file:
        cv = pickle.load(cv_file)

    eval_file = os.path.join(current_path, 'data/test_data_processed.json')
    df = pd.read_json(eval_file, lines=True)

    X = df['doc_text']
    y = df['label']

    X_cv = cv.transform(X)
    predictions = model.predict(X_cv)

    target_names = ['lighting', 'not_lighting']
    test_cm = pd.DataFrame(confusion_matrix(y,predictions), index=target_names, columns=target_names)
    print("\nTest confusion matrix : ")
    print(test_cm)

    print(f"Evaluation completed.")
    test_accuracy = accuracy_score(y, predictions)
    print(f"\nTest accuracy : {round(test_accuracy*100, 3)}%")


if __name__ == "__main__":
     eval()



