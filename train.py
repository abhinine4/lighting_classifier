#  Logistic regression training
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train():
    current_path = os.path.dirname(__file__)

    train_file = os.path.join(current_path, 'data/train_data_processed.json')
    df = pd.read_json(train_file, lines=True)

    X = df['doc_text']
    y = df['label']

    # using 80-20 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=123)

    print('\nTraining Data :', X_train.shape)
    print('Validation Data : ', X_val.shape)

    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    
    ####
    # cls = RandomForestClassifier()
    # cls = XGBClassifier()
    cls = SVC()
    ####

    # RandomForest params
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    #     }

    #  xgboost params
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [3, 5, 7],
    #     'min_child_weight': [1, 3, 5],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0],
    # }

    # SVM params
    param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': [0.1, 1, 'scale', 'auto'],
        }

    grid_search = GridSearchCV(cls, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train_cv, y_train)

    best_params = grid_search.best_params_
    
    # classifier = RandomForestClassifier(**best_params)
    # classifier = XGBClassifier(**best_params)
    # classifier = LogisticRegression(penalty='l2', max_iter=10000)
    classifier = SVC(**best_params)

    classifier.fit(X_train_cv, y_train)

    X_val_cv = cv.transform(X_val)

    predictions = classifier.predict(X_val_cv)

    accuracy = accuracy_score(y_val, predictions)
    print(f"\nTrain accuracy : {round(accuracy*100, 2)}%")

    target_names = ['lighting', 'not_lighting']

    cr = classification_report(y_val, predictions, target_names=target_names)
    print("\nTrain classification report : ")
    print(cr)

    
    cm = pd.DataFrame(confusion_matrix(y_val,predictions), index=target_names, columns=target_names)
    print("\nConfusion matrix : ")
    print(cm)

    with open(current_path + '/model/count_vectorizer.pkl', 'wb') as cv_file:
        pickle.dump(cv, cv_file)

    with open(current_path + '/model/svm_model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)

    print("\nEvaluating on test set")

    test_file = os.path.join(os.path.dirname(__file__), 'data/test_data_processed.json')
    test_df = pd.read_json(test_file, lines=True)

    X_test = test_df['doc_text']
    y_test = test_df['label']

    X_test_cv = cv.transform(X_test)

    test_predictions = classifier.predict(X_test_cv)
    
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\nTest accuracy : {round(test_accuracy*100, 3)}%")
    
    test_cm = pd.DataFrame(confusion_matrix(y_test,test_predictions), index=target_names, columns=target_names)
    print("\nTest confusion matrix : ")
    print(test_cm)

    print(f"Training completed. Models saved as {current_path}/model/")

if __name__ == "__main__":
    train()
    


