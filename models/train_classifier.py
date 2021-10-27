import sys

from sqlalchemy import create_engine
import sqlite3

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

import re, pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

database_filepath = '../data/DisasterResponse.db'


def load_data(database_filepath):
    # load data from database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterTable', conn)

    X = df[['message']]
    y = df[df.columns.difference(['id', 'message', 'original', 'genre'])]
    category_names = list(y.columns.values)
    
    return X,y,category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    
    # lemmatize and remove stop words
    #tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf' , MultiOutputClassifier(forest))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    m = MultiLabelBinarizer().fit(Y_test.to_numpy())
    print(precision_recall_fscore_support(m.transform(Y_test.to_numpy()), m.transform(model.predict(X_test)), average='weighted'))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath + "/classifier.pkl", 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        # train classifier
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()