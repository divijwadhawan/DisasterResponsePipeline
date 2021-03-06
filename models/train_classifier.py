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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
import time

database_filepath = '../data/DisasterResponse.db'

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    '''This function loads data from sqlite database and returns 3 calues:
    1. X - This is a dataframe containing disaster messages
    2. Y - This is dataframe containing 36 categories
    3. categoriy_names - This is a list of categories of Y'''
    # load data from database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterTable', conn)

    X = df[['message']]
    Y = df[df.columns.difference(['id', 'message', 'original', 'genre'])]
    category_names = list(Y.columns.values)
    
    return X,Y,category_names


def tokenize(text):
    '''Tokenize Funtion -
    1. Removes Punctuation from text
    2. Tokenizes the text
    3. Removes Stop words from english language
    4. Lemmatizes tokens
    5. Returns clean tokens'''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # define stop words
    stop_words = stopwords.words('english')
    
    # lemmatize and remove stop words
    #tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' This funtion uses a Decision Tree Classifer and builds a pipeline using tokenizer, 
    TfidTransformer and MultioutputClassifier'''
    #forest = RandomForestClassifier(n_estimators=10, random_state=1)
    #svc = SVC(gamma="scale")

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf' , MultiOutputClassifier(tree.DecisionTreeClassifier()))
    ])

    print("pipeline parameters")

    print(pipeline.get_params())

    # specify parameters for grid search
    parameters = {
##        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)
##        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
##        'features__text_pipeline__tfidf__use_idf': (True, False)
##        'clf__n_estimators': [5, 10, 15],
##        'clf__min_samples_split': [2, 3, 4],
##        'features__transformer_weights': (
##            {'text_pipeline': 1, 'starting_verb': 0.5},
##            {'text_pipeline': 0.5, 'starting_verb': 1},
##            {'text_pipeline': 0.8, 'starting_verb': 1},
##        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv
    
def save_model(model, model_filepath):
    '''Saves the model to model_filepath provided'''
    pickle.dump(model, open(model_filepath + "/classifier.pkl", 'wb'))
    pass

def display_results(y_test, y_pred):
    '''Scores the model for each of the 36 categories'''
    for category in y_test.columns.values:
        print('**Processing {} comments...**'.format(category))
    
        # calculating test accuracy
        print("Precision            Recall               fScore")
        print(precision_recall_fscore_support(y_test[category], y_pred[category], average='weighted'))
        print("\n")

def main():
    start = time.time()

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)

        print(X_train)
        print(Y_train)
        
        print('Building model...')
        model = build_model()

        
        print('Training model...')
        # train classifier
        model.fit(X_train, Y_train)

        print('Predicting test data...')
        prediction = model.predict(X_test)

    
        # calculating test accuracy
        print('Evaluating model...')
        display_results(Y_test, pd.DataFrame(data=prediction, columns=Y.columns.values))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    
    end = time. time()
    print(end - start)
    sys.exit()


if __name__ == '__main__':
    main()