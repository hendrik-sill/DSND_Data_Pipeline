import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_data(database_filepath):
    '''Loads data from specified database and returns
       dataframes containing the category dummy columns, 
       i.e. the target variables, (Y) and the messages (X) as well as a list 
       with category names
    INPUT
        database_filepath - a path to the database file to be used
    OUTPUT
        X - pandas dataframe containing the messages
        Y - pandas dataframe containing category dummy variables
        category_names - a list of category names
    '''
    # Connect to SQLite database at specified location
    # and save the MsgCategoriesTable in a dataframe
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MsgCategoriesTable', con=engine)
    # Split data into X and Y dataframes
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'genre', 'original'])
    # Take care of instances of category related containing a value of 2 
    # and replace them with  1
    Y['related'].replace(2,1, inplace=True)
    # Obtain categories names from Y column names
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
    ''' Takes in a string, replaces URLs, normalises, tokenises and lematises 
        it. Stop words are also taken care of. The tokenised string is then
        returned.
    INPUT
        text - a string
    OUTPUT
        token - the tokenised and cleaned string
    '''
    # Define a regex to find URLS contained in messages
    # Approach as taught in Udacity Data Engineering class
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    text = re.sub(r'[^a-zA-Z0-9]',' ', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # lemmatize and remove stop words by excluding words contained in stop_words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    return tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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