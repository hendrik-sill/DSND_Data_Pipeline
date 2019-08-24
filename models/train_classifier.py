import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

# Before defining functions define some classes to be subsequently used in
# the machine learning pipeline to create additional features

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    ''' This class is used in the machine learning pipeline
        to identify the length of the different messages.
    '''

    def text_length(self, text):
        tokenized = tokenize(text)
        length = len(tokenized)
        return length

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.text_length)
        return pd.DataFrame(X_tagged)
    
class ContainsPlace(BaseEstimator, TransformerMixin):
    ''' This class is used in the machine learning pipeline
        to identify using part of spech whether a given message contains 
        a reference to at least one place.
    '''
    
    def contains_place(self, text):
        tokenised = tokenize(text)
        # Using approach from 
        # https://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list
        # to access tags inside part of speech tagging
        for chunk in nltk.ne_chunk(nltk.pos_tag(tokenised)):
            if hasattr(chunk, 'label'):
                if chunk.label() == 'GPE':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.contains_place)
        return pd.DataFrame(X_tagged)

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
    # Convert all Y columns to ints
    Y = Y.apply(pd.to_numeric)
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
    ''' Creates a machine learning pipeline which is then nested into a grid
        search cross validation object
    '''
    # Define a machine learning pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
            ])),
            ('text_length', TextLengthExtractor()),
            ('contains_place', ContainsPlace())
        ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=64)))
    ])
    
    #Define parameters to search for in grid search
    parameters = {
    'features__text_pipeline__vect__max_df':(0.75, 1.0),
    'features__text_pipeline__vect__max_features':(None, 5000),
    'clf__estimator__n_estimators':[100,200],
    'clf__estimator__min_samples_split':[3, 4]
    }
    # Define grid search object using pipeline and parameters
    # Use micro average (to deal with class imbalances) of f1_score 
    # for scoring as suggested at https://knowledge.udacity.com/questions/20810
    cv = GridSearchCV(pipeline, param_grid = parameters,verbose=2, cv = 3,
                      score = make_scorer(f1_score, average = 'micro')) 
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Uses trained multioutput model/ pipeline and evaluates it based on 
       test data. Metrics are printed.
    INPUT
        model - trained sklearn model/ pipeline 
        X_test - test data features
        Y_test - test data labels
        category_names - names of categories used in model/ pipeline
    OUTPUT
        None
    '''
    # Obtain predictions for test data 
    Y_pred = model.predict(X_test)

    # Evaluate test data against labels
    for i, col in enumerate(category_names):
        print('Feature: {}'.format(col))
    
        col_true = list(Y_test.values[:,i])
        col_pred = list(Y_pred[:, i])
        print(classification_report(col_true, col_pred))


def save_model(model, model_filepath):
    '''Takes a trained sklearn model and saves it as a pickle file at the 
       specified filepath.
    INPUT
        model - trained sklearn model
        model_filepath - filepath where the model will be saved
    OUTPUT
        None
    '''
    # Save model as explained at 
    # https://machinelearningmastery.com/save-load-machine-learning-models
    #-python-scikit-learn/
    # Use compression to prevent pickle from becoming too large
    joblib.dump(model, model_filepath, compress = 5)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(X.dtypes)
        print(Y.dtypes)
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