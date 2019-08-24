import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

# Class definitions from train_classifier.py needed for
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

# Replaced original function provided in this script with function used in
# train_classifier.py
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MsgCategoriesTable', con=engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visualisations for the number of category
    # assigned to a single message and the 10 most frequent categories
    category_df = df.drop(columns=['id','message','genre','original'])
    category_df['related'].replace(2,1, inplace=True)
    # Convert all Y columns to integers
    category_df = category_df.apply(pd.to_numeric)
    category_counts = category_df.sum(axis=1).value_counts(ascending = False) 
    category_frequency = category_df.sum(axis=0).sort_values(ascending = False)
    
    
    
    # create visuals
    
    graph_one = []
    graph_one.append(
          Bar(
          x = category_counts.index.tolist(),
          y = category_counts.values.tolist(),
          )
      )
    layout_one = dict(title = 'Number of Categories Assigned to Messages',
                xaxis = dict(title = 'Number of Categories'),
                yaxis = dict(title = 'Number of Observations'),
                )
    
    graph_two = []
    graph_two.append(
          Bar(
          x = category_frequency.index.tolist(),
          y = category_frequency.values.tolist(),
          )
      )
    
    layout_two = dict(title = 'Frequencies of Different Categories',
                xaxis = dict(title = 'Message Category'),
                yaxis = dict(title = 'Number of Observations'),
                )
    
    graphs = []
    graphs.append(dict(data = graph_one, layout = layout_one))
    graphs.append(dict(data=graph_two, layout = layout_two))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()