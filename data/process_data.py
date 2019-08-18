# Import all necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from location specified by user at command line
    INPUT
        messages_filepath - file path to messages provided at command line
        categories_filepath - 
    OUTPUT
        df - a dataframe containing messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    Cleans the dataframe containing the messages and categories data
    INPUT
        df - a pandas dataframe containing the merged messages and
        categories data read in from CSV filees using the load_data function
    OUTPUT
        clean_df - a pandas dataframe containing the cleaned data with
        a dummy column for each message type
    '''
    clean_df = df.copy()
    # Create dataframe with columns for each message type
    categories = clean_df['categories'].str.split(pat=';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # Use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Iterate through all categories columns to generate dummy columns for each
    # one
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # Convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    # Drop the original categories column from `df`
    clean_df = clean_df.drop(columns=['categories'])
    # Concatenate the original dataframe with the new `categories` dataframe
    clean_df = pd.concat([clean_df, categories], axis = 1)
    # Eliminate duplicates
    clean_df = clean_df.drop_duplicates()
    return clean_df

def save_data(df, database_filename):
    '''
    Takes a dataframe containing cleaned data and saves the content in a local
    SQLite database at the filepath passed into the function
    INPUT
        df - a pandas dataframe containing the cleaned data to be read into
        a SQLite database
        database_filename - filepath of the SQLite database to be generated
        from the cleaned dataframee
    OUTPUT
        None
    '''
    # Create an engine and save contents of dataframe
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MsgCategoriesTable', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()