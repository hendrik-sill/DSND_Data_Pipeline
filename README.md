# Disaster Response Pipeline Project

### Required Packages
I used Python 3.6.7 (Anaconda distribution) for this project. If you are using the Anaconda distribution of Python, you will only need to install the plotly package. On other Python distributions, you may also need to install Pandas, Numpy, and scikit-learn.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
