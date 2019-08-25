# Disaster Response Pipeline Project

### Introduction and Motivation
I did this project as a part of the Udacity Data Scientist Nanodegree. It consists of an ETL pipeline, a machine learning pipeline and 
a web app. 

The overall idea of this project is to use the ETL and the machine learning pipelines to lay the foundation for a web app which allows users to enter messages which are then classified.

### Overview

The ETL pipeline (contained in process_data.py) reads in 2 CSV files, one containing messages and another containing category labels for these messages, creates dummies for each category and saves the resulting data in an SQLite database.

The machine learning pipeline (contained in train_classifier.py) takes an SQLite database with messages and category labels. First, the messages are preprocessed using a simple NLP pipeline, then a machine learning pipeline is initialised, optimised and trained using GridSearchCV. Finally, the optimised model is saved as a pickle file. 

The web app shows two simple visualisations of the training data used and hence needs access to the database used for the training process. Users can enter messages in a form which will then be classified using the trained random forest classification algorithm.

### Modelling Process and General Considerations
I tried a number of different classification algorithms: Random Forests, Naive Bayes, AdaBoost (using a DecisionTreeClassifier as the base estimator). The Random Forest algorithm had an ever so slight edge over the other algorithms which were rather close to each other in terms of performance.

In the training process and the grid search I decided to use the F1 score rather than accuracy to account for imbalances in the data. Given the rather large number of categories I decided to only do this instead of perhaps also dealing with this matter by using techniques such as oversampling.

While the overall performance of the model was rather good, extending the scope of the grid search could potentially further improve model performance. Unfortunately, saving the model as a pickle file prevented me from using more than one thread for the grid search. I tried some workaround mentioned in the Udacity knowledge base which unfortunately did not work either. Hence, despite the rather limited scope of the grid search it took about 5 hours to complete the grid search and train the model using an Intel Core i5-6600k. 

### Required Packages
I used Python 3.6.7 (Anaconda distribution) for this project. If you are using the Anaconda distribution of Python, you will only need to install the plotly package. On other Python distributions, you may also need to install Pandas, Numpy, SQLAlchemy and scikit-learn.

### Instructions (partly taken from Udacity):
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Please note that the pickle file of the model was rather large (283 MB) despite a compression setting of 5. I thus had to upload it using GIT LFS.

### Acknowledgements and Credits

This project is primarily based on what I have learned in the Udacity Data Scientist Nanodegree. Parts of the code were supplied by Udacity as a template. Furthermore, credit has to be given to Figure Eight for providing the data used in this project.

### License

This project is licensed under the GNU General Public License Version 3.
