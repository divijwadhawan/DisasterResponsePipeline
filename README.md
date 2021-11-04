# NLP Multiclass Disaster Response Prediction Model with Flask App

1. This Project reads data from 2 different CSV files -
* disaster_messages.csv - Contains text data for messages recorded during time of disaster
* disaster_categories.csv - Contains categories classified for these text messages

2. Data is cleaned and stored in an sqlite database

3. Data is read from this sqlite database and converted into tokens

4. Data is split into train and test, then trainied using Decision Tree multiclass classifier

5. Results are displayed using a flask web app and user can also categorize his own text using this app

## Repository Structure
1.  app
* template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
* run.py  # Flask file that runs app

2. data
* disaster_categories.csv  # data to process 
* disaster_messages.csv  # data to process
* process_data.py
* InsertDatabaseName.db   # database to save clean data to

3. models
* train_classifier.py
* classifier.pkl  # saved model 

- README.md

## How to run

1. Step 1 is to create DisasterResponse.db which can be done executing process_data.py in data folder which can be done with arguments in this way -
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

2. Step 2 is to build and save the model which can be done this way -
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Please note that runtime is approximately 40 minutes due to finding best parameters using GridSearch.

3. Step 3 is to execute run.py flask app -

Open a new terminal window. You should already be in the workspace folder, but if not, then use terminal commands to navigate inside the folder with the run.py file.

Type in the command line:

* python run.py *

Your web app should now be running if there were no errors.

Now, open another Terminal Window.

Type

* env|grep WORK *

In a new web browser window, type in the following:

https://SPACEID-3001.SPACEDOMAIN

## Getting Started

### Dependencies

* Windows, Mac, Linux
* Latest Python Release
* Libraries - sqlalchemy, sqlite3, nltk, re, pickle, pandas, numpy, sklearn (all these libraries can be installed using pip command)

### Installing

* This is public git project which can be downloaded or cloned
* Before executing make sure dependencies are installed and necessary files are downloaded from https://www.kaggle.com/airbnb/seattle
* Please put the files in a folder called "archive" or if you choose a different folder please change the path where files are being read in the code

## Help

If you need help or facing issues, you can write to wwdivij@gmail.com

## Authors

[@divijwadhawan](https://github.com/divijwadhawan)

## Version History

* 0.1
    * Initial Release

## Acknowledgments

* [Udacity](https://classroom.udacity.com/)
