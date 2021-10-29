import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''This function is used to load messages and categories files from their
    filepath to dataframes'''

    # load messages dataset
    messages = pd.read_csv("disaster_messages.csv")
    # load categories dataset
    categories = pd.read_csv("disaster_categories.csv")
    #merging both datasets into a single dataframe
    df = pd.merge(messages,categories)
    df.head()
    return df


def clean_data(df):
    '''this funtion inputs merged dataframe of messages and categories and does the following steps :
    1. converts merged column of 36 categories into 36 different columns
    2. cleans data to extract value for each row and column
    3. drops duplicate rows in the dataframe df'''
    #Spliting categories column into 36 individual columns and saving it as a seprate dataframe
    df_categories = df['categories'].str.split(';',expand=True)
    
    #categories is a list of column names with a not needed character at the end eg: 'related-1' , 'request-0' and so on
    #this character needs cleaning
    df_categories.columns = df_categories.iloc[0]
    
    #a new list of cleaned column names
    clean_category_colnames = []
    
    #adding cleaned column name to empty list 
    for column in df_categories:
        clean_category_colnames.append(column[0:-2])

    #replacing columns of categories with cleaned values
    df_categories.columns = clean_category_colnames

    #We have value of each column in the last character of the string that we need to extract and convert it to numeric
    for column in df_categories:
    # set each value to be the last character of the string
        df_categories[column] = df_categories[column].str[-1:]
    
        #convert column from string to numeric
        df_categories[column] = df_categories[column].astype(int)

    #dropping original categories column which we already split and saved into a new dataframe df_categories
    df = df.drop(columns=['categories'])
    
    #merging original dataframe after dropping 'categories' column with the dataframe df_categories which has split 36 values
    #axis=1 means we concat it column wise
    df = pd.concat([df, df_categories], axis=1)
    df.head()

    #droping duplicate rows from the dataframe
    df = df.drop_duplicates()


    return df


def save_data(df, database_filename):
    '''Saves df into database_filename as an sqlite file'''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterTable', engine, index=False)  


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