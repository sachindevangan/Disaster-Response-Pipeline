import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Loading Data From Multiple Files. 
  
    Merging Them Together To Form A Single DataFrame And Returning It. 
  
    Parameters: 
    messages_filepath (str): Filepath Where Messages Are Stored.
    categories_filepath (str): Filepath Where Categories Are Stored.
    
    Returns: 
    merged_df (DataFrame): Combined DataFrame Object
  
    """

    # reading csv files using pandas
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merging above dataframes 
    merged_df = pd.merge(messages,categories)

    return merged_df

def clean_data(df):
    """ 
    Data Cleaning Process. 
  
    Takes In A Pandas DataFrame And Seperates The Categories Into Individual
    Category Columns And Then Merging Them Back To The DataFrame ,Finally 
    Returning New DataFrame. 
  
    Parameters: 
    df (DataFrame): Pandas DataFrame Object Containing The Data.
  
    Returns: 
    cleaned_df (DataFrame): Pandas DataFrame Object Containing The Cleaned Data.
  
    """

    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True)

    # selecting the first row of the categories dataframe to extract column names
    row = categories.loc[0]

    # creating a list of category column names
    category_colnames = [category[:len(category)-2] for category in row ]

    # renaming the columns of `categories` dataframe
    categories.columns = category_colnames

    # now converting category values to just numbers 0 or 1
    for column in categories:

    	# setting each value to be the last character of the string
    	categories[column] = categories[column].str[-1]
    
    	# converting column from string to numeric
    	categories[column] = pd.to_numeric(categories[column])

    # replacing categories column in df with new category columns.
    df = df.drop('categories',axis = 1)

    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)

    # removing duplicates
    cleaned_df = df.drop_duplicates(keep = 'first')
    
    return cleaned_df


def save_data(df, database_filename):
    """ 
    Save To Database Method. 
  
    Saving The Cleaned Pandas DataFrame Into A SQLite Database. 
  
    Parameters: 
    df (DataFrame): Pandas DataFrame Object Containing The Cleaned Data.
    database_filename (str) : Filename Of Database.
  
    """

    # formatting database name properly
    db_name = 'sqlite:///{}'.format(database_filename)

    # initiating SQLAlchemy Engine
    engine = create_engine(db_name)

    # using pandas to save the DataFrame to the database
    df.to_sql('DisasterResponse', engine, index=False)
 


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
