import pandas as pd
import numpy as np
import datetime
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('stopwords')
nltk.download('punkt')

folder = "files"


def clean_stocks():
    """
    The function below cleans stocks and fills 
    out the missing data using the method defined in paper
    """
    stocks = pd.read_csv(folder+"/tesla_stocks.csv")
    stocks = stocks.rename(columns = str.lower)
    stocks = stocks.rename(columns = {'adj close':'adj'})

    stocks.date = pd.to_datetime(stocks.date)

    len = stocks.date.shape[0]
    missing = pd.DataFrame(columns=stocks.columns)
    for i in range(len-1):
        dif = stocks.date[i+1] - stocks.date[i]
        if dif > datetime.timedelta(days = 1):
            x_vals = stocks.iloc[i]
            y_vals = stocks.iloc[i+1]
            for j in range(1, int(dif.days)):
                date = x_vals[0] + datetime.timedelta(days = 1)
                open = (y_vals[1] + x_vals[1])/2
                high = (y_vals[2] + x_vals[2])/2
                low = (y_vals[3] + x_vals[3])/2
                close = (y_vals[4] + x_vals[4])/2
                adj = (y_vals[5] + x_vals[5])/2
                new_row = {'date': date, 'open' : open, 'high': high, 'low' :low, 'close' : close, 'adj': adj, 'volume' :0}
                x_vals = list(new_row.values())
                missing = missing.append(new_row, ignore_index=True)
    stocks = stocks.append(missing).sort_values("date").reset_index(drop = True)
    stocks = stocks.iloc[:,:-1]

    stocks = cut_dates(stocks)
    stocks = stocks.reset_index(drop = True)
    
    stocks.to_csv(folder+"/clean_stocks.csv", index = False)
    return stocks


def clean(text):
    """Helper function that removes unnecessary symbols in tweets, but keeps stopwords"""
    text = re.sub(r"http\S+", " ", text) # remove urls
    text = re.sub(r"RT ", " ", text) # remove rt
    text = re.sub(r"[^a-zA-Z\'\.\,\d\s]", " ", text) # remove special character except # @ . ,
    text = re.sub(r"[0-9]", " ", text) # remove number
    text = re.sub(r'\t', ' ', text) # remove tabs
    text = re.sub(r'\n', ' ', text) # remove line jump
    text = re.sub(r"\s+", " ", text) # remove extra white space
    text = text.strip()
    return text

def clean_remove_stopwords(text):
    """Helper function that removes unnecessary symbols and stopwords in tweets"""
    text = text.replace('\n',' ') #cleaning newline “\n” from the tweets
    text = re.sub(r'(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+', '', text)
    sw = stopwords.words('english') #you can adjust the language as you desire
    sw.remove('not')
    text = word_tokenize(text)
    text = [word for word in text if not word in sw]
    return ' '.join(text)

def clean_tweets_keep_stopwords():
    """tweets cleaning function that keeps stopwords"""
    tweets = pd.read_csv(folder + '/musk_tweets.csv')
    tweets['tweet'] = tweets['tweet'].map(lambda a: clean(a))
    tweets = tweets.drop(['Unnamed: 0', 'id', 'conversation_id', 'created_at', 
    'timezone', 'place', 'user_id','user_id_str', 
    'username', 'name', 'day', 'hour', 'urls', 'photos', 
    'video' , 'thumbnail', 'quote_url', 'search', 'near', 
    'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 
    'retweet_date', 'translate', 'trans_src', 'trans_dest'], axis=1)
    nan_value = float("NaN")
    tweets['tweet'].replace("", nan_value, inplace=True)
    tweets.dropna(subset = ["tweet"], inplace=True)
    tweets.to_csv(folder + "/clean_tweets_with_stopwords.csv", index = False)
    return tweets

def clean_tweets_remove_stopwords():
    """tweets cleaning function that removes stopwords"""
    tweets = pd.read_csv(folder + '/musk_tweets.csv')
    tweets['tweet'] = tweets['tweet'].map(lambda a: clean_remove_stopwords(a))
    tweets = tweets.drop(['Unnamed: 0', 'id', 'conversation_id', 'created_at', 
    'timezone', 'place', 'user_id','user_id_str', 
    'username', 'name', 'day', 'hour', 'urls', 'photos', 
    'video' , 'thumbnail', 'quote_url', 'search', 'near', 
    'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 
    'retweet_date', 'translate', 'trans_src', 'trans_dest'], axis=1)
    nan_value = float("NaN")
    tweets['tweet'].replace("", nan_value, inplace=True)
    tweets.dropna(subset = ["tweet"], inplace=True)
    tweets.to_csv(folder + "/clean_tweets_without_stopwords.csv", index = False)
    return tweets


def cut_dates(df):
    """This function cuts dates based on the datasets limits for the date"""
    startdate = pd.to_datetime('2011-12-01')
    enddate = pd.to_datetime('2021-03-22')
    df = df.loc[(df.date >= startdate) & (df.date <= enddate)].sort_values('date').reset_index(drop = True)
    return df

def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]