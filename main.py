import data_preprocessing
import correlation

if __name__ == '__main__':
    data_preprocessing.clean_stocks()
    data_preprocessing.clean_tweets_keep_stopwords()
    data_preprocessing.clean_tweets_remove_stopwords()