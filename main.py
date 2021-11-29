import data_preprocessing
import correlation

if __name__ == '__main__':
    # data_preprocessing.clean_stocks()
    tw_with_stopwords = data_preprocessing.clean_tweets_keep_stopwords()
    print(tw_with_stopwords.shape)