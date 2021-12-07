
from IPython.display import display
import pandas as pd

if __name__ == "__main__":
    print("\nPLEASE INPUT ONE OF THE FOLLOWING COMMANDS TO VIEW THE .CSV FILES: \n")
    folder = "files/"
    file_names = {
    "stocks":"tesla_stocks.csv",
    "tweets":"musk_tweets.csv",
    "clean stocks": "clean_stocks.csv",
    "no stopword tweets": "clean_tweets_without_stopwords.csv",
    "with stopword tweets": "clean_tweets_with_stopwords.csv",
    "no stopword clusters": "clusterized_tweets_without_stopwords.csv",
    "with stopword clusters": "clusterized_tweets_with_stopwords.csv",
    "no stopword sentiments": "sentimented_tweets_without_stopwords.csv",
    "with stopword sentiments": "sentimented_tweets_with_stopwords.csv",
    "final results": "cramer_values.csv"}

    print(list(file_names.keys()), "\n")
    print("YOU MIGHT BE INTERESTED IN THE COMMANDS \n\n'stocks' and 'tweets' to see the input files \n'final results' to see the output file \n")
    while(True):
        print("Type 'q' to exit \n")
        command = input("Command: ")
        if command == 'q':
            break
        if command in file_names.keys():
            df = pd.read_csv(folder + file_names[command])
            print("\nThe description of the "+str(file_names[command])+" file")
            display(df.describe())
            print("\nThe first 5 rows of the "+str(file_names[command])+" file")
            display(df.head())
        else:
            print("No such command")