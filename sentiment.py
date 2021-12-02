from textblob import TextBlob
import numpy as np
import pandas as pd
import data_preprocessing
from collections import Counter
from IPython.display import display
import matplotlib.pyplot as plt 
from matplotlib.pyplot import pie

def getPolarity(text):
   return  TextBlob(text).sentiment.polarity


def getSentiment(score):
    if score < 0:
        return -1
    elif score == 0:
        return 0
    else:
        return 1

#Get top-10 words from each cluster  
def top_ten_words(tweets, clusters):
    #Cleaning
    f0 = lambda x: data_preprocessing.remove_stopword(x)
    f1 = lambda x: x.replace('.', '')
    f2 = lambda x: x.replace(',', '')
    f3 = lambda x: str(x).lower().split()
    tweets = map(f1, tweets)
    tweets = map(f2, tweets)
    tweets = map(f3, tweets)
    tweets = map(f0, tweets)

    d = {'Tweet': tweets, 'Cluster': clusters}
    df = pd.DataFrame(d)
    summary = []
    for cluster in np.unique(clusters):
        top = Counter([item for sublist in  df[(df['Cluster'] == cluster)]['Tweet'] for item in sublist])
        temp = pd.DataFrame(top.most_common(10))
        temp.columns = ['Common_words','count']
        summary.append([cluster, df[(df['Cluster'] == cluster)].shape[0], list(temp['Common_words']), list(temp['count'])])
        temp.style.background_gradient(cmap='Blues')
    
    summary_df = pd.DataFrame (summary, columns = ['Cluster', 'Size', 'Common words', 'Counts'])
    summary_df.style.set_properties(subset=['Common Words', 'Counts'], **{'width': '300px'})
    display(summary_df)

def show_semantic_pie(df, method, with_or_without):
    fig = plt.figure(figsize=(7,7))

    fig.patch.set_alpha(1.0)
    sums = df[method].value_counts()
    labels = list(sums.index.values)

    #Convert numerical to categorical labels for visualisation
    d = {0: 'Neutral',-1: 'Negative',1: 'Positive'}
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = d[1]
        elif labels[i] == -1:
            labels[i] = d[-1]
        else:
            labels[i] = d[0]

    pie(sums, labels=labels, autopct='%.1f%%', textprops={'fontsize': 15})

    ax1 = plt.title(method + ': Sentimental analysis of Elon Musk\'s Tweet -' + with_or_without,
            fontsize = 18,
            fontweight = 'heavy',
            loc='center', 
            pad=25)