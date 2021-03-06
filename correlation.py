from IPython.core.display import display
import pandas as pd
import numpy as np
import scipy.stats
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

def cramers_corrected_stat(confusion_matrix):
  """ calculate Cramers V statistic for categorial-categorial association.
      uses correction from Bergsma and Wicher, 
      Journal of the Korean Statistical Society 42 (2013): 323-328
      
      Code was taken from https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix 
  """
  chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
  n = confusion_matrix.sum().sum()
  phi2 = chi2/n
  r,k = confusion_matrix.shape
  phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
  rcorr = r - ((r-1)**2)/(n-1)
  kcorr = k - ((k-1)**2)/(n-1)
  return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def stocks_changer(dataframe, p, delay):
  """
  creates stocks dataframe with categorical values depending on:
  p - gives threshold for 0 value in stocks change indicator
  delay - after how many days do stocks change
  """
  res = dataframe.copy()
  direction = []
  for j in range(0, dataframe.shape[0]-delay):
    sub = dataframe['open'][j+delay] - dataframe['open'][j]
    if np.abs(sub) >= p:
      direction.append(1 if dataframe['open'][j+delay] >= dataframe['open'][j] else -1)
    else:
      direction.append(0)
  for i in range(0,delay):
    direction.append(0)
  res['change'] = direction
  return res

def find_correlation(df, cluster_methods):
  """
  find Cramer's V coefficient for all the methods and parameter combinations
  
  return dataframe consisting of all these params and final correlation value
  """
  percents = [0,0.5,1,1.5,2] # percent change
  delays = [1,2,3] #lag in days
  types = ['textblob_sentiment', 'bertweet_sentiment', 'distilbert_sentiment']
  corr_coef = pd.DataFrame(columns = ['method', 'cluster','type', 'cramer', 'percent_change', 'delay'])
  methods = cluster_methods
  for m in methods:
    clusters = df[m].unique()
    for c in clusters:
        clusterized = df.loc[df[m] == c].reset_index()
        for t in types:
            if t == 'distilbert_sentiment': #it has only -1 and 1 values, therefore needs different condition
              for d in delays:
                changed_stocks = stocks_changer(clusterized, 0, d)
                confusion_matrix = pd.crosstab(changed_stocks[t], changed_stocks.change)
                cramer = round(cramers_corrected_stat(confusion_matrix), 5)
                corr_coef = corr_coef.append(pd.Series([m, c, t, cramer, 0, d], index = corr_coef.columns), ignore_index=True)
            else:
              for d in delays:
                for p in percents:
                    changed_stocks = stocks_changer(clusterized, p, d)
                    confusion_matrix = pd.crosstab(changed_stocks[t], changed_stocks.change)
                    cramer = round(cramers_corrected_stat(confusion_matrix), 5)
                    corr_coef = corr_coef.append(pd.Series([m, c, t, cramer, p, d], index = corr_coef.columns), ignore_index=True)
  return corr_coef


def generate_wordcloud(df, method, cluster):
  """
  utilizes wordcloud library to generate the cloud of the most 
  popular words in the cluster

  max_number of words is 50
  """
  text = df.loc[df[method] == cluster].tweet.str.cat(sep = ' ')
  wordcloud = WordCloud(max_font_size=100, max_words=50, background_color="white").generate(text)
  print(wordcloud.words_.keys())
  wordcloud.to_file("wordclouds/"+str(method)+"_"+str(cluster)+".png")
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()