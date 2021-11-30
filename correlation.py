import pandas as pd
import numpy as np
import scipy.stats

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def stocks_changer(dataframe, p):
  res = dataframe.copy()
  direction = []
  for j in range(0, dataframe.shape[0]-1):
    sub = dataframe['open'][j+1] - dataframe['open'][j]
    if np.abs(sub) >= p:
      direction.append(1 if dataframe['open'][j+1] >= dataframe['open'][j] else -1)
    else:
      direction.append(0)
  direction.append(0)
  res['change'] = direction
  return res

def find_correlation(df):
    percents = [0,0.5,1,1.5,2]
    types = ['sentiment', 'bertwitter_sentiment', 'hf_sentiment']
    corr_coef = pd.DataFrame(columns = ['cluster','type', 'cramer', 'percent_change'])
    clusters = [1]
    for c in clusters:
        clusterized = df.loc[df.cluster == c].reset_index()
        for t in types:
            if t == 'hf_sentiment':
                changed_stocks = stocks_changer(clusterized, 0)
                confusion_matrix = pd.crosstab(changed_stocks[t], changed_stocks.change)
                cramer = round(cramers_corrected_stat(confusion_matrix), 5)
                corr_coef = corr_coef.append(pd.Series([c, t, cramer, 0], index = corr_coef.columns), ignore_index=True)
            else:
                for p in percents:
                    changed_stocks = stocks_changer(clusterized, p)
                    confusion_matrix = pd.crosstab(changed_stocks[t], changed_stocks.change)
                    cramer = round(cramers_corrected_stat(confusion_matrix), 5)
                    corr_coef = corr_coef.append(pd.Series([c, t, cramer, p], index = corr_coef.columns), ignore_index=True)