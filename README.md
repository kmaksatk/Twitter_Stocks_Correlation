<div align="center">

# Clustering and Sentiment Analysis: How Elon Musk Controls the Tesla Stock Prices Using Twitter
<h3 align="center"> ML701 - Machine Learning </h3>
  
</div>

## Description

<p float="center">
  <img src="https://qph.fs.quoracdn.net/main-qimg-c25657afa1b0c6fd10d2e453ef1e114f", width = 49%, height = 250px>
   <img src="https://cdn.wccftech.com/wp-content/uploads/2020/05/TESLA-STOCK-PRICE-11-51-AM-ET-1-MAY-2020-1480x888.png", width = 49%, height = 250px>
</p>
“Tesla’s stock price is too high imo” - this tweet was posted on the 1st May of 2020 by Elon Musk. On the same day, Tesla's stock price had a significant drop by ten percent. The link between Twitter and Stocks markets has been explored before. Most of them dealt with the crowd's opinion and tweets were either labeled manually or selected with certain conditions. The problem with such an approach is that it doesn't consider the whole scale of tweets, and possibly overfits in some cases. Meanwhile, our project focuses on the entire history of Elon Musk's Twitter and applies unsupervised techniques for identifying Tesla-related tweets. Our agenda is to cluster the tweets using unsupervised learning methods to find the Tesla-related tweets, then find their sentiments and finally get the correlation with stocks. Ultimately, we want to make a pipeline that can identify the topic of the new tweet, its sentiment and tell if the stock will increase or decrease. We checked the following hypothesis, does the sentiment of Musk's tweets have any correlation with Tesla's stock prices throughout 10 years.  There are specific cases that show the influence of Musk's social media tool on the stocks market, however, the latter depends on so many other factors like news about their competitors, quality of their products, etc. Our project tried to weigh Twitter's influence on Tesla's economy. For that purpose, we calculated the Cramer's V correlation coefficient. While we understand that the tone of the tweet may not correlate with the increase or decrease of the stocks, our hypothesis is based on several previous works that have tested the same exact connection.


This repository contains: .

## Installation
For this project, you must have the ```cuda-11.2``` support

```yaml
# clone project
git clone https://github.com/kmaksatk/Twitter_Stocks_Correlation.git
cd Twitter_Stocks_Correlation

# create conda environment with necessary libraries
conda env create --name your_env_name --file requirements.yaml
conda activate your_env_name
```
If you have the required version of ```cuda```, but at startup it says that the version of cuda and libraries do not match, try the following commands:
```yaml
# to watch available versions of cuda
module avail
# select the necessary version of cuda
module load cuda-11.2
```
## Project Files Description

<p>This Project includes 3 executable files, 3 text files as well as 2 directories as follows:</p>
<h4>Executable Files:</h4>
<ul>
  <li><b>spam_detector.py</b> - Includes all functions required for classification operations.</li>
  <li><b>train.py</b> - Uses the functions defined in the spam_detector.py file and generates the model.txt file after execution.</li>
  <li><b>test.py</b> - Uses the functions defined in the spam_detector.py file and, after execution, generates the result.txt as well as evaluation.txt files.</li>
</ul>

<h4>Output Files:</h4>
<ul>
  <li><b>model.txt</b> - Contains information about the vocabularies of the train set, such as the frequency and conditional probability of each word in Spam and Ham classes.</li>
  <li><b>result.txt</b> - Contains information about the classified emails of the test set.</li>
  <li><b>evaluation.txt</b> - Contains evaluation results table as well as Confusion Matrix of Spam and Ham classes.</li>
</ul>

<h4>Source Directories:</h4>
<ul>
  <li><b>train directory</b> - Includes all emails for the training phase of the program.</li>
  <li><b>test directory</b> - Includes all emails for the testing phase of the program.</li>
</ul>



## Demo 


## Datasets



## Authors

Adilbek Karmanov (@kdiAAA) 

[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kdiAAA)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adilbek-karmanov/)

Maksat Kengeskanov (@kmaksatk)

[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kmaksatk)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maksat-kengeskanov/)
