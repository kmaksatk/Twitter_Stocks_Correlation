from cuml.cluster import HDBSCAN
import cuml
from tqdm.notebook import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from functools import partial
import hdbscan
import gensim
from collections import Counter
from IPython.display import display
import data_preprocessing

#Word2Vec Encoder
# Original code: https://github.com/KimaruThagna/Nlp_Tuts/blob/master/analytics_vidhya.py 
def word2vec(tweets, size = 1000, window = 5, min_count = 2, sg = 1, hs = 0, negative = 10, workers = 32, seed = 34):

    tokenized_tweet = tweets.apply(lambda x: x.split()) 

    model_w2v = gensim.models.Word2Vec(
                tokenized_tweet,
                size = size, # desired no. of features/independent variables
                window = window, # context window size
                min_count = min_count, # Ignores all words with total frequency lower than 2.                                  
                sg = sg, # 1 for skip-gram model
                hs = hs,
                negative = negative, # for negative sampling
                workers= workers, # no.of cores
                seed = 34) 
    model_w2v.train(tokenized_tweet, total_examples= len(tweets), epochs=20)
    wordvec_arrays = np.zeros((len(tokenized_tweet), 1000)) 
    for i in range(len(tokenized_tweet)):
        wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 1000, model_w2v)
    return wordvec_arrays

def word_vector(tokens, size, model_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

#Original code: https://github.com/dborrelli/chat-intents
def generate_clusters(embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      min_samples,
                      random_state = None):
    """
    Returns HDBSCAN objects after first performing dimensionality reduction using UMAP
    
    Arguments:
        message_embeddings: embeddings to use
        n_neighbors: int, UMAP hyperparameter n_neighbors
        n_components: int, UMAP hyperparameter n_components
        min_cluster_size: int, HDBSCAN hyperparameter min_cluster_size
        min_samples: int, HDBSCAN hyperparameter min_samples
        random_state: int, random seed
        
    Returns:
        clusters: HDBSCAN object of clusters and DBCV score
    """
    
    umap_embeddings = (cuml.UMAP(n_neighbors = n_neighbors, 
                                n_components = n_components,
                                random_state=random_state,
                                min_dist = 0.0)
                            .fit_transform(embeddings))

    clusters = HDBSCAN(min_cluster_size = min_cluster_size, 
                               min_samples = min_samples,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)
    
    return clusters, hdbscan.validity.validity_index(umap_embeddings.get().astype('double'), clusters.labels_.get(), d = n_components)


#Original code: https://github.com/dborrelli/chat-intents
def objective(params, embeddings):
    """
    Objective function for hyperopt to minimize

    Arguments:
        params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'random_state' and
               their values to use for evaluation
        embeddings: embeddings to use


    Returns:
        loss: cost function result incorporating penalties for falling
              outside desired range for number of clusters
        label_count: int, number of unique cluster labels, including noise
        status: string, hypoeropt status

        """
    
    clusters, dbcv = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 min_samples = params['min_samples'],
                                 random_state = params['random_state'])
    
    loss = (-1.0) * dbcv
    label_count = len(np.unique(clusters.labels_))
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


#Original code: https://github.com/dborrelli/chat-intents
def bayesian_search(embeddings, space,  max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

        """
    
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings)
    
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=max_evals, 
                trials=trials)

    best_params = space_eval(space, best)
    print ('best:')
    print (best_params)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters, dbcv = generate_clusters(embeddings, 
                                      n_neighbors = best_params['n_neighbors'], 
                                      n_components = best_params['n_components'], 
                                      min_cluster_size = best_params['min_cluster_size'],
                                      min_samples = best_params['min_samples'],
                                      random_state = best_params['random_state'])
    
    return best_params, best_clusters, trials


#Original code: https://github.com/dborrelli/chat-intents
def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.0):
    """
    Reduce dimensionality of best clusters and plot in 2D

    Arguments:
        embeddings: embeddings to use
        clusteres: HDBSCAN object of clusters
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """
    fig, ax = plt.subplots(figsize=(14, 8))
    umap_data = cuml.UMAP(n_neighbors=n_neighbors, 
                          n_components=2, 
                          min_dist = min_dist, 
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(embeddings.shape[0])
    

    umap_data = umap_data.get()
    labels = clusters.labels_.get()
    clustered = (labels >= 0)
    plt.scatter(umap_data[~clustered, 0],
                umap_data[~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=point_size,
                alpha=0.5)
    plt.scatter(umap_data[clustered, 0],
                umap_data[clustered, 1],
                c=labels[clustered],
                s=point_size,
                cmap='Spectral')
    plt.show()


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
        