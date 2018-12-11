#!/usr/bin/env python
# coding: utf-8

# # Baseline Model

# In[ ]:


import numpy as np
import pandas as pd
import gensim
import os
import glob
import re
from collections import Counter
import ntpath
import sys
import json
import nltk
import random
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.stem import *
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import csv
import json
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel


# In[ ]:


# Run cleandata.py provided first and give the path of the output file generated there to open here
with open("titles.txt", 'r') as dictFile:
    titles = json.load(dictFile)


# ## preprocessing

# In[ ]:


import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')


# In[ ]:


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3:
            stemmed_tokens = stemmer.stem(WordNetLemmatizer().lemmatize(token, pos = 'v'))
            result.append(stemmed_tokens)
    return result


# In[ ]:


# For baseline model we randomly sample 600 titles
# The sample size can be increased to all titles when being run on a competent processor

stemmer = SnowballStemmer('english')
sub_doc = random.sample(list(titles), 600)


# In[ ]:


# Function to perform preprocessing on the all the headlines in sampled dataset
def do_call(doc_raw):
    doc_preprocessed = {}
    for date, doc in doc_raw.items():
        doc_preprocessed[date] = [preprocess(headline) for headline in doc]
        
    return doc_preprocessed


# In[ ]:


doc_raw = {k:titles[k] for k in sub_doc}


# In[ ]:


doc_preprocessed = do_call(doc_raw)


# ## Bag of words conversion for gensim TF-IDF model

# In[ ]:


def get_dict(doc):
    return gensim.corpora.Dictionary(doc)


# In[ ]:


# function to generate bag-of-words for the dataset of headlines
def generate_bow(doc_list):
    doc_dict = get_dict(doc_list)    
    doc_bow = [doc_dict.doc2bow(headline) for headline in doc_list]
        
    return doc_bow, doc_dict


# In[ ]:


doc_all = [headline for k,v in doc_preprocessed.items()
                      for headline in v]


# In[ ]:


# Generate bag-of-words for all headlines sampled
doc_bow, doc_dict = generate_bow(doc_all)


# ## TF-IDF 

# In[ ]:


tfidf = models.TfidfModel(doc_bow)


# In[ ]:


doc_tfidf = tfidf[doc_bow]


# ## train LDA

# In[ ]:


# Using LDAMulticore to perform topic modelling and get the the words in the BOW as topics
n_topics = 15
doc_lda = gensim.models.LdaMulticore(doc_tfidf, num_topics = n_topics, id2word = doc_dict, passes = 3, minimum_probability=0.0)


# In[ ]:


doc_topic_dist_tup = doc_lda.get_document_topics(doc_tfidf, minimum_probability=0.0)


# In[ ]:


doc_topic_prob = []
for i, top_dist in enumerate(doc_topic_dist_tup):
    doc_topic_prob.append(np.asarray(top_dist)[:,1])


# In[ ]:


avg_doc_topic = np.zeros(shape= (len(doc_raw), n_topics), dtype = "float64")
start = 0
for i,day_headline_count in enumerate(map(lambda x : len(x), list(doc_raw.values()))):
    avg_doc_topic[i,:] = np.average(doc_topic_prob[start:(start+day_headline_count)], axis = 0 )
    start += day_headline_count


# In[ ]:


#Change path to the csv file with the historical stock data
ndaq_filepath = './NDAQ.csv' 
data_stock = np.genfromtxt(ndaq_filepath, delimiter=',')


# ## read ndaq, convert dates to match headline keys

# In[ ]:


data_ndaq = pd.read_table(ndaq_filepath, delimiter=',', index_col=0, header = 0)

data_ndaq.index = [date.replace('-', '') for date in data_ndaq.index]

data_ndaq['Difference'] = data_ndaq["Close"] - data_ndaq["Open"]


# In[ ]:


# selecting dates that have stock data in the NDAQ data set
common_dates = np.intersect1d(np.asarray(list(doc_raw.keys())), data_ndaq.index)


# In[ ]:


headline_dates = np.asarray(list(doc_raw.keys()))

headline_idx = np.where(np.isin(headline_dates, common_dates))[0]


# In[ ]:


doc_topic_dist = pd.DataFrame(avg_doc_topic[headline_idx], index=common_dates, dtype="float32")

data_ndaq.Volume = data_ndaq.Volume.astype("float64")

result_concat = pd.concat([doc_topic_dist, data_ndaq.loc[common_dates]], axis =1)


# In[ ]:


# Calculate perplexity of the model
print('\nPerplexity: ', doc_lda.log_perplexity(doc_bow))


# In[ ]:


# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=doc_lda, corpus=doc_bow, dictionary=doc_dict, coherence='u_mass')

coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


# Find the average topic coherence
top_topics = doc_lda.top_topics(doc_bow)
avg_topic_coherence = sum([t[1] for t in top_topics]) / 15
print('Average topic coherence: %.4f.' % avg_topic_coherence)


# In[ ]:


result = result_concat.copy(deep=True)


# In[ ]:


result['Difference'] = result['Difference']


# In[ ]:


# Multiple the topic probability distributio with the difference value
for i in range(0,30):
    result[i] *= result['Difference']


# In[ ]:


result = result.drop(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])


# In[ ]:


# Find the average of probabilistic price change for each topic for all dates 
result.loc['mean'] = result.mean()


# In[ ]:


print(result.loc['mean'])

