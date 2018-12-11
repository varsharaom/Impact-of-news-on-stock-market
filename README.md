Stock market forcasting using news headlines

Data set : Reuter's, NASDAQ
Headlines downloaded from : https://github.com/philipperemy/Reuters-full-data-set
NASDAQ prices under ./data/

Overview :
Preprocess headlines using gensim and nltk. Preprocessing involves removing stop words, stemming and lematizing.

Create a bag of words from the headlines, each headline being a document, across all dates.

Generate a tfidf probability using the bag of words data.

Train LDA using the tfid distribution.


Instructions:

1. Change path in cell 2 accordingly
2. Run all cells sequentially 






