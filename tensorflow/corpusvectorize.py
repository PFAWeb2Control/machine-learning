# coding: utf-8
import sys 
import numpy as np
from preprocess import set_sentence
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords


def vectorize(corpus):
        
        corpus_ = []
        for line in corpus:
                corpus_.append(set_sentence(line))

        stwf=stopwords.words('french')
        stwf.append('les')
        stwf.append('rt')


        vectorizer = CountVectorizer(stop_words=stwf,decode_error ="ignore")
        X = vectorizer.fit_transform(corpus_)
        return X.toarray()

