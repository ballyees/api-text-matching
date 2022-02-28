from multiprocessing.dummy import Process
import docx
import numpy as np
import os
from glob import glob
import re
import pandas as pd
from helper import edit_distance

from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import word_tokenize 
from pythainlp.corpus import thai_stopwords
from pythainlp.util import dict_trie, normalize, isthai
from nltk.stem.porter import PorterStemmer
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.corpus import thai_stopwords
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import Counter

class TextPreprocessor(object):
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            print('Creating the object')
            
            cls._instance = super(TextPreprocessor, cls).__new__(cls)
            # Put any initialization here.
            new_words = {''}
            words = new_words.union(thai_words())
            
            vowels = ('”','“','”','_','!!','–','-','(',')','!','"')
            stopword_set = frozenset(vowels)
            
            cls.TH_stopword = thai_stopwords().union(stopword_set)
            cls.custom_dictionary_trie = dict_trie(words)
            cls.p_stemmer = PorterStemmer()
        return cls._instance

    def clean(cls, word):
        '''
        clean word with PorterStemmer
        '''
        dfTitle = word.strip('!()/\\|}"’‘{_<>[]')
        tokendfTitle = word_tokenize(normalize(dfTitle), custom_dict=cls.custom_dictionary_trie, keep_whitespace=False, engine="newmm")
        Word_in_Title = [word for word in tokendfTitle if not word in cls.TH_stopword]
        Word_in_Title = [cls.p_stemmer.stem(i)  for i in Word_in_Title]
        return  ''.join(Word_in_Title)
    
    def word_preprocessing(cls, paragraphs):
        '''
        word pre-processing for new document
        '''
        doc = np.full(len(paragraphs), '', dtype='O')
        for i, paragraph in enumerate(paragraphs):
            paragraph_wt = word_tokenize(paragraph, keep_whitespace=False)
            paragraph = [word for word in paragraph_wt if isthai(word)]
            doc[i] = cls.clean("".join(paragraph))
        return doc

class Tfidf:
    '''
    workflow: load_dataset -> init_idf
    '''
    def __init__(self, ngram_range=(2, 3), max_features=3000, min_df_factor=0.01) -> None:
        # self.stop_words = [t.encode('utf-8') for t in list(thai_stopwords())]
        # self.stop_words = frozenset(thai_stopwords())
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df_factor = min_df_factor
        self.datasets = None
        self.p = TextPreprocessor()
        
    def text_tokenizer(self, text):
        # terms = [k.strip() for k in text.split(', ') if len(k.strip()) > 0 and k.strip() not in self.stop_words]
        # terms = [k.strip() for k in word_tokenize(text) if len(k.strip()) > 0 and k.strip() not in self.stop_words]
        terms = [k.strip() for k in word_tokenize(text) if len(k.strip()) > 0 and k.strip()]
        
        return [t for t in terms if len(t) > 0 or t is not None]

    def text_processor(self, text: str) -> str:
        omit_key = [ord(c) for c in '*?!&']
        omit_value = [None] * len(omit_key)
        omit = dict(zip(omit_key, omit_value))
        
        return ' '.join([t.strip() for t in text.translate(omit).split("\n") ])
            
    @staticmethod
    def word_preprocessing(paragraphs):
        p = TextPreprocessor()
        doc = np.full(len(paragraphs), '', dtype='O')
        for i, paragraph in enumerate(paragraphs):
            paragraph_wt = word_tokenize(paragraph, keep_whitespace=False)
            paragraph = [word for word in paragraph_wt if isthai(word)]
            doc[i] = p.clean("".join(paragraph))
        return doc
    
    def load_dataset(self, fname, replace=False):
        # get file format
        file_format = fname.split('.')[-1].lower()
        if file_format not in ['csv']:
            raise NotImplementedError(f'format [{file_format}]: not support')
        # load documet
        df = pd.read_csv(fname, sep='|')
        paragraphs = df['Summary'].tolist()
        # initial array of string object (corpus) assume 1 paragraph is document
        doc = np.full(len(paragraphs), '', dtype='O')
        for i, paragraph in enumerate(paragraphs):
            # tokenize word on paragraph
            paragraph_wt = word_tokenize(paragraph, keep_whitespace=False)
            # cut word if not thai word
            paragraph = [word for word in paragraph_wt if isthai(word)]
            # final clean tokenize word and concatenate word
            doc[i] = self.p.clean("".join(paragraph))
        if (self.datasets is None) or (replace is True):
            self.datasets = doc
        else:
            self.datasets = np.hstack([self.datasets, doc])
            
    def init_idf(self):
        if self.datasets is None:
            raise NotInitializedValue('please call load_dataset')
        # initial tfidf
        # print(int(self.datasets.shape[0] * self.min_df_factor))
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.text_tokenizer,
            preprocessor=self.text_processor,
            ngram_range=self.ngram_range,
            # stop_words=self.stop_words,
            # min_df=int(self.datasets.shape[0] * self.min_df_factor),
            # min_df= 2 ,
            max_features=self.max_features,
            token_pattern=None
        )
        # fit the dataset
        self.fit(self.datasets)
        
    def get_vectorizer(self):
        return self.tfidf_vectorizer
    
    def fit(self, corpus):
        # overide method fit on TfidfVectorizer
        self.tfidf_vectorizer.fit(corpus)
        
    def transform(self, corpus):
        # overide method transform on TfidfVectorizer
        return self.tfidf_vectorizer.transform(corpus)
        
    def fit_transform(self, corpus):
        # overide method fit_transform on TfidfVectorizer
        self.fit(corpus)
        return self.transform(corpus)
    
    def get_feature_names(self):
        # overide method get_feature_names on TfidfVectorizer
        return np.array(self.tfidf_vectorizer.get_feature_names())
    
    def idf_(self, corpus=None, topn=20):
        """get idf word and score

        Args:
            corpus ([None], np.ndarray): corpus of document. Defaults to None.
            topn (int): maximum idf sequence. Defaults to 20.

        Returns:
            dict: word is key and idf score is value
        """        
        if corpus is None:
            corpus = self.datasets
        corpus_idf = self.transform(corpus).toarray()
        corpus_idf = corpus_idf.sum(axis=0)
        rank_idx = np.argsort(corpus_idf)[::-1]
        return dict(zip(self.get_feature_names()[rank_idx[:topn]], corpus_idf[rank_idx[:topn]]))

class TextCategory:
    def __init__(self) -> None:
        idf = Tfidf()
        idf.load_dataset('dataLDA_01.csv')
        idf.init_idf()
        self.vectorizer = idf
        idf_transform = idf.transform(idf.datasets)
        # create LDA for every tag on dataset
        self.LDA = LDA(n_components=5, learning_decay=0.5, n_jobs=-1, random_state=1234)
        self.LDA.fit(idf_transform)
        self.unique_keyword = pd.read_csv('unique_keyword.csv')['unique'].to_numpy()
        
    def fit_with_tag(self, corpus):
        idf_transform = self.vectorizer.fit_transform(corpus)
        self.LDA.fit(idf_transform)
        
    def transform_with_tag(self, corpus):
        idf_transform = self.vectorizer.transform(corpus)
        return self.LDA.transform(idf_transform)
    
    def fit_transform_with_tag(self, corpus):
        idf_transform = self.vectorizer.fit_transform(corpus)
        return self.LDA.fit_transform(idf_transform)
    
    def get_topics(self, corpus, topn=10):
        """get LDA topic from unknown tag with probability based on Tfidf
        Args:
            corpus (np.ndarray of unicode): corpus of document
            topn (int): maximum topics sequence. Defaults to 10.

        Returns:
            [np.ndarray]: topics 
        """        
        corpus = [*filter(lambda c: c != '', corpus)]
        idf_transform = self.vectorizer.fit_transform(corpus)
        self.LDA.fit(idf_transform)
        topics_feature = self.vectorizer.get_feature_names()
        top_topics_idx = np.fliplr(np.argsort(self.LDA.components_))[..., :topn]
        remove_space_func = np.vectorize(lambda s: re.sub(r'\s+', '', s))
        lda = []
        topics = []
        is_first = True
        for values, idx in zip(self.LDA.components_, top_topics_idx):
            tmp = []
            # sum_prob = np.sum(values[idx])
            for i, s in zip(idx, topics_feature[idx]):
                s = re.sub(r'\s+', '', s)
                # tmp.append(f'{s} [{values[i]/sum_prob:.3f}]')
                if is_first is True:
                    topics.append(s)
                tmp.append(f'{s} [{values[i]:.3f}]')
            lda.append(', '.join(tmp))
            is_first = False
        print(topics)
            # lda.append(dict(zip(topics_feature[idx], values[idx])))
        print('--------------------'*10)
        # lda = list(zip(topics_feature[top_topics_idx], self.LDA[tag_name].components_[..., top_topics_idx]))
        # for (key, value) in lda:
        topics_in_keyword = []
        for t in topics:
            isin_unique = np.vectorize(lambda s: t[:3] in s)
            word_idx = isin_unique(self.unique_keyword)
            word = self.unique_keyword[word_idx]
            if word.shape[0] != 0:
                if t in word:
                    topics_in_keyword.append(t)
                else:
                    distance = edit_distance(word, t)
                    idx = np.argmin(distance)
                    topics_in_keyword.append(word[idx])
                    print(distance)
            else:
                distance = edit_distance(self.unique_keyword, t)
                # print(distance)
                idx = np.argmin(distance)
                topics_in_keyword.append(self.unique_keyword[idx])
        # print('--------------------'*10)
        # print(lda)
        print('------------'*10)
        # print(topics_in_keyword)
        return lda
        # topics = remove_space_func(topics_feature[top_topics_idx])
        # return self.get_topics_with_tag(tag_name, corpus, topn)

class NotInitializedValue(Exception):
    pass

if __name__ == '__main__':
    
    tc = TextCategory()
    print('------------'*5)
    corpus = tc.vectorizer.datasets
    # print(tc.LDA.get_params())
    # print(corpus)
    # print('------------'*5)
    topics = tc.get_topics(corpus)
    # print(topics, len(topics))
    # print('------------'*5)