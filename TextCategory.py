import docx
import numpy as np
import os
from glob import glob
import re

from pythainlp import word_tokenize, Tokenizer, sent_tokenize
from pythainlp.corpus.common import thai_words, provinces
from pythainlp.tokenize import word_tokenize 
from pythainlp.corpus import thai_stopwords, get_corpus
from pythainlp.util import dict_trie, normalize, isthai
from nltk.stem.porter import PorterStemmer
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.corpus import thai_stopwords
from sklearn.decomposition import LatentDirichletAllocation as LDA



class Tfidf:
    '''
    workflow: load_dataset -> init_idf
    '''
    def __init__(self, ngram_range=(2, 3), max_features=3000, min_df_factor=0.01) -> None:
        # self.stop_words = [t.encode('utf-8') for t in list(thai_stopwords())]
        self.stop_words = frozenset(thai_stopwords())
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df_factor = min_df_factor
        self.datasets = None
        
    def text_tokenizer(self, text):
        # terms = [k.strip() for k in text.split(', ') if len(k.strip()) > 0 and k.strip() not in self.stop_words]
        terms = [k.strip() for k in word_tokenize(text) if len(k.strip()) > 0 and k.strip() not in self.stop_words]
        return [t for t in terms if len(t) > 0 or t is not None]

    def text_processor(self, text: str) -> str:
        omit_key = [ord(c) for c in '*?!&']
        omit_value = [None] * len(omit_key)
        omit = dict(zip(omit_key, omit_value))
        
        return ' '.join([t.strip() for t in text.translate(omit).split("\n") ])
    
    def load_dataset_from_directory(self, directory):
        doc_path = glob(os.path.join(directory, '*.doc'))
        docx_path = glob(os.path.join(directory, '*.docx'))
        for fname in [*doc_path, *docx_path]:
            self.load_dataset(fname)
    
    def load_dataset(self, fname, replace=False):
        file_format = fname.split('.')[-1].lower()
        if file_format not in ['doc', 'docx']:
            raise NotImplementedError(f'format [{file_format}]: not support')
        docx_file = docx.Document(fname)
        paragraphs = [paragraph.text for paragraph in docx_file.paragraphs]
        doc = np.full(len(paragraphs), '', dtype='O')
        for i, paragraph in enumerate(paragraphs):
            paragraph_wt = word_tokenize(paragraph, keep_whitespace=False)
            paragraph = [word for word in paragraph_wt if isthai(word)]
            doc[i] = "".join(paragraph)
        if (self.datasets is None) or (replace is True):
            self.datasets = doc
        else:
            self.datasets = np.hstack([self.datasets, doc])
            
    def init_idf(self):
        if self.datasets is None:
            raise NotInitializedValue('please call load_dataset')
        
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.text_tokenizer,
            preprocessor=self.text_processor,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            min_df=int(self.datasets.shape[0] * self.min_df_factor),
            max_features=self.max_features
        )
        self.fit(self.datasets)
        
    def get_vectorizer(self):
        return self.tfidf_vectorizer
    
    def fit(self, corpus):
        self.tfidf_vectorizer.fit(corpus)
        
    def transform(self, corpus):
        return self.tfidf_vectorizer.transform(corpus)
        
    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    
    def get_feature_names(self):
        return np.array(self.tfidf_vectorizer.get_feature_names())
    
    def idf_(self, corpus=None, topn=20):
        if corpus is None:
            corpus = self.datasets
        corpus_idf = self.transform(corpus).toarray()
        corpus_idf = corpus_idf.sum(axis=0)
        rank_idx = np.argsort(corpus_idf)[::-1]
        return dict(zip(self.get_feature_names()[rank_idx[:topn]], corpus_idf[rank_idx[:topn]]))

class TextCategory:
    def __init__(self) -> None:
        self.vectorizers = {}
        self.LDA = {}
        for pw in os.walk('LDA'):
            if len(pw[1]) == 0: # walk to datasets folder
                tag = os.path.split(pw[0])[-1]
                idf = Tfidf()
                idf.load_dataset_from_directory(pw[0])
                idf.init_idf()
                self.vectorizers[tag] = idf
                idf_transform = idf.transform(idf.datasets)
                self.LDA[tag] = LDA(n_components=5, learning_decay=0.5, n_jobs=-1, random_state=1234)
                self.LDA[tag].fit(idf_transform)
        self.tags = list(self.LDA.keys())
        
    def fit_with_tag(self, tag, corpus):
        idf_vectorizer = self.vectorizers[tag]
        idf_transform = idf_vectorizer.fit_transform(corpus)
        self.LDA[tag].fit(idf_transform)
        
    def transform_with_tag(self, tag, corpus):
        idf_vectorizer = self.vectorizers[tag]
        idf_transform = idf_vectorizer.transform(corpus)
        return self.LDA[tag].transform(idf_transform)
    
    def fit_transform_with_tag(self, tag, corpus):
        idf_vectorizer = self.vectorizers[tag]
        idf_transform = idf_vectorizer.fit_transform(corpus)
        return self.LDA[tag].fit_transform(idf_transform)
        
    def get_topics(self, tag, corpus, topn=10):
        self.fit_with_tag(tag, corpus)
        topics_feature = self.vectorizers[tag].get_feature_names()
        top_topics_idx = np.fliplr(np.argsort(self.LDA[tag].components_))[..., :topn]
        remove_space_func = np.vectorize(lambda s: re.sub('\s+', '', s))
        topics = remove_space_func(topics_feature[top_topics_idx])
        return [', '.join(t) for t in topics]
    def get_tag_from_index(self, idx=0):
        return self.tags[idx]
class NotInitializedValue(Exception):
    pass

if __name__ == '__main__':
    tc = TextCategory()
    tag = tc.get_tag_from_index(0)
    corpus = tc.vectorizers[tag].datasets
    print(tag)
    topics = tc.get_topics(tag, corpus)
    print(topics)
    print('------------'*5)