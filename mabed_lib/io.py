# coding: utf-8
import pickle
import codecs

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def save_corpus(corpus, file_path):
    pickle.dump(corpus, open(file_path, 'wb'))


def load_corpus(file_path):
    return pickle.load(open(file_path, 'rb'))


def load_stopwords(file_path):
    stopwords_file = codecs.open(file_path, 'r', 'utf-8')
    stopwords = stopwords_file.readlines()
    return [line.rstrip('\n') for line in stopwords]
