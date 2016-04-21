# coding: utf-8
import re
import string
from datetime import timedelta

import nltk
import numpy as np
import pandas
from nltk import FreqDist

import utils

__authors__ = "Adrien Guille, Nicolas Dugué"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:
    MAX_FEATURES = 5000
    TWITTER_TOKENS = ['rt', 'via', '@', '..', '...']
    STOPWORDS_FILE = '../stopwords_en.txt'
    PUNCTUATION = list(string.punctuation)

    def __init__(self, source_file_path, min_absolute_frequency=4, max_relative_frequency=0.5):
        # load stop words
        self.stop_words = self.TWITTER_TOKENS
        self.stop_words.extend(self.PUNCTUATION)
        self.stop_words.extend(utils.load_stopwords(self.STOPWORDS_FILE))
        self.stop_words = set(self.stop_words)
        print '   Stop words:', self.stop_words

        # load corpus
        self.df = pandas.read_csv(source_file_path, sep='\t', encoding='utf-8')
        self.df['date'] = pandas.to_datetime(self.df.date)
        self.size = self.df.count(0)[0]
        self.start_date = self.df['date'].min()
        self.end_date = self.df['date'].max()
        print '   Corpus: %i tweets, spanning from %s to %s' % (self.size,
                                                                self.start_date,
                                                                self.end_date)

        # extract features
        all_tweets = []
        for i in range(0, self.size):
            all_tweets.extend(self.tokenize(self.df.iloc[i]['text']))
        freq_distribution = FreqDist(all_tweets)
        self.vocabulary = {}
        j = 0
        for word, frequency in freq_distribution.most_common(self.MAX_FEATURES):
            if word not in self.stop_words:
                self.vocabulary[word] = j
                j += 1
        print '   Vocabulary: %i unique tokens' % len(self.vocabulary)

        self.time_slice_count = None
        self.tweet_count = None
        self.global_freq = None
        self.mention_freq = None
        self.time_slice_length = None

    def discretize(self, time_slice_length):
        self.time_slice_length = time_slice_length

        # compute the total number of time-slices
        time_delta = (self.end_date - self.start_date)
        time_delta = time_delta.total_seconds()/60
        self.time_slice_count = int(time_delta/float(time_slice_length)) + 1

        # initialize data structures
        self.tweet_count = np.zeros(self.time_slice_count, dtype=np.int)
        self.global_freq = np.zeros((len(self.vocabulary), self.time_slice_count), dtype=np.short)
        self.mention_freq = np.zeros((len(self.vocabulary), self.time_slice_count), dtype=np.short)

        # prepare a new column
        time_slices = []

        # iterate through the corpus
        for i in range(0, self.size):
            tweet_date = self.df.iloc[i]['date']
            time_delta = (tweet_date - self.start_date)
            time_delta = time_delta.total_seconds()/60
            time_slice = int(time_delta/time_slice_length)
            time_slices.append(time_slice)
            self.tweet_count[time_slice] = self.tweet_count.item(time_slice) + 1
            words = self.tokenize(self.df.iloc[i]['text'])
            mention = '@' in words
            for word in set(words):
                if self.vocabulary.get(word) is not None:
                    row = self.vocabulary[word]
                    column = time_slice
                    self.global_freq[row, column] = self.global_freq.item((row, column)) + 1
                    if mention:
                        self.mention_freq[row, column] = self.mention_freq.item((row, column)) + 1

        # add the new column in the data frame
        self.df['time_slice'] = np.array(time_slices)

    def cooccurring_words(self, event, p):
        # remove characters that could be interpreted as regular expressions by pandas
        main_word = event[2].replace(')', '').replace('(', '').replace('?', '').replace('*', '').replace('+', '')

        # identify tweets related to the event
        filtered_df_0 = self.df.loc[self.df['time_slice'].isin(range(event[1][0], event[1][1]))]
        filtered_df = filtered_df_0.loc[filtered_df_0['text'].str.contains(main_word)]
        related_tweets = []
        for i in range(0, filtered_df.count(0)[0]):
            related_tweets.extend(self.tokenize(filtered_df.iloc[i]['text']))
        freq_distribution = FreqDist(related_tweets)

        # compute word frequency
        top_cooccurring_words = []
        for word, frequency in freq_distribution.most_common(self.MAX_FEATURES):
            if word != main_word and word not in self.stop_words:
                if self.vocabulary.get(word) is not None:
                    top_cooccurring_words.append(word)
                    if len(top_cooccurring_words) == p:
                        # return the p words that co-occur the most with the main word
                        return top_cooccurring_words

    def to_date(self, time_slice):
        a_date = self.start_date + timedelta(minutes=time_slice*self.time_slice_length)
        return a_date

    def print_vocabulary(self):
        for entry in self.vocabulary:
            print entry.get_word()

    @staticmethod
    def tokenize(text):
        text_without_url = re.sub(r'(?:https?\://)\S+', '', text)
        tokens = nltk.wordpunct_tokenize(text_without_url)
        clean_text = nltk.Text(tokens)
        words = [w.lower() for w in clean_text]
        return words

if __name__ == '__main__':
    my_corpus = Corpus('/Users/adrien/Desktop/messages1.csv')
    my_corpus.discretize(30)
