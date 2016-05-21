# coding: utf-8
import re
import string
from datetime import timedelta
from multiprocessing import cpu_count, Pool

import numpy as np
from scipy.sparse import *
import pandas
from nltk import FreqDist, Text, wordpunct_tokenize

import mabed_lib.io as utils

__authors__ = "Adrien Guille, Nicolas Dugué"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:
    MAX_FEATURES = 5000
    TWITTER_TOKENS = ['rt', 'via', '@', '..', '...']
    STOPWORDS_FILE = 'stopwords_en.txt'
    PUNCTUATION = list(string.punctuation)

    def __init__(self, source_file_path, min_absolute_frequency=4, max_relative_frequency=0.5):
        # load stop words
        self.stop_words = self.TWITTER_TOKENS
        self.stop_words.extend(self.PUNCTUATION)
        self.stop_words.extend(utils.load_stopwords(self.STOPWORDS_FILE))
        self.stop_words = set(self.stop_words)
        print('   Stop words:', self.stop_words)

        # load corpus
        self.df = pandas.read_csv(source_file_path, sep='\t', encoding='utf-8')
        self.df['date'] = pandas.to_datetime(self.df.date)
        self.size = self.df.count(0)[0]
        self.start_date = self.df['date'].min()
        self.end_date = self.df['date'].max()
        print('   Corpus: %i tweets, spanning from %s to %s' % (self.size,
                                                                self.start_date,
                                                                self.end_date))

        # extract features
        tweets = []
        for i in range(0, self.size):
            tweets.extend(self.tokenize(self.df.iloc[i]['text']))
        freq_distribution = FreqDist(tweets)
        self.vocabulary = {}
        j = 0
        for word, frequency in freq_distribution.most_common(self.MAX_FEATURES+200):
            if word not in self.stop_words:
                if frequency > min_absolute_frequency and float(frequency/self.size) < max_relative_frequency:
                    self.vocabulary[word] = j
                    j += 1
            if len(self.vocabulary) == self.MAX_FEATURES:
                break
        print('   Vocabulary: %i unique tokens' % len(self.vocabulary))

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
        self.time_slice_count = int(time_delta // self.time_slice_length) + 1

        # parallelize tweet partitioning using a pool of processes (number of processes = number of cores).
        nb_processes = cpu_count()
        nb_tweets_per_process = self.size // nb_processes
        portions = []
        for i in range(0, self.size, nb_tweets_per_process):
            j = i + nb_tweets_per_process if i + nb_tweets_per_process < self.size else self.size
            portions.append((i, j))
        p = Pool()
        results = p.map(self.discretize_job, portions)
        results.sort(key=lambda x: x[0])

        # insert the time-slices number in the data frame and compute the final frequency matrices
        time_slices = []
        self.tweet_count = np.zeros(self.time_slice_count, dtype=np.int)
        self.global_freq = csr_matrix((len(self.vocabulary), self.time_slice_count), dtype=np.short)
        self.mention_freq = csr_matrix((len(self.vocabulary), self.time_slice_count), dtype=np.short)
        for a_tuple in results:
            time_slices.extend(a_tuple[1])
            self.tweet_count = np.add(self.tweet_count, a_tuple[2])
            self.global_freq = np.add(self.global_freq, a_tuple[3])
            self.mention_freq = np.add(self.mention_freq, a_tuple[4])
        self.df['time_slice'] = np.array(time_slices)

    def discretize_job(self, portion):
        # initialize data structures
        time_slices = []
        tweet_count = np.zeros(self.time_slice_count, dtype=np.int)
        # dictionary-of-keys based sparse matrix that allow incremental construction
        global_freq = dok_matrix((len(self.vocabulary), self.time_slice_count), dtype=np.short)
        mention_freq = dok_matrix((len(self.vocabulary), self.time_slice_count), dtype=np.short)

        # iterate through the corpus
        for i in range(portion[0], portion[1]):
            # identify the time-slice that corresponds to the tweet
            tweet_date = self.df.iloc[i]['date']
            time_delta = (tweet_date - self.start_date)
            time_delta = time_delta.total_seconds()/60
            time_slice = int(time_delta/self.time_slice_length)
            time_slices.append(time_slice)
            tweet_count[time_slice] = tweet_count.item(time_slice) + 1

            # tokenize the tweet and compute word frequency
            words = self.tokenize(self.df.iloc[i]['text'])
            mention = '@' in words
            for word in set(words):
                if self.vocabulary.get(word) is not None:
                    row = self.vocabulary[word]
                    column = time_slice
                    global_freq[row, column] += 1
                    if mention:
                        mention_freq[row, column] += 1
        # return the results (dok matrices are converted to csr matrices because dok matrices can't be pickled)
        return portion[0], time_slices, tweet_count, global_freq.tocsr(), mention_freq.tocsr()

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
            print(entry.get_word())

    @staticmethod
    def tokenize(text):
        text_without_url = re.sub(r'(?:https?\://)\S+', '', text)
        tokens = wordpunct_tokenize(text_without_url)
        clean_text = Text(tokens)
        words = [w.lower() for w in clean_text]
        return words
