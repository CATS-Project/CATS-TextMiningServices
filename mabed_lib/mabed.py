# coding: utf-8
import timeit
import networkx as nx
import numpy as np
import stats
from corpus import Corpus

__authors__ = "Adrien Guille, Nicolas Dugué"
__email__ = "adrien.guille@univ-lyon2.fr"


class MABED:

    def __init__(self, corpus):
        self.corpus = corpus
        self.event_graph = None
        self.redundancy_graph = None
        self.events = None

    def run(self, k=10, theta=0.6, sigma=0.5):
        basic_events = self.phase1()
        return self.phase2(basic_events, k, theta, sigma)

    def phase1(self):
        print 'Phase 1...'
        basic_events = []

        # iterate through the vocabulary
        for word, word_index in self.corpus.vocabulary.iteritems():
            mention_freq = self.corpus.mention_freq[word_index, :]
            total_mention_freq = np.sum(mention_freq)

            # compute the time-series that describes the evolution of mention-anomaly
            anomaly = []
            for i in range(0, self.corpus.time_slice_count):
                anomaly.append(self.anomaly(i, mention_freq[i], total_mention_freq))

                # identify the interval that maximizes the magnitude of impact
                interval_max = self.maximum_contiguous_subsequence_sum(anomaly)
                if interval_max is not None:
                    mag = np.sum(anomaly[interval_max[0]:interval_max[1]])
                    basic_event = (mag, interval_max, word, anomaly)
                    basic_events.append(basic_event)

        print 'Phase 1: %d events' % len(basic_events)
        return basic_events

    def maximum_contiguous_subsequence_sum(self, anomaly):
        max_ending_here = max_so_far = 0
        a = b = a_ending_here = 0
        for idx, ano in enumerate(anomaly):
            max_ending_here = max(0, max_ending_here + ano)
            if max_ending_here == 0:
                # a new bigger sum may start from here
                a_ending_here = idx
            if max_ending_here > max_so_far:
                # the new sum from a_ending_here to idx is bigger !
                a = a_ending_here+1
                max_so_far = max_ending_here
                b = idx
        max_interval = (a, b)
        return max_interval

    def phase2(self, basic_events, k=10, theta=0.7, sigma=0.5):
        print 'Phase 2...'

        # sort the events detected during phase 1 according to their magnitude of impact
        basic_events.sort(key=lambda tup: tup[0], reverse=True)

        # create the event graph (directed) and the redundancy graph (undirected)
        self.event_graph = nx.DiGraph(name='Event graph')
        self.redundancy_graph = nx.Graph(name='Redundancy graph')
        i = 0
        unique_events = 0
        refined_events = []

        # phase 2 goes on until the top k (distinct) events have been identified
        while unique_events < k and i < len(basic_events):
            basic_event = basic_events[i]
            main_word = basic_event[2]
            candidate_words = self.corpus.cooccurring_words(basic_event, 10)
            main_word_freq = self.corpus.global_freq[self.corpus.vocabulary[main_word], :]
            related_words = []

            # identify candidate words based on co-occurrence
            if candidate_words is not None:
                for candidate_word in candidate_words:
                    candidate_word_freq = self.corpus.global_freq[self.corpus.vocabulary[candidate_word], :]

                    # compute correlation and filter according to theta
                    weight = (stats.erdem_correlation(main_word_freq, candidate_word_freq) + 1) / 2
                    if weight > theta:
                        related_words.append((candidate_word, weight))

                if len(related_words) > 1:
                    refined_event = (basic_event[0], basic_event[1], main_word, related_words, basic_event[3])
                    # check if this event is distinct from those already stored in the event graph
                    if self.update_graphs(refined_event, sigma):
                        refined_events.append(refined_event)
                        unique_events += 1
            i += 1
        # merge redundant events and save the result
        self.events = self.merge_redundant_events(refined_events)

    def anomaly(self, time_slice, observation, total_mention_freq):
        # compute the expected frequency of the given word at this time-slice
        expectation = float(self.corpus.tweet_count[time_slice]) * (float(total_mention_freq)/(float(self.corpus.size)))

        # return the difference between the observed frequency and the expected frequency
        return observation - expectation

    def update_graphs(self, event, sigma):
        redundant = False
        main_word = event[2]
        if self.event_graph.has_node(main_word):
            for related_word, weight in event[3]:
                if self.event_graph.has_edge(main_word, related_word):
                    interval_0 = self.event_graph.node[related_word]['interval']
                    interval_1 = event[1]
                    if stats.overlap_coefficient(interval_0, interval_1) > sigma:
                        self.redundancy_graph.add_node(main_word, description=event)
                        self.redundancy_graph.add_node(related_word, description=event)
                        self.redundancy_graph.add_edge(main_word, related_word)
                        redundant = True
        if not redundant:
            self.event_graph.add_node(event[2], interval=event[1], main_term=True)
            for related_word, weight in event[3]:
                self.event_graph.add_edge(related_word, event[2], weight=weight)
        return not redundant

    def merge_redundant_events(self, events):
        # compute the connected components in the redundancy graph
        components = []
        for c in nx.connected_components(self.redundancy_graph):
            components.append(c)
        final_events = []

        # merge redundant events
        for event in events:
            main_word = event[2]
            main_term = main_word
            for component in components:
                if main_word in component:
                    main_term = ', '.join(component)
                    break
            final_event = (event[0], event[1], main_term, event[3], event[4])
            final_events.append(final_event)
            self.print_event(final_event)
        return final_events

    def print_event(self, event):
        related_words = []
        for related_word, weight in event[3]:
            related_words.append(related_word+'('+str("{0:.2f}".format(weight))+')')
        print '%s - %s: %s (%s)' % (str(self.corpus.to_date(event[1][0])),
                                    str(self.corpus.to_date(event[1][1])),
                                    event[2],
                                    ', '.join(related_words))

if __name__ == '__main__':
    print 'Loading corpus...'
    start_time = timeit.default_timer()
    my_corpus = Corpus('../input/messages1.csv')
    elapsed = timeit.default_timer() - start_time
    print 'Corpus loaded in %f seconds.' % elapsed

    time_slice_length = 30
    print 'Partitioning tweets into %d-minute time-slices...' % time_slice_length
    start_time = timeit.default_timer()
    my_corpus.discretize(time_slice_length)
    print '   Time-slices: %i' % my_corpus.time_slice_count
    elapsed = timeit.default_timer() - start_time
    print 'Partitioning done in %f seconds.' % elapsed

    print 'Running MABED...'
    start_time = timeit.default_timer()
    mabed = MABED(my_corpus)
    top_k = 20
    mabed.run(k=top_k, theta=0.6, sigma=0.6)
    elapsed = timeit.default_timer() - start_time
    print 'Top %d events detected in %f seconds.' % (k, elapsed)
