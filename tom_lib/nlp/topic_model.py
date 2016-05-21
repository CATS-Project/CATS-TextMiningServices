# coding: utf-8
import itertools
from abc import ABCMeta, abstractmethod
import numpy as np
import tom_lib.stats
from gensim import models, matutils
from scipy import spatial, sparse, cluster
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF

from tom_lib.structure.corpus import Corpus

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"


class TopicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus  # a Corpus object
        self.document_topic_matrix = None  # document x topic matrix
        self.topic_word_matrix = None  # topic x word matrix
        self.nb_topics = None

    @abstractmethod
    def infer_topics(self, num_topics=10):
        pass

    def greene_metric(self, min_num_topics=10, step=5, max_num_topics=50, top_n_words=10, tao=10):
        """
        Implements Greene metric to compute the optimal number of topics. Taken from How Many Topics?
        Stability Analysis for Topic Models from Greene et al. 2014.
        :param step:
        :param min_num_topics: Minimum number of topics to test
        :param max_num_topics: Maximum number of topics to test
        :param top_n_words: Top n words for topic to use
        :param tao: Number of sampled models to build
        :return: A list of len (max_num_topics - min_num_topics) with the stability of each tested k
        """
        stability = []
        # Build reference topic model
        # Generate tao topic models with tao samples of the corpus
        for k in np.arange(min_num_topics, max_num_topics + 1, step):
            self.infer_topics(k)
            reference_rank = [list(zip(*self.top_words(i, top_n_words))[0]) for i in range(k)]
            agreement_score_list = []
            for t in range(tao):
                tao_corpus = Corpus(source_file_path=self.corpus._source_file_path,
                                    language=self.corpus._language,
                                    n_gram=self.corpus._n_gram,
                                    vectorization=self.corpus._vectorization,
                                    max_relative_frequency=self.corpus._max_relative_frequency,
                                    min_absolute_frequency=self.corpus._min_absolute_frequency,
                                    preprocessor=self.corpus._preprocessor,
                                    sample=True)
                tao_model = type(self)(tao_corpus)
                tao_model.infer_topics(k)
                tao_rank = [list(zip(*tao_model.top_words(i, top_n_words))[0]) for i in range(k)]
                agreement_score_list.append(tom_lib.stats.agreement_score(reference_rank, tao_rank))
            stability.append(np.mean(agreement_score_list))
        return stability

    def arun_metric(self, min_num_topics=10, max_num_topics=50, iterations=10):
        """
        Implements Arun metric to estimate the optimal number of topics:
        Arun, R., V. Suresh, C. V. Madhavan, and M. N. Murthy
        On finding the natural number of topics with latent dirichlet allocation: Some observations.
        In PAKDD (2010), pp. 391–402.
        :param min_num_topics: Minimum number of topics to test
        :param max_num_topics: Maximum number of topics to test
        :param iterations: Number of iterations per value of k
        :return: A list of len (max_num_topics - min_num_topics) with the average symmetric KL divergence for each k
        """
        kl_matrix = []
        for j in range(iterations):
            kl_list = []
            l = np.array([sum(self.corpus.vector_for_document(doc_id)) for doc_id in range(self.corpus.size)])
            norm = np.linalg.norm(l)
            for i in range(min_num_topics, max_num_topics + 1):
                self.infer_topics(i)
                c_m1 = np.linalg.svd(self.topic_word_matrix.todense(), compute_uv=False)
                c_m2 = l.dot(self.document_topic_matrix.todense())
                c_m2 += 0.0001
                c_m2 /= norm
                kl_list.append(tom_lib.stats.symmetric_kl(c_m1.tolist(), c_m2.tolist()[0]))
            kl_matrix.append(kl_list)
        ouput = np.array(kl_matrix)
        return ouput.mean(axis=0)

    def brunet_metric(self, min_num_topics=10, max_num_topics=50, iterations=10):
        """
        Implements a consensus-based metric to estimate the optimal number of topics:
        Brunet, J.P., Tamayo, P., Golub, T.R., Mesirov, J.P.
        Metagenes and molecular pattern discovery using matrix factorization.
        Proc. National Academy of Sciences 101(12) (2004), pp. 4164–4169
        :param min_num_topics:
        :param max_num_topics:
        :param iterations:
        :return:
        """
        cophenetic_correlation = []
        for i in range(min_num_topics, max_num_topics+1):
            average_C = np.zeros((self.corpus.size, self.corpus.size))
            for j in range(iterations):
                self.infer_topics(i)
                for p in range(self.corpus.size):
                    for q in range(self.corpus.size):
                        if self.most_likely_topic_for_document(p) == self.most_likely_topic_for_document(q):
                            average_C[p, q] += float(1./iterations)

            clustering = cluster.hierarchy.linkage(average_C, method='average')
            Z = cluster.hierarchy.dendrogram(clustering, orientation='right')
            index = Z['leaves']
            average_C = average_C[index, :]
            average_C = average_C[:, index]
            (c, d) = cluster.hierarchy.cophenet(Z=clustering, Y=spatial.distance.pdist(average_C))
            # plt.clf()
            # f, ax = plt.subplots(figsize=(11, 9))
            # ax = sns.heatmap(average_C)
            # plt.savefig('reorderedC.png')
            cophenetic_correlation.append(c)
        return cophenetic_correlation

    def print_topics(self, num_words=10):
        frequency = self.topics_frequency()
        for topic_id in range(self.nb_topics):
            word_list = []
            for weighted_word in self.top_words(topic_id, num_words):
                word_list.append(weighted_word[0])
            print('topic %i: %s (%f)' % (topic_id, ' '.join(word_list), frequency[topic_id]))

    def top_words(self, topic_id, num_words):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weighted_words = [()] * len(self.corpus.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weighted_words[word_id] = (self.corpus.word_for_id(word_id), weight)
        weighted_words.sort(key=lambda x: x[1], reverse=True)
        return weighted_words[:num_words]

    def top_documents(self, word_id, num_docs):
        vector = self.topic_word_matrix[:, word_id]
        cx = vector.tocoo()
        distribution = [0.0] * self.nb_topics
        for topic_id, col, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            distribution[topic_id] = weight
        return distribution

    def word_distribution_for_topic(self, topic_id):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weights = [0.0] * self.nb_topics
        for row, topic_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weights[topic_id] = weight
        return weights

    def topic_distribution_for_document(self, doc_id):
        vector = self.document_topic_matrix[doc_id]
        cx = vector.tocoo()
        weights = [0.0] * self.nb_topics
        for row, topic_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weights[topic_id] = weight
        return weights

    def topic_distribution_for_word(self, word_id):
        vector = self.topic_word_matrix[:, word_id]
        cx = vector.tocoo()
        distribution = [0.0] * self.nb_topics
        for topic_id, col, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            distribution[topic_id] = weight
        return distribution

    def topic_distribution_for_author(self, author_name):
        all_weights = []
        for document_id in self.corpus.documents_by_author(author_name):
            all_weights.append(self.topic_distribution_for_document(document_id))
        ouput = np.array(all_weights)
        return ouput.mean(axis=0)

    def most_likely_topic_for_document(self, doc_id):
        weights = self.topic_distribution_for_document(doc_id)
        return weights.index(max(weights))

    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = [0.0] * self.nb_topics
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids(date)
        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0 / len(ids)
        return frequency

    def documents_for_topic(self, topic_id):
        doc_ids = []
        for doc_id in range(self.corpus.size):
            most_likely_topic = self.most_likely_topic_for_document(doc_id)
            if most_likely_topic == topic_id:
                doc_ids.append(doc_id)
        return doc_ids

    def documents_per_topic(self):
        topic_associations = {}
        for i in range(self.corpus.size):
            topic_id = self.most_likely_topic_for_document(i)
            if topic_associations.get(topic_id):
                documents = topic_associations[topic_id]
                documents.append(i)
                topic_associations[topic_id] = documents
            else:
                documents = [i]
                topic_associations[topic_id] = documents
        return topic_associations

    def similar_documents(self, doc_id, num_docs):
        doc_weights = self.topic_distribution_for_document(doc_id)
        similarities = []
        for a_doc_id in range(self.corpus.size):
            if a_doc_id != doc_id:
                similarity = 1.0 - spatial.distance.cosine(doc_weights, self.topic_distribution_for_document(a_doc_id))
                similarities.append((a_doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]

    def affiliation_repartition(self, topic_id):
        counts = {}
        doc_ids = self.documents_for_topic(topic_id)
        for i in doc_ids:
            affiliations = set(self.corpus.affiliations(i))
            for affiliation in affiliations:
                if str(affiliation) not in ['nan', '@gmail.com', '@yahoo.fr']:
                    if counts.get(affiliation):
                        count = counts[affiliation] + 1
                        counts[affiliation] = count
                    else:
                        counts[affiliation] = 1
        tuples = []
        for affiliation, count in counts.items():
            tuples.append((affiliation, count))
        tuples.sort(key=lambda x: x[1], reverse=True)
        return tuples


class LatentDirichletAllocation(TopicModel):
    def infer_topics(self, num_topics=10):
        if self.corpus.gensim_vector_space is None:
            self.corpus.gensim_vector_space = matutils.Sparse2Corpus(self.corpus.sklearn_vector_space,
                                                                     documents_columns=False)
        self.nb_topics = num_topics
        lda = models.LdaModel(corpus=self.corpus.gensim_vector_space,
                              iterations=10000,
                              num_topics=num_topics)
        tmp_topic_word_matrix = list(lda.show_topics(num_topics=num_topics,
                                                     num_words=len(self.corpus.vocabulary),
                                                     formatted=False))
        row = []
        col = []
        data = []
        for topic_id in range(self.nb_topics):
            topic_description = tmp_topic_word_matrix[topic_id]
            for word_id, probability in topic_description[1]:
                row.append(topic_id)
                col.append(int(word_id))
                data.append(probability)
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        self.document_topic_matrix = sparse.csr_matrix(
            np.transpose(matutils.corpus2dense(lda[self.corpus.gensim_vector_space],
                                               num_topics,
                                               self.corpus.size)))
        self.corpus.gensim_vector_space = None


class LatentSemanticAnalysis(TopicModel):
    def infer_topics(self, num_topics=10):
        if self.corpus.gensim_vector_space is None:
            self.corpus.gensim_vector_space = matutils.Sparse2Corpus(self.corpus.sklearn_vector_space,
                                                                     documents_columns=False)
        self.nb_topics = num_topics
        lsa = models.LsiModel(corpus=self.corpus.gensim_vector_space,
                              id2word=self.corpus.vocabulary,
                              num_topics=num_topics)
        tmp_topic_word_matrix = list(lsa.show_topics(num_topics=num_topics,
                                                     num_words=len(self.corpus.vocabulary),
                                                     formatted=False))
        row = []
        col = []
        data = []
        for topic_id in range(self.nb_topics):
            topic_description = tmp_topic_word_matrix[topic_id]
            for weight, word_id in topic_description:
                row.append(topic_id)
                col.append(word_id)
                data.append(weight)
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        self.document_topic_matrix = np.transpose(matutils.corpus2dense(lsa[self.corpus.gensim_vector_space],
                                                                        num_topics,
                                                                        self.corpus.size))
        self.corpus.gensim_vector_space = None


class NonNegativeMatrixFactorization(TopicModel):
    def infer_topics(self, num_topics=10):
        self.nb_topics = num_topics
        nmf = NMF(n_components=num_topics)
        topic_document = nmf.fit_transform(self.corpus.sklearn_vector_space)
        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for (topic_idx, topic) in enumerate(nmf.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        row = []
        col = []
        data = []
        doc_count = 0
        for doc in topic_document:
            topic_count = 0
            for topic_weight in doc:
                row.append(doc_count)
                col.append(topic_count)
                data.append(topic_weight)
                topic_count += 1
            doc_count += 1
        self.document_topic_matrix = coo_matrix((data, (row, col)),
                                                shape=(self.corpus.size, self.nb_topics)).tocsr()
