# coding: utf-8
import itertools

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def to_dense_vector(sparse_vector, vector_length):
    cx = sparse_vector.tocoo()
    dense_vector = [0.0] * vector_length
    for row, col, dat in itertools.zip_longest(cx.row, cx.col, cx.data):
        dense_vector[col] = dat
    return dense_vector
