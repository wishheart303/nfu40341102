# -*- coding: iso-8859-15 -*-

from __future__ import division # want 3 / 2 == 1.5
import re, math, random # regexes, math functions, random numbers
import matplotlib.pyplot as plt # pyplot
from collections import defaultdict, Counter
from functools import partial

# 
# functions for working with vectors
# 2017/03/23

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

a1 = [1, 2]
a2 = [3, 4]
print("vector_add(a1) = " + str(vector_add(a1, a2)))
print("vector_subtract(a2) = " + str(vector_subtract(a1, a2)))

def vector_sum(vectors):
    return reduce(vector_add, vectors)

a3 = [[1, 2], [3, 4], [5, 6]]
print("vector_sum(a3) = " + str(vector_sum(a3)))

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

print("scalar_multiply(10, a1) = " + str(scalar_multiply(10, a1)))

# this isn't right if you don't from __future__ import division
def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

print("vector_mean(a3) = " + str(vector_mean(a3)))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

print("dot(a1, a2) = " + str(dot(a1, a2)))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

print("magnitude(a1) = " + str(magnitude(a1)))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
   return math.sqrt(squared_distance(v, w))

