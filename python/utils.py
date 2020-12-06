"""
Utility functions.
"""

from itertools import accumulate, repeat, chain, islice, tee
from functools import reduce
from collections import deque
from operator import itemgetter

def identity(x):
    return x

def call(x, f):
    return f(x)

def pipe(*fs):
    f, *_fs = fs
    return lambda *args, **kwargs: reduce(call, _fs, f(*args, **kwargs))
    
def fmap(f):
    return lambda xs: tuple(map(f, xs))

def ffmap(f):
    return fmap(fmap(f))

def graph(f):
    return lambda x: (x, f(x))

def iterate(f):
    return lambda x: accumulate(repeat(x), lambda x, _: f(x))
    
def scan(f, ys, x):
    return accumulate(ys, f, initial=x)

def tail(iterable, n=1):
    return islice(iterable, n, None)

def last(iterable):
    return deque(iterable, maxlen=1).pop()
    
def take(n, iterable):
    return tuple(islice(iterable, n))
    
def transpose(xs):
    return zip(*xs)

flatten = chain.from_iterable

def len_(iterable):
    return sum(1 for _ in iterable)

def const(x):
    return lambda: x

def pairs(xs):
    xs, _xs = tee(xs)
    return zip(xs, tail(_xs))

fst = itemgetter(0)
snd = itemgetter(1)
