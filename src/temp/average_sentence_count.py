import numpy as np
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def synopsis_sentence_count():
    synopsis_sentence_counts = []
    for ncode in corpus_accessor.ncodes:
        synopsis_len = len(corpus_accessor.get_synopsis_lines(ncode))
        synopsis_sentence_counts.append(synopsis_len)
    max_sentence_count = max(synopsis_sentence_counts)
    min_sentence_count = min(synopsis_sentence_counts)
    avg_sentence_cuont = np.average(synopsis_sentence_counts)
    bins = np.arange(0, 30, 1)
    hist, bins = np.histogram(synopsis_sentence_counts, bins=bins)
    print('Max sentence count: ', max_sentence_count)
    print('Min sentence count: ', min_sentence_count)
    print('Average sentence count: ', avg_sentence_cuont)
    print('Histgram')
    for val, key in zip(hist, bins):
        print('sentence count: {} , novel count: {}'.format(key, val))

def contents_sentence_count():
    contents_sentence_counts = []
    for ncode in corpus_accessor.ncodes:
        contents_len = len(corpus_accessor.get_contents_lines(ncode))
        contents_sentence_counts.append(contents_len)
    max_sentence_count = max(contents_sentence_counts)
    min_sentence_count = min(contents_sentence_counts)
    avg_sentence_cuont = np.average(contents_sentence_counts)
    bins = np.arange(0, 5000, 100)
    hist, bins = np.histogram(contents_sentence_counts, bins=bins)
    print('Max sentence count: ', max_sentence_count)
    print('Min sentence count: ', min_sentence_count)
    print('Average sentence count: ', avg_sentence_cuont)
    print('Histgram')
    for val, key in zip(hist, bins):
        print('sentence count: {} , novel count: {}'.format(key, val))

def summarization_rate():
    rates = []
    for ncode in corpus_accessor.ncodes:
        contents_len = len(corpus_accessor.get_contents_lines(ncode))
        synopsis_len = len(corpus_accessor.get_synopsis_lines(ncode))
        if contents_len == 0 or synopsis_len == 0: continue
        rate = synopsis_len / contents_len
        rates.append(rate)
    max_rate = max(rates)
    min_rate = min(rates)
    avg_rate = np.average(rates)
    bins = np.arange(0, 0.5, 0.01)
    hist, bins = np.histogram(rates, bins=bins, density=True)
    print('Max rate: ', max_rate)
    print('Min rate: ', min_rate)
    print('Average sentence count: ', avg_rate)
    print('Histgram')
    for val, key in zip(hist, bins):
        print('rate: {} , novel count: {}'.format(key, val))


if __name__ == '__main__':
    # synopsis_sentence_count()
    # contents_sentence_count()
    summarization_rate()
