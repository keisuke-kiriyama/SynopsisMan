import numpy as np
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def synopsis_sentence_count():
    total = len(corpus_accessor.ncodes)
    synopsis_sentence_counts = np.zeros(total)
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        synopsis_len = len(corpus_accessor.get_synopsis_lines(ncode))
        synopsis_sentence_counts[i] = synopsis_len
    max_sentence_count = max(synopsis_sentence_counts)
    min_sentence_count = min(synopsis_sentence_counts)
    avg_sentence_count = np.average(synopsis_sentence_counts)
    median_sentence_count = np.median(synopsis_sentence_counts)
    bins = np.arange(min_sentence_count, max_sentence_count+1, 1)
    h, b = np.histogram(synopsis_sentence_counts, bins=bins, density=True)
    print('[INFO] SYNOPSIS SENTENCE')
    print('Max sentence count: ', max_sentence_count)
    print('Min sentence count: ', min_sentence_count)
    print('Median sentence count: ', median_sentence_count)
    print('Average sentence count: ', avg_sentence_count)
    print('Histgram')
    for val, key in zip(h, b):
        print('sentence count: {} , novel count: {:.3f}%'.format(key, val * 100))
    print('\n')

def contents_sentence_count():
    total = len(corpus_accessor.ncodes)
    contents_sentence_counts = np.zeros(total)
    for i, ncode in enumerate(corpus_accessor.ncodes):

        print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        contents_len = len(corpus_accessor.get_contents_lines(ncode))
        contents_sentence_counts[i] = contents_len
    max_sentence_count = max(contents_sentence_counts)
    min_sentence_count = min(contents_sentence_counts)
    avg_sentence_count = np.average(contents_sentence_counts)
    median_sentence_count = np.median(contents_sentence_counts)
    bins = np.arange(0, 5000, 100)
    h, b = np.histogram(contents_sentence_counts, bins=bins, density=False)
    print('[INFO] CONTENTS SENTENCE')
    print('Max sentence count: ', max_sentence_count)
    print('Min sentence count: ', min_sentence_count)
    print('Average sentence count: ', avg_sentence_count)
    print('Median sentence count: ', median_sentence_count)
    print('Histgram')
    for val, key in zip(h, b):
        print('sentence count: {} , novel count: {:.3f}%'.format(key, val / total * 100))
    print('\n')

def summarization_rate():
    total = len(corpus_accessor.ncodes)
    rates = np.zeros(total)
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        contents_len = len(corpus_accessor.get_contents_lines(ncode))
        synopsis_len = len(corpus_accessor.get_synopsis_lines(ncode))
        if contents_len == 0 or synopsis_len == 0: continue
        rate = synopsis_len / contents_len
        rates[i] = rate
    max_rate = max(rates)
    min_rate = min(rates)
    avg_rate = np.average(rates)
    bins = np.arange(0, 0.05, 0.0005)
    hist, bins = np.histogram(rates, bins=bins, density=False)
    print('[INFO] SUMMARIZATION RATE')
    print('Max rate: ', max_rate)
    print('Min rate: ', min_rate)
    print('Average sentence count: ', avg_rate)
    print('Histgram')
    for val, key in zip(hist, bins):
        print('summarization rate: {:.3f} , novel count: {:.3f}%'.format(key * 100, val / total * 100))
    print('\n')


if __name__ == '__main__':
    synopsis_sentence_count()
    contents_sentence_count()
    summarization_rate()
