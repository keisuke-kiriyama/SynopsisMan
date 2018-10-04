import numpy as np
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def synopsis_sentence_count():
    synopsis_sentence_counts = [len(corpus_accessor.get_synopsis_lines(ncode)) for ncode in corpus_accessor.ncodes]
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


if __name__ == '__main__':
    synopsis_sentence_count()
