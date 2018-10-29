import numpy as np

from util.corpus_accessor import CorpusAccessor

accessor = CorpusAccessor()

def count_words():
    total = len(accessor.exist_ncodes)
    counts = []
    for i, ncode in enumerate(accessor.exist_ncodes):
        print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        contents = accessor.get_wakati_contents_lines(ncode)
        for sentence in contents:
            counts.append(len(sentence))

    print('average: ', np.average(counts))
    print('max: ', np.max(counts))
    print('min: ', np.min(counts))




if __name__ == '__main__':
    count_words()

