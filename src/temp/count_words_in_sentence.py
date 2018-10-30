import numpy as np
import matplotlib.pyplot as plt

from util.corpus_accessor import CorpusAccessor

accessor = CorpusAccessor()

def count_words():
    total = len(accessor.exist_ncodes)
    counts = []
    for i, ncode in enumerate(accessor.exist_ncodes):
        print('[INFO] PROGRESS: {:.1f}'.format(i/400 * 100))
        contents = accessor.get_wakati_contents_lines(ncode)
        for sentence in contents:
            counts.append(len(sentence))
        if i == 1: break

    print('average: ', np.average(counts))
    print('max: ', np.max(counts))
    print('min: ', np.min(counts))
    print('std: ', np.std(counts))



    bins = np.arange(0, 100, 1)
    h, b = np.histogram(counts, bins=bins)

    for h, b in zip(h,b):
        print(b, ' ', h)

    plt.hist(h, bins=b)
    plt.show()





if __name__ == '__main__':
    count_words()

