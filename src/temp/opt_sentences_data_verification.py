import numpy as np
import click

from data_supplier import opt_sentences_data_supplier
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

@click.group()
def cmd():
    pass

@cmd.command()
def verificate_rouge_score():
    scores = []
    total = len(corpus_accessor.exist_ncodes)
    for i, ncode in enumerate(corpus_accessor.exist_ncodes):
        print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        data = opt_sentences_data_supplier.load(ncode)
        scores.append(data['rouge']['r'])
    bins = np.arange(0, 1, 0.05)
    h, b = np.histogram(scores, bins=bins)
    for key, value in zip(b, h):
        print('{:.2f}: {:.1f}%'.format(key, value / total * 100))

@cmd.command()
def verificate_sentence_count():
    count = []
    total = len(corpus_accessor.exist_ncodes)
    for i, ncode in enumerate(corpus_accessor.exist_ncodes):
        if i % 100 == 0:
            print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        data = opt_sentences_data_supplier.load(ncode)
        count.append(len(data['opt_sentence_index']))
    bins = np.arange(1, 7, 1)
    h, b = np.histogram(count, bins=bins)
    for key, value in zip(b, h):
        print('{:.2f}: {}'.format(key, value))

@cmd.command()
def verificate_long_short_novel():
    """
    あらすじが１文の場合と6文の場合で長編と短編の割合をみる
    """
    one_long = 0
    one_short = 0
    six_long = 0
    six_short = 0
    total = len(corpus_accessor.exist_ncodes)
    for i, ncode in enumerate(corpus_accessor.exist_ncodes):
        if i % 100 == 0:
            print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        data = opt_sentences_data_supplier.load(ncode)
        length = len(data['opt_sentence_index'])
        if length == 1 and corpus_accessor.is_long(ncode):
            one_long += 1
        elif length == 1 and not corpus_accessor.is_long(ncode):
            one_short += 1
        elif length == 6 and corpus_accessor.is_long(ncode):
            six_long += 1
        elif length == 6 and not corpus_accessor.is_long(ncode):
            six_short += 1
    print('one_long: ', one_long)
    print('one_short: ', one_short)
    print('six_long: ', six_long)
    print('six_short', six_short)



def main():
    cmd()

if __name__ == '__main__':
    main()