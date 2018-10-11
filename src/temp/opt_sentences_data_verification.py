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
    total = len(corpus_accessor.active_ncodes)
    for i, ncode in enumerate(corpus_accessor.active_ncodes):
        print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        data = opt_sentences_data_supplier.load(ncode)
        scores.append(data['rouge']['r'])
    bins = np.arange(0, 1, 0.05)
    h, b = np.histogram(scores, bins=bins)
    for key, value in zip(b, h):
        print('{:.2f}: {}'.format(key, value))


def main():
    cmd()

if __name__ == '__main__':
    main()