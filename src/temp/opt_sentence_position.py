import numpy as np
import click

from data_supplier import opt_sentences_data_supplier
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

@click.group()
def cmd():
    pass


@cmd.command()
def verificate_opt_sentence_position():
    total_ncodes = len(corpus_accessor.exist_ncodes)
    result = np.zeros(10, dtype=int)
    for i, ncode in enumerate(corpus_accessor.exist_ncodes):
        if i % 100 == 0:
            print('[INFO] PROGRESS: {:.1f}'.format(i/total_ncodes*100))
        contents_lines = corpus_accessor.get_contents_lines(ncode)
        positions = opt_sentences_data_supplier.load(ncode)['opt_sentence_index'] / len(contents_lines)
        rounded_positions = [round(value, 1) for value in positions]
        bins = np.arange(0, 1.1, 0.1)
        h, _ = np.histogram(rounded_positions, bins=bins)
        result += h

        if i == 20:
            break
    print(result)


def main():
    cmd()

if __name__ == '__main__':
    main()
