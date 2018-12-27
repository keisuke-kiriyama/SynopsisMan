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
    total = len(corpus_accessor.exist_ncodes)
    for i, ncode in enumerate(corpus_accessor.exist_ncodes):
        if i % 100 == 0:
            print('[INFO] PROGRESS: {:.1f}'.format(i/total*100))
        contents_lines = corpus_accessor.get_contents_lines(ncode)
        data = opt_sentences_data_supplier.load(ncode)['opt_sentence_index']

        print(len(contents_lines))
        print(data)

        print(data/len(contents_lines))


        return


def main():
    cmd()

if __name__ == '__main__':
    main()
