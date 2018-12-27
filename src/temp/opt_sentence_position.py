import numpy as np
import click

from data_supplier import opt_sentences_data_supplier
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

@click.group()
def cmd():
    pass


@cmd.command()
def test():
    print('test')


def main():
    cmd()

if __name__ == '__main__':
    main()
