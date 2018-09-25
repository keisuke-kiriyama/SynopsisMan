import click
import time

from util import paths
import preprocess as p
from embedding.word_embedding import create_path_line_sentences_files

@click.group()
def cmd():
    pass

@cmd.command()
def check_paths():
    """
    設定されたPATHの確認
    """
    paths.check()

@cmd.command()
def preprocess():
    """
    スクレイピングしたデータを正しく文分割したデータに変換する
    """
    start = time.time()

    p.contents.execute()
    p.meta.execute()

    elapsed_time = time.time() - start
    print("elapsed_time:{:3f}".format(elapsed_time) + "[sec]")

@cmd.command()
@click.option('--refresh', '-r', default=False)
def train_word_embedding_model(refresh):
    start = time.time()
    if refresh:
        create_path_line_sentences_files()

    elapsed_time = time.time() - start
    print("elapsed_time:{:3f}".format(elapsed_time) + "[sec]")


def main():
    cmd()

if __name__ == '__main__':
    main()