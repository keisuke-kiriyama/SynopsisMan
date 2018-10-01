import click
import time

from util import paths
import preprocess as p
from embedding.word_embedding import create_path_line_sentences_files, train_word_embedding_model, test_word_embedding_model
from data_construction import word_embedding_avg_vector
from data_construction import similarity_between_contents_and_synopsis_sentence
from data_construction import position_of_sentence

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
    p.contents.execute()
    p.meta.execute()

@cmd.command()
@click.option('--refresh', is_flag=True)
def word_embedding(refresh):
    """
    Word Embedding Modelの学習
    """
    if refresh:
        create_path_line_sentences_files()

    train_word_embedding_model()

@cmd.command()
@click.option('--word', '-w')
def test_word_embedding(word):
    test_word_embedding_model(word)

@cmd.command()
def construct_word_embedding_avg_vector():
    """
    文中の単語の分散表現ベクトルの平均ベクトルのデータを構築する
    """
    word_embedding_avg_vector.construct()

@cmd.command()
def construct_similarity_data():
    """
    本文中の各文とあらすじ文各文の類似度のデータを構築する
    """
    similarity_between_contents_and_synopsis_sentence.construct()

@cmd.command()
def construct_position_of_sentence_data():
    """
    本文各文の出現位置のデータを構築する
    """
    position_of_sentence.construct()


def main():

    start = time.time()
    cmd()

    elapsed_time = time.time() - start
    print("elapsed_time:{:3f}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
    main()