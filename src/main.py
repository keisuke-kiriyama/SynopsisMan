import click
import time
import os
import re
import sys

from util import paths
from util.corpus_accessor import CorpusAccessor
import preprocess as p
from embedding.word_embedding import create_path_line_sentences_files, train_word_embedding_model, test_word_embedding_model
import data_construction
from summarizer.lstm_summarizer import LSTMSummarizer
from summarizer.dnn_summarizer import DNNSummarizer
from data_supplier.lstm_vector_supplier import LSTMVectorSupplier
from data_supplier.dnn_vector_supplier import DNNVectorSupplier
from evaluate import rouge_evaluation
from rouge import Rouge
from util.text_processor import wakati

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
def remove_tsubame_log():
    """
    tsubameで実行時に作成されるログファイルを削除する
    """
    for file_name in os.listdir(paths.SRC_DIR_PATH):
        splited = file_name.split('.')
        if not len(splited) == 2: continue
        pattern = '^[e,o]\d+'
        if re.match(pattern, splited[1]):
            file_path = os.path.join(paths.SRC_DIR_PATH, file_name)
            print('remove: ', file_path)
            os.remove(file_path)

@cmd.command()
def data_mkdir():
    """
    データ用のディレクトリを一斉作成
    """
    paths.mkdir()

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
def construct_embedding_matrix():
    """
    単語の分散表現のマトリックス構築
    """
    data_construction.embedding_matrix.construct()

@cmd.command()
def construct_word_embedding_vectors():
    """
    文中の単語の分散表現ベクトルのリストのデータを構築する
    """
    data_construction.word_embedding_vectors.construct()

@cmd.command()
def construct_word_embedding_avg_vector():
    """
    文中の単語の分散表現ベクトルの平均ベクトルのデータを構築する
    """
    data_construction.word_embedding_avg_vector.construct()

@cmd.command()
def construct_word_indexes():
    """
    文中の単語のインデックスを要素とするリストのデータを構築する
    """
    data_construction.word_indexes.construct()

@cmd.command()
def construct_similarity_data():
    """
    本文中の各文とあらすじ文各文の類似度のデータを構築する
    """
    data_construction.similarity_between_contents_and_synopsis_sentence.construct()

@cmd.command()
def construct_rouge_similarity_data():
    """
    本文中の各文とあらすじ文各文のROUGEのデータを構築する
    """
    data_construction.rouge_between_contents_and_synopsis_sentence.construct()

@cmd.command()
def construct_position_of_sentence_data():
    """
    本文各文の出現位置のデータを構築する
    """
    data_construction.position_of_sentence.construct()

@cmd.command()
def construct_is_serif_data():
    """
    本文各文がセリフか否かのデータを構築する
    """
    data_construction.is_serif.construct()

@cmd.command()
def construct_is_include_person_data():
    """
    本文各文に人名が含まれるか否かのデータを構築する
    """
    data_construction.is_include_person.construct()

@cmd.command()
def construct_sentence_length_data():
    """
    本文各文の文長データの構築
    """
    data_construction.sentence_length.construct()

@cmd.command()
@click.option('--short_rate', '-s', default=0.051)
@click.option('--long_rate', '-l', default=0.013)
@click.option('--min_sentence_count', '-min', default=1)
@click.option('--max_sentence_count', '-max', default=6)
def construct_opt_sentences_data(short_rate, long_rate, min_sentence_count, max_sentence_count):
    data_construction.opt_sentences.construct(short_rate=short_rate,
                                              long_rate=long_rate,
                                              min_sentence_count=min_sentence_count,
                                              max_sentence_count=max_sentence_count)

@cmd.command()
@click.option('--threshold', '-t', default=0.3)
def construct_active_ncodes_data(threshold):
    data_construction.active_ncodes.construct(threshold)

@cmd.command()
@click.option('--genre', '-g', default='general')
@click.option('--importance', '-i', default='cos_sim')
@click.option('--position', is_flag=True)
@click.option('--serif', is_flag=True)
@click.option('--person', is_flag=True)
@click.option('--sentence_length', is_flag=True)
def dnn_summarizer_fit(genre,
                        importance,
                        position=False,
                        serif=False,
                        person=False,
                        sentence_length=False):
    summarizer = DNNSummarizer()
    supplier = DNNVectorSupplier(genre,
                                  importance=importance,
                                  use_data_of_position_of_sentence=position,
                                  use_data_of_is_serif=serif,
                                  use_data_of_is_include_person=person,
                                  use_data_of_sentence_length=sentence_length)
    summarizer.set_supplier(supplier)
    summarizer.fit()

@cmd.command()
@click.option('--genre', '-g', default='general')
@click.option('--importance', '-i', default='cos_sim')
@click.option('--position', is_flag=True)
@click.option('--serif', is_flag=True)
@click.option('--person', is_flag=True)
@click.option('--sentence_length', is_flag=True)
def lstm_summarizer_fit(genre,
                        importance,
                        position=False,
                        serif=False,
                        person=False,
                        sentence_length=False):
    summarizer = LSTMSummarizer()
    supplier = LSTMVectorSupplier(genre,
                                  importance=importance,
                                  use_data_of_position_of_sentence=position,
                                  use_data_of_is_serif=serif,
                                  use_data_of_is_include_person=person,
                                  use_data_of_sentence_length=sentence_length)
    summarizer.set_supplier(supplier)
    summarizer.fit()

@cmd.command()
@click.option('--genre', '-g', default='general')
@click.option('--importance', '-i', default='cos_sim')
@click.option('--model_name', '-m', default='')
@click.option('--position', is_flag=True)
@click.option('--serif', is_flag=True)
@click.option('--person', is_flag=True)
@click.option('--sentence_length', is_flag=True)
def dnn_summarizer_re_fit(genre,
                        importance,
                        model_name='',
                        position=False,
                        serif=False,
                        person=False,
                        sentence_length=False):
    summarizer = DNNSummarizer()
    supplier = DNNVectorSupplier(genre,
                                  importance=importance,
                                  use_data_of_position_of_sentence=position,
                                  use_data_of_is_serif=serif,
                                  use_data_of_is_include_person=person,
                                  use_data_of_sentence_length=sentence_length)

    summarizer.set_supplier(supplier)
    summarizer.re_fit(model_name)

@cmd.command()
@click.option('--genre', '-g', default='general')
@click.option('--importance', '-i', default='cos_sim')
@click.option('--model_name', '-m', default='')
@click.option('--position', is_flag=True)
@click.option('--serif', is_flag=True)
@click.option('--person', is_flag=True)
@click.option('--sentence_length', is_flag=True)
def lstm_summarizer_re_fit(genre,
                        importance,
                        model_name='',
                        position=False,
                        serif=False,
                        person=False,
                        sentence_length=False):
    summarizer = LSTMSummarizer()
    supplier = LSTMVectorSupplier(genre,
                                  importance=importance,
                                  use_data_of_position_of_sentence=position,
                                  use_data_of_is_serif=serif,
                                  use_data_of_is_include_person=person,
                                  use_data_of_sentence_length=sentence_length)

    summarizer.set_supplier(supplier)
    summarizer.re_fit(model_name)

@cmd.command()
@click.option('--genre', '-g', default='general')
@click.option('--importance', '-i', default='cos_sim')
@click.option('--position', is_flag=True)
@click.option('--serif', is_flag=True)
@click.option('--person', is_flag=True)
@click.option('--sentence_length', is_flag=True)
def evaluate_rouge_score(genre,
                         importance,
                         position=False,
                         serif=False,
                         person=False,
                         sentence_length=False):

    rouge_evaluation.evaluate(genre,
                              importance=importance,
                              use_data_of_position_of_sentence=position,
                              use_data_of_is_serif=serif,
                              use_data_of_is_include_person=person,
                              use_data_of_sentence_length=sentence_length)

@cmd.command()
@click.option('--importance', '-i', default='cos_sim')
@click.option('--ncode', '-n', default='')
@click.option('--position', is_flag=True)
@click.option('--serif', is_flag=True)
@click.option('--person', is_flag=True)
@click.option('--sentence_length', is_flag=True)
def generate(importance,
             ncode,
             position=False,
             serif=False,
             person=False,
             sentence_length=False):
    corpus_accessor = CorpusAccessor()
    genre = corpus_accessor.get_genre(ncode)
    if len(genre) == 0:
        print('non genre')
        return
    s = LSTMSummarizer()
    supplier = LSTMVectorSupplier(genre,
                                  importance,
                                  use_data_of_position_of_sentence=position,
                                  use_data_of_is_serif=serif,
                                  use_data_of_is_include_person=person,
                                  use_data_of_sentence_length=sentence_length)
    s.set_supplier(supplier)
    s.set_trained_model()
    print(s.generate(ncode=ncode))

@cmd.command()
@click.option('--importance', '-i', default='cos_sim')
@click.option('--start', '-s', default=0)
@click.option('--end', '-e', default=30)
def multi_generate(importance, start, end):
    corpus_accessor = CorpusAccessor()
    output_file_path = 'result_start_' + str(start) + '_end_' + str(end) + '.txt'
    file = open(output_file_path, 'w')


    love_story_s = LSTMSummarizer()
    love_story_supplier = LSTMVectorSupplier('love_story',
                                  importance,
                                  use_data_of_position_of_sentence=True,
                                  use_data_of_is_serif=True,
                                  use_data_of_is_include_person=True,
                                  use_data_of_sentence_length=True)
    love_story_s.set_supplier(love_story_supplier)
    love_story_s.set_trained_model()

    fantasy_s = LSTMSummarizer()
    fantasy_supplier = LSTMVectorSupplier('fantasy',
                                  importance,
                                  use_data_of_position_of_sentence=True,
                                  use_data_of_is_serif=True,
                                  use_data_of_is_include_person=True,
                                  use_data_of_sentence_length=True)
    fantasy_s.set_supplier(fantasy_supplier)
    fantasy_s.set_trained_model()

    literature_s = LSTMSummarizer()
    literature_supplier = LSTMVectorSupplier('literature',
                                  importance,
                                  use_data_of_position_of_sentence=True,
                                  use_data_of_is_serif=True,
                                  use_data_of_is_include_person=True,
                                  use_data_of_sentence_length=True)
    literature_s.set_supplier(literature_supplier)
    literature_s.set_trained_model()

    sf_s = LSTMSummarizer()
    sf_supplier = LSTMVectorSupplier('sf',
                                  importance,
                                  use_data_of_position_of_sentence=True,
                                  use_data_of_is_serif=True,
                                  use_data_of_is_include_person=True,
                                  use_data_of_sentence_length=True)
    sf_s.set_supplier(sf_supplier)
    sf_s.set_trained_model()

    sys.setrecursionlimit(20000)
    rouge = Rouge()

    for i, ncode in enumerate(corpus_accessor.exist_ncodes[start:end]):
        print('processed ncode count: ', i)

        genre = corpus_accessor.get_genre(ncode)
        if len(genre) == 0:
            print('non genre')
            continue
        ref = ''.join(corpus_accessor.get_synopsis_lines(ncode))

        synopsis = ''
        if genre == 'love_story':
            synopsis = love_story_s.generate(ncode)
        elif genre == 'fantasy':
            synopsis = fantasy_s.generate(ncode)
        elif genre == 'literature':
            synopsis = literature_s.generate(ncode)
        elif genre == 'sf':
            synopsis = sf_s.generate(ncode)

        score = rouge.get_scores(wakati(synopsis), wakati(ref), False)[0]['rouge-1']['r']

        file.write(ncode + '\n')
        file.write(genre + '\n')
        file.write('score: ' + str(score) + '\n')
        file.write(ref + '\n\n')
        file.write(synopsis + '\n\n\n')
    file.close()

def main():
    cmd()

if __name__ == '__main__':
    main()

