import numpy as np
import os
from gensim.models import word2vec
import joblib

from util.corpus_accessor import CorpusAccessor
from util import text_processor
from util.paths import WORD_EMBEDDING_VECTORS_CONTENTS_PATH, WORD_EMBEDDING_MODEL_PATH

data_accessor = CorpusAccessor()

if os.path.isfile(WORD_EMBEDDING_MODEL_PATH):
    word_embedding_model = word2vec.Word2Vec.load(WORD_EMBEDDING_MODEL_PATH)

def convert_word_embedding_vectors(sentence):
    """
    文を文中の単語の文さんベクトルのリストに変換
    """
    wakati_line = text_processor.wakati(sentence).split()
    return [word_embedding_model.__dict__['wv'][word] for word in wakati_line]

def sentence_to_word_embedding_vectors(ncode):
    """
    小説本文を、文中における各単語の分散表現のベクトルのリストに変換する
    データは文番号をkey、文ベクトルをvalueとする辞書で保存される
    [1: tensor, 2: tensor, ... , n: tensor]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    contents_lines = data_accessor.get_contents_lines(ncode)
    synopsis_lines = data_accessor.get_synopsis_lines(ncode)
    if not contents_lines or not synopsis_lines:
        return

    # 本文各文のベクトル化
    contents_line_vectors = dict()
    for line_idx, line in enumerate(contents_lines):
        if line_idx % 50 == 0:
            print('contents progress: {:.1f}%'.format(line_idx / len(contents_lines) * 100))
        tensor = convert_word_embedding_vectors(line)
        contents_line_vectors[line_idx] = tensor

    # データの保存
    contents_file_path = os.path.join(WORD_EMBEDDING_VECTORS_CONTENTS_PATH, ncode + '.txt')
    print('[INFO] saving data: {}'.format(ncode))
    with open(contents_file_path, 'wb') as cf:
        joblib.dump(contents_line_vectors, cf, compress=3)


def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(data_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        sentence_to_word_embedding_vectors(ncode)

if __name__ == '__main__':
    construct()

