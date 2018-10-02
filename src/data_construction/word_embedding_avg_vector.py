import numpy as np
import os
from gensim.models import word2vec
import joblib

from util.corpus_accessor import CorpusAccessor
from util import text_processor
from util.paths import WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH, WORD_EMBEDDING_AVG_VECTOR_META_PATH, WORD_EMBEDDING_MODEL_PATH

data_accessor = CorpusAccessor()

print('[INFO] loading word embedding model...')
if os.path.isfile(WORD_EMBEDDING_MODEL_PATH):
    word_embedding_model = word2vec.Word2Vec.load(WORD_EMBEDDING_MODEL_PATH)

def convert_avg_vector(line):
    """
    文を文中の各単語の平均ベクトルに変換
    """
    wakati_line = text_processor.wakati(line).split()
    word_vectors = np.array([word_embedding_model.__dict__['wv'][word] for word in wakati_line])
    return np.average(word_vectors, axis=0)

def sentence_to_word_embedding_avg_vector(ncode):
    """
    小説本文とあらすじ文の各文を、文中における各単語の分散表現の平均ベクトルに変換する
    データは文番号をkey、文ベクトルをvalueとする辞書で保存される
    [1: 文ベクトル, 2: 文ベクトル, ... , n: 文ベクトル]
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
        try:
            vector = convert_avg_vector(line)
        except KeyError as err:
            print(err)
            continue
        except:
            print('[Error] continue to add sentence vectors')
            continue
        contents_line_vectors[line_idx] = vector

    # あらすじ文のベクトル化
    synopsis_line_vectors = dict()
    for line_idx, line in enumerate(synopsis_lines):
        print('synopsis progress: {:.1f}%'.format(line_idx / len(contents_lines) * 100))
        try:
            vector = convert_avg_vector(line)
        except KeyError as err:
            print(err)
            continue
        except:
            print('[Error] continue to add sentence vectors')
            continue
        synopsis_line_vectors[line_idx] = vector

    # データの保存
    contents_file_path = os.path.join(WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH, ncode + '.txt')
    synopsis_file_path = os.path.join(WORD_EMBEDDING_AVG_VECTOR_META_PATH, ncode + '.txt')
    print('[INFO] saving data: {}'.format(ncode))
    with open(contents_file_path, 'wb') as cf:
        joblib.dump(contents_line_vectors, cf, compress=3)
    with open(synopsis_file_path, 'wb') as sf:
        joblib.dump(synopsis_line_vectors, sf, compress=3)

def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(data_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        sentence_to_word_embedding_avg_vector(ncode)








