import os
import joblib

from util.paths import SENTENCE_LENGTH_CONTENTS_PATH
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def construct_sentence_length_data(ncode):
    """
    本文各文の文長(最大文長で正規化)
    データは文番号をkey、正規化された文長をvalueとする辞書で保存される
    [1: [0,1], 2: [0,1], ... , n: [0,1]]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    synopsis_lines = corpus_accessor.get_synopsis_lines(ncode)
    if not contents_lines or not synopsis_lines:
        return
    raw_length_data = [len(line) for line in contents_lines]
    max_len = max(raw_length_data)
    length_data = {idx: length/max_len for idx, length in enumerate(raw_length_data)}

    file_path = os.path.join(SENTENCE_LENGTH_CONTENTS_PATH, ncode + '.txt')
    print('[INFO] saving data: {}'.format(ncode))
    with open(file_path, 'wb') as f:
        joblib.dump(length_data, f, compress=1)


def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        construct_sentence_length_data(ncode)