import numpy as np
import os
import joblib

from util.paths import POSITION_OF_SENTENCE_CONTENTS_DIR_PATH
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def construct_position_of_sentence_data(ncode):
    """
    小説本文各文の出現位置のデータを構築
    データは文番号をkey、文の出現位置[0:1]をvalueとする辞書で保存される
    [1: 類似度, 2: 類似度, ... , n: 類似度]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    contents_len = len(corpus_accessor.get_contents_lines(ncode))
    if contents_len == 0:
        return
    position_data = np.arange(0, 1, 1 / contents_len, dtype=float)
    position_dict = {idx: position for idx, position in enumerate(position_data)}

    file_path = os.path.join(POSITION_OF_SENTENCE_CONTENTS_DIR_PATH, ncode + '.txt')
    with open(file_path, 'wb') as f:
        joblib.dump(position_dict, f, compress=1)

def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(corpus_accessor.ncodes):
        construct_position_of_sentence_data(ncode)




