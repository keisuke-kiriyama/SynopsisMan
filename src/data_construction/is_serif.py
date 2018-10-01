import os
import joblib

from util.paths import IS_SERIF_CONTENTS_DIR_PATH
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def construct_is_serif_data(ncode):
    """
    小説本文各文がセリフか否かのデータを構築
    データは文番号をkey、セリフか否かの0or1をvalueとする辞書で保存される
    [1: 0or1, 2: 0or1, ... , n: 0or1]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    is_serif = {idx: int('「' in line) for idx, line in enumerate(contents_lines)}

    file_path = os.path.join(IS_SERIF_CONTENTS_DIR_PATH, ncode + '.txt')
    print('[INFO] saving data: {}'.format(ncode))
    with open(file_path, 'wb') as f:
        joblib.dump(is_serif, f, compress=1)

def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        construct_is_serif_data(ncode)



