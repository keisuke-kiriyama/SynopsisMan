import os
import joblib
import re

from util.paths import IS_INCLUDE_PERSON_CONTENTS_PATH
from util.corpus_accessor import CorpusAccessor
from util.text_processor import tagger

corpus_accessor = CorpusAccessor()

tagger = tagger()

def construct_is_include_person_data(ncode):
    """
    小説本文各文が人名を含むか否かのデータを構築
    データは文番号をkey、セリフか否かの0or1をvalueとする辞書で保存される
    [1: 0or1, 2: 0or1, ... , n: 0or1]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    if not contents_lines:
        return
    is_include_person_dict = dict()
    for line_idx, line in enumerate(contents_lines):
        morph_info = [re.split('[,\t]', morph)[3] for morph in tagger.parse(line).split('\n') if morph not in ['', 'EOS']]
        is_include_person_dict[line_idx] = int('人名' in morph_info)

    file_path = os.path.join(IS_INCLUDE_PERSON_CONTENTS_PATH, ncode + '.txt')
    print('[INFO] saving data: {}'.format(ncode))
    with open(file_path, 'wb') as f:
        joblib.dump(is_include_person_dict, f, compress=1)

def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        construct_is_include_person_data(ncode)
