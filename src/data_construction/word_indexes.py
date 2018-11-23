import os
from gensim.models import word2vec
import joblib

from util.corpus_accessor import CorpusAccessor
from util import text_processor
from util.paths import WORD_INDEXES_CONTENTS_PATH, WORD_EMBEDDING_MODEL_PATH


class WordIndexesConstructor():

    def __init__(self):
        self.data_accessor = CorpusAccessor()
        if os.path.isfile(WORD_EMBEDDING_MODEL_PATH):
            self.word_embedding_model = word2vec.Word2Vec.load(WORD_EMBEDDING_MODEL_PATH)

    def convert_index_list(self, line):
        if self.word_embedding_model is None:
            raise ValueError("there is not word embedding model")
        words = text_processor.wakati(line).split()
        index_list = [self.word_embedding_model.wv.vocab[word].index + 1 for word in words]
        return index_list


    def sentence_to_word_indexes(self, ncode):
        """
        小説本文各文を、文中の各単語のインデックスのリストに変換する
        データは文番号をkey、インデックスのリストをvalueとする辞書で保存される
        [1: list, 2: list, ... , n: list]
        """
        print('[PROCESS NCODE]: {}'.format(ncode))
        contents_file_path = os.path.join(WORD_INDEXES_CONTENTS_PATH, ncode + '.txt')
        # if os.path.isfile(contents_file_path):
        #     return

        contents_lines = self.data_accessor.get_contents_lines(ncode)
        synopsis_lines = self.data_accessor.get_synopsis_lines(ncode)
        if not contents_lines or not synopsis_lines:
            return

        index_data = dict()
        for line_idx, line in enumerate(contents_lines):
            if line_idx % 50 == 0:
                print('contents progress: {:.1f}%'.format(line_idx / len(contents_lines) * 100))
            index_list = self.convert_index_list(line)
            index_data[line_idx] = index_list

        print(index_data)

        # データの保存
        # print('[INFO] saving data: {}'.format(ncode))
        # with open(contents_file_path, 'wb') as cf:
        #     joblib.dump(index_data, cf, compress=3)

    def construct(self):
        """
        全小説のデータを構築する
        """
        for i, ncode in enumerate(self.data_accessor.ncodes):
            print('[INFO] num of constructed data: {}'.format(i))
            self.sentence_to_word_indexes(ncode)


def construct():
    constructor = WordIndexesConstructor()
    constructor.construct()

