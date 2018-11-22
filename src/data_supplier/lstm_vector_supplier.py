import os
import joblib
import numpy as np
from gensim.models import word2vec

import data_supplier
from data_supplier.active_ncodes_supplier import ncodes_train_test_split
from util.paths import LSTM_TRAINED_MODEL_DIR_PATH
from util.corpus_accessor import CorpusAccessor
from util.paths import WORD_EMBEDDING_MODEL_PATH
from util import text_processor

data_accessor = CorpusAccessor()

class LSTMVectorSupplier:

    def __init__(self,
                 genre='general',
                 importance='cos_sim',
                 use_data_of_position_of_sentence=False,
                 use_data_of_is_serif=False,
                 use_data_of_is_include_person=False,
                 use_data_of_sentence_length=False):
        """
        このクラスの初期化時に使用する素性を指定する
        :param word_embedding_avg_vector: bool
        単語の分散表現ベクトルの平均ベクトル
        :param position_of_sentence: bool
        文の出現位置
        :param is_serif: bool
        セリフか否か
        :param is_include_person: bool
        人名を含むか否か
        :param sentence_length: bool
        文の文字数
        """
        if not importance in ['cos_sim', 'rouge']:
            raise ValueError('[ERROR]importance must be cos_soim or rouge')

        if not os.path.isfile(WORD_EMBEDDING_MODEL_PATH):
            raise ValueError('[ERROR] word embedding model is not exist')
        self.word_embedding_model = word2vec.Word2Vec.load(WORD_EMBEDDING_MODEL_PATH)
        self.vocabulary_size = len(self.word_embedding_model.wv.vocab)

        self.genre = genre
        self.importance = importance
        self.use_data_of_position_of_sentence = use_data_of_position_of_sentence
        self.use_data_of_is_serif = use_data_of_is_serif
        self.use_data_of_is_include_person = use_data_of_is_include_person
        self.use_data_of_sentence_length = use_data_of_sentence_length

        # Feature vector dimension
        self.word_embedding_vector_dim = 200
        position_of_sentence_dim = 1
        is_serif_dim = 1
        is_include_person_dim = 1
        sentence_length_dim = 1

        self.multi_feature_dim = 0
        if use_data_of_position_of_sentence:
            self.multi_feature_dim += position_of_sentence_dim
        if use_data_of_is_serif:
            self.multi_feature_dim += is_serif_dim
        if use_data_of_is_include_person:
            self.multi_feature_dim += is_include_person_dim
        if use_data_of_sentence_length:
            self.multi_feature_dim += sentence_length_dim

        # NCode
        self.train_ncodes, self.test_ncodes, self.validation_ncodes = ncodes_train_test_split(genre=genre,
                                                                                              validation_size=0.01,
                                                                                              test_size=0.2)
        # Num of sentences used per batch
        self.batch_size = 50
        # Max num of words in sentence
        self.max_count_of_words = 200
        # Shape of per batch
        self.word_index_batch_shape = (self.batch_size, self.max_count_of_words)
        # Shape of multi features vector
        self.multi_feature_batch_shape = (self.batch_size, self.multi_feature_dim)


    def trained_model_dir_path(self):
        """
        使用する素性に基づいて保存するディレクトリのpathを決定する
        """
        feature_dir_name = self.importance + '_'
        feature_dir_name += 'emb_'
        if self.use_data_of_position_of_sentence:
            feature_dir_name += 'pos_'
        if self.use_data_of_is_serif:
            feature_dir_name += 'ser_'
        if self.use_data_of_is_include_person:
            feature_dir_name += 'per_'
        if self.use_data_of_sentence_length:
            feature_dir_name += 'len_'
        path = os.path.join(LSTM_TRAINED_MODEL_DIR_PATH, self.genre)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, feature_dir_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def get_trained_model_path(self):
        """
        使用する素性に基づいて訓練済みデータのpathを取得する
        :return:
        """
        dir_path = self.trained_model_dir_path()
        if not os.path.isdir(dir_path) or len(os.listdir(dir_path)) == 0:
            raise ValueError("Nothing trained model")

        trained_model_candidate = os.listdir(dir_path)
        vloss_dict = dict()
        for candidate in trained_model_candidate:
            if not 'model' in candidate: continue
            vloss = str(candidate.split('vloss')[1]).replace('.hdf5', '')
            vloss_dict[vloss] = candidate

        min_vloss = min(vloss_dict.keys())
        model_path = os.path.join(dir_path, vloss_dict[min_vloss])
        return model_path

    def train_steps_per_epoch(self):
        train_sentence_count = self.total_sentence_count(self.train_ncodes)
        return int(train_sentence_count / self.batch_size)

    def validation_steps_per_epoch(self):
        validation_sentence_count = self.total_sentence_count(self.validation_ncodes)
        return int(validation_sentence_count / self.batch_size)

    def total_sentence_count(self, ncodes):
        total = 0
        if self.importance == 'cos_sim':
            supplier = data_supplier.similarity_data_supplier
        elif self.importance == 'rouge':
            supplier = data_supplier.rouge_data_supplier
        else:
            raise ValueError('[ERROR]importance must be cos_soim or rouge')
        for i, ncode in enumerate(ncodes):
            if i % 1000 == 0:
                print('[INFO] sentence counting progress: {:.1f}%'.format(i / len(ncodes) * 100))
            total += len(supplier.load(ncode).keys())
        print(total)
        return total

    def word_index_add_one(self, word):
        if not word in self.word_embedding_model.wv.vocab:
            raise ValueError('[ERROR] word does not exist in vocabulary', word)
        return self.word_embedding_model.wv.vocab[word].index + 1

    def train_data_generator(self):
        return self.data_generator(self.train_ncodes)

    def validation_data_generator(self):
        return self.data_generator(self.validation_ncodes)

    def test_data_generator(self):
        return self.data_generator(self.test_ncodes)

    def data_generator(self, ncodes):
        while 1:
            word_index_batch_data = np.empty(self.word_index_batch_shape)
            multi_feature_batch_data = np.empty(self.multi_feature_batch_shape)
            label_batch_data = np.empty(self.batch_size)
            position_in_batch = 0

            for ncode in ncodes:
                contents_lines = data_accessor.get_contents_lines(ncode)

                if self.importance == 'cos_sim':
                    similarity_data = data_supplier.similarity_data_supplier.load(ncode)
                elif self.importance == 'rouge':
                    similarity_data = data_supplier.rouge_data_supplier.load(ncode)
                else:
                    raise ValueError('[ERROR]importance must be cos_soim or rouge')
                sentence_count = len(similarity_data)

                if not len(contents_lines) == sentence_count:
                    raise ValueError('[ERROR] num of contents lines is not equal to similarity data count')

                data_of_word_indexes = data_supplier.word_indexes_supplier.load(ncode)
                data_of_position_of_sentence = None
                data_of_is_serif = None
                data_of_is_include_person = None
                data_of_sentence_length = None

                if self.use_data_of_position_of_sentence:
                    data_of_position_of_sentence = data_supplier.position_of_sentence_data_supplier.load(ncode)
                if self.use_data_of_is_serif:
                    data_of_is_serif = data_supplier.is_serif_data_supplier.load(ncode)
                if self.use_data_of_is_include_person:
                    data_of_is_include_person = data_supplier.is_include_person_data_supplier.load(ncode)
                if self.use_data_of_sentence_length:
                    data_of_sentence_length = data_supplier.sentence_length_data_supplier.load(ncode)

                for index in range(sentence_count):
                    # 文中の単語インデックスの系列ベクトル構築
                    word_index_sequence = np.zeros(self.max_count_of_words, dtype='int32')
                    word_indexes = data_of_word_indexes[index]
                    word_indexes_length = min(len(word_indexes), self.max_count_of_words)
                    word_index_sequence[0: word_indexes_length] = word_indexes

                    # 追加の素性ベクトルの構築
                    multi_feature_vector = []
                    if self.use_data_of_position_of_sentence:
                        multi_feature_vector.append(data_of_position_of_sentence[index])
                    if self.use_data_of_is_serif:
                        multi_feature_vector.append(data_of_is_serif[index])
                    if self.use_data_of_is_include_person:
                        multi_feature_vector.append(data_of_is_include_person[index])
                    if self.use_data_of_sentence_length:
                        multi_feature_vector.append(data_of_sentence_length[index])

                    if not len(multi_feature_vector) == self.multi_feature_dim:
                        raise ValueError("[ERROR] not equal length of input vector({}) and input vector size({})"
                                         .format(len(multi_feature_vector), self.multi_feature_dim))

                    word_index_batch_data[position_in_batch] = word_index_sequence
                    multi_feature_batch_data[position_in_batch] = multi_feature_vector
                    label_batch_data[position_in_batch] = similarity_data[index]
                    position_in_batch += 1

                    if position_in_batch == self.batch_size:
                        yield ({'sequence': word_index_batch_data, 'features': multi_feature_batch_data},
                               {'main_output': label_batch_data, 'aux_output': label_batch_data})
                        word_index_batch_data = np.empty(self.word_index_batch_shape)
                        multi_feature_batch_data = np.empty(self.multi_feature_batch_shape)
                        label_batch_data = np.empty(self.batch_size)
                        position_in_batch = 0


    def test_data_input(self, ncode):
        contents_lines = data_accessor.get_contents_lines(ncode)

        if self.importance == 'cos_sim':
            similarity_data = data_supplier.similarity_data_supplier.load(ncode)
        elif self.importance == 'rouge':
            similarity_data = data_supplier.rouge_data_supplier.load(ncode)
        else:
            raise ValueError('[ERROR]importance must be cos_soim or rouge')
        sentence_count = len(similarity_data)

        if not len(contents_lines) == sentence_count:
            raise ValueError('[ERROR] num of contents lines is not equal to similarity data count')

        data_of_position_of_sentence = None
        data_of_is_serif = None
        data_of_is_include_person = None
        data_of_sentence_length = None

        if self.use_data_of_position_of_sentence:
            data_of_position_of_sentence = data_supplier.position_of_sentence_data_supplier.load(ncode)
        if self.use_data_of_is_serif:
            data_of_is_serif = data_supplier.is_serif_data_supplier.load(ncode)
        if self.use_data_of_is_include_person:
            data_of_is_include_person = data_supplier.is_include_person_data_supplier.load(ncode)
        if self.use_data_of_sentence_length:
            data_of_sentence_length = data_supplier.sentence_length_data_supplier.load(ncode)

        word_index_data = []
        multi_feature_data = []

        for index in range(sentence_count):
            # 文中の単語インデックスの系列ベクトル構築
            word_index_sequence = np.zeros(self.max_count_of_words, dtype='int32')
            words = text_processor.wakati(contents_lines[index]).split()
            for i, word in enumerate(words):
                if i == self.max_count_of_words: break
                word_index_sequence[i] = self.word_index_add_one(word)

            # 追加の素性ベクトルの構築
            multi_feature_vector = []
            if self.use_data_of_position_of_sentence:
                multi_feature_vector.append(data_of_position_of_sentence[index])
            if self.use_data_of_is_serif:
                multi_feature_vector.append(data_of_is_serif[index])
            if self.use_data_of_is_include_person:
                multi_feature_vector.append(data_of_is_include_person[index])
            if self.use_data_of_sentence_length:
                multi_feature_vector.append(data_of_sentence_length[index])

            if not len(multi_feature_vector) == self.multi_feature_dim:
                raise ValueError("[ERROR] not equal length of input vector({}) and input vector size({})"
                                 .format(len(multi_feature_vector), self.multi_feature_dim))

            word_index_data.append(word_index_sequence)
            multi_feature_data.append(multi_feature_vector)

        return {'sequence': np.array(word_index_data), 'features': np.array(multi_feature_data)}



if __name__ == '__main__':
    sup = LSTMVectorSupplier('general',
                             importance='cos_sim',
                             use_data_of_position_of_sentence=True,
                             use_data_of_is_serif=True,
                             use_data_of_is_include_person=True,
                             use_data_of_sentence_length=True)
    gen = sup.train_data_generator()
    for i in range(5):
        print(next(gen))
        print('\n')



