import os
import joblib
import numpy as np

from util.corpus_accessor import CorpusAccessor
from util.paths import TRAIN_NCODES_FILE_PATH, VALIDATION_NCODES_FILE_PATH, TEST_NCODES_FILE_PATH
import data_supplier

corpus_accessor = CorpusAccessor()

class VectorSupplier:

    def __init__(self,
                 use_data_of_word_embedding_avg_vector=False,
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

        self.use_data_of_word_embedding_avg_vector = use_data_of_word_embedding_avg_vector
        self.use_data_of_position_of_sentence = use_data_of_position_of_sentence
        self.use_data_of_is_serif = use_data_of_is_serif
        self.use_data_of_is_include_person = use_data_of_is_include_person
        self.use_data_of_sentence_length = use_data_of_sentence_length

        # Feature vector dimension
        word_embedding_avg_vector_dim = 200
        position_of_sentence_dim = 1
        is_serif_dim = 1
        is_include_person_dim = 1
        sentence_length_dim = 1

        # Input Vector Size
        self.input_vector_size = 0
        if use_data_of_word_embedding_avg_vector:
            self.input_vector_size += word_embedding_avg_vector_dim
        if use_data_of_position_of_sentence:
            self.input_vector_size += position_of_sentence_dim
        if use_data_of_is_serif:
            self.input_vector_size += is_serif_dim
        if use_data_of_is_include_person:
            self.input_vector_size += is_include_person_dim
        if use_data_of_sentence_length:
            self.input_vector_size += sentence_length_dim

        # NCode
        self.train_ncodes, self.validation_ncodes, self.test_ncodes = self.ncodes_train_test_split(validation_size=0.01,
                                                                                                   test_size=0.2)

        # Num of sentences used per batch
        self.batch_size = 50
        # Shape of per batch
        self.batch_shape = (self.batch_size, self.input_vector_size)
        # Total sentence count
        self.train_sentence_count = self.total_sentence_count(self.train_ncodes)
        # num of batch of sample
        self.steps_per_epoch = int(self.train_sentence_count / self.batch_size)


    def ncodes_train_test_split(self, validation_size = 0.01, test_size=0.2):
        """
        訓練データとテストデータのncodeを返す
        """
        if os.path.isfile(TRAIN_NCODES_FILE_PATH) \
                and os.path.isfile(TEST_NCODES_FILE_PATH) \
                and os.path.isfile(VALIDATION_NCODES_FILE_PATH):
            print('[INFO] loading splited ncodes data...')
            with open(TRAIN_NCODES_FILE_PATH, 'rb') as train_f:
                train_ncodes = joblib.load(train_f)
            with open(VALIDATION_NCODES_FILE_PATH, 'rb') as validation_f:
                validation_ncodes = joblib.load(validation_f)
            with open(TEST_NCODES_FILE_PATH, 'rb') as test_f:
                test_ncodes = joblib.load(test_f)

        else:
            active_ncodes = corpus_accessor.get_active_ncodes()
            temp_ncodes = active_ncodes[:int(len(active_ncodes) * (1 - test_size))]
            test_ncodes = active_ncodes[int(len(active_ncodes) * (1 - test_size)):]
            train_ncodes = temp_ncodes[:int(len(temp_ncodes) * (1 - validation_size))]
            validation_ncodes = temp_ncodes[int(len(temp_ncodes) * (1 - validation_size)):]
            print('[INFO] saving splited ncodes data...')
            with open(TRAIN_NCODES_FILE_PATH, 'wb') as train_f:
                joblib.dump(train_ncodes, train_f, compress=3)
            with open(VALIDATION_NCODES_FILE_PATH, 'wb') as validation_f:
                joblib.dump(validation_ncodes, validation_f, compress=3)
            with open(TEST_NCODES_FILE_PATH, 'wb') as test_f:
                joblib.dump(test_ncodes, test_f, compress=3)

        print('[INFO] train ncodes count: {}'.format(len(train_ncodes)))
        print('[INFO] validation ncodes count: {}'.format(len(validation_ncodes)))
        print('[INFO] test ncodes count: {}'.format(len(test_ncodes)))
        return train_ncodes, validation_ncodes, test_ncodes

    def total_sentence_count(self, ncodes):
        total = 0
        for i, ncode in enumerate(ncodes):
            if i % 1000 == 0:
                print('[INFO] sentence counting progress: {:.1f}%'.format(i / len(ncodes) * 100))
            total += len(data_supplier.similarity_data_supplier.load(ncode).keys())
        print(total)
        return total

    def train_data_generator(self):
        while 1:
            input_batch_data = np.empty(self.batch_shape)
            label_batch_data = np.empty(self.batch_size)
            position_in_batch = 0

            for ncode in self.train_ncodes:
                similarity_data = data_supplier.similarity_data_supplier.load(ncode)
                sentence_count = len(similarity_data)

                data_of_word_embedding_avg_vector = None
                data_of_position_of_sentence = None
                data_of_is_serif = None
                data_of_is_include_person = None
                data_of_sentence_length = None

                if self.use_data_of_word_embedding_avg_vector:
                    data_of_word_embedding_avg_vector = data_supplier.word_embedding_avg_vector_data_supplier.load(ncode)
                if self.use_data_of_position_of_sentence:
                    data_of_position_of_sentence = data_supplier.position_of_sentence_data_supplier.load(ncode)
                if self.use_data_of_is_serif:
                    data_of_is_serif = data_supplier.is_serif_data_supplier.load(ncode)
                if self.use_data_of_is_include_person:
                    data_of_is_include_person = data_supplier.is_include_person_data_supplier.load(ncode)
                if self.use_data_of_sentence_length:
                    data_of_sentence_length = data_supplier.sentence_length_data_supplier.load(ncode)

                for index in range(sentence_count):
                    input_vector = []
                    if self.use_data_of_word_embedding_avg_vector:
                        input_vector.extend(data_of_word_embedding_avg_vector[index])
                    if self.use_data_of_position_of_sentence:
                        input_vector.append(data_of_position_of_sentence[index])
                    if self.use_data_of_is_serif:
                        input_vector.append(data_of_is_serif[index])
                    if self.use_data_of_is_include_person:
                        input_vector.append(data_of_is_include_person[index])
                    if self.use_data_of_sentence_length:
                        input_vector.append(data_of_sentence_length[index])

                    if not len(input_vector) == self.input_vector_size:
                        raise ValueError("[ERROR] not equal length of input vector({}) and input vector size({})"
                                         .format(len(input_vector), self.input_vector_size))

                    input_batch_data[position_in_batch] = input_vector
                    label_batch_data[position_in_batch] = similarity_data[index]
                    position_in_batch += 1

                    if position_in_batch == self.batch_size:
                        yield input_batch_data, label_batch_data
                        input_batch_data = np.empty(self.batch_shape)
                        label_batch_data = np.empty(self.batch_size)
                        position_in_batch = 0








if __name__ == '__main__':
    sup = VectorSupplier(use_data_of_word_embedding_avg_vector=True,
                          use_data_of_position_of_sentence=True,
                          use_data_of_is_serif=False,
                          use_data_of_is_include_person=False,
                          use_data_of_sentence_length=False)

    sup.total_sentence_count(sup.test_ncodes)





