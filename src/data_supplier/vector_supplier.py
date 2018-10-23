import os
import joblib
import numpy as np

import data_supplier
from data_supplier.active_ncodes_supplier import ncodes_train_test_split
from util.paths import DNN_TRAINED_MODEL_DIR_PATH


class VectorSupplier:

    def __init__(self,
                 genre='general',
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
        self.genre = genre

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
        self.train_ncodes, self.test_ncodes, self.validation_ncodes = ncodes_train_test_split(genre=genre,
                                                                                              validation_size=0.01,
                                                                                              test_size=0.2)
        # Num of sentences used per batch
        self.batch_size = 50
        # Shape of per batch
        self.batch_shape = (self.batch_size, self.input_vector_size)

    def trained_model_dir_path(self):
        feature_dir_name = ''
        if self.use_data_of_word_embedding_avg_vector:
            feature_dir_name += 'emb_'
        if self.use_data_of_position_of_sentence:
            feature_dir_name += 'pos_'
        if self.use_data_of_is_serif:
            feature_dir_name += 'ser_'
        if self.use_data_of_is_include_person:
            feature_dir_name += 'per_'
        if self.use_data_of_sentence_length:
            feature_dir_name += 'len_'
        path = os.path.join(DNN_TRAINED_MODEL_DIR_PATH, self.genre)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, feature_dir_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def get_trained_model_path(self):
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
        for i, ncode in enumerate(ncodes):
            if i % 1000 == 0:
                print('[INFO] sentence counting progress: {:.1f}%'.format(i / len(ncodes) * 100))
            total += len(data_supplier.similarity_data_supplier.load(ncode).keys())
        print(total)
        return total

    def train_data_generator(self):
        return self.data_generator(self.train_ncodes)

    def validation_data_generator(self):
        return self.data_generator(self.validation_ncodes)

    def test_data_generator(self):
        return self.data_generator(self.test_ncodes)

    def data_generator(self, ncodes):
        while 1:
            input_batch_data = np.empty(self.batch_shape)
            label_batch_data = np.empty(self.batch_size)
            position_in_batch = 0

            for ncode in ncodes:
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
                        if not type(data_of_word_embedding_avg_vector[index]) == np.ndarray:
                            continue
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

    def test_data_input(self, ncode):
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

        tensor = []

        for index in range(sentence_count):
            input_vector = []
            if self.use_data_of_word_embedding_avg_vector:
                if not type(data_of_word_embedding_avg_vector[index]) == np.ndarray:
                    tensor.append(list(np.zeros(self.input_vector_size)))
                    continue
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

            tensor.append(input_vector)
        return np.array(tensor)


if __name__ == '__main__':
    sup = VectorSupplier('love_story',
                         use_data_of_word_embedding_avg_vector=True,
                         use_data_of_position_of_sentence=True,
                         use_data_of_is_serif=True,
                         use_data_of_is_include_person=False,
                         use_data_of_sentence_length=False)

    ncode = 'n0002ei'
    tensor = sup.test_data_input(ncode)
    print(type(tensor))
    print(len(tensor))
    print(len(tensor[0]))


