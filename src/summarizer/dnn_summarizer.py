import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc

import keras
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model

from data_supplier.vector_supplier import VectorSupplier
from util.corpus_accessor import CorpusAccessor
from util.paths import DNN_TRAINED_MODEL_DIR_PATH


corpus_accessor = CorpusAccessor()

class DNNSummarizer:

    def __init__(self):
        pass

    def set_supplier(self, supplier):
        self.supplier = supplier

        # DNN MODEL PROPERTY
        self.activation = 'relu'
        self.p_keep = 0.5

    def set_trained_model(self):
        """
        文の選択肢とスコアのリストを引数にとり、要約率を満たすようにあらすじを生成する
        """
        if self.supplier is None:
            raise ValueError("[ERROR] vector supplier haven't set yet")

        trained_model_path = self.supplier.get_trained_model_path()
        print('[INFO] trained model path: ', trained_model_path)
        if not os.path.isfile(trained_model_path):
            raise ValueError("[ERROR] trained model does not exist")

        self.trained_model = load_model(trained_model_path)

    def inference(self):
        """
        DNNで重回帰分析を行うモデルを構築する
        :return: Sequential
        """
        if self.supplier is None:
            raise ValueError("[ERROR] vector supplier haven't set yet")
        max_count_of_words = self.supplier.max_count_of_words
        word_emb_dim = self.supplier.word_embedding_vector_dim

        main_input = Input(shape=(max_count_of_words, word_emb_dim), dtype='float', name='embedding')
        lstm_out = LSTM(200, input_shape=(self.supplier.word_embedding_batch_shape))(main_input)

        # 文をエンコードするLSTMを訓練するための補助出力
        auxiliary_output = Dense(1, activation=self.activation, name='aux_output')(lstm_out)

        features_input = Input(shape=(self.supplier.multi_feature_dim,), name='features')
        x = keras.layers.concatenate([lstm_out, features_input])

        x = Dense(800, activation=self.activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(.3)(x)
        x = Dense(800, activation=self.activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(.3)(x)
        x = Dense(800, activation=self.activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(.3)(x)

        main_output = Dense(1, activation='linear', name='main_output')(x)
        model = Model(inputs=[main_input, features_input], outputs=[main_output, auxiliary_output])

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                      loss_weights=[1., 0.2])
        return model

    def fit(self):
        """
        Feed Foward Neural Netを用いた訓練
        :return:
        """
        if self.supplier is None:
            raise ValueError("[ERROR] vector supplier haven't set yet")
        epochs = 100
        model = self.inference()

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10)
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.supplier.trained_model_dir_path(),
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        model.fit_generator(
            self.supplier.train_data_generator(),
            steps_per_epoch=self.supplier.train_steps_per_epoch(),
            validation_data=self.supplier.validation_data_generator(),
            validation_steps=self.supplier.validation_steps_per_epoch(),
            epochs=epochs,
            shuffle=True,
            callbacks=[early_stopping, checkpoint])

    def generate_synopsis(self, ncode, short_rate, long_rate, min_sentence_count, max_sentence_count):
        contents_lines = corpus_accessor.get_contents_lines(ncode)
        if not contents_lines:
            raise ValueError("[ERROR] ncode does not exist")
        if not len(contents_lines) > max_sentence_count:
            print("[ERROR] contents lines is too short to generate synopsis")

        contents_lines = np.array(contents_lines)
        contents_len = len(''.join(contents_lines))

        # 用いる要約率の閾値
        rate = long_rate if corpus_accessor.is_long(ncode) else short_rate

        # 学習済みモデルに依る学習
        self.set_trained_model()
        if self.trained_model is None:
            raise ValueError('[ERROR] trained model have not set yet')
        test_data_input = self.supplier.test_data_input(ncode)

        prediction = self.trained_model.predict(test_data_input).T[0]
        high_score_line_indexes = np.argsort(-prediction)[:max_sentence_count]

        # 要約率を満たすようにあらすじを作成
        synopsis = contents_lines[high_score_line_indexes[:min_sentence_count]]
        for sentence_index in high_score_line_indexes[min_sentence_count:]:
            if len(''.join(np.append(synopsis, contents_lines[sentence_index]))) / contents_len < rate:
                synopsis = np.append(synopsis, contents_lines[sentence_index])
            else:
                break
        synopsis = ''.join(synopsis)

        return synopsis





if __name__ == '__main__':
    s = DNNSummarizer()
    supplier = VectorSupplier('general',
                              use_data_of_position_of_sentence=True,
                              use_data_of_is_serif=True,
                              use_data_of_is_include_person=True,
                              use_data_of_sentence_length=True)
    s.set_supplier(supplier)
    print(s.generate_synopsis('n0019bv', 0.051, 0.013, 1, 6))

