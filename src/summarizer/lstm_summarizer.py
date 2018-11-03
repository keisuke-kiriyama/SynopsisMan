import os
import numpy as np
import joblib
import keras
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc

from data_supplier.lstm_vector_supplier import LSTMVectorSupplier
from util.corpus_accessor import CorpusAccessor
from util.paths import EMBEDDING_MATRIX_PATH


corpus_accessor = CorpusAccessor()

class LSTMSummarizer:

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
        LSTM&DNNで重回帰分析を行うモデルを構築する
        :return: Sequential
        """
        if self.supplier is None:
            raise ValueError("[ERROR] vector supplier haven't set yet")

        if not os.path.isfile(EMBEDDING_MATRIX_PATH):
            raise ValueError("[ERROR] embedding matrix hax not been constructed.")
        with open(EMBEDDING_MATRIX_PATH, 'rb') as f:
            embedding_matrix = joblib.load(f)

        max_count_of_words = self.supplier.max_count_of_words
        vocabulary_size = self.supplier.vocabulary_size
        embedding_vector_dim = self.supplier.word_embedding_vector_dim

        main_input = Input(shape=(max_count_of_words,), dtype='int32', name='sequence')
        x = Embedding(input_dim=vocabulary_size + 1,
                      output_dim=embedding_vector_dim,
                      input_length=max_count_of_words,
                      weights=[embedding_matrix],
                      trainable=False,
                      mask_zero=True,
                      name='embedding')(main_input)
        lstm_out = LSTM(64, input_shape=(self.supplier.word_index_batch_shape))(x)

        # 文をエンコードするLSTMを訓練するための補助出力
        auxiliary_output = Dense(1, activation='linear', name='aux_output')(lstm_out)

        features_input = Input(shape=(self.supplier.multi_feature_dim,), name='features')
        x = keras.layers.concatenate([lstm_out, features_input])

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

    def generate(self, ncode):
        contents_lines = corpus_accessor.get_contents_lines(ncode)
        synopsis_lines = corpus_accessor.get_synopsis_lines(ncode)
        if not contents_lines or not synopsis_lines:
            return

        contents_lines = np.array(contents_lines)
        # 参照あらすじの長さ
        ref_length = len(''.join(synopsis_lines))
        # 最低文数
        min_sentence_count = 1


        # 学習済みモデルに依る学習
        self.set_trained_model()
        if self.trained_model is None:
            raise ValueError('[ERROR] trained model have not set yet')
        test_data_input = self.supplier.test_data_input(ncode)

        prediction = np.array(self.trained_model.predict(test_data_input)[0].T[0])
        high_score_line_indexes = np.argsort(-prediction)

        # 要約率を満たすようにあらすじを作成
        synopsis = contents_lines[high_score_line_indexes[:min_sentence_count]]
        for sentence_index in high_score_line_indexes[min_sentence_count:]:
            if len(''.join(np.append(synopsis, contents_lines[sentence_index]))) <= ref_length:
                synopsis = np.append(synopsis, contents_lines[sentence_index])
            else:
                break
        return ''.join(synopsis)


if __name__ == '__main__':
    s = LSTMSummarizer()
    supplier = LSTMVectorSupplier('sf',
                                  'cos_sim',
                                  use_data_of_position_of_sentence=True,
                                  use_data_of_is_serif=True,
                                  use_data_of_is_include_person=True,
                                  use_data_of_sentence_length=True)
    s.set_supplier(supplier)
    print(s.generate('n0013da'))

