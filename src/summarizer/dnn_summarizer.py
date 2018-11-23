import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense

from data_supplier.dnn_vector_supplier import DNNVectorSupplier
from util.corpus_accessor import CorpusAccessor


corpus_accessor = CorpusAccessor()

class DNNSummarizer:

    def __init__(self):
        pass

    def set_supplier(self, supplier):
        self.supplier = supplier

        # DNN MODEL PROPERTY
        self.n_in = self.supplier.input_vector_size
        self.n_hiddens = [800, 800]
        self.n_out = 1
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
        model = Sequential()
        for i, input_dim in enumerate(([self.n_in] + self.n_hiddens)[:-1]):
            model.add(Dense(self.n_hiddens[i], input_dim=input_dim))
            model.add(BatchNormalization())
            model.add(Activation(self.activation))
            model.add(Dropout(self.p_keep))
        model.add(Dense(self.n_out))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
        return model

    def fit(self):
        """
        Feed Foward Neural Netを用いた訓練
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

    def re_fit(self, model_name):
        """
        モデルの再学習
        """
        trained_model_file_path = os.path.join(self.supplier.trained_model_dir_path(), model_name)
        if not os.path.isfile(trained_model_file_path):
            raise ValueError("[ERROR] trained model does not exist")

        trained_model = load_model(trained_model_file_path)
        initial_epoch = int(model_name.split('_')[1]) + 1
        epochs = 100 - initial_epoch

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10)
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.supplier.trained_model_dir_path(),
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)

        print('[INFO] trained model path: {}'.format(trained_model_file_path))
        print('[INFO] initial_epoch: {}'.format(initial_epoch))
        trained_model.fit_generator(
            self.supplier.train_data_generator(),
            initial_epoch=initial_epoch,
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
        if self.trained_model is None:
            raise ValueError('[ERROR] trained model have not set yet')
        test_data_input = self.supplier.test_data_input(ncode)

        prediction = self.trained_model.predict(test_data_input).T[0]
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
    s = DNNSummarizer()
    supplier = DNNVectorSupplier('general',
                                  'cos_sim',
                                  use_data_of_position_of_sentence=True,
                                  use_data_of_is_serif=True,
                                  use_data_of_is_include_person=True,
                                  use_data_of_sentence_length=True)
    s.set_supplier(supplier)
    print(s.generate('n0013da'))

