import os
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc

from data_supplier.vector_supplier import VectorSupplier
from util.paths import DNN_TRAINED_MODEL_DIR_PATH

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
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
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


if __name__ == '__main__':
    s = DNNSummarizer()
    s.fit()
