import os
import numpy as np
from gensim.models import word2vec
import joblib

from util.paths import WORD_EMBEDDING_MODEL_PATH, EMBEDDING_MATRIX_PATH

if os.path.isfile(WORD_EMBEDDING_MODEL_PATH):
    word_embedding_model = word2vec.Word2Vec.load(WORD_EMBEDDING_MODEL_PATH)

def construct():
    if word_embedding_model is None:
        print('[ERROR] there is not word embedding model')

    embedding_matrix = np.zeros(shape=(len(word_embedding_model.wv.vocab) + 1, word_embedding_model.vector_size))
    total = len(word_embedding_model.wv.vocab)
    for i, word in enumerate(word_embedding_model.wv.vocab.keys()):
        if i % 50 == 0:
            print('contents progress: {:.1f}%'.format(i / total * 100))
        embedding_matrix[word_embedding_model.wv.vocab[word].index + 1] = word_embedding_model.wv[word]
    # データの保存
    print('[INFO] saving data')
    with open(EMBEDDING_MATRIX_PATH, 'wb') as cf:
        joblib.dump(embedding_matrix, cf, compress=3)

if __name__ == '__main__':
    construct()
