import os
from os import path, pardir

FILE_PATH = path.abspath(__file__)
UTIL_PATH = path.dirname(FILE_PATH)
SRC_DIR_PATH = path.abspath(path.join(UTIL_PATH, pardir))
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
DATA_DIR_PATH = os.environ['DATA_DIR_PATH']
MODEL_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'model'))

ACTIVE_NCODES_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'active_ncodes'))

ORIGIN_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'origin'))
ORIGIN_CONTENTS_DATA_DIR_PATH = path.abspath(path.join(ORIGIN_DATA_DIR_PATH, 'contents'))
ORIGIN_META_DATA_DIR_PATH = path.abspath(path.join(ORIGIN_DATA_DIR_PATH, 'meta'))

PREPROCESSED_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'preprocessed'))
PREPROCESSED_CONTENTS_DATA_DIR_PATH = path.abspath(path.join(PREPROCESSED_DATA_DIR_PATH, 'contents'))
PREPROCESSED_META_DATA_DIR_PATH = path.abspath(path.join(PREPROCESSED_DATA_DIR_PATH, 'meta'))

PATH_LINE_SENTENCES_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'path_line_sentences'))

WORD_EMBEDDING_VECTORS_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'word_embedding_vectors'))
WORD_EMBEDDING_VECTORS_CONTENTS_PATH = path.abspath(path.join(WORD_EMBEDDING_VECTORS_DIR_PATH, 'contents'))

WORD_EMBEDDING_AVG_VECTOR_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'word_embedding_avg_vector'))
WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH = path.abspath(path.join(WORD_EMBEDDING_AVG_VECTOR_DIR_PATH, 'contents'))
WORD_EMBEDDING_AVG_VECTOR_META_PATH = path.abspath(path.join(WORD_EMBEDDING_AVG_VECTOR_DIR_PATH, 'meta'))

SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'similarity_between_contents_and_synopsis_sentence'))
ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'rouge_between_contents_and_synopsis_sentence'))

POSITION_OF_SENTENCE_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'position_of_sentence'))
POSITION_OF_SENTENCE_CONTENTS_DIR_PATH = path.abspath(path.join(POSITION_OF_SENTENCE_DATA_DIR_PATH, 'contents'))

IS_SERIF_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'is_serif'))
IS_SERIF_CONTENTS_DIR_PATH = path.abspath(path.join(IS_SERIF_DATA_DIR_PATH, 'contents'))

IS_INCLUDE_PERSON_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'is_include_person'))
IS_INCLUDE_PERSON_CONTENTS_PATH = path.abspath(path.join(IS_INCLUDE_PERSON_DIR_PATH, 'contents'))

SENTENCE_LENGTH_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'sentence_length'))
SENTENCE_LENGTH_CONTENTS_PATH = path.abspath(path.join(SENTENCE_LENGTH_DIR_PATH, 'contents'))

OPT_SENTENCES_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'opt_sentences'))
OPT_SENTENCES_DATA_DIR_PATH = path.abspath(path.join(OPT_SENTENCES_DIR_PATH, 'short_0.051_long_0.013_min_1_max_6'))

WORD_EMBEDDING_MODEL_DIR_PATH = path.abspath(path.join(MODEL_DIR_PATH, 'word_embedding'))
WORD_EMBEDDING_MODEL_PATH = path.abspath(path.join(WORD_EMBEDDING_MODEL_DIR_PATH, 'word_embedding.model'))

EMBEDDING_MATRIX_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'embedding_matrix'))
EMBEDDING_MATRIX_PATH = path.abspath(path.join(EMBEDDING_MATRIX_DIR_PATH, 'embedding_matrix.txt'))

DNN_TRAINED_MODEL_DIR_PATH = path.abspath(path.join(MODEL_DIR_PATH, 'dnn_trained_model'))
LSTM_TRAINED_MODEL_DIR_PATH = path.abspath(path.join(MODEL_DIR_PATH, 'lstm_trained_model'))


def check():
    print("FILE_PATH: ", FILE_PATH)
    print("UTIL_PATH: ", UTIL_PATH)
    print("SRC_DIR_PATH: ", SRC_DIR_PATH)
    print("PROJECT_ROOT: ", PROJECT_ROOT)
    print("DATA_DIR_PATH: ", DATA_DIR_PATH)
    print("ACTIVE_NCODES_DIR_PATH: ", ACTIVE_NCODES_DIR_PATH)
    print("ORIGIN_DATA_DIR_PATH: ", ORIGIN_DATA_DIR_PATH)
    print("ORIGIN_CONTENTS_DATA_DIR_PATH: ", ORIGIN_CONTENTS_DATA_DIR_PATH)
    print("ORIGIN_META_DATA_DIR_PATH: ", ORIGIN_META_DATA_DIR_PATH)
    print("PREPROCESSED_DATA_DIR_PATH: ", PREPROCESSED_DATA_DIR_PATH)
    print("PREPROCESSED_CONTENTS_DATA_DIR_PATH: ", PREPROCESSED_CONTENTS_DATA_DIR_PATH)
    print("PREPROCESSED_META_DATA_DIR_PATH: ", PREPROCESSED_META_DATA_DIR_PATH)
    print("PATH_LINE_SENTENCES_DIR_PATH: ", PATH_LINE_SENTENCES_DIR_PATH)
    print("WORD_EMBEDDING_VECTORS_DIR_PATH: ", WORD_EMBEDDING_VECTORS_DIR_PATH)
    print("WORD_EMBEDDING_VECTORS_CONTENTS_PATH: ", WORD_EMBEDDING_VECTORS_CONTENTS_PATH)
    print("WORD_EMBEDDING_AVG_VECTOR_DIR_PATH: ", WORD_EMBEDDING_AVG_VECTOR_DIR_PATH)
    print("WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH: ", WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH)
    print("WORD_EMBEDDING_AVG_VECTOR_META_PATH: ", WORD_EMBEDDING_AVG_VECTOR_META_PATH)
    print("SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH: ", SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)
    print("ROUGE_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH: ", ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)
    print("POSITION_OF_SENTENCE_DATA_DIR_PATH: ", POSITION_OF_SENTENCE_DATA_DIR_PATH)
    print("POSITION_OF_SENTENCE_CONTENTS_DIR_PATH: ", POSITION_OF_SENTENCE_CONTENTS_DIR_PATH)
    print("IS_SERIF_DATA_DIR_PATH: ", IS_SERIF_DATA_DIR_PATH)
    print("IS_SERIF_CONTENTS_DIR_PATH:f ", IS_SERIF_CONTENTS_DIR_PATH)
    print("IS_INCLUDE_PERSON_DIR_PATH: ", IS_INCLUDE_PERSON_DIR_PATH)
    print("IS_INCLUDE_PERSON_CONTENTS_PATH: ", IS_INCLUDE_PERSON_CONTENTS_PATH)
    print("SENTENCE_LENGTH_DIR_PATH: ", SENTENCE_LENGTH_DIR_PATH)
    print("SENTENCE_LENGTH_CONTENTS_PATH: ", SENTENCE_LENGTH_CONTENTS_PATH)
    print("OPT_SENTENCES_DIR_PATH: ", OPT_SENTENCES_DIR_PATH)
    print("OPT_SENTENCES_DATA_DIR_PATH: ", OPT_SENTENCES_DATA_DIR_PATH)
    print("MODEL_DIR_PATH: ", MODEL_DIR_PATH)
    print("WORD_EMBEDDING_MODEL_PATH: ", WORD_EMBEDDING_MODEL_PATH)
    print("EMBEDDING_MATRIX_DIR_PATH: ", EMBEDDING_MATRIX_DIR_PATH)
    print("EMBEDDING_MATRIX_PATH: ", EMBEDDING_MATRIX_PATH)
    print("DNN_TRAINED_MODEL_DIR_PATH: ", DNN_TRAINED_MODEL_DIR_PATH)
    print("LSTM_TRAINED_MODEL_DIR_PATH: ", LSTM_TRAINED_MODEL_DIR_PATH)

def mkdir():
    if not path.isdir(ORIGIN_DATA_DIR_PATH):
        os.mkdir(ORIGIN_DATA_DIR_PATH)
    if not path.isdir(ORIGIN_CONTENTS_DATA_DIR_PATH):
        os.mkdir(ORIGIN_CONTENTS_DATA_DIR_PATH)
    if not path.isdir(ORIGIN_META_DATA_DIR_PATH):
        os.mkdir(ORIGIN_META_DATA_DIR_PATH)
    if not path.isdir(ACTIVE_NCODES_DIR_PATH):
        os.mkdir(ACTIVE_NCODES_DIR_PATH)
    if not path.isdir(PREPROCESSED_DATA_DIR_PATH):
        os.mkdir(PREPROCESSED_DATA_DIR_PATH)
    if not path.isdir(PREPROCESSED_CONTENTS_DATA_DIR_PATH):
        os.mkdir(PREPROCESSED_CONTENTS_DATA_DIR_PATH)
    if not path.isdir(PREPROCESSED_META_DATA_DIR_PATH):
        os.mkdir(PREPROCESSED_META_DATA_DIR_PATH)
    if not path.isdir(PATH_LINE_SENTENCES_DIR_PATH):
        os.mkdir(PATH_LINE_SENTENCES_DIR_PATH)
    if not path.isdir(WORD_EMBEDDING_VECTORS_DIR_PATH):
        os.mkdir(WORD_EMBEDDING_VECTORS_DIR_PATH)
    if not path.isdir(WORD_EMBEDDING_VECTORS_CONTENTS_PATH):
        os.mkdir(WORD_EMBEDDING_VECTORS_CONTENTS_PATH)
    if not path.isdir(WORD_EMBEDDING_AVG_VECTOR_DIR_PATH):
        os.mkdir(WORD_EMBEDDING_AVG_VECTOR_DIR_PATH)
    if not path.isdir(WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH):
        os.mkdir(WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH)
    if not path.isdir(WORD_EMBEDDING_AVG_VECTOR_META_PATH):
        os.mkdir(WORD_EMBEDDING_AVG_VECTOR_META_PATH)
    if not path.isdir(SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH):
        os.mkdir(SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)
    if not path.isdir(ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH):
        os.mkdir(ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)
    if not path.isdir(POSITION_OF_SENTENCE_DATA_DIR_PATH):
        os.mkdir(POSITION_OF_SENTENCE_DATA_DIR_PATH)
    if not path.isdir(POSITION_OF_SENTENCE_CONTENTS_DIR_PATH):
        os.mkdir(POSITION_OF_SENTENCE_CONTENTS_DIR_PATH)
    if not path.isdir(IS_SERIF_DATA_DIR_PATH):
        os.mkdir(IS_SERIF_DATA_DIR_PATH)
    if not path.isdir(IS_SERIF_CONTENTS_DIR_PATH):
        os.mkdir(IS_SERIF_CONTENTS_DIR_PATH)
    if not path.isdir(IS_INCLUDE_PERSON_DIR_PATH):
        os.mkdir(IS_INCLUDE_PERSON_DIR_PATH)
    if not path.isdir(IS_INCLUDE_PERSON_CONTENTS_PATH):
        os.mkdir(IS_INCLUDE_PERSON_CONTENTS_PATH)
    if not path.isdir(SENTENCE_LENGTH_DIR_PATH):
        os.mkdir(SENTENCE_LENGTH_DIR_PATH)
    if not path.isdir(SENTENCE_LENGTH_CONTENTS_PATH):
        os.mkdir(SENTENCE_LENGTH_CONTENTS_PATH)
    if not path.isdir(OPT_SENTENCES_DIR_PATH):
        os.mkdir(OPT_SENTENCES_DIR_PATH)
    if not path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    if not path.isdir(WORD_EMBEDDING_MODEL_DIR_PATH):
        os.mkdir(WORD_EMBEDDING_MODEL_DIR_PATH)
    if not path.isdir(EMBEDDING_MATRIX_DIR_PATH):
        os.mkdir(EMBEDDING_MATRIX_DIR_PATH)
    if not path.isdir(DNN_TRAINED_MODEL_DIR_PATH):
        os.mkdir(DNN_TRAINED_MODEL_DIR_PATH)
    if not path.isdir(LSTM_TRAINED_MODEL_DIR_PATH):
        os.mkdir(LSTM_TRAINED_MODEL_DIR_PATH)
