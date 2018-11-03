import sys
import numpy as np
from rouge import Rouge
from util.corpus_accessor import CorpusAccessor
from util.text_processor import wakati
from summarizer.lstm_summarizer import LSTMSummarizer
from summarizer.dnn_summarizer import DNNSummarizer
from data_supplier.lstm_vector_supplier import LSTMVectorSupplier
from data_supplier.dnn_vector_supplier import DNNVectorSupplier
from evaluate import lead_synopsis
from evaluate import opt_synopsis
from evaluate import random_synopsis

corpus_accessor = CorpusAccessor()

def evaluate(genre='general',
             importance='cos_sim',
             use_data_of_position_of_sentence=False,
             use_data_of_is_serif=False,
             use_data_of_is_include_person=False,
             use_data_of_sentence_length=False):
    dnn_summarizer = DNNSummarizer()
    dnn_vector_supplier = DNNVectorSupplier(genre,
                                 importance,
                                 use_data_of_position_of_sentence=use_data_of_position_of_sentence,
                                 use_data_of_is_serif=use_data_of_is_serif,
                                 use_data_of_is_include_person=use_data_of_is_include_person,
                                 use_data_of_sentence_length=use_data_of_sentence_length)
    dnn_summarizer.set_supplier(dnn_vector_supplier)

    lstm_summarizer = LSTMSummarizer()
    lstm_vector_supplier = LSTMVectorSupplier(genre,
                                  importance,
                                  use_data_of_position_of_sentence=use_data_of_position_of_sentence,
                                  use_data_of_is_serif=use_data_of_is_serif,
                                  use_data_of_is_include_person=use_data_of_is_include_person,
                                  use_data_of_sentence_length=use_data_of_sentence_length)
    lstm_summarizer.set_supplier(lstm_vector_supplier)

    test_ncodes = lstm_vector_supplier.test_ncodes
    total = len(test_ncodes)
    print('[INFO] test ncodes count: ', total)

    # ROUGE-1
    opt_rouge_one_scores = []               # 類似度上位から文選択(理論上の上限値)
    lead_rouge_one_scores = []              # 文章の先頭からoptの文数分選択
    random_rouge_one_scores = []            # ランダムに文を選択
    dnn_rouge_one_scores = []               # DNNによるあらすじ
    lstm_rouge_one_scores = []              # LSTMによるあらすじ

    # ROUGE-2
    opt_rouge_two_scores = []               # 類似度上位から文選択(理論上の上限値)
    lead_rouge_two_scores = []              # 文章の先頭からoptの文数分選択
    random_rouge_two_scores = []            # ランダムに文を選択
    dnn_rouge_two_scores = []               # DNNによるあらすじ
    lstm_rouge_two_scores = []              # LSTMによるあらすじ

    sys.setrecursionlimit(20000)
    rouge = Rouge()
    for i, ncode in enumerate(test_ncodes):
        print('[INFO] processing ncode: ', ncode)
        print('[INFO] progress: {:.1f}%'.format(i / total * 100))

        ref = wakati(''.join(corpus_accessor.get_synopsis_lines(ncode)))
        opt = wakati(opt_synopsis.generate(ncode))
        lead = wakati(lead_synopsis.generate(ncode))
        random = wakati(random_synopsis.generate(ncode))
        dnn_hyp = wakati(dnn_summarizer.generate(ncode))
        lstm_hyp = wakati(lstm_summarizer.generate(ncode))

        opt_score = rouge.get_scores(opt, ref, False)
        lead_score = rouge.get_scores(lead, ref, False)
        random_score = rouge.get_scores(random, ref, False)
        dnn_score = rouge.get_scores(dnn_hyp, ref, False)
        lstm_score = rouge.get_scores(lstm_hyp, ref, False)

        opt_rouge_one_scores.append(opt_score[0]['rouge-1']['r'])
        lead_rouge_one_scores.append(lead_score[0]['rouge-1']['r'])
        random_rouge_one_scores.append(random_score[0]['rouge-1']['r'])
        dnn_rouge_one_scores.append(dnn_score[0]['rouge-1']['r'])
        lstm_rouge_one_scores.append(lstm_score[0]['rouge-1']['r'])

        opt_rouge_two_scores.append(opt_score[0]['rouge-2']['r'])
        lead_rouge_two_scores.append(lead_score[0]['rouge-2']['r'])
        random_rouge_two_scores.append(random_score[0]['rouge-2']['r'])
        dnn_rouge_two_scores.append(dnn_score[0]['rouge-2']['r'])
        lstm_rouge_two_scores.append(lstm_score[0]['rouge-2']['r'])

    print('[RESULT]')
    print('ROUGE-1')
    print('opt: {}'.format(np.average(opt_rouge_one_scores)))
    print('lead: {}'.format(np.average(lead_rouge_one_scores)))
    print('random: {}'.format(np.average(random_rouge_one_scores)))
    print('dnn: {}'.format(np.average(dnn_rouge_one_scores)))
    print('lstm: {}'.format(np.average(lstm_rouge_one_scores)))
    print('\n')

    print('ROUGE-2')
    print('opt: {}'.format(np.average(opt_rouge_two_scores)))
    print('lead: {}'.format(np.average(lead_rouge_two_scores)))
    print('random: {}'.format(np.average(random_rouge_two_scores)))
    print('dnn: {}'.format(np.average(dnn_rouge_two_scores)))
    print('lstm: {}'.format(np.average(lstm_rouge_two_scores)))
    print('\n')


if __name__ == '__main__':
    evaluate()