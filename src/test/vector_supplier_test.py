import unittest
import numpy as np

from data_supplier.vector_supplier import VectorSupplier
import data_supplier

class VectorSupplierTest(unittest.TestCase):

    def test_train_data_generator(self):
        sup = VectorSupplier(use_data_of_word_embedding_avg_vector=True,
                          use_data_of_position_of_sentence=True,
                          use_data_of_is_serif=True,
                          use_data_of_is_include_person=True,
                          use_data_of_sentence_length=True)

        train_ncodes = ['n0019bv', 'n0013da', 'n0056dd', 'n0047ec']

        train_data_generator = sup.data_generator(train_ncodes)

        tensor = np.empty((0, 204))
        label = []

        for ncode in train_ncodes:
            data_of_word_embedding_avg_vector = data_supplier.word_embedding_avg_vector_data_supplier.load(ncode)
            data_of_position_of_sentence = data_supplier.position_of_sentence_data_supplier.load(ncode)
            data_of_is_serif = data_supplier.is_serif_data_supplier.load(ncode)
            data_of_is_include_person = data_supplier.is_include_person_data_supplier.load(ncode)
            data_of_sentence_length = data_supplier.sentence_length_data_supplier.load(ncode)

            data_of_similarity = data_supplier.similarity_data_supplier.load(ncode)

            for index in range(len(data_of_is_serif.keys())):
                input_vector = []
                input_vector.extend(data_of_word_embedding_avg_vector[index])
                input_vector.append(data_of_position_of_sentence[index])
                input_vector.append(data_of_is_serif[index])
                input_vector.append(data_of_is_include_person[index])
                input_vector.append(data_of_sentence_length[index])
                tensor = np.append(tensor, [input_vector], axis=0)

                label.append(data_of_similarity[index])

        data = None
        for index in range(tensor.shape[0]):

            if index % sup.batch_size == 0:
                data = next(train_data_generator)
                if tensor.shape[0] - index < sup.batch_size:
                    break

            self.assertTrue(np.allclose(tensor[index], data[0][index % sup.batch_size]))
            self.assertTrue(label[index] == data[1][index % sup.batch_size])







if __name__ == '__main__':
    unittest.main()
