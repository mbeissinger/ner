"""
Character-level recurrent model for named entity recognition.
"""
from __future__ import print_function, division
# normal imports
import math
import itertools
import numpy
import theano.tensor as T
import pickle
# opendeep imports
import opendeep
from opendeep.data import NumpyDataset
from opendeep.models import LSTM
from opendeep.monitor import Monitor, Plot
from opendeep.optimization import RMSProp
from opendeep.utils.misc import numpy_one_hot
# internal imports
# from data_processing import parse_text_files


class OpenNER():
    vocab_file = 'vocab.pkl'
    entity_vocab_file = 'entity_vocab.pkl'
    lstm_file = 'trained_lstm.pkl'

    def __init__(self):
        self.vocab = pickle.load(open(self.vocab_file, 'rb'))
        self.entity_vocab = pickle.load(open(self.entity_vocab_file, 'rb'))
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.inverse_entity_vocab = {v: k for k, v in self.entity_vocab.items()}

        # define our model! we are using lstm.
        hidden_size = 128

        self.lstm = LSTM(input_size=len(self.vocab),
                         hidden_size=hidden_size,
                         output_size=len(self.entity_vocab),
                         hidden_activation='tanh',
                         inner_hidden_activation='sigmoid',
                         activation='softmax',
                         weights_init='uniform',
                         weights_interval='montreal',
                         r_weights_init='orthogonal',
                         clip_recurrent_grads=5.,
                         noise='dropout',
                         noise_level=0.5,
                         direction='bidirectional',
                         cost_function='nll',
                         cost_args={'one_hot': True})

        self.lstm.load_params(self.lstm_file)

    # if you want debugging output from opendeep
    opendeep.config_root_logger()

    @staticmethod
    def __get_entities(data, predictions, vocab, entity_vocab):
        # find contiguous entity characters across timesteps
        entities = []
        for i, batch in enumerate(predictions):
            previous_label = 0
            continuous_string = ""
            for j, label in enumerate(batch):
                # if not continuous, reset
                if label != previous_label:
                    entity = continuous_string
                    # add only if the label is an entity
                    if previous_label != 0:
                        label_string = entity_vocab.get(previous_label)
                        entities.append((entity, label_string))
                    continuous_string = ""
                data_char = vocab.get(numpy.argmax(data[i, j]))
                continuous_string += data_char
                previous_label = label

        return entities

    @staticmethod
    def __process_str(data_str, vocab):
        # process the raw input data string

        data = []
        for data_char in data_str:
            if data_char in vocab:
                data.append(vocab[data_char])

        data = numpy_one_hot(numpy.asarray(data), n_classes=numpy.amax(vocab.values()) + 1)

        seq, dim = data.shape
        data = numpy.reshape(data, (1, seq, dim))

        return data

    def predict(self, query):

        data = self.__process_str(query, self.vocab)

        character_probs = self.lstm.run(data)
        # this has the shape (timesteps, batches, data), so swap axes to (batches, timesteps, data)
        character_probs = numpy.swapaxes(character_probs, 0, 1)
        # now extract the guessed entities
        predictions = numpy.argmax(character_probs, axis=2)

        entities = self.__get_entities(data, predictions, self.inverse_vocab, self.inverse_entity_vocab)

        return entities

    def getEntities(self, query):
        entities = self.predict(query)
        uniques = set(entities)
        entities = [{'type': a, 'text': t, 'count': entities.count((t, a))} for (t, a) in uniques]
        return entities


if __name__ == "__main__":
    ner = OpenNER()
    yhat = ner.predict("iphone 6 is a pretty cool phone")
    print(yhat)
