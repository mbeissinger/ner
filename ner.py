"""
Character-level recurrent model for named entity recognition.
"""
from __future__ import print_function, division
# normal imports
import string
import numpy
import pickle
# opendeep imports
from opendeep.models import LSTM
from opendeep.utils.misc import numpy_one_hot


class OpenNER:
    vocab_file = 'vocab.pkl'
    entity_vocab_file = 'entity_vocab.pkl'
    lstm_file = 'trained_epoch_94.pkl'

    def __init__(self):
        self.vocab = pickle.load(open(self.vocab_file, 'rb'))
        self.entity_vocab = pickle.load(open(self.entity_vocab_file, 'rb'))
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.inverse_entity_vocab = {v: k for k, v in self.entity_vocab.items()}

        # define our model! we are using lstm.
        self.lstm = LSTM(input_size=len(self.vocab),
                         hidden_size=128,
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

        # # if you want debugging output from opendeep
        # opendeep.config_root_logger()

    @staticmethod
    def __get_entities(data, predictions, vocab_inv, entity_vocab):
        # find contiguous entity characters across timesteps
        non_entity_label = entity_vocab.get('O')
        entities = []
        for i, query in enumerate(predictions):
            previous_label = non_entity_label
            entity_string = ""
            for j, label in enumerate(query):
                # find entity start point (expand to space character) and extract the continuous entity
                if label != non_entity_label and label != previous_label:
                    entity_start = j
                    while vocab_inv.get(numpy.argmax(data[i, entity_start])) not in string.whitespace and entity_start >= 0:
                        entity_start -= 1
                    # move start point forward one to get out of whitespace or back to 0 index
                    entity_start += 1
                    # now from the start point, extract continuous until whitespace or punctuation
                    entity_idx = entity_start
                    while entity_idx < len(query) and \
                            (
                                query[entity_idx] == label or
                                entity_idx == entity_start or
                                (
                                    entity_idx > entity_start and
                                    vocab_inv.get(numpy.argmax(data[i, entity_idx])) not in string.whitespace + string.punctuation and
                                    vocab_inv.get(numpy.argmax(data[i, entity_idx-1])) not in string.whitespace + string.punctuation
                                )
                            ):
                        entity_string += vocab_inv.get(numpy.argmax(data[i, entity_idx]))
                        entity_idx += 1
                    # get rid of trailing matched punctuation
                    if entity_string[-1] in string.punctuation:
                        entity_string = entity_string[:-1]
                    # add the entity stripped of whitespace in beginning and end, and reset the string
                    entities.append(entity_string.strip())
                    entity_string = ""

                previous_label = label

        return entities

    @staticmethod
    def __process_str(data_str, vocab):
        # process the raw input data string
        data = []
        for data_char in data_str:
            data.append(vocab.get(data_char, 0))

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

        entities = self.__get_entities(data, predictions, self.inverse_vocab, self.entity_vocab)

        return entities

    def getEntities(self, query):
        entities = self.predict(query)
        uniques = set(entities)
        entities = [{'type': a, 'text': t, 'count': entities.count((t, a))} for (t, a) in uniques]
        return entities


if __name__ == "__main__":
    ner = OpenNER()
    print(ner.predict("does the iphone 6 have good battery compared to the galaxy s6?"))
    print(ner.predict("iphone 8 is a pretty cool phone"))
    print(ner.predict("I'm loving this new iphone 1200!!"))
    print(ner.predict(" this iphone is the most fun phone ever made"))
