"""
Character-level recurrent model for named entity recognition.
"""
from __future__ import print_function, division
# normal imports
import string
import numpy
import cPickle as pickle
import theano.tensor as T
import re
# opendeep imports
from opendeep.models import LSTM
from opendeep.optimization import RMSProp
from opendeep.monitor import Monitor, Plot
from opendeep.data import TextDataset
from opendeep.utils.misc import numpy_one_hot

def get_dataset(path_to_data='data/tokenized/'):
    # our input data is going to be .txt files in a folder that are formatted as follows:
    # each line is a new token (word) separated from a class label with a tab character.
    # our preprocessing includes converting to lowercase, splitting into characters, and repeating
    # the label for each character. Because punctuation counts as a word, we are doing special
    # rules with adding spaces around punctuation tokens to build a more accurate language model

    class StringProcessor:
        """
        This is a helper class (normally we would just do functions for preprocessing) to preprocess
        our text files (line by line) into the appropriate input and target data. The class is used
        because we needed to keep track of state when processing line by line.
        """
        def __init__(self):
            self.previous_label = ''
            self.space_before_punct = ['(', '``', '[', '{', '$', '#', '&']
            self.space_after_punct = ['&']
            self.previous_char = ''

        def process_line(self, line):
            chars, label = line.split('\t', 1)
            chars = chars.lower()
            label = label.rstrip()
            labels = [label] * len(chars)
            if (not chars[0] in string.punctuation or chars[0] in self.space_before_punct) and \
                    (not self.previous_char in self.space_before_punct or self.previous_char in self.space_after_punct):
                chars = ' ' + chars
                if label == self.previous_label:
                    labels = [label] + labels
                else:
                    labels = ['O'] + labels

            self.previous_label = label
            self.previous_char = chars[-1]

            return chars, labels

        def get_inputs(self, line):
            return self.process_line(line)[0]

        def get_labels(self, line):
            return self.process_line(line)[1]

    # now that we defined our preprocessor, create a new TextDataset (works over files)
    # a TextDataset is an OpenDeep class that creates one-hot encodings of inputs and outputs automatically
    # and keeps them in vocab and entity_vocab dictionaries.
    processor = StringProcessor()
    dataset = TextDataset(path=path_to_data,
                          inputs_preprocess=lambda line: processor.get_inputs(line),
                          targets_preprocess=lambda line: processor.get_labels(line),
                          level="char", sequence_length=120)

    return dataset


class OpenNER:
    vocab_file = 'vocab.pkl'
    entity_vocab_file = 'entity_vocab.pkl'
    lstm_file = 'trained_epoch_94.pkl'

    def __init__(self):
        self.vocab = pickle.load(open(self.vocab_file, 'rb'))
        self.entity_vocab = pickle.load(open(self.entity_vocab_file, 'rb'))
        #
        # self.data = get_dataset()
        # self.vocab = self.data.vocab
        # self.entity_vocab = self.data.label_vocab
        # # save the computed dictionaries to use for converting inputs and outputs from running the model.
        # with open('vocab.pkl', 'wb') as f:
        #     pickle.dump(self.data.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('entity_vocab.pkl', 'wb') as f:
        #     pickle.dump(self.data.label_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

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

        self.load_params(self.lstm_file)

        # self.lstm.save_run('lstm_run.pkl')

    def train(self):
        # train the lstm on our dataset!
        # let's monitor the error %
        # output is in shape (n_timesteps, n_sequences, data_dim)
        # calculate the mean prediction error over timesteps and batches
        predictions = T.argmax(self.lstm.get_outputs(), axis=2)
        actual = T.argmax(self.lstm.get_targets()[0].dimshuffle(1, 0, 2), axis=2)
        char_error = T.mean(T.neq(predictions, actual))

        # optimizer - RMSProp generally good for recurrent nets, lr taken from Karpathy's char-rnn project.
        # you can also load these configuration arguments from a file or dictionary (parsed from json)
        optimizer = RMSProp(
            dataset=self.data,
            epochs=250,
            batch_size=50,
            save_freq=10,
            learning_rate=2e-3,
            lr_decay="exponential",
            lr_decay_factor=0.97,
            decay=0.95,
            grad_clip=None,
            hard_clip=False
        )

        # monitors
        char_errors = Monitor(name='char_error', expression=char_error, train=True, valid=True, test=True)
        plot = Plot(monitor_channels=[char_errors], bokeh_doc_name='ner')

        self.lstm.train(optimizer=optimizer, plot=plot)

    def load_params(self, params_file):
        self.lstm.load_params(params_file)


    @staticmethod
    def __get_entities(data, predictions, vocab_inv, entity_vocab):
        entity_vocab_inv = {v:k for k,v in entity_vocab.items()}
        # find contiguous entity characters across timesteps
        non_entity_label = entity_vocab.get('O')
        entities = []
        for i, query in enumerate(predictions):
            previous_label = non_entity_label
            entity_string = ""
            used_indices = set()
            for j, label in enumerate(query):
                # find entity start point (expand to space character) and extract the continuous entity
                if label != non_entity_label and label != previous_label and j not in used_indices:
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
                        used_indices.add(entity_idx)
                        entity_idx += 1
                    # get rid of trailing matched punctuation
                    if entity_string[-1] in string.punctuation:
                        entity_string = entity_string[:-1]
                    # add the entity stripped of whitespace in beginning and end, and reset the string
                    entities.append((entity_string.strip(), entity_vocab_inv.get(label)))
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


def __process_str(data_str, vocab):
    # process the raw input data string
    data = []
    for data_char in data_str:
        data.append(vocab.get(data_char, 0))

    data = numpy_one_hot(numpy.asarray(data), n_classes=numpy.amax(vocab.values()) + 1)

    seq, dim = data.shape
    data = numpy.reshape(data, (1, seq, dim))

    return data

if __name__ == "__main__":
    print('hi')
    from opendeep import config_root_logger
    config_root_logger()
    ner = OpenNER()
    print(ner.predict("does the iphone 6 have good battery compared to the galaxy s6?"))
    print(ner.predict("iphone 8 is a pretty cool phone"))
    print(ner.predict("I'm loving this new iphone 1200!!"))
    print(ner.predict(" this iphone is the most fun phone ever made"))
    print(ner.predict("i like the iphone 6 so much that i might not buy a new android"))
    print(ner.predict("6 reasons to get the new iphone"))
    print(ner.predict("i'm switching from android galaxy s6 to apple iphone 6 plus"))
