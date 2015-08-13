"""
Character-level recurrent model for named entity recognition.
"""
from __future__ import print_function, division
# normal imports
import numpy
import theano.tensor as T
import pickle
import string
# opendeep imports
import opendeep
from opendeep.data import TextDataset
from opendeep.models import LSTM
from opendeep.monitor import Monitor, Plot
from opendeep.optimization import RMSProp
from opendeep.utils.misc import numpy_one_hot


def get_entities(data, predictions, vocab_inv, entity_vocab):
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

def process_str(data_str, vocab):
    # process the raw input data string

    data = []
    for data_char in data_str:
        if data_char in vocab:
            data.append(vocab[data_char])

    data = numpy_one_hot(numpy.asarray(data), n_classes=numpy.amax(vocab.values()) + 1)

    seq, dim = data.shape
    data = numpy.reshape(data, (1, seq, dim))

    return data

def predict(query, model, vocab, inverse_vocab, entity_vocab):

    data = process_str(query, vocab)

    character_probs = model.run(data)
    # this has the shape (timesteps, batches, data), so swap axes to (batches, timesteps, data)
    character_probs = numpy.swapaxes(character_probs, 0, 1)
    # now extract the guessed entities
    predictions = numpy.argmax(character_probs, axis=2)

    entities = get_entities(data, predictions, inverse_vocab, entity_vocab)

    return entities


class StringProcessor:
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


def main():
    # uncomment this dataset to process text files into dataset
    processor = StringProcessor()
    data = TextDataset(path='data/tokenized/',
                       inputs_preprocess=lambda line: processor.get_inputs(line),
                       targets_preprocess=lambda line: processor.get_labels(line),
                       level="char", sequence_length=120)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(data.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('entity_vocab.pkl', 'wb') as f:
        pickle.dump(data.label_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    vocab = data.vocab
    label_vocab = data.label_vocab

    lstm = LSTM(input_size=len(vocab),
                hidden_size=128,
                output_size=len(label_vocab),
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

    # load params
    # lstm.load_params('trained_lstm.pkl')

    # train the lstm on our dataset!
    # let's monitor the error %
    # output is in shape (n_timesteps, n_sequences, data_dim)
    # calculate the mean prediction error over timesteps and batches
    predictions = T.argmax(lstm.get_outputs(), axis=2)
    actual = T.argmax(lstm.get_targets()[0].dimshuffle(1, 0, 2), axis=2)
    char_error = T.mean(T.neq(predictions, actual))

    # uncomment these lines for training!

    # optimizer - RMSProp generally good for recurrent nets, lr taken from Karpathy's char-rnn project.
    optimizer = RMSProp(
        dataset=data,
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

    # plot the monitor
    # plot = Plot('Chars', monitor_channels=[char_errors], open_browser=True)

    lstm.train(optimizer=optimizer, monitor_channels=[char_errors])

    return lstm, vocab, label_vocab


if __name__ == "__main__":
    # if you want debugging output from opendeep
    opendeep.config_root_logger()

    # run the experiment!
    lstm, vocab, entity_vocab = main()

    inverse_vocab = {v: k for k, v in vocab.items()}

    # predict some stuff
    print(predict("I love the iphone 6!", lstm, vocab, inverse_vocab, entity_vocab))
