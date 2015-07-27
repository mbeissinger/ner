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

def get_entities(data, predictions, vocab, entity_vocab):
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


def process_line(line):
    chars, label = line.split('\t', 1)
    chars = chars.lower()
    label = label.rstrip()
    labels = [label] * len(chars)
    space_punct = ['(', '``', '[', '{', '$', '#']
    if not chars[0] in string.punctuation or chars[0] in space_punct:
        chars = ' ' + chars
        labels = ['O'] + labels

    return chars, labels

def get_inputs(line):
    return process_line(line)[0]

def get_labels(line):
    return process_line(line)[1]


def main():
    data = TextDataset(path='data/',
                       inputs_preprocess=lambda line: get_inputs(line),
                       targets_preprocess=lambda line: get_labels(line),
                       level="char", sequence_length=120)

    print(data.vocab)
    print(data.label_vocab)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(data.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('entity_vocab.pkl', 'wb') as f:
        pickle.dump(data.label_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    # define our model! we are using lstm.
    hidden_size = 128

    lstm = LSTM(input_size=len(data.vocab),
                hidden_size=hidden_size,
                output_size=len(data.label_vocab),
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
    # lstm.load_params('trained_epoch_40.pkl')

    # train the lstm on our dataset!
    # let's monitor the error %
    # output is in shape (n_timesteps, n_sequences, data_dim)
    # calculate the mean prediction error over timesteps and batches
    predictions = T.argmax(lstm.get_outputs(), axis=2)
    actual = T.argmax(lstm.get_targets()[0].dimshuffle(1, 0, 2), axis=2)
    char_error = T.mean(T.neq(predictions, actual))

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
    plot = Plot('Chars', monitor_channels=[char_errors], open_browser=True)

    lstm.train(optimizer=optimizer, plot=plot)


if __name__ == "__main__":
    # if you want debugging output from opendeep
    opendeep.config_root_logger()
    # run the experiment!
    main()
