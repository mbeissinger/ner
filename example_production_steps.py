"""
Example creating an LSTM for character-level language models.
"""
from __future__ import print_function, division
# normal imports
import numpy
import theano.tensor as T
import pickle
import string
# opendeep imports
from opendeep.data import TextDataset
from opendeep.models import LSTM
from opendeep.monitor import Monitor
from opendeep.optimization import RMSProp
from opendeep.utils.misc import numpy_one_hot


#################
# Training data #
#################
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

    # save the computed dictionaries to use for converting inputs and outputs from running the model.
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(dataset.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('entity_vocab.pkl', 'wb') as f:
        pickle.dump(dataset.label_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


##############################
# Step 1 and 2: create model #
##############################
def create_model(init_config_file=None, vocab={}, label_vocab={}):
    # load from a configuration file, or define the model configuration
    if init_config_file is not None:
        with open(init_config_file, 'rb') as f:
            init_config = pickle.load(f)
    else:
        init_config = {
            'input_size': len(vocab),
            'hidden_size': 128,
            'output_size': len(label_vocab),
            'hidden_activation': 'tanh',
            'inner_hidden_activation': 'sigmoid',
            'activation': 'softmax',
            'weights_init': 'uniform',
            'weights_interval': 'montreal',
            'r_weights_init': 'orthogonal',
            'clip_recurrent_grads': 5.,
            'noise': 'dropout',
            'noise_level': 0.5,
            'direction': 'bidirectional',
            'cost_function': 'nll',
            'cost_args': {'one_hot': True}
        }

    # instantiate the model!
    lstm = LSTM(**init_config)

    return lstm

############################
# Step 2a: train the model #
############################
def train_model(model, dataset):
    # train the lstm on our dataset!
    # let's monitor the error %
    # output is in shape (n_timesteps, n_sequences, data_dim)
    # calculate the mean prediction error over timesteps and batches
    predictions = T.argmax(model.get_outputs(), axis=2)
    actual = T.argmax(model.get_targets()[0].dimshuffle(1, 0, 2), axis=2)
    char_error = T.mean(T.neq(predictions, actual))

    # optimizer - RMSProp generally good for recurrent nets, lr taken from Karpathy's char-rnn project.
    # you can also load these configuration arguments from a file or dictionary (parsed from json)
    optimizer = RMSProp(
        dataset=dataset,
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

    model.train(optimizer=optimizer, monitor_channels=[char_errors])

#################################
# Step 3: load model parameters #
#################################
def load_model_params(model, param_file='outputs/lstm/trained_epoch_10.pkl'):
    # load params
    model.load_params(param_file)

################
# Step 4: compile model's run function (run on dummy data)
################
# parse a string into some input data
def string_to_data(query, vocab):
    # process the raw input data string
    data = []
    # get the integer encodings
    for data_char in query:
        data.append(vocab.get(data_char, 0))

    # convert the integers to one-hot arrays
    data = numpy_one_hot(numpy.asarray(data), n_classes=numpy.amax(vocab.values()) + 1)

    # make 3D for model input
    seq, dim = data.shape
    data = numpy.reshape(data, (1, seq, dim))

    return data


def run_model_dummy(model, data):
    dummy_character_probabilities = model.run(data)

############
# Step 5: run on real data
############
def run_model(model, data, vocab, label_vocab):
    # in our case here, data will be given from the user as a string. we have to process with the encoding
    # and decoding vocabulary to do anything meaningful with the results
    def _get_entities(data, predictions, vocab_inv, entity_vocab):
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
                    while vocab_inv.get(
                            numpy.argmax(data[i, entity_start])) not in string.whitespace and entity_start >= 0:
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
                                    vocab_inv.get(numpy.argmax(data[i, entity_idx - 1])) not in string.whitespace + string.punctuation
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

    data = string_to_data(data, vocab)

    character_probs = model.run(data)
    # this has the shape (timesteps, batches, data), so swap axes to (batches, timesteps, data)
    character_probs = numpy.swapaxes(character_probs, 0, 1)
    # now extract the guessed entities
    predictions = numpy.argmax(character_probs, axis=2)

    entities = _get_entities(data, predictions, {v:k for k,v in vocab.items()}, label_vocab)

    return entities
